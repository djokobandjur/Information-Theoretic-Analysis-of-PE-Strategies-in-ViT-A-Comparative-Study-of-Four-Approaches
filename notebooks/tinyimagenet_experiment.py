#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TinyImageNet Training Script for PE Strategy Comparison
=============================================================================

Trains ViT-Base on TinyImageNet for the four PE strategies (Learned,
Sinusoidal, RoPE, ALiBi) across three seeds (42, 123, 456), following
the IDENTICAL configuration used for ImageNet-100 training to enable
direct comparison.

Dataset:    TinyImageNet (200 classes, native 64x64 -> upscaled to 224x224)
            ~100K train / ~10K val images
Model:      ViT-Base (12 layers, 768 dim, 12 heads, patch 16x16 -> 196 patches)
Resolution: 224x224 (upscale from 64x64 by factor 3.5)
            Rationale: keeps the SAME architecture as IN-100, enables
            direct cross-dataset comparison, much milder upscale than
            CIFAR-100 (32 -> 224 = 7x).
Seeds:      3 independent runs per configuration
PE types:   Learned, Sinusoidal, RoPE, ALiBi (12 models total)

Compute:    NVIDIA RTX PRO 6000 Blackwell Server Edition, ~18-19h per model 

Usage:
    # First time only -- download dataset (~250MB, into Drive for reuse)
    python tinyimagenet_experiment.py --download_only

    # Train all 12 models (4 PE x 3 seeds) sequentially
    python tinyimagenet_experiment.py

    # Train a single configuration
    python tinyimagenet_experiment.py --pe_type rope --seed 42

    # Resume from previous run (skip already-trained models)
    python tinyimagenet_experiment.py --resume
=============================================================================
"""

import os
import sys
import json
import math
import time
import argparse
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import model classes from the patched main script. This expects
# full_scale_experiment.py to be in the same directory (or on PYTHONPATH)
# and to have been patched with the 2D-ALiBi additions (although we do
# not need alibi_2d here -- the original 4 PE methods are sufficient).
from full_scale_experiment import VisionTransformer


# ============================================================================
# CONFIG  (matches ImageNet-100 training EXACTLY, except dataset paths)
# ============================================================================

TINYIN_CONFIG = {
    'img_size':         224,    # upscaled from 64 (factor 3.5)
    'patch_size':        16,    # 224/16 = 14x14 = 196 patches  -- same as IN-100
    'num_classes':      200,    # TinyImageNet has 200 classes
    'embed_dim':        768,
    'depth':             12,
    'num_heads':         12,
    'mlp_ratio':        4.0,
    'dropout':          0.1,
    'epochs':           300,
    'batch_size':       128,
    'lr':              3e-4,
    'warmup_epochs':     20,
    'weight_decay':     0.1,
    'label_smoothing':  0.1,
    'mixup_alpha':      0.8,
}

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS    = [42, 123, 456]

# Paths
DRIVE_BASE   = '/content/drive/MyDrive'
DATA_DIR     = os.path.join(DRIVE_BASE, 'tinyimagenet')           # dataset
RESULTS_DIR  = os.path.join(DRIVE_BASE, 'Trained models_TinyImageNet')  # checkpoints

# Dataset URL
TINYIN_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA SETUP -- download and reorganize TinyImageNet for ImageFolder
# ============================================================================

def download_and_prepare(data_dir):
    """Download TinyImageNet and reorganize val/ into ImageFolder structure."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path  = os.path.join(data_dir, 'tiny-imagenet-200.zip')
    extracted = os.path.join(data_dir, 'tiny-imagenet-200')

    if os.path.isdir(extracted) and os.path.isdir(os.path.join(extracted, 'val_reorganised')):
        print(f"  TinyImageNet already prepared at {extracted}")
        return extracted

    if not os.path.isfile(zip_path) and not os.path.isdir(extracted):
        print(f"  Downloading TinyImageNet (~250MB) -> {zip_path}")
        urllib.request.urlretrieve(TINYIN_URL, zip_path)

    if not os.path.isdir(extracted):
        print(f"  Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)

    # The val/ split comes as a flat folder of images plus a TSV mapping
    # file. We need it in ImageFolder layout: val_reorganised/{class}/{img}.JPEG
    val_dir          = os.path.join(extracted, 'val')
    val_annotations  = os.path.join(val_dir, 'val_annotations.txt')
    val_images_dir   = os.path.join(val_dir, 'images')
    val_reorg_dir    = os.path.join(extracted, 'val_reorganised')

    if not os.path.isdir(val_reorg_dir):
        print(f"  Reorganising val split into ImageFolder layout ...")
        os.makedirs(val_reorg_dir, exist_ok=True)
        with open(val_annotations) as f:
            for line in f:
                fname, cls = line.strip().split('\t')[:2]
                cls_dir = os.path.join(val_reorg_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(val_images_dir, fname)
                dst = os.path.join(cls_dir, fname)
                if os.path.isfile(src) and not os.path.isfile(dst):
                    os.link(src, dst)  # hardlink; saves disk space

    # Sanity-check
    train_dir = os.path.join(extracted, 'train')
    n_train_classes = len([d for d in os.listdir(train_dir)
                            if os.path.isdir(os.path.join(train_dir, d))])
    n_val_classes   = len([d for d in os.listdir(val_reorg_dir)
                            if os.path.isdir(os.path.join(val_reorg_dir, d))])
    print(f"  Train classes: {n_train_classes}, Val classes: {n_val_classes}")
    assert n_train_classes == 200 and n_val_classes == 200, \
        "Expected 200 classes in both splits"

    return extracted


def get_loaders(extracted_dir, batch_size, num_workers):
    """Build train and val DataLoaders with the same augmentations as
    the ImageNet-100 training, but resizing 64x64 -> 224x224."""
    # NOTE: TinyImageNet train images are 64x64. We resize to 224x224 to
    # keep the SAME architecture as IN-100 (patch_size=16, 14x14 grid).
    # Per-channel statistics are from ImageNet (since we're upscaling
    # natural images and the augmentation pipeline expects this).
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),  # mild crop after upscale
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(
        os.path.join(extracted_dir, 'train'),
        transform=train_transform,
    )
    # Use the reorganised val
    val_ds = datasets.ImageFolder(
        os.path.join(extracted_dir, 'val_reorganised'),
        transform=val_transform,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=(num_workers > 0),
                                drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=(num_workers > 0))
    return train_loader, val_loader


# ============================================================================
# TRAINING UTILITIES  (matches IN-100 training)
# ============================================================================

def mixup_data(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def cosine_schedule(epoch, warmup, total, base_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_config(pe_type, seed, cfg, train_loader, val_loader, save_dir):
    """Train one (pe_type, seed) configuration to convergence."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = VisionTransformer(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'],
        num_classes=cfg['num_classes'],
        embed_dim=cfg['embed_dim'], depth=cfg['depth'],
        num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'],
        dropout=cfg['dropout'], pe_type=pe_type,
    ).to(device)

    # torch.compile for speed (matches IN-100 setup)
    try:
        model = torch.compile(model)
        print(f"  torch.compile enabled")
    except Exception as e:
        print(f"  torch.compile NOT available: {e}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=cfg['lr'],
                                    weight_decay=cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    os.makedirs(save_dir, exist_ok=True)
    history = {'train_loss': [], 'val_acc': [], 'epoch_time': []}
    best_acc = 0.0

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        model.train()
        lr = cosine_schedule(epoch - 1, cfg['warmup_epochs'],
                              cfg['epochs'], cfg['lr'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        running_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), \
                              labels.to(device, non_blocking=True)
            mixed_x, y_a, y_b, lam = mixup_data(images, labels,
                                                  alpha=cfg['mixup_alpha'])
            out = model(mixed_x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        train_loss = running_loss / max(1, n_batches)
        val_acc    = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                        os.path.join(save_dir, 'best_model.pth'))

        # Print every epoch with explicit flush so Colab can stream it
        print(f"    Epoch {epoch:3d}/{cfg['epochs']}: loss={train_loss:.3f} "
              f"acc={val_acc:.2f}% best={best_acc:.2f}% ({epoch_time:.1f}s, "
              f"lr={lr:.5f})", flush=True)

        # Also incrementally write training_history.json after every epoch
        # so progress is recoverable if Colab disconnects mid-run.
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)

    print(f"  Done. Best val acc: {best_acc:.2f}%", flush=True)
    return best_acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TinyImageNet PE comparison')
    parser.add_argument('--data_dir',   type=str, default=DATA_DIR)
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--pe_type',    type=str, default=None,
                          choices=PE_TYPES)
    parser.add_argument('--seed',       type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=TINYIN_CONFIG['batch_size'])
    parser.add_argument('--epochs',     type=int, default=TINYIN_CONFIG['epochs'])
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--download_only', action='store_true',
                          help='Download dataset and exit')
    parser.add_argument('--resume', action='store_true',
                          help='Skip configurations with existing best_model.pth')
    args = parser.parse_args()

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    cfg = TINYIN_CONFIG.copy()
    cfg['batch_size'] = args.batch_size
    cfg['epochs']     = args.epochs

    print(f"Config: {json.dumps(cfg, indent=2)}")

    # 1) Prepare dataset
    extracted = download_and_prepare(args.data_dir)
    if args.download_only:
        print("Download complete. Exiting (--download_only).")
        return

    # 2) Loaders
    print(f"Building data loaders ...")
    train_loader, val_loader = get_loaders(extracted, args.batch_size,
                                             args.num_workers)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3) Decide configurations to run
    pe_list   = [args.pe_type] if args.pe_type else PE_TYPES
    seed_list = [args.seed]    if args.seed    else SEEDS

    # 4) Train each config
    summary = {}
    for pe_type in pe_list:
        for seed in seed_list:
            tag      = f"{pe_type}_seed{seed}"
            save_dir = os.path.join(args.output_dir, tag)
            best_pth = os.path.join(save_dir, 'best_model.pth')

            if args.resume and os.path.isfile(best_pth):
                print(f"\n>>> Skipping {tag} (best_model.pth exists)")
                continue

            print(f"\n>>> Training {tag}")
            print(f"    output: {save_dir}")
            t_start = time.time()
            best_acc = train_one_config(pe_type, seed, cfg,
                                          train_loader, val_loader, save_dir)
            elapsed = time.time() - t_start
            summary[tag] = {'best_acc': best_acc,
                            'wall_time_seconds': elapsed,
                            'wall_time_hours': elapsed / 3600.0}
            print(f"  Wall time: {elapsed/3600:.2f}h")

    # 5) Print summary
    if summary:
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for tag, info in summary.items():
            print(f"  {tag:25s}: {info['best_acc']:6.2f}%  "
                  f"({info['wall_time_hours']:.1f}h)")

        with open(os.path.join(args.output_dir, '_training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
