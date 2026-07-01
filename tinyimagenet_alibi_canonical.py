"""
tinyimagenet_alibi_canonical.py
================================
Re-trains 1D-ALiBi AND 2D-ALiBi on TinyImageNet under the SAME canonical
protocol used for the CIFAR-100 12-seed ALiBi re-training. This produces
two strictly paired single-seed results so the missing 2D-ALiBi cell in
Table 3 (TinyImageNet row) can be filled with a directly comparable
number, and so the 1D vs. 2D ALiBi comparison on TinyImageNet uses the
identical training code (the only experimental variable being the
distance metric in the ALiBi bias).

Why this exists
---------------
The existing TinyImageNet results in Table 3 (Learned 52.19, Sinusoidal
54.11, RoPE 56.73, 1D-ALiBi 53.41) were produced by
tinyimagenet_experiment.py, which uses a slightly different recipe than
the manuscript describes (it has torch.compile=ON and NO gradient
clipping). To make the 2D-vs-1D ALiBi comparison on TinyImageNet
bit-comparable to the CIFAR-100 ALiBi comparison, this script re-trains
both 1D-ALiBi and 2D-ALiBi under the canonical protocol. The other
three PE types on TinyImageNet are left as single-seed entries in
Table 3 since they are not part of the 2D-vs-1D test.

Canonical protocol (frozen, matches cifar100_alibi_10seeds.py):
  - grad-clip 1.0  (as the manuscript Sec. 4 describes)
  - NaN-loss guard
  - torch.manual_seed + np.random.seed + torch.cuda.manual_seed_all
  - NO torch.compile  (deterministic numerics across paired runs)
  - 300 epochs, batch 128, lr 3e-4, warmup 20, wd 0.1, mixup 0.8,
    label smoothing 0.1, cosine schedule

Output
------
/content/drive/MyDrive/Trained models_TinyImageNet_v2/
    alibi_seed{N}/best_model.pth, training_history.json       (1D-ALiBi)
    alibi_2d_seed{N}/best_model.pth, training_history.json    (2D-ALiBi)

Old results under 'Trained models_TinyImageNet/' are NOT touched.

Compute (G4 / RTX 6000 Blackwell Server Edition)
-------------------------------------------------
~19 h / seed x 2 PE types x N seeds. With SEEDS=[42] (default): ~38 h
total. Easy to extend to [42, 123, 456] by editing the SEEDS list below
(then ~114 h, split across multiple Colab sessions; resume logic skips
already-completed runs).

Prerequisites
-------------
1. /content/full_scale_experiment_v2.py  (output of apply_2d_alibi_patch.py)
   This patched module must expose VisionTransformer accepting
   pe_type='alibi' AND pe_type='alibi_2d'.
2. TinyImageNet must be present at the configured DATA_DIR (or this
   script will download and reorganise it the first time, ~250 MB).
"""

import os
import sys
import json
import math
import time
import zipfile
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Import the PATCHED VisionTransformer (must support 'alibi_2d') ---
sys.path.insert(0, '/content')
from full_scale_experiment_v2 import VisionTransformer

# ============================================================
# CANONICAL CONFIG -- matches CIFAR-100 12-seed protocol and the
# manuscript training recipe (Sec. 4).
# ============================================================
TINYIN_CONFIG = {
    'img_size':         224,    # upscaled from 64 (factor 3.5)
    'patch_size':        16,    # 224/16 = 14x14 = 196 patches  (N=197)
    'num_classes':      200,
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
    'grad_clip':        1.0,    # canonical: ON (matches manuscript)
}

PE_TYPES = ['alibi', 'alibi_2d']   # 1D first, then 2D, paired per seed
SEEDS    = [456]                     # default: single-seed; add 123, 456 to extend

# Paths (NEW v2 dir; original 'Trained models_TinyImageNet' is untouched)
DRIVE_BASE   = '/content/drive/MyDrive'
DATA_DIR     = os.path.join(DRIVE_BASE, 'tinyimagenet')
RESULTS_DIR  = os.path.join(DRIVE_BASE, 'Trained models_TinyImageNet_v2')

TINYIN_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ============================================================
# FAIL-FAST: verify the patched module actually supports 'alibi_2d'.
# Done up-front so we don't waste ~19 h discovering a bad patch.
# ============================================================
print("\n[Sanity] Verifying full_scale_experiment_v2.py supports 'alibi_2d'...")
try:
    _probe = VisionTransformer(
        img_size=TINYIN_CONFIG['img_size'],
        patch_size=TINYIN_CONFIG['patch_size'],
        num_classes=TINYIN_CONFIG['num_classes'],
        embed_dim=TINYIN_CONFIG['embed_dim'],
        depth=2,                         # tiny probe, just to instantiate
        num_heads=TINYIN_CONFIG['num_heads'],
        mlp_ratio=TINYIN_CONFIG['mlp_ratio'],
        dropout=TINYIN_CONFIG['dropout'],
        pe_type='alibi_2d',
    )
    alibi_module = _probe.blocks[0].attn.alibi
    assert alibi_module.__class__.__name__ == 'TwoDALiBi', (
        f"Expected TwoDALiBi, got {alibi_module.__class__.__name__}. "
        f"Patch likely not applied correctly."
    )
    assert hasattr(alibi_module, 'dist_2d'), "TwoDALiBi missing dist_2d buffer"
    expected_N = (TINYIN_CONFIG['img_size'] // TINYIN_CONFIG['patch_size']) ** 2 + 1
    assert alibi_module.dist_2d.shape[-1] == expected_N, (
        f"dist_2d has shape {alibi_module.dist_2d.shape}, expected last "
        f"dim {expected_N} (N=197 for 14x14 grid + CLS)"
    )
    del _probe
    print(f"  OK: patched module supports 'alibi_2d', "
          f"TwoDALiBi is wired (N={expected_N}).")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  Cannot proceed -- re-run apply_2d_alibi_patch.py before training.")
    sys.exit(1)


# ============================================================
# DATA SETUP -- download + reorganise val/ into ImageFolder layout
# (identical to tinyimagenet_experiment.py, kept verbatim so we hit
# the same dataset bytes as the existing TinyImageNet runs)
# ============================================================
def download_and_prepare(data_dir):
    """Download TinyImageNet and reorganise val/ into ImageFolder structure."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path  = os.path.join(data_dir, 'tiny-imagenet-200.zip')
    extracted = os.path.join(data_dir, 'tiny-imagenet-200')

    if os.path.isdir(extracted) and os.path.isdir(
            os.path.join(extracted, 'val_reorganised')):
        print(f"  TinyImageNet already prepared at {extracted}")
        return extracted

    if not os.path.isfile(zip_path) and not os.path.isdir(extracted):
        print(f"  Downloading TinyImageNet (~250MB) -> {zip_path}")
        urllib.request.urlretrieve(TINYIN_URL, zip_path)

    if not os.path.isdir(extracted):
        print(f"  Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)

    val_dir         = os.path.join(extracted, 'val')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')
    val_images_dir  = os.path.join(val_dir, 'images')
    val_reorg_dir   = os.path.join(extracted, 'val_reorganised')

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
    """Build train and val DataLoaders. IDENTICAL to tinyimagenet_experiment.py
    so we use the same bytes as the existing TinyImageNet results."""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(
        os.path.join(extracted_dir, 'train'), transform=train_transform)
    val_ds = datasets.ImageFolder(
        os.path.join(extracted_dir, 'val_reorganised'), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=(num_workers > 0),
                                drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=(num_workers > 0))
    return train_loader, val_loader


# ============================================================
# TRAINING UTILITIES (canonical -- identical to cifar100_alibi_12seeds.py)
# ============================================================
def set_all_seeds(seed):
    """Set every RNG source so per-seed paired runs are deterministic."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_one_config(pe_type, seed, cfg, train_loader, val_loader):
    """Train one (pe_type, seed) configuration end-to-end."""
    set_all_seeds(seed)

    model = VisionTransformer(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'],
        num_classes=cfg['num_classes'],
        embed_dim=cfg['embed_dim'], depth=cfg['depth'],
        num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'],
        dropout=cfg['dropout'], pe_type=pe_type,
    ).to(device)

    # NOTE: deliberately NO torch.compile -- keeps numerics deterministic
    #       and bit-comparable between 1D-ALiBi and 2D-ALiBi at the same seed.

    optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=cfg['lr'],
                                    weight_decay=cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    # Cosine schedule via LambdaLR (matches CIFAR canonical path)
    def lr_lambda(epoch):
        if epoch < cfg['warmup_epochs']:
            return (epoch + 1) / cfg['warmup_epochs']
        progress = (epoch - cfg['warmup_epochs']) / max(
            1, cfg['epochs'] - cfg['warmup_epochs'])
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = os.path.join(RESULTS_DIR, f"{pe_type}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    history = {'train_loss': [], 'val_acc': [], 'epoch_time': []}
    best_acc = 0.0

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        model.train()
        running_loss, n_batches = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed_x, y_a, y_b, lam = mixup_data(
                images, labels, alpha=cfg['mixup_alpha'])
            out = model(mixed_x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

            # NaN guard (canonical)
            if torch.isnan(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Grad-clip (canonical -- ON, matches manuscript Sec. 4)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg['grad_clip'])
            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1

        scheduler.step()
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

        current_lr = scheduler.get_last_lr()[0]
        print(f"    Epoch {epoch:3d}/{cfg['epochs']}: loss={train_loss:.3f} "
              f"acc={val_acc:.2f}% best={best_acc:.2f}% "
              f"({epoch_time:.1f}s, lr={current_lr:.5f})", flush=True)

        # Save history incrementally so partial runs are recoverable
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)

    print(f"  Done. Best val acc: {best_acc:.2f}%")
    return best_acc


def already_complete(pe_type, seed, total_epochs):
    """Return (True, n_epochs) if this (pe_type, seed) finished all epochs."""
    save_dir = os.path.join(RESULTS_DIR, f"{pe_type}_seed{seed}")
    hist_path = os.path.join(save_dir, 'training_history.json')
    ckpt_path = os.path.join(save_dir, 'best_model.pth')
    if not (os.path.isfile(hist_path) and os.path.isfile(ckpt_path)):
        return False, 0
    try:
        with open(hist_path) as f:
            h = json.load(f)
        n_epochs = len(h.get('val_acc', []))
        return (n_epochs >= total_epochs), n_epochs
    except Exception:
        return False, 0


# ============================================================
# MAIN -- seed-major loop so per-seed (1D, 2D) pairs run back-to-back
# ============================================================
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_runs = len(PE_TYPES) * len(SEEDS)
    print("\n" + "=" * 70)
    print(f"  Training {PE_TYPES} x {len(SEEDS)} seeds on TinyImageNet")
    print(f"  Total runs: {total_runs}  (paired by seed)")
    print("=" * 70)
    print(f"  Patch grid: 14x14 (196 patches, N=197 with CLS)")
    print(f"  Image size: {TINYIN_CONFIG['img_size']} (upsampled from 64)")
    print(f"  Epochs/seed: {TINYIN_CONFIG['epochs']}")
    print(f"  Grad clip: {TINYIN_CONFIG['grad_clip']}")
    print(f"  Output: {RESULTS_DIR}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    # 1) Prepare dataset
    extracted = download_and_prepare(DATA_DIR)

    # 2) Loaders (built once -- both PE types share the same data pipeline)
    print(f"\nBuilding data loaders ...")
    train_loader, val_loader = get_loaders(
        extracted, TINYIN_CONFIG['batch_size'], num_workers=4)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3) Training -- seed-major (pairs (1D, 2D) at the same seed back-to-back)
    summary = {}
    run_idx = 0
    for seed in SEEDS:
        for pe_type in PE_TYPES:
            run_idx += 1
            tag = f"{pe_type}_seed{seed}"

            done, n_epochs = already_complete(
                pe_type, seed, TINYIN_CONFIG['epochs'])
            if done:
                save_dir = os.path.join(RESULTS_DIR, tag)
                with open(os.path.join(save_dir,
                                        'training_history.json')) as f:
                    h = json.load(f)
                acc = max(h['val_acc'])
                print(f"\n[{run_idx}/{total_runs}] {tag}: "
                      f"already complete ({n_epochs} epochs, best={acc:.2f}%) "
                      f"-- skipping")
                summary[tag] = {'best_acc': acc, 'status': 'cached'}
                continue

            if n_epochs > 0:
                print(f"\n[{run_idx}/{total_runs}] {tag}: partial "
                      f"({n_epochs} epochs) -- restarting from scratch")

            print(f"\n[{run_idx}/{total_runs}] >>> Training {tag}")
            t_start = time.time()
            try:
                best_acc = train_one_config(
                    pe_type, seed, TINYIN_CONFIG, train_loader, val_loader)
                elapsed = time.time() - t_start
                summary[tag] = {
                    'best_acc': best_acc,
                    'wall_time_hours': elapsed / 3600.0,
                    'status': 'trained',
                }
                print(f"  Wall time: {elapsed/3600:.2f}h")
            except Exception as e:
                print(f"  FAILED: {e}")
                summary[tag] = {'status': 'failed', 'error': str(e)}
                continue

            with open(os.path.join(
                    RESULTS_DIR, '_alibi_canonical_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    # 4) Final summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for tag, info in summary.items():
        if info.get('status') == 'failed':
            print(f"  {tag:25s}: FAILED ({info.get('error', '?')})")
        else:
            t = info.get('wall_time_hours', 0)
            print(f"  {tag:25s}: {info['best_acc']:6.2f}%  "
                  f"({t:.1f}h, {info.get('status', '?')})")

    with open(os.path.join(
            RESULTS_DIR, '_alibi_canonical_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to "
          f"{os.path.join(RESULTS_DIR, '_alibi_canonical_summary.json')}")
