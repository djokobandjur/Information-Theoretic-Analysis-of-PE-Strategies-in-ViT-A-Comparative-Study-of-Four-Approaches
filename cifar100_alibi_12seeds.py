"""
cifar100_alibi_12seeds.py
=========================
Re-trains 1D-ALiBi-style AND 2D-ALiBi-style on CIFAR-100 across 12 seeds, under
a SINGLE canonical training protocol so that per-seed pairs are
protocol-matched and directly comparable (same code path, same seed set,
and the intended difference being the distance metric in the attention bias).

Why this exists
---------------
The original CIFAR-100 1D-ALiBi-style run was trained via cifar100_experiment.py
(with grad-clip 1.0, no torch.compile, no cuda.manual_seed_all). The
original CIFAR-100 2D-ALiBi-style run was trained via cifar100_2d_alibi_only.py
(no grad-clip, with torch.compile, with cuda.manual_seed_all). The
manuscript comparison requires the only intended intervention between
1D and 2D ALiBi-style to be the distance metric; this script enforces that
by running both variants through the same canonical code path.

The canonical protocol (frozen for ALL 24 runs):
  - grad-clip 1.0  (as the manuscript describes for 1D-ALiBi)
  - NaN-loss guard (same as cifar100_experiment.py)
  - torch.manual_seed + np.random.seed + torch.cuda.manual_seed_all
  - NO torch.compile (deterministic numerics across runs)
  - 300 epochs, batch 128, lr 3e-4, warmup 20, wd 0.1, mixup 0.8,
    label smoothing 0.1, cosine schedule

Output
------
/content/drive/My Drive/pe_experiment/results_cifar100_v2/
    alibi_seed{N}/best_model.pth, training_history.json       (1D-ALiBi-style)
    alibi_2d_seed{N}/best_model.pth, training_history.json    (2D-ALiBi-style)
    _alibi_12seeds_summary.json

Old results in results_cifar100/ are NOT touched.

Compute
-------
~3.7 h per run x 24 runs. Has resume logic: any run already
completed (training_history.json with >= 300 epochs) is skipped, so this
script can be interrupted and re-launched across multiple Colab sessions.

Prerequisites
-------------
1. /content/full_scale_experiment_v2.py  (output of apply_2d_alibi_patch.py)
   This patched module must expose VisionTransformer that accepts
   pe_type='alibi' AND pe_type='alibi_2d'.
2. CIFAR-100 (auto-downloads via torchvision)
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Import the PATCHED VisionTransformer (must support 'alibi_2d') ---
sys.path.insert(0, '/content')
from full_scale_experiment_v2 import VisionTransformer

# ============================================================
# CANONICAL CONFIG -- matches manuscript Section 4 for CIFAR-100
# ============================================================
CIFAR_CONFIG = {
    'img_size':         32,
    'patch_size':        4,        # 32/4 = 8x8 = 64 patches; N=65 with CLS
    'num_classes':     100,
    'embed_dim':       768,
    'depth':            12,
    'num_heads':        12,
    'mlp_ratio':        4.0,
    'dropout':          0.1,
    'epochs':          300,
    'batch_size':      128,
    'lr':              3e-4,
    'warmup_epochs':    20,
    'weight_decay':     0.1,
    'label_smoothing':  0.1,
    'mixup_alpha':      0.8,
    'grad_clip':        1.0,       # canonical: ON (matches 1D-ALiBi original)
}

PE_TYPES = ['alibi', 'alibi_2d']   # 1D first, then 2D, per-seed paired
SEEDS    = [42, 123, 456, 1, 5, 7, 11, 13, 21, 99, 2024, 31337]  # 12 canonical seeds

SUMMARY_FILENAME = '_alibi_12seeds_summary.json'

DRIVE_BASE  = '/content/drive/My Drive/pe_experiment'
RESULTS_DIR = os.path.join(DRIVE_BASE, 'results_cifar100_v2')   # NEW dir
DATA_DIR    = '/content/cifar100'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ============================================================
# FAIL-FAST: verify the patched module actually supports alibi_2d
# ============================================================
print("\n[Sanity] Verifying full_scale_experiment_v2.py supports 'alibi_2d'...")
try:
    _probe = VisionTransformer(
        img_size=CIFAR_CONFIG['img_size'],
        patch_size=CIFAR_CONFIG['patch_size'],
        num_classes=CIFAR_CONFIG['num_classes'],
        embed_dim=CIFAR_CONFIG['embed_dim'],
        depth=2,                         # tiny probe, just to instantiate
        num_heads=CIFAR_CONFIG['num_heads'],
        mlp_ratio=CIFAR_CONFIG['mlp_ratio'],
        dropout=CIFAR_CONFIG['dropout'],
        pe_type='alibi_2d',
    )
    # Verify the 2D-ALiBi class is actually wired up (not silently fallback)
    alibi_module = _probe.blocks[0].attn.alibi
    assert alibi_module.__class__.__name__ == 'TwoDALiBi', (
        f"Expected TwoDALiBi, got {alibi_module.__class__.__name__}. "
        f"Patch likely not applied correctly."
    )
    # Verify 2D distance buffer exists and is the right shape
    assert hasattr(alibi_module, 'dist_2d'), "TwoDALiBi missing dist_2d buffer"
    expected_N = (CIFAR_CONFIG['img_size'] // CIFAR_CONFIG['patch_size']) ** 2 + 1
    assert alibi_module.dist_2d.shape[-1] == expected_N, (
        f"dist_2d has shape {alibi_module.dist_2d.shape}, expected last "
        f"dim {expected_N}"
    )
    del _probe
    print("  OK: patched module supports 'alibi_2d', TwoDALiBi is wired.")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  Cannot proceed -- re-run apply_2d_alibi_patch.py before training.")
    sys.exit(1)


# ============================================================
# DATA LOADERS (identical to cifar100_experiment.py)
# ============================================================
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD  = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_dataset = datasets.CIFAR100(DATA_DIR, train=True,  download=True,
                                    transform=train_transform)
val_dataset   = datasets.CIFAR100(DATA_DIR, train=False, download=True,
                                    transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=CIFAR_CONFIG['batch_size'],
                           shuffle=True,  num_workers=4, pin_memory=True,
                           persistent_workers=True, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=CIFAR_CONFIG['batch_size'],
                           shuffle=False, num_workers=4, pin_memory=True,
                           persistent_workers=True)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# ============================================================
# TRAINING UTILITIES (canonical -- merges both old scripts)
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


def train_one_config(pe_type, seed, cfg):
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

    # Cosine schedule via LambdaLR -- matches cifar100_experiment.py
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

            # NaN guard (canonical -- matches cifar100_experiment.py)
            if torch.isnan(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Grad-clip (canonical -- ON, matches manuscript Sec. 4 for 1D)
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
    """Return True if this (pe_type, seed) finished all epochs."""
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
# MAIN -- iterate seed-major so per-seed pairs train back-to-back
# ============================================================
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_runs = len(PE_TYPES) * len(SEEDS)
    print("\n" + "=" * 70)
    print(f"  Training {PE_TYPES} x {len(SEEDS)} seeds on CIFAR-100")
    print(f"  Total runs: {total_runs}  (paired by seed)")
    print("=" * 70)
    print(f"  Patch grid: 8x8 (64 patches, N=65 with CLS)")
    print(f"  Epochs/seed: {CIFAR_CONFIG['epochs']}")
    print(f"  Grad clip: {CIFAR_CONFIG['grad_clip']}")
    print(f"  Output: {RESULTS_DIR}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    summary = {}
    run_idx = 0

    # Seed-major loop: train (1D@seed_k, 2D@seed_k) back-to-back.
    # This minimises the "elapsed wall time" between paired runs, which
    # keeps the comparison clean even under CUDA driver / library updates.
    for seed in SEEDS:
        for pe_type in PE_TYPES:
            run_idx += 1
            tag = f"{pe_type}_seed{seed}"

            done, n_epochs = already_complete(
                pe_type, seed, CIFAR_CONFIG['epochs'])
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
                best_acc = train_one_config(pe_type, seed, CIFAR_CONFIG)
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
                # Continue to next config rather than abort the whole batch
                continue

            # Save summary incrementally after each run
            with open(os.path.join(RESULTS_DIR,
                                    SUMMARY_FILENAME), 'w') as f:
                json.dump(summary, f, indent=2)

    # ----- Final summary -----
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

    with open(os.path.join(RESULTS_DIR,
                            SUMMARY_FILENAME), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to "
          f"{os.path.join(RESULTS_DIR, SUMMARY_FILENAME)}")
