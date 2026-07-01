"""
cifar100_2d_alibi_matched_12seeds.py
=====================================
Train a magnitude-matched 2D-ALiBi control on CIFAR-100 across the same
12 canonical seeds used for the paired 1D-vs-2D ALiBi analysis.

Purpose
-------
This script tests whether the observed 2D-ALiBi effects are due to the
2D Euclidean topology itself or partly due to weaker effective bias magnitude.
It keeps the 2D Euclidean distance matrix but scales the ALiBi slopes so that
mean patch-patch bias magnitude matches 1D-ALiBi on the same 8x8 CIFAR grid.

Implemented control
-------------------
Standard 1D-ALiBi:
    bias_1d(i,j,h) = - slope_h * |i - j|

Standard 2D-ALiBi:
    bias_2d(i,j,h) = - slope_h * d_2d(i,j)

Magnitude-matched 2D-ALiBi:
    bias_2d_matched(i,j,h) = - slope_h * alpha * d_2d(i,j)

where alpha is computed from the patch-patch distance matrices:
    alpha = mean_{i,j patches} |i-j| / mean_{i,j patches} d_2d(i,j)

For CIFAR-100 with img_size=32 and patch_size=4, the patch grid is 8x8 and
alpha ~= 5.1561. The CLS row/column is not used for matching because the
primary MI estimator is CLS-excluded patch-only.

Protocol
--------
This script follows cifar100_alibi_12seeds.py's canonical CIFAR-100 protocol:
  - 300 epochs, batch 128, lr 3e-4, warmup 20, weight decay 0.1
  - mixup 0.8, label smoothing 0.1
  - grad-clip 1.0
  - NaN-loss guard
  - torch.manual_seed + np.random.seed + torch.cuda.manual_seed_all
  - NO torch.compile

Output
------
/content/drive/My Drive/pe_experiment/results_cifar100_v2/
    alibi_2d_matched_seed{N}/best_model.pth
    alibi_2d_matched_seed{N}/training_history.json
    alibi_2d_matched_seed{N}/match_metadata.json
    alibi_2d_matched_seed{N}/last_checkpoint.pth

The checkpoint is compatible with pe_type='alibi_2d': instantiate the model as
alibi_2d and load the state dict. The scaled slopes are stored in the state dict.

Prerequisite
------------
/content/full_scale_experiment_v2.py must exist and support pe_type='alibi_2d'
via the TwoDALiBi patch.

Usage in Colab
--------------
    !python /content/cifar100_2d_alibi_matched_12seeds.py

Optional quick smoke test:
    set CIFAR_CONFIG['epochs'] = 1 temporarily and run one seed.
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Import the PATCHED VisionTransformer (must support 'alibi_2d') ---
sys.path.insert(0, '/content')
try:
    from full_scale_experiment_v2 import VisionTransformer
except Exception as e:
    raise RuntimeError(
        "Could not import VisionTransformer from /content/full_scale_experiment_v2.py. "
        "Run/apply apply_2d_alibi_patch.py first so that pe_type='alibi_2d' is supported."
    ) from e


# ============================================================
# CANONICAL CIFAR-100 CONFIG
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
    'grad_clip':        1.0,
}

# We train only the new matched control. The internal model is alibi_2d;
# the public output folder name marks it as matched.
PUBLIC_PE_TYPE = 'alibi_2d_matched'
INTERNAL_MODEL_PE_TYPE = 'alibi_2d'

SEEDS = [1, 5, 7, 11, 13, 21, 42, 99, 123, 456, 2024, 31337]

DRIVE_BASE  = '/content/drive/My Drive/pe_experiment'
RESULTS_DIR = os.path.join(DRIVE_BASE, 'results_cifar100_v2')
DATA_DIR    = '/content/cifar100'

# Save a resumable checkpoint after every epoch. This is useful on Colab.
SAVE_LAST_CHECKPOINT_EVERY_EPOCH = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=CIFAR_CONFIG['epochs'])
    parser.add_argument('--seeds', type=str, default=','.join(map(str, SEEDS)),
                        help='Comma-separated seeds, e.g. 1,5,7 or all default 12.')
    parser.add_argument('--force_restart', action='store_true',
                        help='Ignore last_checkpoint.pth and restart incomplete runs from scratch.')
    parser.add_argument('--no_epoch_resume', action='store_true',
                        help='Do not resume from last_checkpoint.pth; still skips completed runs.')
    return parser.parse_args()


args = parse_args()
CIFAR_CONFIG['epochs'] = args.epochs
SEEDS = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
RESULTS_DIR = args.results_dir
DATA_DIR = args.data_dir


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ============================================================
# DISTANCE-MATCHING UTILITIES
# ============================================================
def compute_patch_distance_match_scale(img_size: int, patch_size: int) -> Dict[str, float]:
    """Compute alpha = mean 1D raster distance / mean 2D Euclidean distance.

    Uses patch-patch pairs only. Diagonal zeros are included; including or
    excluding them gives the same ratio because both distance matrices have
    zero diagonal and the same number of pairs.
    """
    side = img_size // patch_size
    assert side * patch_size == img_size, "img_size must be divisible by patch_size"
    n = side * side

    coords = np.array([(i // side, i % side) for i in range(n)], dtype=np.float64)
    raster_dists = []
    euclid_dists = []
    for i in range(n):
        for j in range(n):
            raster_dists.append(abs(i - j))
            euclid_dists.append(float(np.linalg.norm(coords[i] - coords[j])))

    mean_1d = float(np.mean(raster_dists))
    mean_2d = float(np.mean(euclid_dists))
    alpha = mean_1d / mean_2d
    return {
        'grid_side': side,
        'num_patches': n,
        'mean_patch_patch_1d_raster_distance': mean_1d,
        'mean_patch_patch_2d_euclidean_distance': mean_2d,
        'match_scale_alpha': float(alpha),
    }


MATCH = compute_patch_distance_match_scale(
    CIFAR_CONFIG['img_size'], CIFAR_CONFIG['patch_size'])
ALIBI_2D_MATCH_SCALE = MATCH['match_scale_alpha']

print("\n[Magnitude matching]")
print(f"  grid: {MATCH['grid_side']}x{MATCH['grid_side']} patches")
print(f"  mean 1D raster distance: {MATCH['mean_patch_patch_1d_raster_distance']:.6f}")
print(f"  mean 2D Euclidean distance: {MATCH['mean_patch_patch_2d_euclidean_distance']:.6f}")
print(f"  alpha = mean_1d / mean_2d = {ALIBI_2D_MATCH_SCALE:.9f}")


def scale_2d_alibi_slopes_(model: nn.Module, scale: float) -> None:
    """In-place slope scaling for every TwoDALiBi block."""
    with torch.no_grad():
        n_scaled = 0
        for block_idx, block in enumerate(model.blocks):
            if not hasattr(block.attn, 'alibi'):
                raise RuntimeError(f"Block {block_idx} has no attn.alibi module")
            alibi = block.attn.alibi
            if not hasattr(alibi, 'dist_2d'):
                raise RuntimeError(
                    f"Block {block_idx} ALiBi module has no dist_2d; expected TwoDALiBi."
                )
            alibi.slopes.mul_(scale)
            n_scaled += 1
    if n_scaled != model.depth:
        raise RuntimeError(f"Scaled {n_scaled} blocks, expected {model.depth}")


def collect_first_layer_slope_info(model: nn.Module) -> Dict[str, object]:
    alibi = model.blocks[0].attn.alibi
    slopes = alibi.slopes.detach().cpu().view(-1).numpy().tolist()
    return {
        'first_layer_slopes': [float(x) for x in slopes],
        'first_layer_slope_min': float(np.min(slopes)),
        'first_layer_slope_max': float(np.max(slopes)),
    }


# ============================================================
# FAIL-FAST SANITY CHECKS
# ============================================================
print("\n[Sanity] Verifying full_scale_experiment_v2.py supports 'alibi_2d'...")
try:
    _probe = VisionTransformer(
        img_size=CIFAR_CONFIG['img_size'],
        patch_size=CIFAR_CONFIG['patch_size'],
        num_classes=CIFAR_CONFIG['num_classes'],
        embed_dim=CIFAR_CONFIG['embed_dim'],
        depth=2,
        num_heads=CIFAR_CONFIG['num_heads'],
        mlp_ratio=CIFAR_CONFIG['mlp_ratio'],
        dropout=CIFAR_CONFIG['dropout'],
        pe_type=INTERNAL_MODEL_PE_TYPE,
    )
    _alibi = _probe.blocks[0].attn.alibi
    assert _alibi.__class__.__name__ == 'TwoDALiBi', (
        f"Expected TwoDALiBi, got {_alibi.__class__.__name__}. Patch likely not applied."
    )
    assert hasattr(_alibi, 'dist_2d'), "TwoDALiBi missing dist_2d buffer"
    expected_N = (CIFAR_CONFIG['img_size'] // CIFAR_CONFIG['patch_size']) ** 2 + 1
    assert _alibi.dist_2d.shape[-1] == expected_N, (
        f"dist_2d has shape {_alibi.dist_2d.shape}, expected last dim {expected_N}"
    )

    _before = collect_first_layer_slope_info(_probe)
    scale_2d_alibi_slopes_(_probe, ALIBI_2D_MATCH_SCALE)
    _after = collect_first_layer_slope_info(_probe)
    ratio_check = _after['first_layer_slope_max'] / _before['first_layer_slope_max']
    assert abs(ratio_check - ALIBI_2D_MATCH_SCALE) < 1e-5, (
        f"Slope scaling failed: got ratio {ratio_check}, expected {ALIBI_2D_MATCH_SCALE}"
    )
    del _probe
    print("  OK: alibi_2d exists and matched slope scaling works.")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  Cannot proceed -- re-run/apply apply_2d_alibi_patch.py before training.")
    sys.exit(1)


# ============================================================
# DATA LOADERS
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

train_dataset = datasets.CIFAR100(DATA_DIR, train=True, download=True,
                                  transform=train_transform)
val_dataset   = datasets.CIFAR100(DATA_DIR, train=False, download=True,
                                  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=CIFAR_CONFIG['batch_size'],
                          shuffle=True, num_workers=4, pin_memory=True,
                          persistent_workers=True, drop_last=False)
val_loader   = DataLoader(val_dataset, batch_size=CIFAR_CONFIG['batch_size'],
                          shuffle=False, num_workers=4, pin_memory=True,
                          persistent_workers=True)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# ============================================================
# TRAINING UTILITIES
# ============================================================
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, object]:
    state = {
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
    }
    if torch.cuda.is_available():
        state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    if 'torch_rng_state' in state:
        torch.set_rng_state(state['torch_rng_state'])
    if 'numpy_rng_state' in state:
        np.random.set_state(state['numpy_rng_state'])
    if torch.cuda.is_available() and 'cuda_rng_state_all' in state:
        torch.cuda.set_rng_state_all(state['cuda_rng_state_all'])


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
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(images)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def build_model(cfg: Dict[str, object]) -> nn.Module:
    """Build internal alibi_2d model and convert it to matched scale."""
    model = VisionTransformer(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'],
        num_classes=cfg['num_classes'],
        embed_dim=cfg['embed_dim'], depth=cfg['depth'],
        num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'],
        dropout=cfg['dropout'], pe_type=INTERNAL_MODEL_PE_TYPE,
    ).to(device)
    scale_2d_alibi_slopes_(model, ALIBI_2D_MATCH_SCALE)
    return model


def make_scheduler(optimizer, cfg):
    def lr_lambda(epoch):
        if epoch < cfg['warmup_epochs']:
            return (epoch + 1) / cfg['warmup_epochs']
        progress = (epoch - cfg['warmup_epochs']) / max(
            1, cfg['epochs'] - cfg['warmup_epochs'])
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_paths(seed: int) -> Dict[str, str]:
    save_dir = os.path.join(RESULTS_DIR, f"{PUBLIC_PE_TYPE}_seed{seed}")
    return {
        'save_dir': save_dir,
        'best': os.path.join(save_dir, 'best_model.pth'),
        'last': os.path.join(save_dir, 'last_checkpoint.pth'),
        'history': os.path.join(save_dir, 'training_history.json'),
        'metadata': os.path.join(save_dir, 'match_metadata.json'),
    }


def write_metadata(seed: int, model: nn.Module, paths: Dict[str, str]) -> None:
    os.makedirs(paths['save_dir'], exist_ok=True)
    metadata = {
        'public_pe_type': PUBLIC_PE_TYPE,
        'internal_model_pe_type_for_loading': INTERNAL_MODEL_PE_TYPE,
        'seed': seed,
        'matching_rule': 'patch-patch mean distance matching, CLS excluded',
        'match': MATCH,
        'config': CIFAR_CONFIG,
        'note': (
            "Instantiate as pe_type='alibi_2d' and load this state_dict; "
            "scaled slopes are stored in the checkpoint buffers."
        ),
        'slope_info_after_scaling': collect_first_layer_slope_info(model),
    }
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)


def already_complete(seed: int, total_epochs: int) -> Tuple[bool, int]:
    paths = get_paths(seed)
    if not (os.path.isfile(paths['history']) and os.path.isfile(paths['best'])):
        return False, 0
    try:
        with open(paths['history']) as f:
            h = json.load(f)
        n_epochs = len(h.get('val_acc', []))
        return (n_epochs >= total_epochs), n_epochs
    except Exception:
        return False, 0


def save_last_checkpoint(paths, epoch, model, optimizer, scheduler, history, best_acc):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_acc': best_acc,
        'public_pe_type': PUBLIC_PE_TYPE,
        'internal_model_pe_type_for_loading': INTERNAL_MODEL_PE_TYPE,
        'match': MATCH,
        'rng_state': capture_rng_state(),
    }
    torch.save(ckpt, paths['last'])


def maybe_resume(paths, model, optimizer, scheduler, force_restart=False, no_epoch_resume=False):
    if force_restart or no_epoch_resume or not os.path.isfile(paths['last']):
        return 1, {'train_loss': [], 'val_acc': [], 'epoch_time': []}, 0.0

    print(f"  Resuming from {paths['last']}")
    ckpt = torch.load(paths['last'], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    history = ckpt.get('history', {'train_loss': [], 'val_acc': [], 'epoch_time': []})
    best_acc = float(ckpt.get('best_acc', max(history.get('val_acc', [0.0]))))
    if 'rng_state' in ckpt:
        restore_rng_state(ckpt['rng_state'])
    start_epoch = int(ckpt['epoch']) + 1
    print(f"  Resume point: epoch {start_epoch}, best={best_acc:.2f}%")
    return start_epoch, history, best_acc


def train_one_seed(seed: int, cfg: Dict[str, object]) -> float:
    """Train one magnitude-matched 2D-ALiBi model."""
    set_all_seeds(seed)
    paths = get_paths(seed)
    os.makedirs(paths['save_dir'], exist_ok=True)

    model = build_model(cfg)
    write_metadata(seed, model, paths)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    scheduler = make_scheduler(optimizer, cfg)

    start_epoch, history, best_acc = maybe_resume(
        paths, model, optimizer, scheduler,
        force_restart=args.force_restart,
        no_epoch_resume=args.no_epoch_resume,
    )

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        t0 = time.time()
        model.train()
        running_loss, n_batches = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=cfg['mixup_alpha'])
            out = model(mixed_x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(1, n_batches)
        val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        history['train_loss'].append(float(train_loss))
        history['val_acc'].append(float(val_acc))
        history['epoch_time'].append(float(epoch_time))

        if val_acc > best_acc:
            best_acc = float(val_acc)
            torch.save(model.state_dict(), paths['best'])

        with open(paths['history'], 'w') as f:
            json.dump(history, f, indent=2)

        if SAVE_LAST_CHECKPOINT_EVERY_EPOCH:
            save_last_checkpoint(paths, epoch, model, optimizer, scheduler, history, best_acc)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"    Epoch {epoch:3d}/{cfg['epochs']}: "
            f"loss={train_loss:.3f} acc={val_acc:.2f}% best={best_acc:.2f}% "
            f"({epoch_time:.1f}s, lr={current_lr:.5f})",
            flush=True,
        )

    print(f"  Done. Best val acc: {best_acc:.2f}%")
    return best_acc


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 76)
    print(f"  Training {PUBLIC_PE_TYPE} x {len(SEEDS)} seeds on CIFAR-100")
    print("=" * 76)
    print("  Internal model pe_type: alibi_2d")
    print(f"  Slope match scale alpha: {ALIBI_2D_MATCH_SCALE:.9f}")
    print(f"  Patch grid: {MATCH['grid_side']}x{MATCH['grid_side']} "
          f"({MATCH['num_patches']} patches, N={MATCH['num_patches'] + 1} with CLS)")
    print(f"  Epochs/seed: {CIFAR_CONFIG['epochs']}")
    print(f"  Grad clip: {CIFAR_CONFIG['grad_clip']}")
    print(f"  Output: {RESULTS_DIR}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 76)

    summary = {}
    summary_path = os.path.join(RESULTS_DIR, '_alibi_2d_matched_12seeds_summary.json')

    for idx, seed in enumerate(SEEDS, 1):
        tag = f"{PUBLIC_PE_TYPE}_seed{seed}"
        done, n_epochs = already_complete(seed, CIFAR_CONFIG['epochs'])
        if done and not args.force_restart:
            paths = get_paths(seed)
            with open(paths['history']) as f:
                h = json.load(f)
            acc = max(h['val_acc'])
            print(f"\n[{idx}/{len(SEEDS)}] {tag}: already complete "
                  f"({n_epochs} epochs, best={acc:.2f}%) -- skipping")
            summary[tag] = {'best_acc': float(acc), 'status': 'cached'}
        else:
            if n_epochs > 0 and args.force_restart:
                print(f"\n[{idx}/{len(SEEDS)}] {tag}: force restart requested")
            elif n_epochs > 0:
                print(f"\n[{idx}/{len(SEEDS)}] {tag}: partial history found "
                      f"({n_epochs}/{CIFAR_CONFIG['epochs']} epochs); attempting epoch-level resume")

            print(f"\n[{idx}/{len(SEEDS)}] >>> Training {tag}")
            t_start = time.time()
            try:
                best_acc = train_one_seed(seed, CIFAR_CONFIG)
                elapsed = time.time() - t_start
                summary[tag] = {
                    'best_acc': float(best_acc),
                    'wall_time_hours': elapsed / 3600.0,
                    'status': 'trained',
                    'match_scale_alpha': ALIBI_2D_MATCH_SCALE,
                }
                print(f"  Wall time: {elapsed/3600:.2f}h")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                summary[tag] = {'status': 'failed', 'error': f"{type(e).__name__}: {e}"}

        with open(summary_path, 'w') as f:
            json.dump({
                'public_pe_type': PUBLIC_PE_TYPE,
                'internal_model_pe_type_for_loading': INTERNAL_MODEL_PE_TYPE,
                'match': MATCH,
                'config': CIFAR_CONFIG,
                'seeds': SEEDS,
                'runs': summary,
            }, f, indent=2)

    print("\n" + "=" * 76)
    print("TRAINING SUMMARY")
    print("=" * 76)
    for tag, info in summary.items():
        if info.get('status') == 'failed':
            print(f"  {tag:30s}: FAILED ({info.get('error', '?')})")
        else:
            t = info.get('wall_time_hours', 0.0)
            print(f"  {tag:30s}: {info['best_acc']:6.2f}% "
                  f"({t:.1f}h, {info.get('status', '?')})")

    print(f"\nSummary written to {summary_path}")
    print("\nLoading note for MI/analysis scripts:")
    print("  instantiate VisionTransformer(..., pe_type='alibi_2d')")
    print("  then load alibi_2d_matched_seed*/best_model.pth")
    print("  the matched slopes are stored in the checkpoint state_dict.")
