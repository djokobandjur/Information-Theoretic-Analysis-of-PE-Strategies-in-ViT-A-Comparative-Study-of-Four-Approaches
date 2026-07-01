"""
figures_revision.py
====================
Regenerates Figures 2, 3, 4a/4b, 5a/5b for the TPAMI revision with
proper TPAMI/IEEE-compliant formatting:
  - 300 DPI raster (PNG) + vector (PDF) output
  - Font size >= 9 pt at final figure size (R3 requirement)
  - Two-track formatting: embedding-space figures show Learned+Sin
    only; attention-space (intrinsic) figures show RoPE/ALiBi/2D-ALiBi

Outputs to /content/figures_revision/:
  fig2_cosine_similarity.png  +  .pdf
  03_pca_tsne.png             +  .pdf     (Learned + Sinusoidal only)
  fig4a_embedding_entropy.png +  .pdf
  fig4b_intrinsic_entropy.png +  .pdf
  fig5a_embedding_variance.png + .pdf
  fig5b_intrinsic_structure.png + .pdf
  figS3_distance_distortion.png + .pdf

Prerequisites:
  - /content/full_scale_experiment_v2.py  (patched VisionTransformer
    that supports 'alibi_2d')
  - Trained checkpoints at:
    /content/drive/MyDrive/Trained models_ImageNet100/
        {pe_type}_seed42/best_model.pth
    for pe_type in {learned, sinusoidal, rope, alibi, alibi_2d}

USAGE in Colab:
    !python /content/figures_revision.py
"""

import os
import sys
import math
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from scipy.stats import entropy as sp_entropy

# ============================================================================
# CONFIG
# ============================================================================
CKPT_ROOT = "/content/drive/MyDrive/Trained models_ImageNet100"
OUTPUT_DIR = "/content/figures_revision"
PE_TYPES = ["learned", "sinusoidal", "rope", "alibi", "alibi_2d"]
SEED = 42

# Path to the aggregated per-layer MI JSON written by revision_analysis_v4.ipynb
# (canonical discrete estimator: _mutual_information_discrete +
# compute_mi_per_layer_v2). Used only by Figure 6; if missing, Figure 6 is
# skipped with a warning.
MI_AGG_PATH = (
    "/content/drive/MyDrive/revision_results/imagenet100/_aggregate.json"
)

# Figure 3 composite inputs. Panel (a) uses the ImageNet-100 aggregate
# per-layer MI JSON. Panel (b) uses machine-readable CIFAR-100 paired MI
# summary JSONs produced by the MI-control analysis scripts. These defaults
# deliberately avoid manual audit files, so the figure is repo-reproducible.
FIG3_IMAGENET_MI_AGG_PATH = (
    "/content/drive/MyDrive/revision_results/imagenet100/_aggregate.json"
)
FIG3_CIFAR_FIXED_MI_PATH = (
    "/content/drive/MyDrive/revision_results/mi_cls_control/"
    "cifar100_canonical_n12/paired_alibi_mi_summary.json"
)
FIG3_CIFAR_MATCHED_MI_PATH = (
    "/content/drive/MyDrive/revision_results/mi_cls_control/"
    "cifar100_canonical_matched2d_n12/paired_alibi_mi_summary.json"
)
FIG3_OUTPUT_NAME = "fig_main_mi_control"

# Figure 7 input: saved noise-ablation / PE-removal analysis JSON.
FIG7_ANALYSIS_DATA_PATH = (
    "/content/drive/MyDrive/pe_experiment/results/analysis_data.json"
)

# IN-100 architecture (matches existing training)
IN100_ARCH = {
    "img_size": 224, "patch_size": 16,
    "num_patches_per_side": 14,
    "num_positions": 197,
    "num_classes": 100,
    "embed_dim": 768, "depth": 12, "num_heads": 12,
    "mlp_ratio": 4.0, "dropout": 0.1,
}

# TPAMI-friendly matplotlib style
rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size":            11,
    "axes.labelsize":       12,
    "axes.titlesize":       12,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      10,
    "figure.titlesize":     13,
    "savefig.dpi":          300,
    "figure.dpi":           150,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "axes.grid":            False,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
})

# Color palette consistent across all figures
COLORS = {
    "learned":    "#1f77b4",   # blue
    "sinusoidal": "#ff7f0e",   # orange
    "rope":       "#2ca02c",   # green
    "alibi":      "#d62728",   # red
    "alibi_2d":   "#9467bd",   # purple
}
DISPLAY_NAME = {
    "learned":    "Learned",
    "sinusoidal": "Sinusoidal",
    "rope":       "RoPE",
    "alibi":      "1D-ALiBi",
    "alibi_2d":   "2D-ALiBi",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# MODEL LOADING
# ============================================================================
sys.path.insert(0, "/content")
try:
    from full_scale_experiment_v2 import VisionTransformer
except ImportError as _import_err:
    VisionTransformer = None
    print("  [WARN] Could not import full_scale_experiment_v2. "
          "Checkpoint-dependent figures will fail, but --only-fig3 can still run.")


def load_model(pe_type, arch, seed=SEED):
    """Load a trained model checkpoint."""
    if VisionTransformer is None:
        raise ImportError("full_scale_experiment_v2.VisionTransformer is required "
                          "for checkpoint-dependent figures. Use --only-fig3 "
                          "for the JSON-only MI figure.")
    ckpt_path = os.path.join(CKPT_ROOT, f"{pe_type}_seed{seed}", "best_model.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = VisionTransformer(
        img_size=arch["img_size"], patch_size=arch["patch_size"],
        num_classes=arch["num_classes"],
        embed_dim=arch["embed_dim"], depth=arch["depth"],
        num_heads=arch["num_heads"], mlp_ratio=arch["mlp_ratio"],
        dropout=arch["dropout"], pe_type=pe_type,
    )

    state = torch.load(ckpt_path, map_location=device)
    # Strip compile prefixes if any
    new_state = {(k[10:] if k.startswith("_orig_mod.") else k): v
                  for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)  # allow missing PE keys
    model = model.to(device).eval()
    return model


# ============================================================================
# DATA EXTRACTION (robust to attribute naming)
# ============================================================================
def _find_pe_tensor(module, target_shape_pattern):
    """Find a tensor on `module` whose shape matches the pattern.
    
    target_shape_pattern: list of expected dims. Use -1 as wildcard.
    Example: [-1, 768] matches any [N, 768] tensor.
    
    Searches: named_parameters -> named_buffers -> direct attributes.
    Returns the tensor (detached, on CPU), or None.
    """
    def shape_matches(shape, pattern):
        if len(shape) != len(pattern):
            return False
        return all(p == -1 or s == p for s, p in zip(shape, pattern))
    
    # 1. Try named_parameters
    for name, p in module.named_parameters(recurse=False):
        if shape_matches(p.shape, target_shape_pattern):
            print(f"     [found PE in '{name}' (param), shape={tuple(p.shape)}]")
            return p.detach().cpu()
    
    # 2. Try named_buffers
    for name, b in module.named_buffers(recurse=False):
        if shape_matches(b.shape, target_shape_pattern):
            print(f"     [found PE in '{name}' (buffer), shape={tuple(b.shape)}]")
            return b.detach().cpu()
    
    # 3. Try direct attributes (for things stored as plain attr)
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            val = getattr(module, name)
        except AttributeError:
            continue
        if isinstance(val, torch.Tensor) and shape_matches(val.shape, target_shape_pattern):
            print(f"     [found PE in '{name}' (attr), shape={tuple(val.shape)}]")
            return val.detach().cpu()
    
    return None


def extract_pe_matrix(model, pe_type, num_positions):
    """Returns the PE matrix as a numpy array.

    For LEARNED/SINUSOIDAL: returns the PE matrix [N, 768].
    For ROPE: returns concatenated cos/sin features [N, 2*head_dim].
    For ALiBi/2D-ALiBi: returns the bias tensor [H, N, N].
    """
    embed_dim = IN100_ARCH["embed_dim"]
    n_heads = IN100_ARCH["num_heads"]
    head_dim = embed_dim // n_heads
    
    with torch.no_grad():
        if pe_type in ("learned", "sinusoidal"):
            pos_enc = model.pos_encoding
            # Try common shapes:
            # [1, N, d], [N, d], or [N+1, d]
            pe = None
            for pattern in ([1, num_positions, embed_dim],
                              [num_positions, embed_dim],
                              [1, -1, embed_dim],
                              [-1, embed_dim]):
                pe = _find_pe_tensor(pos_enc, pattern)
                if pe is not None:
                    break
            if pe is None:
                # Fallback: search recursively
                print(f"     [searching submodules of pos_encoding...]")
                for sub_name, sub_module in pos_enc.named_modules():
                    if sub_name == "":
                        continue
                    pe = _find_pe_tensor(sub_module, [-1, embed_dim])
                    if pe is not None:
                        break
            if pe is None:
                raise RuntimeError(f"Could not find PE tensor in {pe_type} model")
            pe = pe.numpy()
            if pe.ndim == 3:
                pe = pe[0]  # [1, N, d] -> [N, d]
            return pe

        elif pe_type == "rope":
            # RoPE often stores cos/sin as buffers
            attn = model.blocks[0].attn
            rope = attn.rope if hasattr(attn, "rope") else attn
            # Look for cos/sin buffers
            cos = _find_pe_tensor(rope, [-1, head_dim])
            sin = None
            # If cos found, find sin
            if cos is not None:
                for name, b in rope.named_buffers(recurse=False):
                    if name.lower().startswith("sin") and b.shape == cos.shape:
                        sin = b.detach().cpu()
                        break
            if cos is None or sin is None:
                # Recursive search
                for sub_name, sub_module in rope.named_modules():
                    if sub_name == "":
                        continue
                    cos_cand = _find_pe_tensor(sub_module, [-1, head_dim])
                    if cos_cand is not None:
                        cos = cos_cand
                        for name, b in sub_module.named_buffers(recurse=False):
                            if name.lower().startswith("sin") and b.shape == cos.shape:
                                sin = b.detach().cpu()
                                break
                    if cos is not None and sin is not None:
                        break
            
            if cos is None or sin is None:
                # Last resort: compute cos/sin from rope.theta or inv_freq
                print("     [computing cos/sin from inv_freq/theta]")
                inv_freq = None
                # Search rope and submodules for inv_freq or theta
                for module_to_check in [rope] + [m for _, m in rope.named_modules() if _ != ""]:
                    # Try inv_freq first (param or buffer)
                    if hasattr(module_to_check, "inv_freq"):
                        try:
                            val = module_to_check.inv_freq
                            if isinstance(val, torch.Tensor):
                                inv_freq = val.detach().cpu().numpy().astype(np.float32)
                                print(f"     [found inv_freq, shape={inv_freq.shape}]")
                                break
                        except AttributeError:
                            pass
                    # Try theta
                    if hasattr(module_to_check, "theta"):
                        try:
                            theta_val = module_to_check.theta
                            if isinstance(theta_val, torch.Tensor):
                                theta_np = theta_val.detach().cpu().numpy().astype(np.float32)
                            else:
                                theta_np = np.asarray(theta_val, dtype=np.float32)
                            inv_freq = 1.0 / theta_np
                            print(f"     [computed inv_freq from theta, shape={inv_freq.shape}]")
                            break
                        except AttributeError:
                            pass

                if inv_freq is None:
                    # Final fallback: compute the canonical RoPE inv_freq from head_dim
                    print(f"     [using canonical inv_freq for head_dim={head_dim}]")
                    inv_freq = 1.0 / (10000.0 ** (
                        np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

                # Compute cos/sin from inv_freq
                pos = np.arange(num_positions, dtype=np.float32)
                angles = np.outer(pos, inv_freq)   # [N, head_dim/2]
                cos = np.cos(angles).astype(np.float32)
                sin = np.sin(angles).astype(np.float32)
                # Each frequency band fills two consecutive dims in head_dim
                cos = np.repeat(cos, 2, axis=1)[:, :head_dim]
                sin = np.repeat(sin, 2, axis=1)[:, :head_dim]
            else:
                cos = cos.numpy().squeeze()
                sin = sin.numpy().squeeze()
                if cos.ndim > 2:
                    # Reshape to [N, head_dim] taking first head
                    cos = cos.reshape(-1, head_dim)[:num_positions]
                    sin = sin.reshape(-1, head_dim)[:num_positions]
            
            pe = np.concatenate([cos, sin], axis=-1)
            return pe

        elif pe_type in ("alibi", "alibi_2d"):
            attn = model.blocks[0].attn
            alibi = attn.alibi if hasattr(attn, "alibi") else attn
            
            # Find slopes (per-head scalar)
            slopes = None
            for name, p in alibi.named_parameters(recurse=False):
                if "slope" in name.lower() or p.numel() == n_heads:
                    slopes = p.detach().cpu().squeeze().numpy()
                    print(f"     [found slopes in '{name}', shape={tuple(p.shape)}]")
                    break
            if slopes is None:
                for name, b in alibi.named_buffers(recurse=False):
                    if "slope" in name.lower() or b.numel() == n_heads:
                        slopes = b.detach().cpu().squeeze().numpy()
                        print(f"     [found slopes in '{name}' buffer]")
                        break
            
            # Find distance matrix
            dist = None
            for name, b in alibi.named_buffers(recurse=False):
                if "dist" in name.lower() or "rel" in name.lower():
                    sh = tuple(b.shape)
                    # Should be [N, N] or [1, 1, N, N]
                    if sh[-2:] == (num_positions, num_positions):
                        dist = b.detach().cpu().squeeze().numpy()
                        print(f"     [found dist in '{name}', shape={sh}]")
                        break
            if dist is None:
                # Try direct attributes
                for name in dir(alibi):
                    if name.startswith("_"):
                        continue
                    if "dist" in name.lower() or "rel" in name.lower():
                        try:
                            val = getattr(alibi, name)
                        except AttributeError:
                            continue
                        if isinstance(val, torch.Tensor):
                            sh = tuple(val.shape)
                            if sh[-2:] == (num_positions, num_positions):
                                dist = val.detach().cpu().squeeze().numpy()
                                print(f"     [found dist in attr '{name}', shape={sh}]")
                                break
            
            if slopes is None or dist is None:
                raise RuntimeError(
                    f"Could not find slopes or dist for {pe_type}; "
                    f"slopes={slopes is not None}, dist={dist is not None}"
                )
            
            # Construct bias: bias[h, i, j] = -slopes[h] * dist[i, j]
            bias = -slopes[:, None, None] * dist[None, :, :]
            return bias

        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")


def cosine_similarity_matrix(pe_or_bias, pe_type):
    """Return N x N cosine similarity matrix for visualisation."""
    if pe_type in ("learned", "sinusoidal", "rope"):
        # PE is [N, d]
        v = pe_or_bias
        norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        v = v / norm
        return v @ v.T
    elif pe_type in ("alibi", "alibi_2d"):
        # Use head-mean of bias as the "similarity-like" structure
        # (negative distance, so larger = closer in token-pair space)
        bias = pe_or_bias.mean(axis=0)   # [N, N]
        # Normalise to [-1, 1] for visual comparability
        b_max = np.abs(bias).max() + 1e-12
        return bias / b_max
    else:
        raise ValueError(pe_type)


def per_dim_entropy(pe_matrix, n_bins=64):
    """Shannon entropy per embedding dimension (in bits).

    Canonical implementation matching revision_analysis_v4.ipynb
    (compute_dimension_entropy): n_bins=64, density=True, re-normalised
    to a probability distribution before scipy.stats.entropy.
    """
    N, D = pe_matrix.shape
    ent = np.zeros(D)
    for d in range(D):
        hist, _ = np.histogram(pe_matrix[:, d], bins=n_bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        ent[d] = sp_entropy(hist, base=2)
    return ent


def per_dim_variance(pe_matrix):
    """Per-dimension variance."""
    return pe_matrix.var(axis=0)


def per_head_intrinsic_entropy(bias, n_bins=64):
    """Shannon entropy of each head's bias values (flattened across N x N).

    Canonical implementation matching revision_analysis_v4.ipynb
    (analyze_alibi_intrinsic): n_bins=64, density=True, re-normalised
    to a probability distribution before scipy.stats.entropy.
    """
    H = bias.shape[0]
    ent = np.zeros(H)
    for h in range(H):
        vals = bias[h].flatten()
        hist, _ = np.histogram(vals, bins=n_bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        ent[h] = sp_entropy(hist, base=2)
    return ent


def rope_active_dims(model, N):
    """Fraction of RoPE rotation frequencies with wavelength < N."""
    rope = model.blocks[0].attn.rope
    # The rope.theta is computed as 10000^(2i/head_dim) for i in [0, head_dim/2)
    # active = wavelength < N => 2*pi/theta < N
    head_dim = IN100_ARCH["embed_dim"] // IN100_ARCH["num_heads"]
    freqs = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
    wavelengths = 2 * np.pi / freqs
    return wavelengths, wavelengths < N


# ============================================================================
# FIGURE GENERATION
# ============================================================================
def make_figure2_cosine_similarity(data):
    """2x2 grid: cosine similarity matrices for Learned, Sin, RoPE, 1D-ALiBi."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    titles = {
        "learned":    "(a) Learned PE",
        "sinusoidal": "(b) Sinusoidal PE",
        "rope":       "(c) RoPE (cos/sin features)",
        "alibi":      "(d) 1D-ALiBi (head-mean bias)",
    }
    for ax, pe_type in zip(axes.flat,
                             ["learned", "sinusoidal", "rope", "alibi"]):
        cs = data[pe_type]["cos_sim"]
        # Symmetric colormap
        vmax = np.abs(cs).max()
        im = ax.imshow(cs, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        aspect="equal", interpolation="nearest")
        ax.set_title(titles[pe_type], fontsize=12)
        ax.set_xlabel("Position $j$")
        ax.set_ylabel("Position $i$")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    plt.suptitle(
        "Cosine Similarity / Bias Structure of Positional Encoding "
        "(ImageNet-100, seed 42, $N{=}197$)",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    save_figure(fig, "fig2_cosine_similarity")
    plt.close(fig)


def make_figure3_pca_tsne(data):
    """2x2 grid: PCA (top) and t-SNE (bottom) projections of Learned and
    Sinusoidal additive PE vectors on ImageNet-100.

    Attention-space methods (RoPE, 1D-ALiBi, 2D-ALiBi) are intentionally
    excluded: they do not modify the additive token embedding, so PCA of
    the token vectors would reflect only the baseline embedding
    statistics rather than any positional signature. This corresponds to
    the embedding-space vs. attention-space methodological distinction
    enforced throughout the revised manuscript.

    Output:  03_pca_tsne.png + 03_pca_tsne.pdf
    Caption-referenced in supplementary Section S1 (Figure S1).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    additive_types = ["learned", "sinusoidal"]
    # 196 patch positions (CLS excluded), color-encoded by 1-based
    # raster index (1..196) for readability in the published figure.
    n_patches = IN100_ARCH["num_patches_per_side"] ** 2
    positions = np.arange(1, n_patches + 1)

    for col, pe_type in enumerate(additive_types):
        pe = data[pe_type]["pe"]              # [N, 768] = [197, 768]
        patches = pe[1:]                       # drop CLS -> [196, 768]

        # PCA: first 2 principal components
        pca = PCA(n_components=2)
        proj_pca = pca.fit_transform(patches)
        var_pct = pca.explained_variance_ratio_.sum() * 100.0

        # t-SNE: 2D embedding (fixed random_state for reproducibility,
        # matches the seed used in the original Round 1 script).
        tsne = TSNE(n_components=2, perplexity=30,
                     random_state=42, init="pca")
        proj_tsne = tsne.fit_transform(patches)

        # --- top row: PCA ---
        sc1 = axes[0, col].scatter(
            proj_pca[:, 0], proj_pca[:, 1],
            c=positions, cmap="viridis",
            s=30, edgecolors="k", linewidth=0.3,
        )
        axes[0, col].set_title(
            f"{DISPLAY_NAME[pe_type]} PE \u2014 PCA "
            f"({var_pct:.1f}% var.)"
        )
        axes[0, col].set_xlabel("PC1")
        axes[0, col].set_ylabel("PC2")
        plt.colorbar(sc1, ax=axes[0, col], label="Position", shrink=0.85)

        # --- bottom row: t-SNE ---
        sc2 = axes[1, col].scatter(
            proj_tsne[:, 0], proj_tsne[:, 1],
            c=positions, cmap="viridis",
            s=30, edgecolors="k", linewidth=0.3,
        )
        axes[1, col].set_title(f"{DISPLAY_NAME[pe_type]} PE \u2014 t-SNE")
        axes[1, col].set_xlabel("t-SNE 1")
        axes[1, col].set_ylabel("t-SNE 2")
        plt.colorbar(sc2, ax=axes[1, col], label="Position", shrink=0.85)

    fig.suptitle(
        "PCA and t-SNE Projections of Additive PE Vectors \u2014 "
        "ImageNet-100 (ViT-Base)",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    save_figure(fig, "03_pca_tsne")
    plt.close(fig)


def make_figure4a_embedding_entropy(data):
    """Per-dim Shannon entropy for Learned + Sinusoidal."""
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))

    for pe_type in ["learned", "sinusoidal"]:
        ent = data[pe_type]["per_dim_entropy"]
        ax.plot(ent, label=f"{DISPLAY_NAME[pe_type]} "
                            f"(mean = {ent.mean():.2f} bits)",
                color=COLORS[pe_type], linewidth=1.2, alpha=0.9)

    ax.set_xlabel("Embedding dimension index $d$")
    ax.set_ylabel("Per-dimension entropy (bits)")
    ax.set_title("Embedding-space Entropy Profile (additive PE methods)",
                  fontsize=12)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 768)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "fig4a_embedding_entropy")
    plt.close(fig)


def make_figure4b_intrinsic_entropy(data):
    """Per-head Shannon entropy for 1D-ALiBi and 2D-ALiBi.

    RoPE is intentionally excluded: its rotation schedule is shared
    across heads by construction (same frequencies broadcast over the
    head dimension), so per-head intrinsic entropy is not a
    well-defined diagnostic. See Table tab:rope-intrinsic in the
    manuscript for RoPE's scalar intrinsic diagnostics
    (active-band fraction, intrinsic rank, effective wavelengths).
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))

    # Bar chart: per-head entropy across all 12 heads, mean line
    pe_list = ["alibi", "alibi_2d"]
    n_pe = len(pe_list)
    n_heads = 12
    bar_width = 0.35   # wider bars since only 2 PE types now

    for i, pe_type in enumerate(pe_list):
        ent = data[pe_type]["per_head_entropy"]
        positions = np.arange(n_heads) + i * bar_width
        ax.bar(positions, ent, width=bar_width,
                label=f"{DISPLAY_NAME[pe_type]} "
                       f"(mean = {ent.mean():.2f} bits)",
                color=COLORS[pe_type], alpha=0.85,
                edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Attention head index")
    ax.set_ylabel("Per-head intrinsic entropy (bits)")
    ax.set_title("Per-head Intrinsic Entropy of ALiBi Variants",
                  fontsize=12)
    ax.set_xticks(np.arange(n_heads) + bar_width / 2)
    ax.set_xticklabels([str(h) for h in range(n_heads)])
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "fig4b_intrinsic_entropy")
    plt.close(fig)


def make_figure5a_embedding_variance(data):
    """Per-dim variance for Learned + Sinusoidal."""
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))

    for pe_type in ["learned", "sinusoidal"]:
        var = data[pe_type]["per_dim_variance"]
        ax.plot(var, label=f"{DISPLAY_NAME[pe_type]} "
                            f"($\\bar{{\\sigma^2}} = {var.mean():.4f}$)",
                color=COLORS[pe_type], linewidth=1.0, alpha=0.85)

    ax.set_xlabel("Embedding dimension index $d$")
    ax.set_ylabel("Per-dimension variance $\\sigma^2$")
    ax.set_title("Embedding-space Variance Profile (additive PE methods)",
                  fontsize=12)
    ax.set_yscale("log")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 768)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_figure(fig, "fig5a_embedding_variance")
    plt.close(fig)


def make_figure5b_intrinsic_structure(data):
    """Slope schedule + effective receptive field per head."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: slope schedule (ALiBi 1D, ALiBi 2D)
    ax = axes[0]
    for pe_type in ["alibi", "alibi_2d"]:
        slopes = data[pe_type]["slopes"]
        ax.plot(slopes, marker="o", markersize=6, linewidth=1.5,
                label=DISPLAY_NAME[pe_type], color=COLORS[pe_type])
    ax.set_xlabel("Attention head index $h$")
    ax.set_ylabel("Slope $m_h$")
    ax.set_title("(a) ALiBi slope schedule", fontsize=12)
    ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(range(12))

    # Right: RoPE wavelengths + active region (N=197 threshold)
    ax = axes[1]
    wavelengths = data["rope"]["wavelengths"]
    active = data["rope"]["active_mask"]
    head_dim = IN100_ARCH["embed_dim"] // IN100_ARCH["num_heads"]
    band_idx = np.arange(len(wavelengths))

    ax.bar(band_idx[active], wavelengths[active],
            color=COLORS["rope"], alpha=0.85, edgecolor="black",
            linewidth=0.5, label=f"Active ({active.sum()}/{len(active)})")
    ax.bar(band_idx[~active], wavelengths[~active],
            color="lightgrey", alpha=0.6, edgecolor="black",
            linewidth=0.5, label=f"Inactive ({(~active).sum()}/{len(active)})")
    ax.axhline(y=197, color="red", linestyle="--", linewidth=1.0,
                label="$N{=}197$ threshold")

    ax.set_xlabel("RoPE frequency band index $i$")
    ax.set_ylabel("Wavelength $\\lambda_i$ (positions)")
    ax.set_title("(b) RoPE rotation wavelengths (head dim $d_h{=}64$)",
                  fontsize=12)
    ax.set_yscale("log")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(
        "Attention-space Intrinsic Structure "
        "(ImageNet-100, $N{=}197$)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, "fig5b_intrinsic_structure")
    plt.close(fig)


def save_figure(fig, name):
    """Save both PNG (300 dpi) and PDF (vector) versions."""
    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}.png + .pdf")



# ============================================================================
# FIGURE 3: MI taxonomy + CIFAR-100 2D-ALiBi control analysis
# ============================================================================
def _unwrap_singleton_summary_block(obj):
    """Return the most likely summary-bearing block from a JSON object."""
    if isinstance(obj, dict) and "summary" in obj:
        return obj
    if isinstance(obj, dict) and "estimators" in obj:
        return obj
    if isinstance(obj, dict) and len(obj) == 1:
        only = next(iter(obj.values()))
        if isinstance(only, dict):
            return only
    return obj


def _estimator_aliases():
    return {
        "incl": [
            "incl_cls_all_tokens", "cls_inclusive", "cls-inclusive",
            "cls_incl", "incl_cls", "all_tokens", "cls"
        ],
        "patch": [
            "excl_cls_patch_only", "patch_only", "patch-only",
            "cls_excluded_patch_only", "cls-excluded patch-only",
            "excl_cls", "patch"
        ],
    }


def _find_estimator_block(summary, family):
    """Find a CLS-inclusive or patch-only estimator block in a summary dict."""
    aliases = _estimator_aliases()[family]
    if not isinstance(summary, dict):
        raise ValueError("Estimator summary must be a dictionary")

    # Exact / normalized key lookup first.
    norm_to_key = {
        str(k).lower().replace("_", "-").replace(" ", "-"): k
        for k in summary.keys()
    }
    for alias in aliases:
        norm = alias.lower().replace("_", "-").replace(" ", "-")
        if norm in norm_to_key:
            return summary[norm_to_key[norm]]

    # Fallback: substring lookup.
    for k, v in summary.items():
        key = str(k).lower()
        if family == "incl" and (
            "incl" in key or "cls-inclusive" in key or "all_tokens" in key
        ):
            return v
        if family == "patch" and (
            "patch" in key or "excl" in key or "cls-excluded" in key
        ):
            return v

    raise KeyError(f"Could not find {family} estimator in keys: {list(summary.keys())}")


def _extract_mean_std_from_method_block(block, method_aliases):
    """Extract (mean, std) from a combined JSON estimator block."""
    if not isinstance(block, dict):
        raise ValueError("Method block must be a dictionary")
    for alias in method_aliases:
        if alias in block and isinstance(block[alias], dict):
            m = block[alias]
            mean = m.get("mean", m.get("mi_mean", m.get("value")))
            std = m.get("std", m.get("sd", m.get("mi_std", 0.0)))
            if mean is not None:
                return float(mean), float(std)
    # Case-insensitive fallback.
    low = {str(k).lower(): k for k in block.keys()}
    for alias in method_aliases:
        if alias.lower() in low and isinstance(block[low[alias.lower()]], dict):
            m = block[low[alias.lower()]]
            mean = m.get("mean", m.get("mi_mean", m.get("value")))
            std = m.get("std", m.get("sd", m.get("mi_std", 0.0)))
            if mean is not None:
                return float(mean), float(std)
    raise KeyError(f"Could not extract mean/std for aliases={method_aliases}")


def _parse_alibi_mi_json(path):
    """Parse JSON summary produced by the MI suite.

    Expected native schema:
      {cohort_key: {summary: {
          incl_cls_all_tokens: {mi_1d_mean, mi_1d_std, mi_2d_mean, mi_2d_std},
          excl_cls_patch_only: {...}
      }}}

    Also accepts a compact combined schema:
      {estimators: {cls_inclusive: {"1D-ALiBi": {mean,std}, ...},
                    patch_only: {...}}}
    """
    with open(path, "r") as f:
        obj = json.load(f)
    block = _unwrap_singleton_summary_block(obj)

    if isinstance(block, dict) and "estimators" in block:
        est = block["estimators"]
        out = {}
        for family, out_key in [("incl", "incl_cls_all_tokens"),
                                ("patch", "excl_cls_patch_only")]:
            b = _find_estimator_block(est, family)
            out[out_key] = {
                "one_d": _extract_mean_std_from_method_block(
                    b, ["1D-ALiBi", "alibi", "1d", "1D"]),
                "fixed_2d": _extract_mean_std_from_method_block(
                    b, ["Fixed 2D-ALiBi", "fixed_2d", "alibi_2d_fixed"])
                    if any(str(k).lower() in {"fixed 2d-alibi", "fixed_2d", "alibi_2d_fixed"}
                           for k in b.keys()) else None,
                "matched_2d": _extract_mean_std_from_method_block(
                    b, ["Matched 2D-ALiBi", "matched_2d", "alibi_2d_matched"])
                    if any(str(k).lower() in {"matched 2d-alibi", "matched_2d", "alibi_2d_matched"}
                           for k in b.keys()) else None,
            }
        return out

    if not isinstance(block, dict) or "summary" not in block:
        raise ValueError(f"Unsupported MI JSON schema in {path}")

    summary = block["summary"]
    out = {}
    for family, out_key in [("incl", "incl_cls_all_tokens"),
                            ("patch", "excl_cls_patch_only")]:
        b = _find_estimator_block(summary, family)
        out[out_key] = {
            "one_d": (float(b["mi_1d_mean"]), float(b.get("mi_1d_std", 0.0))),
            "two_d": (float(b["mi_2d_mean"]), float(b.get("mi_2d_std", 0.0))),
        }
    return out


def _load_alibi_mi_summary(path):
    """Load a machine-readable ALiBi MI summary JSON.

    Reproducibility note: manual Markdown audit files are intentionally not
    accepted here. Figure 3 must be regenerated from JSON outputs written by
    the analysis pipeline.
    """
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not path.lower().endswith(".json"):
        raise ValueError(
            f"Expected a JSON MI summary for Figure 3, got: {path}. "
            "Run the MI-control analysis first and pass its "
            "paired_alibi_mi_summary.json output."
        )
    return _parse_alibi_mi_json(path)


def _compose_cifar_control_values(fixed_path, matched_path):
    """Return plotting means/stds for 1D, fixed-2D, matched-2D controls."""
    fixed = _load_alibi_mi_summary(fixed_path)
    matched = _load_alibi_mi_summary(matched_path)
    if fixed is None or matched is None:
        raise ValueError(
            "Figure 3 panel (b) needs both machine-readable JSON inputs: "
            "--fig3-cifar-fixed-mi-json and --fig3-cifar-matched-mi-json. "
            "Expected files are the paired_alibi_mi_summary.json outputs "
            "from the MI-control analysis cohorts."
        )

    out = {}
    for key in ["incl_cls_all_tokens", "excl_cls_patch_only"]:
        one_d = fixed[key].get("one_d") or matched[key].get("one_d")
        fixed_2d = fixed[key].get("two_d") or fixed[key].get("fixed_2d")
        matched_2d = matched[key].get("two_d") or matched[key].get("matched_2d")
        if one_d is None or fixed_2d is None or matched_2d is None:
            raise ValueError(f"Incomplete CIFAR control values for {key}")
        out[key] = {
            "1D-ALiBi": one_d,
            "Fixed 2D-ALiBi": fixed_2d,
            "Matched 2D-ALiBi": matched_2d,
        }
    return out


def _load_imagenet_mi_taxonomy(mi_agg_path):
    """Load per-layer ImageNet-100 MI means/stds from aggregate JSON.

    Accepts either:
      1. _aggregate.json schema with top-level PE keys, each containing
         mi_per_layer.mean_per_layer / std_per_layer; or
      2. _master_summary.json schema with top-level "imagenet100" and
         per-seed entries such as learned_seed42, each containing
         mi_per_layer.values.
    """
    if not os.path.isfile(mi_agg_path):
        raise FileNotFoundError(mi_agg_path)
    with open(mi_agg_path, "r") as f:
        agg = json.load(f)

    # Support _master_summary.json by aggregating imagenet100 seed entries.
    if isinstance(agg, dict) and "imagenet100" in agg:
        im = agg["imagenet100"]
        curves = {}
        for pe_type in ["learned", "sinusoidal", "rope", "alibi"]:
            vals = []
            prefix = f"{pe_type}_seed"
            for key, entry in im.items():
                if not str(key).startswith(prefix):
                    continue
                block = entry.get("mi_per_layer", {}) if isinstance(entry, dict) else {}
                v = block.get("values", block.get("mean_per_layer", None))
                if v is not None:
                    vals.append(np.asarray(v, dtype=np.float64))
            if vals:
                arr = np.vstack(vals)
                curves[pe_type] = (arr.mean(axis=0), arr.std(axis=0, ddof=1)
                                   if arr.shape[0] > 1 else np.zeros(arr.shape[1]))
            else:
                print(f"  [WARN] {pe_type} missing in imagenet100 master summary; skipping")
        if curves:
            return curves

    curves = {}
    for pe_type in ["learned", "sinusoidal", "rope", "alibi"]:
        if pe_type not in agg:
            print(f"  [WARN] {pe_type} missing in ImageNet MI JSON; skipping")
            continue
        block = agg[pe_type].get("mi_per_layer", agg[pe_type])
        mean = block.get("mean_per_layer", block.get("mean", block.get("mi_mean")))
        std = block.get("std_per_layer", block.get("std", None))
        if mean is None:
            # Some per-seed JSONs use mi_per_layer.values.
            mean = block.get("values", None)
        if mean is None:
            print(f"  [WARN] {pe_type} has no mean_per_layer; skipping")
            continue
        mean = np.asarray(mean, dtype=np.float64)
        if std is None:
            std = np.zeros_like(mean)
        else:
            std = np.asarray(std, dtype=np.float64)
        curves[pe_type] = (mean, std)
    if not curves:
        raise ValueError(f"No usable MI curves found in {mi_agg_path}")
    return curves


def make_figure3_mi_taxonomy_and_2d_control(
        imagenet_mi_agg_path=FIG3_IMAGENET_MI_AGG_PATH,
        cifar_fixed_mi_path=FIG3_CIFAR_FIXED_MI_PATH,
        cifar_matched_mi_path=FIG3_CIFAR_MATCHED_MI_PATH,
        output_dir=OUTPUT_DIR,
        output_name=FIG3_OUTPUT_NAME):
    """Generate main Figure 3 as PNG (300 dpi) and PDF.

    Panel (a): ImageNet-100 per-layer position-attention MI taxonomy.
    Panel (b): CIFAR-100 canonical paired 1D / fixed-2D / matched-2D
               ALiBi MI control, for CLS-inclusive and patch-only estimators.
    """
    curves = _load_imagenet_mi_taxonomy(imagenet_mi_agg_path)
    controls = _compose_cifar_control_values(cifar_fixed_mi_path,
                                             cifar_matched_mi_path)

    # Manual layout is used here instead of constrained_layout so that
    # the two panels remain compact and the legends do not obscure data.
    fig = plt.figure(figsize=(10.8, 4.25), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.12)

    # --- Panel (a): MI taxonomy curves ---
    ax = fig.add_subplot(gs[0, 0])
    layers = np.arange(1, 13)
    for pe_type in ["learned", "sinusoidal", "rope", "alibi"]:
        if pe_type not in curves:
            continue
        mean, std = curves[pe_type]
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, marker="o", markersize=4.4, linewidth=1.7,
                color=COLORS[pe_type], label=DISPLAY_NAME[pe_type])
        ax.fill_between(x, mean - std, mean + std,
                        color=COLORS[pe_type], alpha=0.11, linewidth=0)
    ax.set_title("(a) ImageNet-100 MI taxonomy", loc="left", fontsize=12, pad=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Position--attention MI (bits)")
    ax.set_xticks(layers)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.30)
    # Shift legend left from the extreme top-right corner to avoid
    # covering the high late-layer 1D-ALiBi segment.
    ax.legend(loc="upper right", bbox_to_anchor=(0.67, 0.995),
              frameon=True, framealpha=0.95, borderpad=0.35,
              handlelength=1.4, borderaxespad=0.35)

    # --- Panel (b): CIFAR controls ---
    ax = fig.add_subplot(gs[0, 1])
    estimators = ["incl_cls_all_tokens", "excl_cls_patch_only"]
    estimator_labels = ["CLS-incl.", "Patch-only\n(no CLS)"]
    method_labels = ["1D-ALiBi", "Fixed 2D-ALiBi", "Matched 2D-ALiBi"]
    method_colors = [COLORS["alibi"], COLORS["alibi_2d"], "#8c564b"]

    x = np.arange(len(estimators))
    width = 0.24
    offsets = np.array([-width, 0.0, width])
    for i, method in enumerate(method_labels):
        means = [controls[e][method][0] for e in estimators]
        stds = [controls[e][method][1] for e in estimators]
        ax.bar(x + offsets[i], means, width=width,
               yerr=stds, capsize=3.0, linewidth=0.6,
               edgecolor="black", color=method_colors[i], alpha=0.88,
               label=method)

    ax.set_title("(b) CIFAR-100 ALiBi controls ($n{=}12$)",
                 loc="left", fontsize=12, pad=12)
    ax.set_ylabel("Mean MI over layers (bits)")
    ax.set_xticks(x)
    ax.set_xticklabels(estimator_labels)
    # Add modest headroom so the legend can sit inside the axes without
    # covering the left matched-2D bar.
    ax.set_ylim(0, 6.25)
    ax.grid(True, axis="y", alpha=0.30)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.03),
              frameon=True, framealpha=0.95, borderpad=0.35,
              handlelength=1.2, borderaxespad=0.20)

    # The paper caption carries the full title; omitting the in-figure
    # suptitle prevents collisions with the panel titles.
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18,
                        top=0.88, wspace=0.12)

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{output_name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_name}.png + .pdf")
    plt.close(fig)


# ============================================================================
# FIGURE 6: Per-layer mutual information (canonical Round 2 estimator)
# ============================================================================
# Input source differs from Figures 2/4a/4b/5a/5b: rather than re-computing
# from checkpoints, this figure plots aggregated per-layer MI values that
# were produced by revision_analysis_v4.ipynb (Round 2 canonical estimator,
# _mutual_information_discrete + compute_mi_per_layer_v2). The aggregate
# JSON contains mean and std across 3 seeds per PE type. We deliberately
# do not re-compute here because (a) it requires the validation set and a
# forward pass for each of 5 PE types x 3 seeds (expensive); (b) the
# canonical estimator already ran in revision_analysis_v4.ipynb and its
# numbers are what Table S1 and the manuscript text cite.
def make_figure6_mutual_information(mi_agg_path=MI_AGG_PATH):
    """Plot per-layer MI from the canonical Round 2 estimator aggregate JSON.

    Single panel (MI per layer) only, because the manuscript narrative
    around Figure 6 discusses only MI per layer; attention entropy is not
    referenced and is therefore not plotted to keep the figure focused.
    """
    if not os.path.isfile(mi_agg_path):
        print(f"  [WARN] Figure 6 input not found at {mi_agg_path}")
        print(f"         Skipping Figure 6. (To generate, ensure "
              f"revision_analysis_v4.ipynb has written _aggregate.json.)")
        return

    with open(mi_agg_path) as f:
        agg = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    layers = list(range(1, 13))

    for pe_type in PE_TYPES:
        if pe_type not in agg or "mi_per_layer" not in agg[pe_type]:
            print(f"  [WARN] {pe_type} missing in aggregate JSON; skipping line")
            continue
        block = agg[pe_type]["mi_per_layer"]
        mean = np.array(block["mean_per_layer"])
        std  = np.array(block.get("std_per_layer", np.zeros_like(mean)))
        ax.plot(layers, mean, marker="o", markersize=5, linewidth=1.6,
                color=COLORS[pe_type], label=DISPLAY_NAME[pe_type], alpha=0.9)
        ax.fill_between(layers, mean - std, mean + std,
                        color=COLORS[pe_type], alpha=0.12)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_title("Per-layer position-attention MI on ImageNet-100\n"
                 "(canonical discrete estimator, $n{=}3$ seeds)",
                 fontsize=11)
    ax.set_xticks(layers)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    name = "06_mutual_information"
    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}.png + .pdf")
    plt.close(fig)




# ============================================================================
# FIGURE 7: Noise ablation + PE removal
# ============================================================================
def make_figure7_noise_ablation(analysis_data_path=FIG7_ANALYSIS_DATA_PATH):
    """Regenerate Figure 7 from saved analysis_data.json.

    This uses saved data only. It does not load checkpoints and does not
    recompute noise ablations. Delta annotations are percentage-point
    differences and are therefore labelled as "pp", not "%".
    """
    if not os.path.isfile(analysis_data_path):
        raise FileNotFoundError(
            f"Figure 7 input not found: {analysis_data_path}. "
            "Pass --fig7-analysis-json /path/to/analysis_data.json"
        )

    with open(analysis_data_path, "r") as f:
        data = json.load(f)

    pe_types = [pe for pe in ["learned", "sinusoidal", "rope", "alibi"] if pe in data]
    if not pe_types:
        raise ValueError(
            f"No expected PE blocks found in {analysis_data_path}. "
            "Expected keys include learned, sinusoidal, rope, alibi."
        )

    common_seeds = None
    for pe in pe_types:
        seeds_here = {str(s) for s in data[pe].keys()}
        common_seeds = seeds_here if common_seeds is None else common_seeds & seeds_here
    seeds = sorted(common_seeds, key=lambda s: (len(s), s)) if common_seeds else []
    if not seeds:
        raise ValueError("No common seeds found for Figure 7 in analysis_data.json.")

    ref_block = data[pe_types[0]][seeds[0]]["noise_ablation"]
    noise_levels = ref_block.get("noise_levels", [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: noise curves
    for pe in pe_types:
        means = []
        stds = []
        for idx in range(len(noise_levels)):
            vals = [
                data[pe][s]["noise_ablation"]["accuracies"][idx]
                for s in seeds
            ]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)

        ax1.plot(
            noise_levels,
            means,
            "o-",
            color=COLORS[pe],
            linewidth=2,
            markersize=6,
            label=f"{DISPLAY_NAME[pe]} PE" if pe in ("learned", "sinusoidal") else DISPLAY_NAME[pe],
        )
        ax1.fill_between(
            noise_levels,
            means - stds,
            means + stds,
            color=COLORS[pe],
            alpha=0.15,
        )

    ax1.axhline(
        y=1.0,
        color="red",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Chance (1.0%)",
    )
    ax1.set_xlabel(r"Noise level ($\times \sigma_{\mathrm{PE}}$)", fontsize=14)
    ax1.set_ylabel("Test accuracy (%)", fontsize=14)
    ax1.set_title("Noise robustness — ImageNet-100", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2, 90)

    # Right panel: PE removal bar chart
    baselines = {}
    no_pe_accs = {}
    deltas = {}
    for pe in pe_types:
        base_vals = [
            data[pe][s]["noise_ablation"]["accuracies"][0]
            for s in seeds
        ]
        nope_vals = [
            data[pe][s]["noise_ablation"]["accuracy_no_pe"]
            for s in seeds
        ]
        baselines[pe] = float(np.mean(base_vals))
        no_pe_accs[pe] = float(np.mean(nope_vals))
        deltas[pe] = baselines[pe] - no_pe_accs[pe]

    x = np.arange(len(pe_types))
    width = 0.35

    ax2.bar(
        x - width / 2,
        [baselines[pe] for pe in pe_types],
        width,
        color=[COLORS[pe] for pe in pe_types],
        label="With PE",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.bar(
        x + width / 2,
        [no_pe_accs[pe] for pe in pe_types],
        width,
        color=[COLORS[pe] for pe in pe_types],
        alpha=0.4,
        label="Without PE",
        edgecolor="black",
        linewidth=0.5,
    )

    # Delta labels: percentage points, not percentages.
    for i, pe in enumerate(pe_types):
        delta = deltas[pe]
        y_pos = max(baselines[pe], no_pe_accs[pe]) + 2
        ax2.text(
            i,
            y_pos,
            rf"$\Delta={delta:.1f}$ pp",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="red",
        )

    ax2.set_xlabel("")
    ax2.set_ylabel("Test accuracy (%)", fontsize=14)
    ax2.set_title("Effect of PE removal — ImageNet-100", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{DISPLAY_NAME[pe]} PE" if pe in ("learned", "sinusoidal") else DISPLAY_NAME[pe]
         for pe in pe_types],
        fontsize=14,
    )
    ax2.legend(
    fontsize=12,
    loc="lower left",
    frameon=True,
    framealpha=0.95,
    )
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 95)
    ax1.tick_params(axis="both", labelsize=11)
    ax2.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    save_figure(fig, "07_noise_ablation")
    plt.close(fig)



# ============================================================================
# FIGURE S3: Analytical 1D-vs-2D distance distortion
# ============================================================================
def _distance_distortion_ratios(grid_side):
    """Return r_ij = d_1D / d_2D over all unordered patch pairs on a k x k grid."""
    n = grid_side * grid_side
    idx = np.arange(n, dtype=np.int32)
    rows = idx // grid_side
    cols = idx % grid_side

    d1 = np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    d2 = np.hypot(cols[:, None] - cols[None, :], rows[:, None] - rows[None, :])
    tri = np.triu_indices(n, k=1)
    return d1[tri] / d2[tri]


def make_figureS3_distance_distortion():
    """Generate supplementary Figure S3 analytically (no external inputs).

    This version reproduces the older paper-style design:
      (a) mean distortion growth with grid size, with the 8×8 and 14×14 study
          grids highlighted;
      (b) ratio distributions for the two study grids with a no-distortion
          reference line at r_ij = 1.
    """
    grid_sides = [4, 6, 8, 10, 12, 14, 16, 20, 24, 32]
    summaries = {}
    for k in grid_sides:
        r = _distance_distortion_ratios(k)
        summaries[k] = {
            'ratios': r,
            'mean': float(np.mean(r)),
            'median': float(np.median(r)),
            'max': float(np.max(r)),
            'severe_frac': float(np.mean(r > 2.0)),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # Panel (a): mean distortion vs grid side
    means = [summaries[k]['mean'] for k in grid_sides]
    ax1.plot(
        grid_sides,
        means,
        marker='o',
        linewidth=1.9,
        markersize=5.0,
        color='#3b4d61',
        markerfacecolor='#3b4d61',
        markeredgecolor='#3b4d61',
        alpha=0.95,
    )
    ax1.set_xlabel('Grid side (patches per side)')
    ax1.set_ylabel(r'Mean ratio $\, d_{\mathrm{1D}}/d_{\mathrm{2D}}$')
    ax1.set_title('(a) Distance distortion grows with grid size', fontsize=14)
    ax1.grid(True, alpha=0.30)
    ax1.set_xlim(3.5, 33.5)
    ax1.set_ylim(1.8, 21.2)

    # Highlight the two study grids with open markers and a compact legend.
    y8 = summaries[8]['mean']
    y14 = summaries[14]['mean']
    ax1.scatter([8], [y8], s=220, facecolors='none', edgecolors='#9467bd',
                linewidths=3.0, zorder=5,
                label=f'CIFAR-100 (8×8): mean ratio = {y8:.2f}')
    ax1.scatter([14], [y14], s=220, facecolors='none', edgecolors='#d62728',
                marker='s', linewidths=3.0, zorder=5,
                label=f'TIN/IN-100 (14×14): mean ratio = {y14:.2f}')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95)

    # Panel (b): ratio distributions for the two experimental grids.
    r8 = summaries[8]['ratios']
    r14 = summaries[14]['ratios']
    bins = np.linspace(1.0, 14.0, 25)
    ax2.hist(r8, bins=bins, density=True, alpha=0.50, color='#9467bd',
             edgecolor='#6c4a9c', linewidth=0.35,
             label=f'CIFAR-100 (8×8, N=65)\nmedian = {np.median(r8):.2f}')
    ax2.hist(r14, bins=bins, density=True, alpha=0.50, color='#d95f5f',
             edgecolor='#b03a3a', linewidth=0.35,
             label=f'TIN/IN-100 (14×14, N=197)\nmedian = {np.median(r14):.2f}')
    ax2.axvline(1.0, color='black', linestyle=':', linewidth=1.2)
    ax2.text(1.25, 0.31, 'no distortion  →', color='gray', fontsize=11)
    ax2.set_xlabel(r'Ratio $\, d_{\mathrm{1D}}/d_{\mathrm{2D}}$  (per patch pair)')
    ax2.set_ylabel('Probability density')
    ax2.set_title('(b) 14×14 grid shifts distribution toward higher ratios', fontsize=14)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(0.0, 16.0)
    ax2.set_ylim(0.0, 0.52)
    ax2.legend(
    loc="upper right",
    bbox_to_anchor=(1.02, 0.98),
    frameon=True,
    framealpha=0.95,
    )

    plt.tight_layout()
    save_figure(fig, 'figS3_distance_distortion')
    plt.close(fig)

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Regenerate TPAMI revision figures, including Figure 3, Figure 7, and Supplementary Figure S3."
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Directory for generated figures (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--only-fig3", action="store_true",
        help="Generate only main Figure 3 and exit. This avoids loading checkpoints.",
    )
    parser.add_argument(
        "--skip-fig3", action="store_true",
        help="Skip main Figure 3 when regenerating all other figures.",
    )
    parser.add_argument(
        "--fig3-imagenet-mi-json", default=FIG3_IMAGENET_MI_AGG_PATH,
        help="ImageNet-100 aggregate MI JSON for Figure 3 panel (a).",
    )
    parser.add_argument(
        "--fig3-cifar-fixed-mi-json", default=FIG3_CIFAR_FIXED_MI_PATH,
        help=("CIFAR-100 fixed-slope 1D-vs-2D ALiBi MI summary JSON "
              "for Figure 3 panel (b). Use the JSON output generated by "
              "the MI-control analysis, not a manual audit file."),
    )
    parser.add_argument(
        "--fig3-cifar-matched-mi-json", default=FIG3_CIFAR_MATCHED_MI_PATH,
        help="CIFAR-100 magnitude-matched 1D-vs-2D ALiBi MI summary JSON.",
    )
    parser.add_argument(
        "--fig3-output-name", default=FIG3_OUTPUT_NAME,
        help=f"Output basename for Figure 3 (default: {FIG3_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--fig7-analysis-json", default=FIG7_ANALYSIS_DATA_PATH,
        help="Saved analysis_data.json used to regenerate Figure 7.",
    )
    parser.add_argument(
        "--only-fig7", action="store_true",
        help="Generate only Figure 7 from saved analysis_data.json and exit.",
    )
    parser.add_argument(
        "--skip-fig7", action="store_true",
        help="Skip Figure 7 when regenerating all other figures.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.only_fig3:
        print("=" * 60)
        print("Generating main Figure 3 only")
        print("=" * 60)
        make_figure3_mi_taxonomy_and_2d_control(
            imagenet_mi_agg_path=args.fig3_imagenet_mi_json,
            cifar_fixed_mi_path=args.fig3_cifar_fixed_mi_json,
            cifar_matched_mi_path=args.fig3_cifar_matched_mi_json,
            output_dir=OUTPUT_DIR,
            output_name=args.fig3_output_name,
        )
        print(f"DONE. Figure saved to: {OUTPUT_DIR}")
        return

    if args.only_fig7:
        print("=" * 60)
        print("Generating Figure 7 only")
        print("=" * 60)
        make_figure7_noise_ablation(args.fig7_analysis_json)
        print(f"DONE. Figure saved to: {OUTPUT_DIR}")
        return

    print("=" * 60)
    print("Figures Revision - Generating figures for TPAMI submission")
    print("=" * 60)

    # Pre-flight check
    print("\n1. Pre-flight check: verifying checkpoints...")
    for pe_type in PE_TYPES:
        ckpt = os.path.join(CKPT_ROOT, f"{pe_type}_seed{SEED}",
                              "best_model.pth")
        status = "OK" if os.path.isfile(ckpt) else "MISSING"
        size_mb = os.path.getsize(ckpt) / 1e6 if os.path.isfile(ckpt) else 0
        print(f"     [{status}] {pe_type}: {size_mb:.0f} MB")

    # Data extraction
    print("\n2. Extracting PE matrices and computing metrics...")
    data = {}
    for pe_type in PE_TYPES:
        print(f"   Processing {pe_type}...")
        model = load_model(pe_type, IN100_ARCH, SEED)

        if pe_type in ("learned", "sinusoidal"):
            pe = extract_pe_matrix(model, pe_type, IN100_ARCH["num_positions"])
            data[pe_type] = {
                "pe":                pe,
                "cos_sim":           cosine_similarity_matrix(pe, pe_type),
                "per_dim_entropy":   per_dim_entropy(pe),
                "per_dim_variance":  per_dim_variance(pe),
            }
        elif pe_type == "rope":
            pe = extract_pe_matrix(model, pe_type, IN100_ARCH["num_positions"])
            wavelengths, active = rope_active_dims(model,
                                                     IN100_ARCH["num_positions"])
            # RoPE's rotation schedule is shared across attention heads by
            # construction: the same d_h/2 frequencies are broadcast over
            # the head dimension during attention. As a consequence,
            # "per-head intrinsic entropy" is not a well-defined diagnostic
            # for RoPE (all heads see identical cos/sin features). Its
            # intrinsic structure is summarised by scalar diagnostics
            # (active-band fraction, intrinsic rank, effective wavelengths),
            # reported in Table tab:rope-intrinsic of the manuscript.
            data[pe_type] = {
                "pe":            pe,
                "cos_sim":       cosine_similarity_matrix(pe, pe_type),
                "wavelengths":   wavelengths,
                "active_mask":   active,
            }
        elif pe_type in ("alibi", "alibi_2d"):
            bias = extract_pe_matrix(model, pe_type,
                                        IN100_ARCH["num_positions"])
            # Re-extract slopes via the same robust path (mirrors extract_pe_matrix)
            attn = model.blocks[0].attn
            alibi = attn.alibi if hasattr(attn, "alibi") else attn
            slopes = None
            for name, p in alibi.named_parameters(recurse=False):
                if "slope" in name.lower() or p.numel() == IN100_ARCH["num_heads"]:
                    slopes = p.detach().cpu().squeeze().numpy()
                    break
            if slopes is None:
                for name, b in alibi.named_buffers(recurse=False):
                    if "slope" in name.lower() or b.numel() == IN100_ARCH["num_heads"]:
                        slopes = b.detach().cpu().squeeze().numpy()
                        break
            data[pe_type] = {
                "bias":              bias,
                "cos_sim":           cosine_similarity_matrix(bias, pe_type),
                "per_head_entropy":  per_head_intrinsic_entropy(bias),
                "slopes":            slopes,
            }
        del model
        torch.cuda.empty_cache()

    # Quick sanity print
    print("\n3. Sanity numbers:")
    for pe_type in PE_TYPES:
        d = data[pe_type]
        if pe_type in ("learned", "sinusoidal"):
            print(f"   {pe_type}: entropy mean = "
                  f"{d['per_dim_entropy'].mean():.2f} bits, "
                  f"variance mean = {d['per_dim_variance'].mean():.4f}")
        elif pe_type == "rope":
            # RoPE has no per-head intrinsic entropy (rotation schedule
            # is head-shared); report scalar diagnostics instead.
            n_active = int(d["active_mask"].sum())
            n_total = len(d["active_mask"])
            print(f"   {pe_type}: active rotation bands = "
                  f"{n_active}/{n_total} "
                  f"({100.0 * n_active / n_total:.1f}%)")
        else:
            print(f"   {pe_type}: per-head entropy mean = "
                  f"{d['per_head_entropy'].mean():.2f} bits")
    print(f"   rope: active dimensions = "
          f"{data['rope']['active_mask'].sum()}/{len(data['rope']['active_mask'])}")

    # Generate figures
    print("\n4. Generating figures...")
    make_figure2_cosine_similarity(data)
    make_figure3_pca_tsne(data)
    make_figure4a_embedding_entropy(data)
    make_figure4b_intrinsic_entropy(data)
    make_figure5a_embedding_variance(data)
    make_figure5b_intrinsic_structure(data)
    make_figure6_mutual_information(args.fig3_imagenet_mi_json)
    if not args.skip_fig7:
        make_figure7_noise_ablation(args.fig7_analysis_json)
    make_figureS3_distance_distortion()

    if not args.skip_fig3:
        if args.fig3_cifar_fixed_mi_json and args.fig3_cifar_matched_mi_json:
            make_figure3_mi_taxonomy_and_2d_control(
                imagenet_mi_agg_path=args.fig3_imagenet_mi_json,
                cifar_fixed_mi_path=args.fig3_cifar_fixed_mi_json,
                cifar_matched_mi_path=args.fig3_cifar_matched_mi_json,
                output_dir=OUTPUT_DIR,
                output_name=args.fig3_output_name,
            )
        else:
            print("  [WARN] Skipping main Figure 3: provide both "
                  "--fig3-cifar-fixed-mi-json and "
                  "--fig3-cifar-matched-mi-json, or use --skip-fig3.")

    # Summary
    print("\n" + "=" * 60)
    print(f"DONE. Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  {f}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
