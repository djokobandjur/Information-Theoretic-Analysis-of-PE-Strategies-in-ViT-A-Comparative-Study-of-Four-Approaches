"""
PATCH 1: Add 2D-ALiBi support to full_scale_experiment.py
==========================================================

Apply these THREE small edits to existing full_scale_experiment.py.
Each edit is a search-and-replace; old text must match exactly.

After applying, training is:
    python full_scale_experiment.py \
        --data_dir /path/to/imagenet100_resized \
        --output_dir /path/to/results_2d_alibi \
        --mode train \
        --pe_type alibi_2d \
        --seed 42 \
        --epochs 300

Repeat for --seed 123 and --seed 456.
"""

# ============================================================================
# EDIT 1 of 3
# WHERE: After the ALiBi class definition (around line ~150)
# WHAT:  Add the TwoDALiBi class
# ============================================================================

EDIT_1_FIND = '''class ALiBi(nn.Module):'''  # existing class -- find it

# Then, right BEFORE this class definition, INSERT the following:

EDIT_1_INSERT_BEFORE = '''
class TwoDALiBi(nn.Module):
    """ALiBi adapted to 2D image-patch grids using Euclidean distance.

    Standard 1D ALiBi uses |i - j| in the raster-scan sequence,
    which assigns the same penalty to a horizontal neighbor (distance 1
    in 1D) and a vertical neighbor (distance 14 in 1D for a 14x14 grid).
    2D-ALiBi replaces this with the true grid Euclidean distance:

        d_2D(i, j) = sqrt((r_i - r_j)^2 + (c_i - c_j)^2)

    CLS token convention: distance 0 to all tokens (acts as a global
    aggregator with no spatial locality penalty).

    Slope schedule: same geometric ratio as 1D ALiBi
        m_h = 2^(-8h / H), h in 1..H
    """

    def __init__(self, num_heads, num_patches_per_side=14, include_cls=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_patches_per_side = num_patches_per_side
        self.include_cls = include_cls

        num_patches = num_patches_per_side ** 2
        num_positions = num_patches + (1 if include_cls else 0)

        ratio = 2 ** (-8.0 / num_heads)
        slopes = torch.tensor([ratio ** i for i in range(1, num_heads + 1)])
        self.register_buffer('slopes', slopes.view(1, num_heads, 1, 1))

        # 2D Euclidean distance matrix
        # Patch i (0-indexed within patches) sits at grid (i // S, i % S)
        # Position 0 is CLS (distance 0 to all) when include_cls=True
        dist = torch.zeros(num_positions, num_positions)
        patch_offset = 1 if include_cls else 0
        coords = torch.zeros(num_patches, 2)
        for i in range(num_patches):
            coords[i, 0] = i // num_patches_per_side
            coords[i, 1] = i %  num_patches_per_side

        for i in range(num_patches):
            for j in range(num_patches):
                dr = coords[i, 0] - coords[j, 0]
                dc = coords[i, 1] - coords[j, 1]
                dist[i + patch_offset, j + patch_offset] = torch.sqrt(dr*dr + dc*dc)

        self.register_buffer('dist_2d', dist.unsqueeze(0).unsqueeze(0))
        # Backward-compat with existing ALiBi analysis hooks (some code
        # reads `.rel_dist`); we expose dist_2d under that name too.
        self.register_buffer('rel_dist', dist.unsqueeze(0).unsqueeze(0))

    def get_bias(self, seq_len):
        return -self.slopes * self.dist_2d[:, :, :seq_len, :seq_len]


'''


# ============================================================================
# EDIT 2 of 3
# WHERE: Inside MultiHeadAttention.__init__, around line 171-172
# WHAT:  Add the alibi_2d branch
# ============================================================================

EDIT_2_FIND = '''        elif pe_type == 'alibi':
            self.alibi = ALiBi(num_heads, num_positions)'''

EDIT_2_REPLACE = '''        elif pe_type == 'alibi':
            self.alibi = ALiBi(num_heads, num_positions)
        elif pe_type == 'alibi_2d':
            # num_positions = num_patches + 1 (CLS); recover grid side
            num_patches = num_positions - 1
            side = int(round(num_patches ** 0.5))
            assert side * side == num_patches, \\
                f"alibi_2d requires square patch grid, got num_patches={num_patches}"
            self.alibi = TwoDALiBi(num_heads, num_patches_per_side=side,
                                    include_cls=True)'''


# ============================================================================
# EDIT 3 of 3
# WHERE: Inside MultiHeadAttention.forward, around line 183 and line 196
# WHAT:  Make the 'alibi' check also recognize 'alibi_2d'
# ============================================================================

EDIT_3a_FIND = '''        attn_bias = self.alibi.get_bias(N) if self.pe_type == 'alibi' else None'''

EDIT_3a_REPLACE = '''        attn_bias = self.alibi.get_bias(N) if self.pe_type in ('alibi', 'alibi_2d') else None'''


EDIT_3b_FIND = '''            if self.pe_type == 'alibi':
                attn = attn + attn_bias'''

EDIT_3b_REPLACE = '''            if self.pe_type in ('alibi', 'alibi_2d'):
                attn = attn + attn_bias'''


# ============================================================================
# EDIT 4 of 3 (sorry, four after all):
# WHERE: VisionTransformer.__init__, around line 256
# WHAT:  Make rope/alibi check include alibi_2d
# ============================================================================

EDIT_4_FIND = '''        elif pe_type in ('rope', 'alibi'):
            self.pos_encoding = nn.Identity()  # No additive PE
        else:
            raise ValueError(f"Unknown PE type: {pe_type}")'''

EDIT_4_REPLACE = '''        elif pe_type in ('rope', 'alibi', 'alibi_2d'):
            self.pos_encoding = nn.Identity()  # No additive PE
        else:
            raise ValueError(f"Unknown PE type: {pe_type}")'''


# ============================================================================
# EDIT 5: argparse choices and default pe_types list
# WHERE: parse_args, around line 1399
# WHAT:  Allow alibi_2d in CLI choices
# ============================================================================

EDIT_5a_FIND = '''                       choices=['learned', 'sinusoidal', 'rope', 'alibi'],'''
EDIT_5a_REPLACE = '''                       choices=['learned', 'sinusoidal', 'rope', 'alibi', 'alibi_2d'],'''

# Optionally also expand the default "all" list. If you want to train
# alibi_2d via --mode train --pe_type alibi_2d, the change above is enough.
# If you want to include alibi_2d in the bulk "all PE types" run, also:

EDIT_5b_FIND = '''pe_types = [args.pe_type] if args.pe_type else ['learned', 'sinusoidal', 'rope', 'alibi']'''
EDIT_5b_REPLACE = '''pe_types = [args.pe_type] if args.pe_type else ['learned', 'sinusoidal', 'rope', 'alibi', 'alibi_2d']'''

# (Recommend: do NOT change EDIT_5b — keep the default list to the original
# 4 PE methods. Run alibi_2d explicitly with --pe_type alibi_2d so it doesn't
# get tangled with the existing analysis pipeline.)


# ============================================================================
# Application script (run this to apply all edits automatically)
# ============================================================================

import sys
from pathlib import Path

EDITS = [
    ("INSERT TwoDALiBi class",
     EDIT_1_FIND, None, EDIT_1_INSERT_BEFORE),
    ("MultiHeadAttention.__init__ branch",
     EDIT_2_FIND, EDIT_2_REPLACE, None),
    ("MultiHeadAttention.forward attn_bias",
     EDIT_3a_FIND, EDIT_3a_REPLACE, None),
    ("MultiHeadAttention.forward fallback",
     EDIT_3b_FIND, EDIT_3b_REPLACE, None),
    ("VisionTransformer PE-type whitelist",
     EDIT_4_FIND, EDIT_4_REPLACE, None),
    ("argparse choices",
     EDIT_5a_FIND, EDIT_5a_REPLACE, None),
]


def apply_patch(src_path, dst_path):
    """Apply all edits to src_path and write to dst_path."""
    src = Path(src_path).read_text()
    text = src
    for name, find, replace, insert_before in EDITS:
        if find not in text:
            print(f"  [SKIP] '{name}': pattern not found")
            continue
        if replace is not None:
            text = text.replace(find, replace, 1)
            print(f"  [OK]   '{name}': replaced")
        elif insert_before is not None:
            text = text.replace(find, insert_before + find, 1)
            print(f"  [OK]   '{name}': inserted before")
    Path(dst_path).write_text(text)
    print(f"\nPatched file written to: {dst_path}")
    print(f"Diff size: {len(text) - len(src):+d} characters")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python apply_2d_alibi_patch.py "
              "<input full_scale_experiment.py> <output patched file>")
        print("Example:")
        print("  python apply_2d_alibi_patch.py "
              "full_scale_experiment.py full_scale_experiment_2d.py")
        sys.exit(1)
    apply_patch(sys.argv[1], sys.argv[2])
