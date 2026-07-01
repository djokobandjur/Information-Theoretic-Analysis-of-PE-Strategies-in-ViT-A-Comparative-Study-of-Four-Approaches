#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rerun cross-dataset row/column linear probes for additive positional encodings.

This script is protocol-matched to the probe_analysis(...) function in
full_scale_experiment.py:

  - CLS excluded: pe_matrix[1:]
  - row labels: positions // grid_size
  - column labels: positions % grid_size
  - classifier: LogisticRegression(max_iter=2000, C=1.0)
  - no StandardScaler
  - CV: StratifiedKFold(n_splits=min(5, min_class_count),
                         shuffle=True, random_state=42)
  - per-seed score: cross_val_score mean x 100
  - Learned PE: report mean +/- across-seed std
  - Sinusoidal PE: generated analytically unless a checkpoint template is provided;
                   report fixed value +/- 0.0

Outputs:
  <outdir>/per_seed_probe_results.csv
  <outdir>/probe_summary.csv
  <outdir>/probe_table_supp.tex

Typical use:
  python rerun_cross_dataset_probes_protocol_matched.py \
    --learned-checkpoint-template "results/{dataset}/learned_seed{seed}/best_model.pth" \
    --seeds 42 123 456 \
    --outdir probe_rerun_outputs

If your folders use lowercase dataset names, use aliases, for example:
  python rerun_cross_dataset_probes_protocol_matched.py \
    --learned-checkpoint-template "results/{dataset_alias}/learned_seed{seed}/best_model.pth" \
    --dataset-alias ImageNet-100=imagenet100 CIFAR-100=cifar100 TinyImageNet=tinyimagenet \
    --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


DATASETS = {
    "ImageNet-100": {"grid": 14, "n_positions": 197},
    "CIFAR-100": {"grid": 8, "n_positions": 65},
    "TinyImageNet": {"grid": 14, "n_positions": 197},
}


def parse_dataset_alias(items: Optional[List[str]]) -> Dict[str, str]:
    """Parse CLI aliases like ImageNet-100=imagenet100."""
    alias = {name: name for name in DATASETS}
    if not items:
        return alias

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --dataset-alias item {item!r}. Expected format DatasetName=folder_name"
            )
        left, right = item.split("=", 1)
        left = left.strip()
        right = right.strip()
        if left not in DATASETS:
            raise ValueError(
                f"Unknown dataset alias key {left!r}. Valid keys: {list(DATASETS)}"
            )
        alias[left] = right
    return alias


def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return a cleaned state_dict-like tensor dictionary."""
    try:
        ckpt: Any = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {path}")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, val in ckpt.items():
        if not torch.is_tensor(val):
            continue

        kk = key
        changed = True
        while changed:
            changed = False
            for prefix in ["module.", "model.", "net.", "_orig_mod."]:
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
                    changed = True

        cleaned[kk] = val.detach().cpu()

    if not cleaned:
        raise ValueError(f"No tensors found in checkpoint: {path}")

    return cleaned


def find_positional_tensor(
    state: Dict[str, torch.Tensor],
    n_positions: int,
    d_model_hint: int = 768,
) -> Tuple[str, np.ndarray]:
    """
    Find an additive positional encoding tensor of shape [N, D] or [1, N, D].

    Prioritises keys matching full_scale_experiment.py:
      learned:    pos_encoding.pos_embed, shape [1, N, D]
      sinusoidal: pos_encoding.pe,        shape [1, N, D]
    """
    priority_terms = [
        "pos_encoding.pos_embed",
        "pos_encoding.pe",
        "pos_embed",
        "positional_embedding",
        "position_embedding",
        "positional_encoding",
        "position_encoding",
        "pos_encoding",
        "pos_emb",
    ]

    candidates = []

    for key, tensor in state.items():
        arr = tensor
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr2 = arr.squeeze(0)
        elif arr.ndim == 2:
            arr2 = arr
        else:
            continue

        if arr2.ndim != 2:
            continue

        if arr2.shape[0] == n_positions:
            pe = arr2
            orientation_score = 2
        elif arr2.shape[1] == n_positions:
            pe = arr2.T
            orientation_score = 1
        else:
            continue

        if pe.shape[0] != n_positions:
            continue

        d_score = 2 if pe.shape[1] == d_model_hint else 0
        lower = key.lower()
        name_score = 0
        for idx, term in enumerate(priority_terms):
            if term.lower() in lower:
                name_score = max(name_score, len(priority_terms) - idx)

        score = 100 * name_score + 10 * d_score + orientation_score
        candidates.append((score, key, pe.numpy().astype(np.float64)))

    if not candidates:
        available_shapes = {k: tuple(v.shape) for k, v in state.items() if torch.is_tensor(v)}
        preview = list(available_shapes.items())[:30]
        raise ValueError(
            f"Could not find PE tensor with N={n_positions}. "
            f"First checkpoint tensor keys/shapes: {preview}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, key, pe = candidates[0]
    print(f"[INFO] selected PE tensor: key={key!r}, shape={pe.shape}, score={score}")
    return key, pe


def sinusoidal_encoding(n_positions: int, d_model: int) -> np.ndarray:
    """
    Standard Vaswani-style sinusoidal PE matching full_scale_experiment.py.

    Important: generate the full N=P+1 sequence first, with CLS at position 0;
    the probe then excludes CLS via pe[1:].
    """
    pe = np.zeros((n_positions, d_model), dtype=np.float64)
    position = np.arange(0, n_positions, dtype=np.float64)[:, None]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float64) * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


def make_labels(num_patches: int, grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match full_scale_experiment.py probe_analysis labels."""
    positions = np.arange(num_patches)
    rows = positions // grid_size
    cols = positions % grid_size
    return positions, rows, cols


def probe_task_accuracy(
    patches: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Protocol-matched linear probe for one target label.

    Matches:
      clf = LogisticRegression(max_iter=2000, C=1.0)
      cv = StratifiedKFold(n_splits=min(5, min_class_count),
                           shuffle=True, random_state=42)
      cross_val_score(..., scoring='accuracy')
    """
    clf = LogisticRegression(max_iter=2000, C=1.0)

    min_class_count = int(np.min(np.bincount(labels)))
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        clf.fit(patches, labels)
        acc = float(clf.score(patches, labels) * 100.0)
        return {
            "cv_mean": acc,
            "cv_std": 0.0,
            "n_splits": 1,
            "min_class_count": min_class_count,
            "fallback_train_acc": 1.0,
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, patches, labels, cv=cv, scoring="accuracy")

    return {
        "cv_mean": float(scores.mean() * 100.0),
        "cv_std": float(scores.std() * 100.0),
        "n_splits": int(n_splits),
        "min_class_count": min_class_count,
        "fallback_train_acc": 0.0,
    }


def run_probe_on_pe(
    pe_matrix: np.ndarray,
    grid_size: int,
    random_state: int = 42,
) -> Dict[str, float]:
    """Run row, column, and exact-position probes on an [N, D] PE matrix."""
    num_patches = grid_size * grid_size

    if pe_matrix.shape[0] < num_patches + 1:
        raise ValueError(
            f"PE matrix has too few positions: {pe_matrix.shape[0]} "
            f"for grid {grid_size}x{grid_size} requiring {num_patches + 1}"
        )

    patches = pe_matrix[1 : 1 + num_patches]
    positions, rows, cols = make_labels(num_patches, grid_size)

    row = probe_task_accuracy(patches, rows, random_state=random_state)
    col = probe_task_accuracy(patches, cols, random_state=random_state)
    pos = probe_task_accuracy(patches, positions, random_state=random_state)

    return {
        "row_acc": row["cv_mean"],
        "row_cv_std": row["cv_std"],
        "row_n_splits": row["n_splits"],
        "column_acc": col["cv_mean"],
        "column_cv_std": col["cv_std"],
        "column_n_splits": col["n_splits"],
        "position_acc": pos["cv_mean"],
        "position_cv_std": pos["cv_std"],
        "position_n_splits": pos["n_splits"],
    }


def resolve_checkpoint_path(
    template: str,
    dataset: str,
    dataset_alias: str,
    pe: str,
    seed: int,
) -> Path:
    return Path(
        template.format(
            dataset=dataset,
            dataset_alias=dataset_alias,
            pe=pe,
            seed=seed,
        )
    )


def run_learned(
    dataset: str,
    dataset_alias: str,
    seed: int,
    template: str,
    d_model: int,
    random_state: int,
) -> Dict[str, Any]:
    info = DATASETS[dataset]
    path = resolve_checkpoint_path(template, dataset, dataset_alias, "learned", seed)

    if not path.exists():
        raise FileNotFoundError(f"Missing learned checkpoint: {path}")

    state = load_checkpoint(path)
    pe_key, pe = find_positional_tensor(
        state,
        n_positions=info["n_positions"],
        d_model_hint=d_model,
    )
    res = run_probe_on_pe(pe, grid_size=info["grid"], random_state=random_state)

    return {
        "dataset": dataset,
        "grid": f"{info['grid']}x{info['grid']}",
        "pe_type": "Learned",
        "seed": seed,
        "source": str(path),
        "pe_key": pe_key,
        **res,
    }


def run_sinusoidal(
    dataset: str,
    dataset_alias: str,
    d_model: int,
    random_state: int,
    checkpoint_template: Optional[str] = None,
    seed_for_checkpoint: int = 42,
) -> Dict[str, Any]:
    info = DATASETS[dataset]
    source = "generated"
    pe_key = "generated_sinusoidal"

    pe: Optional[np.ndarray] = None

    if checkpoint_template:
        path = resolve_checkpoint_path(
            checkpoint_template,
            dataset,
            dataset_alias,
            "sinusoidal",
            seed_for_checkpoint,
        )
        if path.exists():
            state = load_checkpoint(path)
            pe_key, pe = find_positional_tensor(
                state,
                n_positions=info["n_positions"],
                d_model_hint=d_model,
            )
            source = str(path)
            print(f"[INFO] using stored sinusoidal PE from {path}")
        else:
            print(f"[WARN] sinusoidal checkpoint not found; generating PE analytically: {path}")

    if pe is None:
        pe = sinusoidal_encoding(n_positions=info["n_positions"], d_model=d_model)

    res = run_probe_on_pe(pe, grid_size=info["grid"], random_state=random_state)

    return {
        "dataset": dataset,
        "grid": f"{info['grid']}x{info['grid']}",
        "pe_type": "Sinusoidal",
        "seed": "fixed",
        "source": source,
        "pe_key": pe_key,
        **res,
    }


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    out = []

    for dataset in ["ImageNet-100", "CIFAR-100", "TinyImageNet"]:
        for pe_type in ["Learned", "Sinusoidal"]:
            sub = results[(results["dataset"] == dataset) & (results["pe_type"] == pe_type)]
            if sub.empty:
                continue

            row_vals = sub["row_acc"].astype(float).to_numpy()
            col_vals = sub["column_acc"].astype(float).to_numpy()
            pos_vals = sub["position_acc"].astype(float).to_numpy()

            if pe_type == "Sinusoidal":
                row_std = 0.0
                col_std = 0.0
                pos_std = 0.0
            else:
                row_std = float(np.std(row_vals, ddof=1)) if len(row_vals) > 1 else np.nan
                col_std = float(np.std(col_vals, ddof=1)) if len(col_vals) > 1 else np.nan
                pos_std = float(np.std(pos_vals, ddof=1)) if len(pos_vals) > 1 else np.nan

            out.append(
                {
                    "dataset": dataset,
                    "grid": sub.iloc[0]["grid"],
                    "pe_type": pe_type,
                    "n": len(sub),
                    "row_mean": float(np.mean(row_vals)),
                    "row_std_across_seeds": row_std,
                    "column_mean": float(np.mean(col_vals)),
                    "column_std_across_seeds": col_std,
                    "position_mean": float(np.mean(pos_vals)),
                    "position_std_across_seeds": pos_std,
                    "row_cv_std_mean": float(sub["row_cv_std"].astype(float).mean()),
                    "column_cv_std_mean": float(sub["column_cv_std"].astype(float).mean()),
                    "position_cv_std_mean": float(sub["position_cv_std"].astype(float).mean()),
                }
            )

    return pd.DataFrame(out)


def fmt_mean_std(mean: float, std: float) -> str:
    if np.isnan(std):
        return f"{mean:.1f}"
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def grid_tex(grid: str) -> str:
    a, b = grid.split("x")
    return rf"${a}{{\\times}}{b}$"


def write_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cross-dataset linear-probe decodability ($\%$) for the two "
        r"additive positional encodings. Learned entries are the mean across "
        r"$3$ seeds with the across-seed standard deviation; each per-seed "
        r"estimate is itself a $5$-fold cross-validation mean stratified by "
        r"the target label. Sinusoidal entries are seed-independent because "
        r"the encoding is analytically fixed, so their across-seed standard "
        r"deviation is zero by construction. Chance is $7.1\%$ for the "
        r"$14{\times}14$ grids and $12.5\%$ for the $8{\times}8$ grid.}"
    )
    lines.append(r"\label{tab:supp-probes}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\hline")
    lines.append(r"Dataset (grid) & PE & Row (\%) & Column (\%) \\")
    lines.append(r"\hline")

    for dataset in ["ImageNet-100", "CIFAR-100", "TinyImageNet"]:
        sub = summary[summary["dataset"] == dataset]
        if sub.empty:
            continue
        first = True
        for pe_type in ["Learned", "Sinusoidal"]:
            row = sub[sub["pe_type"] == pe_type]
            if row.empty:
                continue
            r = row.iloc[0]
            dataset_cell = f"{dataset} ({grid_tex(str(r['grid']))})" if first else ""
            first = False
            row_text = fmt_mean_std(float(r["row_mean"]), float(r["row_std_across_seeds"]))
            col_text = fmt_mean_std(float(r["column_mean"]), float(r["column_std_across_seeds"]))
            lines.append(f"{dataset_cell} & {pe_type} & {row_text} & {col_text} \\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Protocol-matched cross-dataset row/column linear probes."
    )

    parser.add_argument(
        "--learned-checkpoint-template",
        required=True,
        help=(
            "Template for learned checkpoints. Supported placeholders: "
            "{dataset}, {dataset_alias}, {pe}, {seed}. "
            "Example: results/{dataset_alias}/learned_seed{seed}/best_model.pth"
        ),
    )
    parser.add_argument(
        "--sinusoidal-checkpoint-template",
        default=None,
        help=(
            "Optional template for sinusoidal checkpoints. If omitted or if the "
            "file is missing, sinusoidal PE is generated analytically."
        ),
    )
    parser.add_argument(
        "--dataset-alias",
        nargs="*",
        default=None,
        help=(
            "Optional aliases for paths, e.g. "
            "ImageNet-100=imagenet100 CIFAR-100=cifar100 TinyImageNet=tinyimagenet"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ImageNet-100", "CIFAR-100", "TinyImageNet"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Learned PE seeds to evaluate.",
    )
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--cv-random-state", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="probe_rerun_outputs")

    args = parser.parse_args()

    aliases = parse_dataset_alias(args.dataset_alias)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["resolved_dataset_alias"] = aliases
    (outdir / "probe_rerun_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    records = []

    for dataset in args.datasets:
        dataset_alias = aliases[dataset]
        print("\n" + "=" * 72)
        print(f"Dataset: {dataset} (path alias: {dataset_alias})")
        print("=" * 72)

        for seed in args.seeds:
            print(f"[RUN] Learned seed={seed}")
            rec = run_learned(
                dataset=dataset,
                dataset_alias=dataset_alias,
                seed=seed,
                template=args.learned_checkpoint_template,
                d_model=args.d_model,
                random_state=args.cv_random_state,
            )
            records.append(rec)
            print(
                f"      row={rec['row_acc']:.2f} "
                f"column={rec['column_acc']:.2f} "
                f"position={rec['position_acc']:.2f}"
            )

        print("[RUN] Sinusoidal fixed")
        rec = run_sinusoidal(
            dataset=dataset,
            dataset_alias=dataset_alias,
            d_model=args.d_model,
            random_state=args.cv_random_state,
            checkpoint_template=args.sinusoidal_checkpoint_template,
            seed_for_checkpoint=args.seeds[0],
        )
        records.append(rec)
        print(
            f"      row={rec['row_acc']:.2f} "
            f"column={rec['column_acc']:.2f} "
            f"position={rec['position_acc']:.2f}"
        )

    df = pd.DataFrame(records)
    df_path = outdir / "per_seed_probe_results.csv"
    df.to_csv(df_path, index=False)

    summary = summarise(df)
    summary_path = outdir / "probe_summary.csv"
    summary.to_csv(summary_path, index=False)

    tex_path = outdir / "probe_table_supp.tex"
    write_latex_table(summary, tex_path)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {df_path.resolve()}")
    print(f"  {summary_path.resolve()}")
    print(f"  {tex_path.resolve()}")


if __name__ == "__main__":
    main()
