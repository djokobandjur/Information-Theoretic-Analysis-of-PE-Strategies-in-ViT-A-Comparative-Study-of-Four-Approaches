
# ============================================================
# CLS-controlled MI suite for 1D-vs-2D ALiBi
# ============================================================
# Edit only the paths/seeds in this config block. The rest of the notebook
# remains unchanged. This section uses:
#   - compute_mi_per_layer_v2(...)
#   - _mean_mi_over_seeds(...)
# and saves both per-seed and aggregate outputs.

import os
import json
import csv
import time
import numpy as np

MI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mi_cls_control")
os.makedirs(MI_OUTPUT_DIR, exist_ok=True)

# ----------------------------
# EDIT THESE PATHS / SEEDS
# ----------------------------
# Assumed default folder layout inside each root:
#   <models_root>/<pe_type>_seed<seed>/best_model.pth
# where pe_type is "alibi" or "alibi_2d".
#
# If one cohort uses a different folder naming convention, set
# "folder_template" in that cohort, e.g. "{pe_type}/seed_{seed}".
#
# If you need exact per-model folders instead of a root, add:
#   "model_dirs": {
#       "alibi":    {42: "/path/to/alibi_seed42_folder", ...},
#       "alibi_2d": {42: "/path/to/alibi_2d_seed42_folder", ...},
#   }
# In that case "models_root" and "folder_template" are ignored.

CIFAR_CANONICAL_SEEDS = [1, 5, 7, 11, 13, 21, 42, 99, 123, 456, 2024, 31337]

# Canonical 1D-ALiBi / fixed 2D-ALiBi CIFAR root.
# Expected subfolders: alibi_seed{seed}/best_model.pth and alibi_2d_seed{seed}/best_model.pth
MODELS_ROOT_CIFAR_CANONICAL = "/content/drive/My Drive/pe_experiment/results_cifar100_v2"

# Magnitude-matched 2D-ALiBi CIFAR root. In the current training script this is
# usually the same results root, with subfolders alibi_2d_matched_seed{seed}/.
MODELS_ROOT_CIFAR_MATCHED_2D = "/content/drive/My Drive/pe_experiment/results_cifar100_v2"

MODELS_ROOT_TINYIN_CANONICAL = "/content/drive/MyDrive/TODO_TINYIMAGENET_CANONICAL_ROOT"

# Exploratory roots default to the existing notebook roots. Change if needed.
MODELS_ROOT_CIFAR_EXPLORATORY = MODELS_ROOT_CIFAR
MODELS_ROOT_IN100_EXPLORATORY = MODELS_ROOT_IN100

def _matched_cifar_model_dirs(seeds):
    """Model folder map for 1D-ALiBi vs magnitude-matched 2D-ALiBi.

    Edit the folder names here if your Drive layout differs.
    """
    return {
        "alibi": {
            int(s): os.path.join(MODELS_ROOT_CIFAR_CANONICAL,
                                 f"alibi_seed{int(s)}")
            for s in seeds
        },
        # The loader instantiates pe_type='alibi_2d'; the folder name marks
        # that these checkpoints contain magnitude-matched slopes.
        "alibi_2d": {
            int(s): os.path.join(MODELS_ROOT_CIFAR_MATCHED_2D,
                                 f"alibi_2d_matched_seed{int(s)}")
            for s in seeds
        },
    }


MI_COHORTS = [
    {
        "key": "cifar100_canonical_n12",
        "dataset": "CIFAR-100",
        "protocol": "canonical",
        "role": "primary",
        "models_root": MODELS_ROOT_CIFAR_CANONICAL,
        "folder_template": "{pe_type}_seed{seed}",
        "arch": CIFAR_ARCH,
        "val_loader_fn": lambda: get_cifar100_val_loader(
            CIFAR_DATA_DIR, BATCH_SIZE, NUM_WORKERS),
        "seeds": CIFAR_CANONICAL_SEEDS,
        "inferential_ci": True,
    },
    {
        "key": "cifar100_canonical_matched2d_n12",
        "dataset": "CIFAR-100",
        "protocol": "canonical_magnitude_matched",
        "role": "diagnostic_control",
        "model_dirs": _matched_cifar_model_dirs(CIFAR_CANONICAL_SEEDS),
        "arch": CIFAR_ARCH,
        "val_loader_fn": lambda: get_cifar100_val_loader(
            CIFAR_DATA_DIR, BATCH_SIZE, NUM_WORKERS),
        "seeds": CIFAR_CANONICAL_SEEDS,
        "inferential_ci": True,
    },
    {
        "key": "tinyimagenet_canonical_n3",
        "dataset": "TinyImageNet",
        "protocol": "canonical",
        "role": "secondary_descriptive",
        "models_root": MODELS_ROOT_TINYIN_CANONICAL,
        "folder_template": "{pe_type}_seed{seed}",
        "arch": TINYIN_ARCH,
        "val_loader_fn": lambda: get_tinyin_val_loader(
            TINYIN_VAL_DIR, BATCH_SIZE, NUM_WORKERS),
        "seeds": [42, 123, 456],
        "inferential_ci": False,
    },
    {
        "key": "cifar100_exploratory_n3",
        "dataset": "CIFAR-100",
        "protocol": "exploratory",
        "role": "sensitivity_descriptive",
        "models_root": MODELS_ROOT_CIFAR_EXPLORATORY,
        "folder_template": "{pe_type}_seed{seed}",
        "arch": CIFAR_ARCH,
        "val_loader_fn": lambda: get_cifar100_val_loader(
            CIFAR_DATA_DIR, BATCH_SIZE, NUM_WORKERS),
        "seeds": [42, 123, 456],
        "inferential_ci": False,
    },
    {
        "key": "imagenet100_exploratory_n3",
        "dataset": "ImageNet-100",
        "protocol": "exploratory",
        "role": "sensitivity_descriptive",
        "models_root": MODELS_ROOT_IN100_EXPLORATORY,
        "folder_template": "{pe_type}_seed{seed}",
        "arch": IN100_ARCH,
        "val_loader_fn": lambda: get_in100_val_loader(
            IN100_VAL_DIR, BATCH_SIZE, NUM_WORKERS),
        "seeds": [42, 123, 456],
        "inferential_ci": False,
    },
]


def _load_model_for_mi(pe_type, seed, cohort):
    """Load model for the MI suite.

    Supports either:
      1. Standard root/template layout, using existing load_model(...) when
         possible.
      2. Explicit per-model folder mapping via cohort["model_dirs"].
    """
    arch = cohort["arch"]

    # Explicit per-model folders.
    if "model_dirs" in cohort and cohort["model_dirs"] is not None:
        model_dirs = cohort["model_dirs"]
        try:
            folder = model_dirs[pe_type][seed]
        except Exception:
            # Also allow tuple-key mapping: {(pe_type, seed): folder}
            folder = model_dirs[(pe_type, seed)]
        ckpt_path = find_checkpoint(folder)
        model = build_model(pe_type, arch).to(device)
        state = torch.load(ckpt_path, map_location=device)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    # Root/template layout.
    root = cohort["models_root"]
    template = cohort.get("folder_template", "{pe_type}_seed{seed}")

    if template == "{pe_type}_seed{seed}":
        # Use the notebook's original loader for the default layout.
        return load_model(pe_type, seed, root, arch)

    folder = os.path.join(root, template.format(pe_type=pe_type, seed=seed))
    ckpt_path = find_checkpoint(folder)
    model = build_model(pe_type, arch).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _model_mi_cache_path(cohort_key, pe_type, seed, exclude_cls):
    suffix = "excl_cls" if exclude_cls else "incl_cls"
    return os.path.join(MI_OUTPUT_DIR, cohort_key,
                        f"{pe_type}_seed{seed}_{suffix}.json")


def _compute_or_load_model_mi(cohort, pe_type, seed, val_loader, exclude_cls):
    """Return dict with per-layer and mean-over-layers MI for one model."""
    cache_path = _model_mi_cache_path(cohort["key"], pe_type, seed, exclude_cls)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.isfile(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    t0 = time.time()
    model = _load_model_for_mi(pe_type, seed, cohort)
    mi_layers = compute_mi_per_layer_v2(
        model, val_loader, device,
        n_batches=N_BATCHES_MI,
        exclude_cls=exclude_cls,
    )
    if isinstance(mi_layers, torch.Tensor):
        mi_layers = mi_layers.detach().cpu().numpy().tolist()
    mi_layers = [float(x) for x in mi_layers]

    out = {
        "cohort": cohort["key"],
        "dataset": cohort["dataset"],
        "protocol": cohort["protocol"],
        "pe_type": pe_type,
        "seed": int(seed),
        "exclude_cls": bool(exclude_cls),
        "estimator": "excl_cls_patch_only" if exclude_cls else "incl_cls_all_tokens",
        "n_batches_mi": int(N_BATCHES_MI),
        "num_positions": int(cohort["arch"]["num_positions"]),
        "mi_per_layer": mi_layers,
        "mean_over_layers": float(np.mean(mi_layers)),
        "min_layer_mi": float(np.min(mi_layers)),
        "max_layer_mi": float(np.max(mi_layers)),
        "runtime_seconds": float(time.time() - t0),
    }

    with open(cache_path, "w") as f:
        json.dump(out, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def _mean_mi_over_seeds(pe_type, seeds, models_root, arch, val_loader,
                        exclude_cls):
    """Backward-compatible helper from the previous MI check.

    Mean-over-layers MI, averaged across available seeds.
    Uses compute_mi_per_layer_v2(...) exactly as before.
    """
    per_seed = []
    for seed in seeds:
        try:
            model = load_model(pe_type, seed, models_root, arch)
        except Exception as e:
            print(f"      [skip {pe_type} seed{seed}: {type(e).__name__}: {e}]")
            continue
        mi_layers = compute_mi_per_layer_v2(
            model, val_loader, device,
            n_batches=N_BATCHES_MI,
            exclude_cls=exclude_cls,
        )
        per_seed.append(float(np.mean(mi_layers)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not per_seed:
        return None
    return float(np.mean(per_seed))


def _paired_bootstrap_ci(values, n_boot=10000, ci=95, seed=2026):
    """Percentile paired bootstrap CI for the mean paired difference.

    values are paired deltas, e.g. Delta_s = MI_2D_s - MI_1D_s.
    Use only for n large enough to be meaningful here (n>=10).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = values[idx].mean()
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot, [alpha, 100 - alpha])
    return float(lo), float(hi)


def _summarize_paired_rows(rows, estimator, inferential_ci):
    """Summarize seed-paired 1D-vs-2D MI rows for one estimator."""
    sub = [r for r in rows if r["estimator"] == estimator and
           r["mi_1d"] is not None and r["mi_2d"] is not None]
    if not sub:
        return None

    mi_1d = np.array([r["mi_1d"] for r in sub], dtype=np.float64)
    mi_2d = np.array([r["mi_2d"] for r in sub], dtype=np.float64)
    deltas = np.array([r["delta_2d_minus_1d"] for r in sub], dtype=np.float64)
    reductions = np.array([r["reduction_percent"] for r in sub], dtype=np.float64)

    n = len(sub)
    out = {
        "estimator": estimator,
        "n_pairs": int(n),
        "mi_1d_mean": float(mi_1d.mean()),
        "mi_1d_std": float(mi_1d.std(ddof=1)) if n > 1 else 0.0,
        "mi_2d_mean": float(mi_2d.mean()),
        "mi_2d_std": float(mi_2d.std(ddof=1)) if n > 1 else 0.0,
        "delta_2d_minus_1d_mean": float(deltas.mean()),
        "delta_2d_minus_1d_std": float(deltas.std(ddof=1)) if n > 1 else 0.0,
        "delta_2d_minus_1d_min": float(deltas.min()),
        "delta_2d_minus_1d_max": float(deltas.max()),
        "mean_reduction_percent": float(reductions.mean()),
        "std_reduction_percent": float(reductions.std(ddof=1)) if n > 1 else 0.0,
        "n_2d_lower_mi": int(np.sum(deltas < 0)),
        "n_2d_equal_mi": int(np.sum(deltas == 0)),
        "n_2d_higher_mi": int(np.sum(deltas > 0)),
        "ci95_delta_2d_minus_1d": None,
        "ci_note": "descriptive only; no inferential CI reported",
    }

    if inferential_ci and n >= 10:
        out["ci95_delta_2d_minus_1d"] = _paired_bootstrap_ci(deltas)
        out["ci_note"] = "paired percentile bootstrap over seed-paired deltas"
    return out


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def run_paired_alibi_mi_cohort(cohort):
    """Run CLS-inclusive and CLS-excluded MI for paired 1D/2D ALiBi."""
    seeds = cohort["seeds"]
    if not seeds:
        raise ValueError(
            f"{cohort['key']} has no seeds. Fill CIFAR_CANONICAL_SEEDS "
            "or disable this cohort."
        )

    print("\n" + "=" * 78)
    print(f"{cohort['key']} | {cohort['dataset']} | {cohort['protocol']} | "
          f"N={cohort['arch']['num_positions']} | seeds={seeds}")
    print("=" * 78)

    val_loader = cohort["val_loader_fn"]()
    rows = []
    raw = {}

    for seed in seeds:
        raw[str(seed)] = {}
        for exclude_cls in [False, True]:
            estimator = "excl_cls_patch_only" if exclude_cls else "incl_cls_all_tokens"
            per_pe = {}
            for pe_type in ["alibi", "alibi_2d"]:
                try:
                    out = _compute_or_load_model_mi(
                        cohort, pe_type, seed, val_loader, exclude_cls)
                    per_pe[pe_type] = out
                    print(f"  {seed:>6} | {estimator:20s} | {pe_type:8s} "
                          f"MI={out['mean_over_layers']:.4f}")
                except Exception as e:
                    per_pe[pe_type] = {
                        "error": f"{type(e).__name__}: {e}",
                        "mean_over_layers": None,
                    }
                    print(f"  {seed:>6} | {estimator:20s} | {pe_type:8s} "
                          f"FAILED: {type(e).__name__}: {e}")

            mi_1d = per_pe["alibi"].get("mean_over_layers")
            mi_2d = per_pe["alibi_2d"].get("mean_over_layers")

            if mi_1d is None or mi_2d is None or mi_1d == 0:
                delta = None
                reduction = None
            else:
                delta = float(mi_2d - mi_1d)
                reduction = float(100.0 * (mi_1d - mi_2d) / mi_1d)

            row = {
                "cohort": cohort["key"],
                "dataset": cohort["dataset"],
                "protocol": cohort["protocol"],
                "role": cohort["role"],
                "seed": int(seed),
                "estimator": estimator,
                "mi_1d": mi_1d,
                "mi_2d": mi_2d,
                "delta_2d_minus_1d": delta,
                "reduction_percent": reduction,
                "n_batches_mi": int(N_BATCHES_MI),
                "num_positions": int(cohort["arch"]["num_positions"]),
            }
            rows.append(row)
            raw[str(seed)][estimator] = per_pe

    summaries = {
        "incl_cls_all_tokens": _summarize_paired_rows(
            rows, "incl_cls_all_tokens", inferential_ci=cohort["inferential_ci"]),
        "excl_cls_patch_only": _summarize_paired_rows(
            rows, "excl_cls_patch_only", inferential_ci=cohort["inferential_ci"]),
    }

    out = {
        "cohort": cohort["key"],
        "dataset": cohort["dataset"],
        "protocol": cohort["protocol"],
        "role": cohort["role"],
        "seeds": [int(s) for s in seeds],
        "primary_estimator": "excl_cls_patch_only",
        "estimator_note": (
            "CLS-inclusive is retained as a sensitivity diagnostic. "
            "CLS-excluded patch-only is the primary topology-controlled estimator."
        ),
        "n_batches_mi": int(N_BATCHES_MI),
        "num_positions": int(cohort["arch"]["num_positions"]),
        "rows": rows,
        "summary": summaries,
        "raw": raw,
    }

    cohort_dir = os.path.join(MI_OUTPUT_DIR, cohort["key"])
    os.makedirs(cohort_dir, exist_ok=True)
    json_path = os.path.join(cohort_dir, "paired_alibi_mi_summary.json")
    # Repo-friendly alias with the cohort key in the filename, useful when
    # collecting all Figure 3 inputs into a single flat folder.
    alias_json_path = os.path.join(
        MI_OUTPUT_DIR, f"{cohort['key']}_paired_alibi_mi_summary.json")
    rows_csv = os.path.join(cohort_dir, "paired_alibi_mi_per_seed.csv")
    summary_csv = os.path.join(cohort_dir, "paired_alibi_mi_aggregate.csv")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    with open(alias_json_path, "w") as f:
        json.dump(out, f, indent=2)

    _write_csv(rows_csv, rows, [
        "cohort", "dataset", "protocol", "role", "seed", "estimator",
        "mi_1d", "mi_2d", "delta_2d_minus_1d", "reduction_percent",
        "n_batches_mi", "num_positions",
    ])

    summary_rows = []
    for estimator, s in summaries.items():
        if s is not None:
            rr = {"cohort": cohort["key"], "dataset": cohort["dataset"],
                  "protocol": cohort["protocol"], "role": cohort["role"]}
            rr.update(s)
            # csv-friendly CI fields
            ci = rr.pop("ci95_delta_2d_minus_1d", None)
            if ci is None:
                rr["ci95_delta_lo"] = None
                rr["ci95_delta_hi"] = None
            else:
                rr["ci95_delta_lo"] = ci[0]
                rr["ci95_delta_hi"] = ci[1]
            summary_rows.append(rr)

    _write_csv(summary_csv, summary_rows, [
        "cohort", "dataset", "protocol", "role", "estimator", "n_pairs",
        "mi_1d_mean", "mi_1d_std", "mi_2d_mean", "mi_2d_std",
        "delta_2d_minus_1d_mean", "delta_2d_minus_1d_std",
        "delta_2d_minus_1d_min", "delta_2d_minus_1d_max",
        "mean_reduction_percent", "std_reduction_percent",
        "n_2d_lower_mi", "n_2d_equal_mi", "n_2d_higher_mi",
        "ci95_delta_lo", "ci95_delta_hi", "ci_note",
    ])

    print("\n  Summary:")
    for estimator, s in summaries.items():
        if s is None:
            print(f"    {estimator}: n/a")
            continue
        ci_txt = ""
        if s["ci95_delta_2d_minus_1d"] is not None:
            lo, hi = s["ci95_delta_2d_minus_1d"]
            ci_txt = f", 95% CI Delta=[{lo:.4f}, {hi:.4f}]"
        print(
            f"    {estimator}: "
            f"1D={s['mi_1d_mean']:.4f}±{s['mi_1d_std']:.4f}, "
            f"2D={s['mi_2d_mean']:.4f}±{s['mi_2d_std']:.4f}, "
            f"Delta(2D-1D)={s['delta_2d_minus_1d_mean']:.4f}, "
            f"reduction={s['mean_reduction_percent']:.2f}%, "
            f"direction={s['n_2d_lower_mi']}/{s['n_pairs']} lower"
            f"{ci_txt}"
        )

    print(f"\n  -> wrote {json_path}")
    print(f"  -> wrote {rows_csv}")
    print(f"  -> wrote {summary_csv}")

    return out


# ----------------------------
# Run all configured cohorts
# ----------------------------
mi_suite_results = {}
for cohort in MI_COHORTS:
    mi_suite_results[cohort["key"]] = run_paired_alibi_mi_cohort(cohort)

master_mi_path = os.path.join(MI_OUTPUT_DIR, "all_paired_alibi_mi_summary.json")
with open(master_mi_path, "w") as f:
    json.dump(mi_suite_results, f, indent=2)

print("\n✅ CLS-controlled paired ALiBi MI suite complete.")
print(f"Master MI summary: {master_mi_path}")
