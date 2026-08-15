#!/usr/bin/env python3
"""
Verify TPAMI R2 canonical CIFAR-100 three-arm accuracy analysis.

Usage:
    python verify_three_arm_decomposition.py revision_results/three_arm_decomposition.json

The script recomputes the three-arm statistics from the per-seed accuracies
stored in the JSON artifact and checks them against the recorded values.

Requires:
    numpy
    scipy
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats


ABS_TOL = 1e-12
CI_TOL = 1e-12


def close(a, b, tol=ABS_TOL):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def close_pair(got, expected, tol=CI_TOL):
    return len(got) == 2 and all(close(g, e, tol) for g, e in zip(got, expected))


class Verifier:
    def __init__(self):
        self.failures = 0

    def check(self, condition, label, detail=None):
        if condition:
            print(f"[PASS] {label}")
        else:
            self.failures += 1
            suffix = f" -- {detail}" if detail else ""
            print(f"[FAIL] {label}{suffix}")

    def finish(self):
        if self.failures:
            print(f"\nVERIFICATION FAILED: {self.failures} check(s) failed.")
            return 1
        print("\nALL CHECKS PASSED")
        return 0


def exact_sign_test(diff):
    nz = diff[diff != 0]
    positive = int(np.sum(nz > 0))
    negative = int(np.sum(nz < 0))
    p = stats.binomtest(positive, n=len(nz), p=0.5, alternative="two-sided").pvalue
    return positive, negative, int(np.sum(diff == 0)), float(p)


def paired_stats(x, y):
    diff = x - y
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    t_res = stats.ttest_rel(x, y)
    se = sd / math.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    ci = [mean - crit * se, mean + crit * se]
    w = stats.wilcoxon(diff, alternative="two-sided", method="exact")
    pos, neg, zero, sign_p = exact_sign_test(diff)
    dz = mean / sd
    return {
        "diff": diff,
        "n": n,
        "mean": mean,
        "sd": sd,
        "t": float(t_res.statistic),
        "df": n - 1,
        "p": float(t_res.pvalue),
        "ci": [float(ci[0]), float(ci[1])],
        "wilcoxon_stat": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
        "sign_positive": pos,
        "sign_negative": neg,
        "sign_zero": zero,
        "sign_p": sign_p,
        "dz": float(dz),
    }


def paired_bootstrap_ci(diff, resamples, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    ci = np.percentile(means, [2.5, 97.5])
    return [float(ci[0]), float(ci[1])]


def holm_two(p_c2, p_c3):
    items = sorted([("C2", float(p_c2)), ("C3", float(p_c3))], key=lambda z: z[1])
    adjusted = {}
    running = 0.0
    m = 2
    for i, (name, p) in enumerate(items):
        candidate = min(1.0, (m - i) * p)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_three_arm_decomposition.py "
              "revision_results/three_arm_decomposition.json")
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}")
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    v = Verifier()

    # ---- Metadata / structure ----
    expected_seeds = [1, 5, 7, 11, 13, 21, 42, 99, 123, 456, 2024, 31337]
    rows = data["per_seed"]
    row_seeds = [int(r["seed"]) for r in rows]

    v.check(data.get("schema_version") == "1.0", "schema version = 1.0")
    v.check(data.get("no_new_models_trained") is True, "no new models trained flag")
    v.check(
        data.get("metric")
        == "best validation accuracy (%) = max(val_acc) per 300-epoch training history",
        "validation-accuracy metric definition",
    )
    v.check(data.get("seed_order") == expected_seeds, "declared seed order")
    v.check(row_seeds == expected_seeds, "per-seed row alignment: 12/12")

    # ---- Arm data ----
    a1 = np.array([r["accuracy_1d"] for r in rows], dtype=float)
    a2 = np.array([r["accuracy_2d_fixed"] for r in rows], dtype=float)
    am = np.array([r["accuracy_2d_matched"] for r in rows], dtype=float)

    arms = {
        "1d": a1,
        "2d_fixed": a2,
        "2d_matched": am,
    }

    for name, values in arms.items():
        rec = data["arm_summaries"][name]
        v.check(close(np.mean(values), rec["mean"]),
                f"{name} arm mean")
        v.check(close(np.std(values, ddof=1), rec["sample_sd"]),
                f"{name} arm sample SD")

    # ---- Stored per-seed differences ----
    expected_diffs = {
        "C1_fixed_minus_1d_pp": a2 - a1,
        "C2_matched_minus_1d_pp": am - a1,
        "C3_matched_minus_fixed_pp": am - a2,
    }
    for field, diff in expected_diffs.items():
        stored = np.array([r[field] for r in rows], dtype=float)
        v.check(np.allclose(stored, diff, atol=ABS_TOL, rtol=0.0),
                f"per-seed {field}")

    # ---- Recompute C1/C2/C3 ----
    contrasts = {
        "C1": (a2, a1),
        "C2": (am, a1),
        "C3": (am, a2),
    }

    recomputed = {}
    for name, (x, y) in contrasts.items():
        calc = paired_stats(x, y)
        recomputed[name] = calc
        rec = data["contrasts"][name]["stats"]
        t_rec = rec["paired_t"]
        w_rec = rec["wilcoxon_exact_two_sided"]
        s_rec = rec["sign_test_exact_two_sided"]

        v.check(calc["n"] == rec["n"], f"{name} n")
        v.check(close(calc["mean"], rec["mean_difference_pp"]),
                f"{name} mean paired difference")
        v.check(close(calc["sd"], rec["sd_paired_differences_pp"]),
                f"{name} SD of paired differences")
        v.check(close(calc["t"], t_rec["t"]), f"{name} paired t statistic")
        v.check(calc["df"] == t_rec["df"], f"{name} degrees of freedom")
        v.check(close(calc["p"], t_rec["p_two_sided_raw"]),
                f"{name} paired t p-value")
        v.check(close_pair(calc["ci"], t_rec["ci95_t_pp"]),
                f"{name} 95% t-CI")
        v.check(close(calc["wilcoxon_stat"], w_rec["statistic"]),
                f"{name} Wilcoxon statistic")
        v.check(close(calc["wilcoxon_p"], w_rec["p"]),
                f"{name} Wilcoxon exact p-value")
        v.check(calc["sign_positive"] == s_rec["positive"]
                and calc["sign_negative"] == s_rec["negative"]
                and calc["sign_zero"] == s_rec["zero"],
                f"{name} sign counts")
        v.check(close(calc["sign_p"], s_rec["p"]),
                f"{name} exact sign-test p-value")
        v.check(close(calc["dz"], rec["cohens_dz"]),
                f"{name} paired Cohen dz")

    # ---- C1 provenance policy ----
    c1_stats = data["contrasts"]["C1"]["stats"]
    v.check("paired_bootstrap" not in c1_stats,
            "C1 contains no new paired_bootstrap field")
    v.check(
        "C1_bootstrap_policy" in data.get("provenance_check", {}),
        "C1 bootstrap provenance policy recorded",
    )

    # ---- C2/C3 bootstraps ----
    for name in ("C2", "C3"):
        rec = data["contrasts"][name]["stats"]["paired_bootstrap"]
        v.check(rec["resamples"] == 10000,
                f"{name} bootstrap resamples = 10000")
        v.check(rec["rng"] == "numpy.random.default_rng(PCG64)",
                f"{name} RNG identifier")
        v.check(rec["rng_seed"] == 20260814,
                f"{name} RNG seed = 20260814")
        ci = paired_bootstrap_ci(
            recomputed[name]["diff"],
            rec["resamples"],
            rec["rng_seed"],
        )
        v.check(close_pair(ci, rec["ci95_percentile_pp"]),
                f"{name} paired-bootstrap 95% percentile CI")

    # ---- Holm correction over C2/C3 only ----
    raw_c2 = recomputed["C2"]["p"]
    raw_c3 = recomputed["C3"]["p"]
    holm = holm_two(raw_c2, raw_c3)

    v.check(data["multiplicity"]["secondary_family"] == ["C2", "C3"],
            "Holm family = C2/C3 only")
    v.check(data["multiplicity"]["method"] == "Holm-Bonferroni",
            "multiplicity method = Holm-Bonferroni")
    for name in ("C2", "C3"):
        recorded = data["contrasts"][name]["stats"]["paired_t"][
            "p_holm_secondary_family"
        ]
        v.check(close(holm[name], recorded),
                f"{name} Holm-adjusted p-value")

    # ---- Provenance reproduction of R1 primary endpoint ----
    prov = data["provenance_check"]
    rec = prov["recomputed_from_seed_aligned_values"]
    c1 = recomputed["C1"]

    v.check(prov.get("R1_primary_endpoint_reproduced") is True,
            "R1 primary endpoint marked reproduced")
    v.check(close(c1["t"], rec["t"])
            and close(c1["p"], rec["p"])
            and close_pair(c1["ci"], rec["ci95_t_pp"])
            and close(c1["wilcoxon_p"], rec["wilcoxon_p"])
            and close(c1["sign_p"], rec["sign_test_p"]),
            "C1 reproduces recorded R1 seed-aligned endpoint")

    # ---- Report runtime versions (metadata check) ----
    print(f"\nRuntime NumPy: {np.__version__}")
    try:
        import scipy
        print(f"Runtime SciPy: {scipy.__version__}")
    except Exception:
        pass
    print(f"JSON NumPy:    {data.get('software', {}).get('numpy')}")
    print(f"JSON SciPy:    {data.get('software', {}).get('scipy')}")

    return v.finish()


if __name__ == "__main__":
    raise SystemExit(main())
