"""
analyze_alibi_seeds.py
======================
Post-training statistical analysis for the 12-seed 1D-vs-2D ALiBi comparison
on CIFAR-100. Reads training_history.json files from results_cifar100_v2/
and reports:

  - per-seed best val_acc for both PE types
  - paired differences (2D - 1D) for the seeds where both runs completed
  - paired t-test  (df = n-1)
  - Wilcoxon signed-rank test  (non-parametric, conservative on small n)
  - sign test  (most conservative)
  - paired percentile bootstrap CI 
  - a ready-to-paste sentence and a ready-to-paste table row

Usage
-----
Either as a Python script:
    python analyze_alibi_seeds.py
        [--results /path/to/results_cifar100_v2]
        [--min-seeds 10]
        [--bootstrap-resamples 10000]

Or as a Colab cell:
    !python /content/drive/MyDrive/analyze_alibi_seeds.py

Requires only numpy and scipy.
"""

import os
import json
import argparse
import numpy as np
from typing import List, Optional


SEEDS_DEFAULT = [42, 123, 456, 1, 5, 7, 11, 13, 21, 99, 2024, 31337]
RESULTS_DEFAULT = '/content/drive/My Drive/pe_experiment/results_cifar100_v2'


def load_best_acc(results_dir: str, pe_type: str, seed: int,
                    required_epochs: int = 300) -> Optional[float]:
    """Return best val_acc if run finished, else None."""
    hist_path = os.path.join(results_dir, f'{pe_type}_seed{seed}',
                              'training_history.json')
    if not os.path.isfile(hist_path):
        return None
    with open(hist_path) as f:
        h = json.load(f)
    accs = h.get('val_acc', [])
    if len(accs) < required_epochs:
        return None
    return float(max(accs))


def paired_bootstrap_ci(diffs: np.ndarray, n_resamples: int = 10000,
                          alpha: float = 0.05, rng_seed: int = 0):
    """Paired percentile bootstrap CI on the mean of paired differences."""
    rng = np.random.default_rng(rng_seed)
    n = len(diffs)
    if n < 2:
        return (float('nan'), float('nan'), float('nan'))
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (float(diffs.mean()), lo, hi)


def paired_t_test(diffs: np.ndarray, alpha: float = 0.05):
    """Standard paired t-test on differences (one-sample t-test vs. 0)."""
    from scipy import stats
    n = len(diffs)
    if n < 2:
        return None
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0.0)
    mean = float(diffs.mean())
    sem = float(diffs.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    ci_lo = mean - tcrit * sem
    ci_hi = mean + tcrit * sem
    return {
        'mean': mean, 'sd': float(diffs.std(ddof=1)),
        'sem': sem, 't': float(t_stat), 'df': n - 1,
        'p_two_sided': float(p_val), 'ci_lo': ci_lo, 'ci_hi': ci_hi,
    }


def wilcoxon_signed_rank(diffs: np.ndarray):
    """Wilcoxon signed-rank test (paired non-parametric)."""
    from scipy import stats
    if len(diffs) < 2:
        return None
    # Drop exact zeros (Wilcoxon convention)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return None
    try:
        w_stat, p_val = stats.wilcoxon(nonzero, zero_method='wilcox',
                                          alternative='two-sided')
        return {'W': float(w_stat), 'p_two_sided': float(p_val),
                 'n_effective': int(len(nonzero))}
    except Exception as e:
        return {'error': str(e)}


def sign_test(diffs: np.ndarray):
    """Sign test (two-sided binomial on the sign of differences)."""
    from scipy import stats
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n < 1:
        return None
    n_pos = int((nonzero > 0).sum())
    # Two-sided binomial test (scipy >= 1.7 has binomtest; older uses binom_test)
    try:
        res = stats.binomtest(n_pos, n, p=0.5, alternative='two-sided')
        p_val = float(res.pvalue)
    except AttributeError:
        p_val = float(stats.binom_test(n_pos, n, p=0.5,
                                          alternative='two-sided'))
    return {'n_positive': n_pos, 'n_total': n, 'p_two_sided': p_val}


def fmt_diff_list(seeds: List[int], diffs: np.ndarray) -> str:
    return ', '.join(f'{s}:{d:+.2f}' for s, d in zip(seeds, diffs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=RESULTS_DEFAULT,
                          help='results_cifar100_v2 directory')
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS_DEFAULT)),
                          help='Comma-separated list of seed indices')
    parser.add_argument('--required-epochs', type=int, default=300)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--min-seeds', type=int, default=2,
                          help='Minimum paired seeds required to run tests')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    print(f'Results dir: {args.results}')
    print(f'Seeds: {seeds}')
    print(f'Required epochs: {args.required_epochs}\n')

    # ---- Load per-seed best accuracies ----
    rows_1d, rows_2d = [], []
    paired_seeds, diffs_list = [], []

    print('Per-seed best validation accuracy:')
    print('=' * 70)
    print(f"  {'seed':>6}  {'1D-ALiBi':>10}  {'2D-ALiBi':>10}  "
          f"{'2D - 1D':>10}")
    print('-' * 70)
    for seed in seeds:
        a1 = load_best_acc(args.results, 'alibi',    seed,
                            args.required_epochs)
        a2 = load_best_acc(args.results, 'alibi_2d', seed,
                            args.required_epochs)
        rows_1d.append(a1); rows_2d.append(a2)

        s1 = f'{a1:.2f}' if a1 is not None else '   --'
        s2 = f'{a2:.2f}' if a2 is not None else '   --'
        if a1 is not None and a2 is not None:
            d = a2 - a1
            paired_seeds.append(seed)
            diffs_list.append(d)
            sd = f'{d:+.2f}'
        else:
            sd = '   --'
        print(f"  {seed:>6}  {s1:>10}  {s2:>10}  {sd:>10}")

    diffs = np.array(diffs_list, dtype=np.float64)
    n = len(diffs)
    print('-' * 70)
    print(f'  Paired seeds available: {n}/{len(seeds)}')

    if n < args.min_seeds:
        print(f'\n[!] Not enough paired completed runs ({n} < '
              f'{args.min_seeds}). Tests skipped.')
        return

    # ---- Aggregate stats ----
    a1_done = [a for a in rows_1d if a is not None]
    a2_done = [a for a in rows_2d if a is not None]

    print('\nAggregate (over completed runs):')
    print(f'  1D-ALiBi: n={len(a1_done):2d}, '
          f'mean={np.mean(a1_done):.2f} ± {np.std(a1_done, ddof=1):.2f}')
    print(f'  2D-ALiBi: n={len(a2_done):2d}, '
          f'mean={np.mean(a2_done):.2f} ± {np.std(a2_done, ddof=1):.2f}')

    print('\nPaired-difference summary (2D - 1D):')
    print(f'  n           = {n}')
    print(f'  mean        = {diffs.mean():+.3f} pp')
    print(f'  sd          = {diffs.std(ddof=1):.3f} pp')
    print(f'  n_positive  = {int((diffs > 0).sum())} / {n}')
    print(f'  n_negative  = {int((diffs < 0).sum())} / {n}')
    print(f'  range       = [{diffs.min():+.2f}, {diffs.max():+.2f}]')
    print(f'  per-seed    = {fmt_diff_list(paired_seeds, diffs)}')

    # ---- Tests ----
    print('\n' + '=' * 70)
    print('STATISTICAL TESTS  (H0: mean paired difference = 0)')
    print('=' * 70)

    # Paired t-test
    t_res = paired_t_test(diffs)
    if t_res is not None:
        sig_t = 'YES' if t_res['p_two_sided'] < 0.05 else 'no'
        print(f"\n  Paired t-test (df={t_res['df']}):")
        print(f"    t = {t_res['t']:+.3f},  p (two-sided) = "
              f"{t_res['p_two_sided']:.4f}   significant @ 0.05: {sig_t}")
        print(f"    95% t-CI  = [{t_res['ci_lo']:+.2f}, "
              f"{t_res['ci_hi']:+.2f}] pp")
        print(f"    excludes 0: "
              f"{'YES' if t_res['ci_lo'] > 0 or t_res['ci_hi'] < 0 else 'no'}")

    # Wilcoxon signed-rank
    w_res = wilcoxon_signed_rank(diffs)
    if w_res is not None and 'error' not in w_res:
        sig_w = 'YES' if w_res['p_two_sided'] < 0.05 else 'no'
        print(f"\n  Wilcoxon signed-rank "
              f"(n_eff={w_res['n_effective']}):")
        print(f"    W = {w_res['W']:.2f},  p (two-sided) = "
              f"{w_res['p_two_sided']:.4f}   significant @ 0.05: {sig_w}")
    elif w_res is not None:
        print(f"\n  Wilcoxon: error -- {w_res['error']}")

    # Sign test
    s_res = sign_test(diffs)
    if s_res is not None:
        sig_s = 'YES' if s_res['p_two_sided'] < 0.05 else 'no'
        print(f"\n  Sign test:")
        print(f"    {s_res['n_positive']}/{s_res['n_total']} positive,  "
              f"p (two-sided) = {s_res['p_two_sided']:.4f}   "
              f"significant @ 0.05: {sig_s}")

    # Bootstrap (kept for continuity with the manuscript)
    boot_mean, boot_lo, boot_hi = paired_bootstrap_ci(
        diffs, n_resamples=args.bootstrap_resamples)
    sig_b = 'YES' if (boot_lo > 0 or boot_hi < 0) else 'no'
    print(f"\n  Paired percentile bootstrap "
          f"({args.bootstrap_resamples:,} resamples):")
    print(f"    mean = {boot_mean:+.3f} pp,  "
          f"95% CI = [{boot_lo:+.2f}, {boot_hi:+.2f}] pp   "
          f"excludes 0: {sig_b}")

    # ---- Ready-to-paste outputs ----
    print('\n' + '=' * 70)
    print('READY-TO-PASTE FOR MANUSCRIPT')
    print('=' * 70)

    if t_res is not None:
        print('\n  Table 3 row (2D vs. 1D, CIFAR-100):')
        sig_marker = (r'$^{\dagger}$'
                      if t_res['p_two_sided'] < 0.05 else '')
        print(f'    \\textbf{{+{diffs.mean():.2f} pp{sig_marker}}}  &  '
              f'\\textbf{{[{t_res["ci_lo"]:+.2f}, {t_res["ci_hi"]:+.2f}]}}'
              f'  (paired t, n={n})')

        print('\n  Suggested sentence:')
        if t_res['p_two_sided'] < 0.05 and t_res['ci_lo'] > 0:
            print(f"    'On CIFAR-100, 2D-ALiBi yields a mean accuracy gain "
                  f"of +{diffs.mean():.2f} pp over 1D-ALiBi across {n} "
                  f"random seeds (paired t-test: t({t_res['df']})="
                  f"{t_res['t']:.2f}, p={t_res['p_two_sided']:.3f}; "
                  f"95% CI [{t_res['ci_lo']:+.2f}, {t_res['ci_hi']:+.2f}] "
                  f"pp, excluding zero).'")
        else:
            n_pos = int((diffs > 0).sum())
            print(f"    'On CIFAR-100, 2D-ALiBi exhibits a directional "
                  f"improvement over 1D-ALiBi of +{diffs.mean():.2f} pp "
                  f"across {n} random seeds ({n_pos}/{n} seeds positive; "
                  f"paired t-test 95% CI "
                  f"[{t_res['ci_lo']:+.2f}, {t_res['ci_hi']:+.2f}] pp, "
                  f"which {'excludes' if (t_res['ci_lo'] > 0 or t_res['ci_hi'] < 0) else 'includes'} "
                  f"zero).'")


if __name__ == '__main__':
    main()
