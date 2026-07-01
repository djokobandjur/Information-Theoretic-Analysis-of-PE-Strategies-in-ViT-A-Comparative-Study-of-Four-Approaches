# Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers: A Comparative Study of Four Approaches

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19063156.svg)](https://doi.org/10.5281/zenodo.19063156)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code and reproducibility package for the paper:

> **Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers: A Comparative Study of Four Approaches**
>
> Đoko Banđur, Miloš Banđur, and Branimir Jakšić
>
> Department of Electrical Engineering, Faculty of Technical Sciences, University of Priština – Kosovska Mitrovica
>
> *Preprint, 2026* · [doi:10.5281/zenodo.19063156](https://doi.org/10.5281/zenodo.19063156)

> **Scope of this repository.** This repo contains everything required to **reproduce the results**: the training/analysis code, the per-run notebooks, and the raw result files (JSON/CSV) behind every reported table and figure. It deliberately does **not** include the manuscript, the supplementary material, or the response to reviewers. Because the study trains **76 ViT-Base models** (~344 MB each), the trained weights are also not stored in the repository: for each of the 76 models we provide its best checkpoint (`best_model.pth`) and full training history (which records the per-epoch metrics) on Google Drive, linked below. Full-scale ImageNet-1K from-scratch evaluation is outside the methodological scope of the study: the release is designed to reproduce the controlled PE-mechanism comparisons reported for CIFAR-100, TinyImageNet, and ImageNet-100, not a recipe-dependent ImageNet-1K scale-up benchmark.

## Abstract

Vision Transformers rely on positional encoding (PE) to compensate for the permutation invariance of self-attention, yet the information-theoretic properties of different PE strategies remain insufficiently characterised. We present a systematic comparison of four established PE approaches — **Learned**, **Sinusoidal**, **RoPE**, and a **1D-ALiBi-style linear-bias variant** — augmented by a targeted **2D-ALiBi-style** diagnostic intervention, training **ViT-Base** across three datasets spanning the low-to-intermediate data regime: **CIFAR-100**, **TinyImageNet**, and **ImageNet-100**. The study comprises **76 trained models**, organised into a primary cross-method matrix, a canonical paired protocol, and a magnitude-matched 2D-ALiBi-style control. Models are analysed with a **two-track diagnostic suite** — embedding-space analyses (per-dimension variance, entropy, PCA, and protocol-matched row/column probes) for additive PE, and attention-space intrinsic analyses (bias-tensor rank/entropy, slope schedule, nominal bias scale, and RoPE short-wavelength band counting) for attention-space PE — together with attention–position mutual information, noise ablation, and PE-removal experiments. All pairwise accuracy gaps are evaluated with paired bootstrap 95% confidence intervals. The results suggest four distinct encoding regimes and a clear accuracy–robustness trade-off: **RoPE attains the best accuracy on every dataset** (84.51% on ImageNet-100), while **Learned PE is by far the most robust** (72.1% absolute accuracy under full PE removal vs. Sinusoidal's collapse to chance). Motivated by a **raster-distance mismatch** between 1D sequence distance and 2D patch-grid geometry, the 2D-ALiBi-style variant replaces the 1D-ALiBi-style raster-scan distance with the 2D Euclidean patch-grid distance and yields a paired accuracy gain in the canonical CIFAR-100 low-data regime (+0.45 pp, *p* = 0.013), while no statistically reliable difference is detected on ImageNet-100. A magnitude-matched control shows that the fixed-slope MI reduction is scale-sensitive rather than attributable to distance geometry alone.

## Contributions

1. **A two-track diagnostic framework** that respects the architectural distinction between additive PE in the embedding stream (Learned, Sinusoidal) and attention-space PE (RoPE, ALiBi-style variants): embedding-space diagnostics (per-dimension variance, Shannon entropy, PCA, and protocol-matched linear probes) for the former, and attention-space intrinsic diagnostics (bias-tensor rank/entropy, slope schedule, nominal bias scale, and RoPE short-wavelength band counting) for the latter — grounded in Shannon theory and the information bottleneck.
2. **An information-theoretic taxonomy of PE behaviour** classifying the four methods into four qualitatively distinct regimes — *"quiet and ubiquitous"* (Learned), *"loud and structured"* (Sinusoidal), *"graceful decay through rotation"* (RoPE), and *"persistent attention-space bias"* (ALiBi-style) — each with characteristic mutual-information, variance, and probe-decodability signatures.
3. **A quantitative accuracy–robustness trade-off** across PE strategies: per-dimension variance tracks noise robustness, while PE-removal sensitivity tracks the degree of implicit positional learning (Learned PE retains 99.7% of clean accuracy at 1.0×σ noise and 72.1% absolute under full PE removal; Sinusoidal collapses to chance), turning the trade-off into a principled deployment criterion.
4. **A grid-aware ALiBi-style distance-geometry intervention** via **2D-ALiBi-style**, which substitutes the 2D Euclidean patch-grid distance for the 1D-ALiBi-style raster-scan distance, plus a mean-magnitude-matched control. The benefit is data-regime-dependent (paired gain on CIFAR-100, no statistically reliable difference detected on ImageNet-100), and the control shows the mutual-information effect to be scale-sensitive rather than attributable to distance geometry alone.
5. **Cross-dataset validation and scaling patterns.** The four-regime taxonomy recurs descriptively across all three datasets; RoPE's short-wavelength band fraction scales with sequence length (28.1% at *N* = 65 → 37.5% at *N* = 197); the protocol-matched additive-PE row/column probes show that Sinusoidal PE's held-out column-probe failure recurs on every evaluated grid; and the ALiBi-style-vs-others MI ordering is robust to data regime.

In addition, we release the full training code, checkpoints, logs, and JSON/CSV-formatted diagnostic outputs for the cross-method matrix, the canonical paired protocols, the matched-control diagnostic, and the cross-dataset probe rerun, enabling reproduction of every figure and table.

## Central Hypotheses

- **Raster-distance mismatch (primary, tested).** The 1D-ALiBi-style raster-scan distance |*i*−*j*| is misaligned with 2D image patch-grid geometry; a 2D-aware distance metric may help — especially where training data is scarce. Tested directly by introducing a grid-aware 2D-ALiBi-style intervention.
- **Channel trade-off.** Concentrating the positional signal in a few high-variance channels (Sinusoidal) versus distributing it across many low-variance channels (Learned) trades efficiency against robustness.
- **Accuracy ≠ dependence.** Higher accuracy does not imply greater reliance on the explicit positional signal — made visible only under noise and PE removal.

## Key Findings

### Classification accuracy (%) — cross-method matrix

| PE method | ImageNet-100 (*N* = 197) | CIFAR-100 (*N* = 65) | TinyImageNet (*N* = 197)¹ |
|-----------|:------------------------:|:--------------------:|:-------------------------:|
| Learned | 79.44 ± 0.62 | 68.28 ± 0.38 | 52.19 |
| Sinusoidal | 81.46 ± 0.33 | 66.92 ± 0.50 | 54.11 |
| **RoPE** | **84.51 ± 0.41** | **73.30 ± 0.18** | **56.73** |
| 1D-ALiBi-style | 81.05 ± 0.35 | 67.66 ± 0.44 | 53.41 |
| 2D-ALiBi-style *(this work)* | 81.17 ± 0.40 | 68.78 ± 0.47 | — |

¹ ImageNet-100 and CIFAR-100 entries are reported as mean ± std over three seeds. The TinyImageNet cross-method row is the exploratory single-seed matrix (seed 42), produced by `tinyimagenet_experiment.py` under the legacy recipe and used as a tertiary descriptive diagnostic. Its 2D-ALiBi-style entry is intentionally left blank because the TinyImageNet 1D-vs-2D ALiBi-style comparison is run under the *canonical* paired protocol (not the exploratory one) and is reported separately in the "2D-vs-1D ALiBi-style" section below (produced by `tinyimagenet_alibi_canonical.py`). **RoPE is the best method on all three datasets.**

### Robustness on ImageNet-100 — PE removal & noise

| PE method | Acc. under full PE removal | Δ vs. clean | Acc. at 3×σ noise |
|-----------|:--------------------------:|:-----------:|:-----------------:|
| **Learned** | **72.1%** | **−7.4 pp** | **~75%** |
| 1D-ALiBi-style | 58.1% | −22.9 pp | ~4% |
| RoPE | 43.6% | −40.9 pp | ~2% |
| Sinusoidal | 1.3% | −80.1 pp | ~2% |

Learned PE is dramatically more robust (≈10× range in PE dependency across methods), evidencing extensive *implicit* positional learning; Sinusoidal PE depends entirely on its explicit signal.

### 2D-vs-1D ALiBi-style (raster-distance mismatch test)

- **Accuracy:** CIFAR-100 (canonical, *n* = 12 seeds) **+0.45 pp**, paired *t*(11) = 2.94, *p* = 0.013, 95% CI [+0.11, +0.79]; ImageNet-100 **+0.12 pp** (NS); TinyImageNet **+0.75 ± 0.18 pp** (*n* = 3, directional paired check, 3/3 seeds positive).
- **Mutual information:** fixed-slope 2D-ALiBi-style lowers CLS-excluded patch-only attention–position MI by **9.3%** (4.01 → 3.63 bits, 12/12 seeds), but a **mean-magnitude-matched control reverses** the effect (MI rises to 5.36 bits while accuracy improves to 70.28 ± 0.35%) — so the MI reduction is **scale-sensitive rather than attributable to distance geometry alone**.

**TinyImageNet — canonical paired protocol (absolute accuracies).** Reported as a separate paired block, mirroring the manuscript's two-tier TinyImageNet presentation, rather than as a cell in the exploratory cross-method table above (whose 2D-ALiBi-style TinyImageNet entry is therefore left blank). Seed-level validation accuracy (%):

| PE method | seed 42 | seed 123 | seed 456 | mean ± std |
|-----------|:-------:|:--------:|:--------:|:----------:|
| 1D-ALiBi-style | 52.61 | 52.65 | 52.40 | 52.55 ± 0.13 |
| **2D-ALiBi-style** | 53.37 | 53.22 | 53.33 | **53.31 ± 0.08** |

Paired Δ (2D − 1D) = **+0.75 ± 0.18 pp**; all 3/3 seed-level differences positive (range +0.57 to +0.93 pp). Produced by `tinyimagenet_alibi_canonical.py` (seeds run one at a time, 1D and 2D paired per seed).

### Cross-dataset additive-PE linear probes

The row/column probe analysis was rerun with a protocol-matched script, `rerun_cross_dataset_probes_protocol_matched.py`, using CLS-excluded additive PE vectors, `LogisticRegression(max_iter=2000, C=1.0)`, no `StandardScaler`, and stratified 5-fold cross-validation. Learned PE entries are reported as means across three seeds for ImageNet-100 and CIFAR-100; the TinyImageNet Learned PE entry is reported separately as a single-checkpoint descriptive diagnostic, so that inferential claims remain anchored to the multi-seed ImageNet-100/CIFAR-100 analyses and the paired ALiBi-style protocols. Sinusoidal PE is analytically fixed and therefore seed-independent.

| Dataset (grid) | PE | Row (%) | Column (%) |
|---|---|---:|---:|
| ImageNet-100 (14×14) | Learned | 76.2 ± 1.6 | 31.3 ± 6.3 |
|  | Sinusoidal | 89.3 ± 0.0 | 0.0 ± 0.0 |
| CIFAR-100 (8×8) | Learned | 51.9 ± 2.4 | 58.7 ± 0.9 |
|  | Sinusoidal | 86.0 ± 0.0 | 0.0 ± 0.0 |
| TinyImageNet (14×14) | Learned | 81.7 | 49.0 |
|  | Sinusoidal | 89.3 ± 0.0 | 0.0 ± 0.0 |

These results support two points used in the manuscript. First, Sinusoidal PE encodes the raster row strongly but fails to linearly generalise to held-out column labels under the probe protocol on every evaluated grid. This is a protocol-level linear-generalisation failure, not absence of column information. Second, Learned PE retains substantial positional information but its row/column decodability is dataset- and seed-dependent rather than an explicit analytic grid decomposition.

> **Intrinsic ALiBi-style diagnostic note.** Direct attention-space analysis shows that the 1D-ALiBi-style bias tensor has **full** intrinsic rank and 5.72 bits of per-head entropy under the PE-defining bias representation; PCA projections of stacked or zero-padded bias representations are not used as intrinsic attention-space entropy or rank diagnostics.

## Reproducing the Results

This repository uses one official reproduction path: the **end-to-end pipeline**. The pipeline starts from model training, produces checkpoints and machine-readable JSON summaries, and ends with final PNG/PDF figures. For practical review and auditing, the final figure-regeneration step can also be run directly from the released checkpoints and JSON files; this is the final stage of the same pipeline, not a separate reproduction level.

The release intentionally avoids multiple named reproduction levels. Instead, it presents one canonical end-to-end path and then gives the exact final figure-regeneration command for users who want to verify the manuscript figures from the released artifacts.

### Figure index (file ↔ manuscript figure)

Throughout this section, figures are referred to **by file stem**, which matches the output filenames in `figures_revision/`. The mapping to the manuscript figure numbers is:

| File stem | Manuscript |
|---|---|
| `fig2_cosine_similarity` | Figure 1 (main text) |
| `fig5a_embedding_variance` | Figure 2 (main text) |
| `fig_main_mi_control` | Figure 3 (main text; panels a, b) |
| `01_training_comparison` | Figure S1 |
| `07_noise_ablation` | Figure S2 |
| `fig5b_intrinsic_structure` | Figure S3 |
| `fig4a_embedding_entropy` | Figure S4 |
| `fig4b_intrinsic_entropy` | Figure S5 |
| `03_pca_tsne` | Figure S6 |
| `09_layer_entropy` | Figure S7 |
| `figS3_distance_distortion` | Figure S8 |
| `06_mutual_information` | — standalone per-layer MI panel, still emitted by the script but **superseded**: the same content now appears as the left panel of `fig_main_mi_control` (Figure 3a), so it is not a separate manuscript figure |

The main text contains three figures (1–3); all other panels are supplementary (S1–S8).

## End-to-end pipeline

```text
apply_2d_alibi_patch.py
        ↓
full_scale_experiment.py  →  full_scale_experiment_v2.py
        ↓
training scripts / notebooks
        ↓
checkpoints + training_history.json
        ↓
MI / control / robustness analysis scripts and notebooks
        ↓
analysis_data.json + _aggregate.json + paired_alibi_mi_summary.json
        ↓
regenerate_revision_figures.py
        ↓
final PNG/PDF figures in figures_revision/
```

Important: **figures and tables are regenerated from machine-readable JSON outputs, checkpoints, and training histories.**

### Final figure regeneration from released artifacts

The main figure-regeneration entry point is:

```text
regenerate_revision_figures.py
```

It regenerates the final revision/manuscript figures produced from released checkpoints, JSON summaries, training histories, and analytic formulas. Two figures are produced by the main training/analysis pipeline (`full_scale_experiment.py` / `run_13_analysis.ipynb`) rather than by this script: `01_training_comparison` (Figure S1) and `09_layer_entropy` (Figure S7, activation entropy across layers).

**Repository scripts staged by the Colab notebook**

```text
regenerate_revision_figures.py     # only figure-generation entry point
full_scale_experiment.py           # base ViT implementation
apply_2d_alibi_patch.py            # creates full_scale_experiment_v2.py
compute_mi_cls_controls.py         # analysis/JSON-generation helper for Figure 3 summaries
```

`compute_mi_cls_controls.py` belongs to the analysis/JSON-generation layer: it produces the machine-readable MI-control summaries consumed by the Figure 3 path. Once the corresponding `paired_alibi_mi_summary.json` files already exist, the final figure script uses those JSON files as Figure 3 inputs.

**Required figure inputs**

```text
# Checkpoint-dependent figures:
#   fig2_cosine_similarity (Fig 1), 03_pca_tsne (S6),
#   fig5b_intrinsic_structure (S3), fig4a_embedding_entropy (S4),
#   fig5a_embedding_variance (Fig 2), fig4b_intrinsic_entropy (S5)
Trained models_ImageNet100/{learned,sinusoidal,rope,alibi,alibi_2d}_seed42/best_model.pth
full_scale_experiment_v2.py

# fig_main_mi_control panel (a) [Figure 3a]  (also the standalone 06_mutual_information)
revision_results/imagenet100/_aggregate.json

# fig_main_mi_control panel (b) [Figure 3b]
revision_results/mi_cls_control/cifar100_canonical_n12/paired_alibi_mi_summary.json
revision_results/mi_cls_control/cifar100_canonical_matched2d_n12/paired_alibi_mi_summary.json

# 07_noise_ablation [Figure S2]
results/analysis_data.json

# figS3_distance_distortion [Figure S8]
No external input; generated analytically from the 1D-vs-2D distance-distortion formula.
```

**Command**

```bash
python regenerate_revision_figures.py \
  --output-dir figures_revision \
  --fig3-imagenet-mi-json revision_results/imagenet100/_aggregate.json \
  --fig3-cifar-fixed-mi-json revision_results/mi_cls_control/cifar100_canonical_n12/paired_alibi_mi_summary.json \
  --fig3-cifar-matched-mi-json revision_results/mi_cls_control/cifar100_canonical_matched2d_n12/paired_alibi_mi_summary.json \
  --fig7-analysis-json results/analysis_data.json
```

The command above regenerates all final revision figures handled by `regenerate_revision_figures.py` when the checkpoints and JSON files are available, including `figS3_distance_distortion`, which has no external input. The optional CLI flags `--only-fig3` and `--only-fig7` remain useful for debugging, but the release notebook does not need separate cells for them.

**Expected final figure outputs**

```text
figures_revision/
  fig2_cosine_similarity.png
  fig2_cosine_similarity.pdf
  03_pca_tsne.png
  03_pca_tsne.pdf
  fig4a_embedding_entropy.png
  fig4a_embedding_entropy.pdf
  fig4b_intrinsic_entropy.png
  fig4b_intrinsic_entropy.pdf
  fig5a_embedding_variance.png
  fig5a_embedding_variance.pdf
  fig5b_intrinsic_structure.png
  fig5b_intrinsic_structure.pdf
  06_mutual_information.png
  06_mutual_information.pdf
  fig_main_mi_control.png
  fig_main_mi_control.pdf
  07_noise_ablation.png
  07_noise_ablation.pdf
  figS3_distance_distortion.png
  figS3_distance_distortion.pdf
```

### Full training and analysis path

To reproduce the entire experimental corpus from scratch, train the models, run the analysis notebooks/scripts, and then run `regenerate_revision_figures.py` as the final step.

**Training outputs per run**

```text
<dataset>/<method>_seed<seed>/
  best_model.pth
  final_model.pth
  checkpoint_epoch*.pth
  training_history.json
```

**Single-run training example**

```bash
python full_scale_experiment.py \
  --data_dir /path/to/dataset \
  --output_dir /path/to/results \
  --pe_type rope \
  --seed 42 \
  --mode train \
  --epochs 300 \
  --batch_size 128 \
  --num_workers 12 \
  --num_classes 100
```

For 2D-ALiBi-style runs, first create or use the patched model file:

```bash
python apply_2d_alibi_patch.py full_scale_experiment.py full_scale_experiment_v2.py
```

Then run the MI, control, robustness, and table-extraction scripts/notebooks that produce the JSON files listed under [Source-of-truth result files](#source-of-truth-result-files).

### Reproducing the TinyImageNet canonical pair (1D/2D ALiBi-style)

The TinyImageNet canonical 1D/2D ALiBi-style results (the Tier-2 paired block in [Key Findings](#key-findings); outputs under `Trained models_TinyImageNet_v2/`) are produced by `tinyimagenet_alibi_canonical.py`, launched through the self-contained `tin_allinone.ipynb` notebook. This script takes **no CLI arguments** — the run is configured by one edit to the master script plus one switch in the notebook.

**Prerequisites on Drive** (under `MyDrive/` or `MyDrive/pe_experiment/`): `apply_2d_alibi_patch.py`, `full_scale_experiment.py` (unpatched), and `tinyimagenet_alibi_canonical.py` (the master).

**Per-seed procedure** — repeat for each seed in 42, 123, 456:

1. In the master `tinyimagenet_alibi_canonical.py` on Drive, set the seed. This is the **only** edit to the script (`tin_allinone.ipynb` does not touch `SEEDS`):
   ```python
   SEEDS = [42]   # then [123], then [456]
   ```
2. In `tin_allinone.ipynb`, set the variant at the top of the **Setup** cell, then run **Setup** followed by **Train**:
   ```python
   PE_VARIANT = '1d'   # 1D-ALiBi-style
   ```
   The Setup cell mounts Drive, runs `apply_2d_alibi_patch.py` to build `full_scale_experiment_v2.py`, downloads/extracts TinyImageNet **locally** into `/content/` (no Drive I/O race), and writes a per-variant copy `/content/tin_run_1d.py` in which `PE_TYPES` is narrowed to `['alibi']` and `DATA_DIR` is repointed to the local copy; `SEEDS` is taken from the master unchanged.
3. For the paired 2D run at the **same seed**, repeat with `PE_VARIANT = '2d'` (narrows `PE_TYPES` to `['alibi_2d']`). The two variants run in **parallel** (separate Colab sessions/accounts) or sequentially; both write into `Trained models_TinyImageNet_v2/` under disjoint sub-folders `alibi_seed{N}/` and `alibi_2d_seed{N}/`.

Three seeds × {1D, 2D} = the 6-model canonical TinyImageNet corpus. There is **no epoch-level resume**: a session that disconnects mid-run must re-run Setup + Train from scratch; only fully-completed runs (300 epochs) are preserved on Drive. Each session trains one variant for one seed (~19–22 h on RTX 6000 Blackwell).

> **Note.** The master ships with `SEEDS = [456]` (the last seed used in this study); set it to the seed you intend to run. `train_tinyimagenet_alibi_canonical.ipynb` is an alternative that trains both variants **sequentially** in a single session, and `train_tinyimagenet_alibi_parallel.ipynb` splits 1D/2D across two sessions while keeping the dataset on Drive.

## Script and artifact map

### 1. Model-definition and patching layer

| File | Purpose | Required inputs | Generated outputs | Used by |
|---|---|---|---|---|
| `full_scale_experiment.py` | Base ViT training/analysis implementation for Learned, Sinusoidal, RoPE, and 1D-ALiBi-style. | Dataset path, PE type, seed, training/analysis CLI arguments. | Per-run checkpoints, `training_history.json`, and analysis outputs such as `analysis_data.json` when the analysis modes are used. | Training, checkpoint analysis, and the patched 2D-ALiBi-style model-generation step. |
| `apply_2d_alibi_patch.py` | Patches the base experiment file so the model supports 2D-ALiBi-style. | `full_scale_experiment.py`; output filename, usually `full_scale_experiment_v2.py`. | `full_scale_experiment_v2.py`. | 2D-ALiBi-style training and checkpoint-dependent figure generation. |
| `full_scale_experiment_v2.py` | Patched model/training definition with 2D-ALiBi-style support. | Produced by `apply_2d_alibi_patch.py`; datasets and checkpoints when used for training/analysis. | Same type of training outputs as the base experiment script. | 2D-ALiBi-style runs and `regenerate_revision_figures.py`. |

**PE naming note.** JSON keys and the final figure script use `alibi_2d`. Some older CLI paths may use `alibi2d`; check `--help` for the exact accepted spelling in the local script version.

### 2. Training layer

| File / notebook | Experiment | Required inputs | Generated outputs | Why it exists |
|---|---|---|---|---|
| `full_scale_experiment.py` | Primary cross-method matrix for ImageNet-100 and other base runs. | Dataset, `--pe_type`, `--seed`, training hyperparameters. | `best_model.pth`, `final_model.pth`, `checkpoint_epoch*.pth`, `training_history.json`. | Produces the trained models and histories for the main PE comparison. |
| `cifar100_experiment.py` | CIFAR-100 exploratory cross-method matrix. | CIFAR-100 data or torchvision download path; PE type/seed configuration. | CIFAR-100 checkpoints and histories; optional robustness outputs. | Reproduces the low-data cross-method accuracy/diagnostic matrix. |
| `cifar100_2d_alibi_matched_12seeds.py` | CIFAR-100 magnitude-matched 2D-ALiBi-style control, 12 seeds. | CIFAR-100, matched-control hyperparameters, seeds. | Matched 2D-ALiBi-style checkpoints and `training_history.json` files. | Tests the scale sensitivity of the fixed-slope 2D-ALiBi-style MI effect. |
| `tinyimagenet_experiment.py` | TinyImageNet exploratory validation (legacy recipe: `torch.compile` on, **no** gradient clipping). | TinyImageNet data, PE type/seed configuration (`--pe_type`, `--seed`, `--resume`). | Checkpoints + `training_history.json` under `Trained models_TinyImageNet/`. | Produces the exploratory single-seed TinyImageNet accuracies (Learned / Sinusoidal / RoPE / 1D-ALiBi-style) at *N* = 197. |
| `tinyimagenet_alibi_canonical.py` (run via `train_tinyimagenet_alibi_canonical.ipynb`, `train_tinyimagenet_alibi_parallel.ipynb`, or `tin_allinone.ipynb`) | TinyImageNet **canonical paired** 1D/2D ALiBi-style re-training under the matched protocol (fp32, gradient clipping at 1.0, NaN-loss guard, `torch.compile` disabled), so the targeted distance-geometry intervention is evaluated under aligned numerics. | Patched `full_scale_experiment_v2.py` (a fail-fast probe verifies `alibi_2d` / `TwoDALiBi` support before training); TinyImageNet data (auto-downloaded if absent); `SEEDS` / `PE_TYPES` set in-file. | `Trained models_TinyImageNet_v2/{alibi,alibi_2d}_seed{N}/best_model.pth` + `training_history.json`; `_alibi_canonical_summary.json`. | Produces the bit-comparable 1D-vs-2D ALiBi-style TinyImageNet pair — the canonical-protocol counterpart to the exploratory row above, used for the TinyImageNet entry of the 2D-ALiBi-style test. |
| Per-run notebooks in `notebooks/` | Colab/Drive execution wrappers for individual runs. | Google Drive mount, dataset paths, script files. | Same checkpoints/histories as the corresponding script. | Makes long training runs easier to launch and audit in Colab. |

### 3. Analysis and JSON-generation layer

| File / notebook | Purpose | Required inputs | Generated outputs | Consumed by |
|---|---|---|---|---|
| `revision_analysis_v41_mi_suite.ipynb` / `revision_analysis_v41_mi_suite_clean_tinyin.ipynb` | Computes per-layer attention–position mutual information and related diagnostics across datasets/seeds. | Trained checkpoints, validation data, model definition. | `revision_results/imagenet100/_aggregate.json`; `_master_summary.json`; dataset-level diagnostic JSON files. | Figure 3 panel (a), the standalone `06_mutual_information`, numerical audit. |
| `compute_mi_cls_controls.py` | Computes CLS-inclusive and CLS-excluded patch-only MI summaries for 1D-vs-2D ALiBi-style controls. | Canonical 1D-ALiBi-style checkpoints; fixed 2D-ALiBi-style checkpoints; matched 2D-ALiBi-style checkpoints; CIFAR-100 validation data. | `revision_results/mi_cls_control/cifar100_canonical_n12/paired_alibi_mi_summary.json`; `revision_results/mi_cls_control/cifar100_canonical_matched2d_n12/paired_alibi_mi_summary.json`. | Main Figure 3 panel (b), 2D-ALiBi-style MI-control claims, and the final figure-regeneration workflow. |
| `extract_tables_data.py` | **Runs** the noise-ablation and linear-probe analysis on the 12 ImageNet-100 checkpoints and **writes** the consolidated JSON; also prints the Table 2 (noise) and Table 3 (probe) summaries. | The 12 `best_model.pth` under `<results>/{pe}_seed{seed}/`, the ImageNet-100 validation set, and `full_scale_experiment.py` (imports `VisionTransformer`, `noise_ablation`, `probe_analysis`, `extract_positional_embedding`). | **`analysis_data.json`** (written to the results dir, i.e. `.../pe_experiment/results/analysis_data.json`); console Table 2 / Table 3. | `run_13_analysis.ipynb` and `07_noise_ablation` (Figure S2); manual table verification. |
| `rerun_cross_dataset_probes_protocol_matched.py` | Reruns the additive-PE row/column probe analysis across ImageNet-100, CIFAR-100, and TinyImageNet using the protocol-matched held-out-position probe. | Learned PE checkpoints, optional Sinusoidal checkpoints or analytic generation, dataset/grid metadata. | `per_seed_probe_results.csv`, `probe_summary.csv`, `probe_table_supp.tex`, and `probe_rerun_config.json`. | Supplementary probe table, cross-dataset probe claims, and reproducibility audit. |
| `merge_probe_seed_outputs.py` | Merges probe reruns when datasets or seeds are executed in separate jobs. | One or more probe output directories or `per_seed_probe_results.csv` files. | Merged `per_seed_probe_results.csv`, `probe_summary.csv`, and `probe_table_supp.tex`. | Final cross-dataset probe table when TinyImageNet is run as a separate single-seed job. |

### 4. Figure-generation layer

| File | Purpose | Required inputs | Generated outputs | Notes |
|---|---|---|---|---|
| `regenerate_revision_figures.py` | Main figure-regeneration entry point for the final manuscript/revision figures produced from released artifacts and analytic formulas. | `full_scale_experiment_v2.py`, ImageNet-100 checkpoints, the JSON inputs listed in the figure-regeneration section, and no external input for `figS3_distance_distortion`. | PNG and PDF versions of final revision figures in `figures_revision/`, including `figS3_distance_distortion`. | `01_training_comparison` and `09_layer_entropy` are produced separately by the main training/analysis pipeline. |

There are no alternative figure-generation entry points in the final release for these artifact-regenerated figures. The notebook and README should both point to `regenerate_revision_figures.py`; `01_training_comparison` and `09_layer_entropy` remain part of the main training/analysis pipeline.

## Source-of-truth result files

These files are the canonical inputs for table/figure reproduction.

| File | What it contains | Primary use |
|---|---|---|
| `results/analysis_data.json` | Noise-ablation and linear-probe summaries for the 12 ImageNet-100 models (one entry per `pe_type`/`seed`). **Produced by `extract_tables_data.py`.** | Table extraction (Tables 2–3) and `07_noise_ablation` (Figure S2) regeneration. |
| `probe_rerun_merged_final/per_seed_probe_results.csv` | Per-seed protocol-matched row/column/position probe outputs for additive PE across ImageNet-100, CIFAR-100, and TinyImageNet. | Audit trail for the cross-dataset probe analysis. |
| `probe_rerun_merged_final/probe_summary.csv` | Cross-dataset probe means/stds used for the supplementary probe table. | Source of the reported row/column probe values. |
| `probe_rerun_merged_final/probe_table_supp.tex` | LaTeX table emitted by the probe rerun/merge workflow. | Supplementary cross-dataset probe table. |
| `revision_results/imagenet100/_aggregate.json` | ImageNet-100 aggregate per-layer MI means/stds by PE method. | Main Figure 3 panel (a) and `06_mutual_information.png/pdf`. |
| `revision_results/imagenet100/_master_summary.json` | Broader dataset/method/seed diagnostic summary. | Numerical audit and fallback analysis; not the preferred Figure 3 input if `_aggregate.json` is available. |
| `revision_results/mi_cls_control/cifar100_canonical_n12/paired_alibi_mi_summary.json` | CIFAR-100 canonical 1D-vs-fixed-2D ALiBi-style MI summary, 12 seeds. | Main Figure 3 panel (b), fixed 2D bars. |
| `revision_results/mi_cls_control/cifar100_canonical_matched2d_n12/paired_alibi_mi_summary.json` | CIFAR-100 canonical 1D-vs-magnitude-matched-2D ALiBi-style MI summary, 12 seeds. | Main Figure 3 panel (b), matched 2D bars. |
| `training_history.json` | Per-epoch training/validation loss and accuracy for each run. | Accuracy tables, sanity checks, and run-level audit. |
| `best_model.pth` | Best validation checkpoint for a run. | Analysis from pretrained weights and checkpoint-dependent figures. |

> **Path note for `analysis_data.json`.** `extract_tables_data.py` writes the file to the **results directory on Drive** (`/content/drive/My Drive/pe_experiment/results/analysis_data.json`), whereas `run_13_analysis.ipynb` reads it from `/content/analysis_data.json`. When reproducing from scratch, copy the file from the results directory to `/content/` (or adjust the path) before running the figure step, so the consumer finds it.


## Repository Structure

A compact release can be organised as follows. The exact Google Drive paths may differ, but the relative roles of the files should remain the same.

```text
.
├── full_scale_experiment.py
├── apply_2d_alibi_patch.py
├── full_scale_experiment_v2.py              # generated by the patch script
├── regenerate_revision_figures.py           # single figure-generation entry point
├── extract_tables_data.py
├── compute_mi_cls_controls.py
├── rerun_cross_dataset_probes_protocol_matched.py
├── merge_probe_seed_outputs.py
├── cifar100_experiment.py
├── cifar100_2d_alibi_matched_12seeds.py
├── tinyimagenet_experiment.py                # exploratory TinyImageNet matrix (legacy recipe)
├── tinyimagenet_alibi_canonical.py           # canonical paired 1D/2D ALiBi-style TinyImageNet re-training
├── requirements.txt
├── notebooks/
│   ├── revision_analysis_v41_mi_suite.ipynb
│   ├── revision_analysis_v41_mi_suite_clean_tinyin.ipynb
│   ├── regenerate_revision_figures.ipynb
│   ├── run_13_analysis.ipynb              # produces 01_training_comparison
│   ├── train_tinyimagenet.ipynb                      # exploratory TinyImageNet, one (PE, seed) per session
│   ├── train_tinyimagenet_alibi_canonical.ipynb      # canonical 1D+2D ALiBi-style, sequential (seed-paired)
│   ├── train_tinyimagenet_alibi_parallel.ipynb       # canonical 1D OR 2D ALiBi-style, split across two sessions
│   ├── tin_allinone.ipynb                            # canonical ALiBi-style, self-contained single-cell runner
│   └── other per-run training notebooks
├── results/
│   └── analysis_data.json
├── probe_rerun_merged_final/
│   ├── per_seed_probe_results.csv
│   ├── probe_summary.csv
│   └── probe_table_supp.tex
├── revision_results/
│   ├── imagenet100/
│   │   ├── _aggregate.json
│   │   └── _master_summary.json
│   └── mi_cls_control/
│       ├── cifar100_canonical_n12/
│       │   └── paired_alibi_mi_summary.json
│       └── cifar100_canonical_matched2d_n12/
│           └── paired_alibi_mi_summary.json
├── figures_revision/
│   ├── fig_main_mi_control.png
│   ├── fig_main_mi_control.pdf
│   ├── fig2_cosine_similarity.png
│   ├── fig2_cosine_similarity.pdf
│   ├── 03_pca_tsne.png
│   ├── 03_pca_tsne.pdf
│   ├── fig4a_embedding_entropy.png
│   ├── fig4a_embedding_entropy.pdf
│   ├── fig4b_intrinsic_entropy.png
│   ├── fig4b_intrinsic_entropy.pdf
│   ├── fig5a_embedding_variance.png
│   ├── fig5a_embedding_variance.pdf
│   ├── fig5b_intrinsic_structure.png
│   ├── fig5b_intrinsic_structure.pdf
│   ├── 06_mutual_information.png
│   ├── 06_mutual_information.pdf
│   ├── 07_noise_ablation.png
│   ├── 07_noise_ablation.pdf
│   ├── figS3_distance_distortion.png
│   └── figS3_distance_distortion.pdf
├── LICENSE
└── README.md
```

> Per-model artifacts — the best checkpoint (`best_model.pth`) and full training history — are **not** stored here due to their size (76 models); they live on Google Drive (see below).

## Pre-trained Models & Experimental Corpus

The trained weights and training histories are **not** included in this repository (76 models × ~344 MB). For each model the Drive hosts its best checkpoint (`best_model.pth`) and full training history, organised by dataset / protocol / PE method / seed:

**[⬇ Download Checkpoints & Training Histories](https://drive.google.com/drive/folders/1elzYM_4Mwyy4aDTY5NQbmzTMyYxIRo2B?usp=sharing)**

The **76-model corpus** breaks down as follows:

| Tier | Dataset | Configuration | Models |
|------|---------|---------------|:------:|
| Primary cross-method matrix | ImageNet-100 | 5 PE × 3 seeds (42, 123, 456) | 15 |
| Primary cross-method matrix | CIFAR-100 | 5 PE × 3 seeds (42, 123, 456) | 15 |
| Primary cross-method matrix | TinyImageNet | 4 PE × seed 42 | 4 |
| Canonical paired protocol | CIFAR-100 | 1D/2D ALiBi-style × 12 seeds | 24 |
| Canonical paired protocol | TinyImageNet | 1D/2D ALiBi-style × 3 seeds | 6 |
| Magnitude-matched 2D-ALiBi-style control | CIFAR-100 | 12 seeds | 12 |
| | | **Total** | **76** |

The five reported PE rows are Learned, Sinusoidal, RoPE, 1D-ALiBi-style, and 2D-ALiBi-style. Per-method/per-seed validation accuracies are listed in [Key Findings](#key-findings).

> **Note:** Models trained with `torch.compile()` have state_dict keys prefixed with `_orig_mod.` (the exploratory ImageNet-100 and TinyImageNet runs; the canonical paired protocol disables `torch.compile`). Strip the prefix if present:
> ```python
> state = torch.load('best_model.pth', map_location='cpu')
> model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
> ```

## Methodological Scope

Full-scale ImageNet-1K from-scratch evaluation is structurally outside the scope of this investigation. This is a deliberate methodological design choice: ImageNet-1K-scale ViT-Base training would require an independent ecosystem of recipe-selection and stabilisation heuristics, rather than serving as a direct, unconfounded extension of the paired PE comparisons. The results should therefore be interpreted as controlled evidence about PE behaviour under the unified matched protocols and data regimes reproduced here.

## Datasets

Three image-classification datasets spanning the low- to intermediate-data regime; the ViT-Base backbone is identical across all, only the input/patch configuration changes.

| Dataset | Classes | Train / Val | Patch | Grid | *N* (tokens) |
|---------|:-------:|:-----------:|:-----:|:----:|:------------:|
| **ImageNet-100** (primary) | 100 | ~127K / 5K | 16×16 | 14×14 | 197 |
| **CIFAR-100** (low-data) | 100 | 50K / 10K | 4×4 | 8×8 | 65 |
| **TinyImageNet** (intermediate) | 200 | 100K / 10K | 16×16 | 14×14 | 197 |

- **ImageNet-100** — the 100-class subset of ILSVRC-2012 defined by [Tian et al. (2020)](https://arxiv.org/abs/1906.05849). Class list: [imagenet100.txt](https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt). Images at 224×224.
- **CIFAR-100** — native 32×32 images with a 4×4 patch size (8×8 grid), avoiding the ~7× upsampling artefacts of resizing to 224×224 (standard ViT-on-CIFAR protocol).
- **TinyImageNet** — native 64×64 upsampled to 224×224 (~3.5×), retaining the ImageNet-100 patch configuration for direct diagnostic comparability.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-Base |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dim | 768 |
| MLP ratio | 4.0 |
| Dropout | 0.1 |
| Parameters | ~85.9M |
| RoPE head dim / frequencies | 64 / 32 bands (θᵢ = 10000⁻²ⁱ/ᵈʰ) |
| ALiBi-style slopes | *m*ₕ = 2⁻⁸⁽ʰ⁺¹⁾/ᴴ, *h* ∈ {0,…,*H*−1}, *H* = 12 |

Patch size, grid, and sequence length *N* depend on the dataset (see [Datasets](#datasets)). A learnable `[CLS]` token is prepended before positional information is injected.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3×10⁻⁴ |
| Weight decay | 0.1 |
| Warmup epochs | 20 |
| Total epochs | 300 |
| Batch size | 128 |
| Label smoothing | 0.1 |
| Mixup α | 0.8 |
| LR schedule | Cosine annealing |
| Seeds (primary) | 42, 123, 456 |
| Seeds (CIFAR-100 canonical, *n* = 12) | 42, 123, 456, 1, 5, 7, 11, 13, 21, 99, 2024, 31337 |

**Two protocols.** The *exploratory* matrix uses dataset-specific throughput settings (ImageNet-100: `torch.amp` mixed precision + `torch.compile`; TinyImageNet: fp32 + `torch.compile`; CIFAR-100: fp32, no `torch.compile`). The *canonical paired protocol* re-trains 1D/2D ALiBi-style under matched settings (fp32, gradient clipping at 1.0, NaN-loss guard, `torch.compile` disabled) so the targeted distance-geometry intervention is evaluated under aligned numerics; it is used for the CIFAR-100 12-seed ALiBi-style comparison (`cifar100_alibi_*seeds`) and, via `tinyimagenet_alibi_canonical.py` (outputs under `Trained models_TinyImageNet_v2/`, left disjoint from the legacy `Trained models_TinyImageNet/`), for the TinyImageNet 1D/2D ALiBi-style pair. The protocols share the same optimiser, schedule, batch size, augmentation, and base hyperparameters.

## Quick Start

### Training a single model

```bash
python full_scale_experiment.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/results \
    --pe_type rope \
    --seed 42 \
    --mode train \
    --epochs 300 \
    --batch_size 128 \
    --num_workers 12 \
    --num_classes 100
```

`--pe_type` accepts `learned`, `sinusoidal`, `rope`, `alibi` (1D), and `alibi2d` (2D-ALiBi-style). Set `--num_classes` to 100 (ImageNet-100 / CIFAR-100) or 200 (TinyImageNet). Run `python full_scale_experiment.py --help` for the full option list (dataset/patch configuration and protocol flags).

### Running analysis

```bash
python full_scale_experiment.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/results \
    --mode analyze \
    --num_classes 100
```

This runs standalone: `full_scale_experiment.py` imports `StratifiedKFold` and strips the `_orig_mod.` checkpoint prefix internally, so the earlier runtime patch cells in `run_13_analysis.ipynb` are no longer required. Figures produced by this script (e.g. `01_training_comparison`, `09_layer_entropy`) are written at 300 dpi with ImageNet-100 titles.

### Extracting table data (noise ablation + probes)

```bash
python extract_tables_data.py
```

This runs the noise-ablation and linear-probe analysis on the 12 ImageNet-100 checkpoints, **writes `analysis_data.json`** to the results directory, and prints the Table 2 (noise) and Table 3 (probe) summaries. (It imports the model and analysis functions from `full_scale_experiment.py`, so that file must be importable, e.g. on `sys.path`/`/content`.)

### Rerunning the cross-dataset additive-PE probes

The cross-dataset row/column probe table is regenerated with `rerun_cross_dataset_probes_protocol_matched.py`. The script matches the manuscript probe protocol: CLS is excluded, row labels are `position // grid_size`, column labels are `position % grid_size`, the classifier is `LogisticRegression(max_iter=2000, C=1.0)`, there is no `StandardScaler`, and the score is a stratified 5-fold cross-validation mean.

ImageNet-100 and CIFAR-100 Learned PE use three seeds:

```bash
python rerun_cross_dataset_probes_protocol_matched.py \
  --learned-checkpoint-template "/content/drive/MyDrive/Trained models_{dataset_alias}/learned_seed{seed}/best_model.pth" \
  --dataset-alias ImageNet-100=ImageNet100 CIFAR-100=CIFAR100 TinyImageNet=TinyImageNet \
  --datasets ImageNet-100 CIFAR-100 \
  --seeds 42 123 456 \
  --outdir probe_rerun_imagenet_cifar
```

TinyImageNet Learned PE is reported as a single-seed descriptive diagnostic:

```bash
python rerun_cross_dataset_probes_protocol_matched.py \
  --learned-checkpoint-template "/content/drive/MyDrive/Trained models_{dataset_alias}/learned_seed{seed}/best_model.pth" \
  --dataset-alias ImageNet-100=ImageNet100 CIFAR-100=CIFAR100 TinyImageNet=TinyImageNet \
  --datasets TinyImageNet \
  --seeds 42 \
  --outdir probe_rerun_tinyimagenet
```

If the probe jobs are run separately, merge them before using the table:

```bash
python merge_probe_seed_outputs.py \
  probe_rerun_imagenet_cifar \
  probe_rerun_tinyimagenet \
  --outdir probe_rerun_merged_final
```

The final merged outputs are `probe_rerun_merged_final/per_seed_probe_results.csv`, `probe_rerun_merged_final/probe_summary.csv`, and `probe_rerun_merged_final/probe_table_supp.tex`.

## Requirements

```
torch>=2.0
torchvision
timm
numpy
scikit-learn
scipy
matplotlib
tqdm
```

Experiments were run on a single **NVIDIA RTX 6000 Pro Blackwell (96 GB)** GPU per run with **PyTorch 2.1**; random seeds control PyTorch, NumPy, and the CUDA RNG state.

## Diagnostic Suite

A **two-track** framework matched to the architecture of each PE family:

**Embedding-space (additive PE — Learned, Sinusoidal)**
1. **Per-dimension variance** — signal-amplitude profiles
2. **Shannon entropy per dimension** — information-capacity allocation
3. **Cosine-similarity matrices** — pairwise PE-vector structure
4. **PCA / t-SNE projections** — information geometry
5. **Linear probes** — row/column decodability under 5-fold CV (held-out-position protocol)

**Attention-space (RoPE, ALiBi-style variants)**
6. **Bias-tensor rank & value entropy** (ALiBi-style variants), **short-wavelength rotation bands & wavelengths** (RoPE)
7. **Slope schedule & nominal bias scale**

**Cross-cutting**
8. **Attention–position mutual information** — corrected discrete *I*(query position; argmax target) estimator, with CLS-inclusive (sensitivity) and CLS-excluded patch-only (primary) variants
9. **Noise ablation** — robustness under PE perturbation (0.1×–5×σ), family-specific injection sites
10. **PE removal** — implicit-positional-learning / dependency spectrum
11. **Cross-dataset validation & scaling patterns**

All pairwise accuracy comparisons use paired bootstrap 95% CIs (exact 27-point enumeration at *n* = 3; 10,000 resamples and a paired *t*-test at *n* = 12), with the Wilcoxon signed-rank and sign tests as secondary checks.

## Citation

```bibtex
@article{bandjur2026information,
  title   = {Information-Theoretic Analysis of Positional Encoding Strategies
             in Vision Transformers: A Comparative Study of Four Approaches},
  author  = {Banđur, Đoko and Banđur, Miloš and Jakšić, Branimir},
  year    = {2026},
  note    = {Preprint},
  doi     = {10.5281/zenodo.19063156}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ImageNet-100 subset defined by [Tian et al. (2020)](https://arxiv.org/abs/1906.05849)
- CIFAR-100 ([Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)) and TinyImageNet ([Le & Yang, 2015](http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf))
- Training recipe adapted from [DeiT](https://github.com/facebookresearch/deit) (Touvron et al., 2021)
- RoPE implementation inspired by [RoFormer](https://arxiv.org/abs/2104.09864) (Su et al., 2024); ALiBi from [Press et al. (2022)](https://arxiv.org/abs/2108.12409)
