# Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19063156.svg)](https://doi.org/10.5281/zenodo.19063156)

Official implementation for the paper:

> **Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers: A Comparative Study of Four Approaches**
>
> [Đoko Banđur, Miloš Banđur, and Branimir Jakšić]
>
> *Preprint, 2026*

## Abstract

We present a comprehensive information-theoretic comparison of four positional encoding (PE) strategies in ViT-Base models trained on ImageNet-100: **Learned PE**, **Sinusoidal PE**, **RoPE**, and **ALiBi**. Training 12 models (4 PE types × 3 seeds × 300 epochs), we apply nine diagnostic analyses including Shannon entropy profiling, noise ablation, linear probes, and layer-wise mutual information tracking. Our results reveal four qualitatively distinct encoding regimes with dramatically different robustness profiles: Learned PE maintains 75% accuracy at 3×σ noise while Sinusoidal PE collapses entirely without its explicit signal (Δ=80.1pp). RoPE achieves the highest accuracy (84.51±0.32%).

## Key Findings

| PE Type | Accuracy | PE Removal Δ | Noise Robustness (3×σ) |
|---------|----------|-------------|----------------------|
| **RoPE** | **84.51±0.32%** | −41.0pp | 2.3% |
| Sinusoidal | 81.46±0.26% | −80.1pp | 2.0% |
| ALiBi | 81.05±0.28% | −22.9pp | 4.0% |
| Learned | 79.44±0.49% | **−7.4pp** | **75.2%** |

## Repository Structure

```
├── full_scale_experiment.py      # Main training and analysis pipeline
├── extract_tables_data.py        # Extract noise ablation & probe data
├── regenerate_figures_7_8.py     # Regenerate Figures 7 & 8 from saved data
├── notebooks/
│   ├── run_01_learned_seed42.ipynb
│   ├── run_02_sinusoidal_seed42.ipynb
│   ├── ...
│   ├── run_12_alibi_seed456.ipynb
│   └── run_13_analysis.ipynb     # Full analysis notebook
├── results/
│   └── analysis_data.json        # Verified analysis results (noise ablation, probes)
└── README.md
```

## Pre-trained Models

All 12 trained ViT-Base models (best checkpoint per configuration) are available on Google Drive:

**[⬇ Download Pre-trained Models](https://drive.google.com/drive/folders/1gPwVSE0qctWVeaGwCv3eGQdQR4IK6Xds?usp=sharing)**

| Model | File | Accuracy | Size |
|-------|------|----------|------|
| Learned PE, seed=42 | `learned_seed42/best_model.pth` | 79.68% | ~344 MB |
| Learned PE, seed=123 | `learned_seed123/best_model.pth` | 79.90% | ~344 MB |
| Learned PE, seed=456 | `learned_seed456/best_model.pth` | 78.74% | ~344 MB |
| Sinusoidal PE, seed=42 | `sinusoidal_seed42/best_model.pth` | 81.84% | ~344 MB |
| Sinusoidal PE, seed=123 | `sinusoidal_seed123/best_model.pth` | 81.30% | ~344 MB |
| Sinusoidal PE, seed=456 | `sinusoidal_seed456/best_model.pth` | 81.24% | ~344 MB |
| RoPE, seed=42 | `rope_seed42/best_model.pth` | 84.96% | ~344 MB |
| RoPE, seed=123 | `rope_seed123/best_model.pth` | 84.18% | ~344 MB |
| RoPE, seed=456 | `rope_seed456/best_model.pth` | 84.38% | ~344 MB |
| ALiBi, seed=42 | `alibi_seed42/best_model.pth` | 81.16% | ~344 MB |
| ALiBi, seed=123 | `alibi_seed123/best_model.pth` | 81.34% | ~344 MB |
| ALiBi, seed=456 | `alibi_seed456/best_model.pth` | 80.66% | ~344 MB |

> **Note:** Models were saved with `torch.compile()`, so state_dict keys have `_orig_mod.` prefix. Load with:
> ```python
> state = torch.load('best_model.pth', map_location='cpu')
> model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
> ```

## Dataset

We use **ImageNet-100**, the 100-class subset defined by [Tian et al. (2020)](https://arxiv.org/abs/1906.05849):
- **Source:** ILSVRC-2012 (requires original ImageNet access)
- **Class list:** [imagenet100.txt](https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt)
- **Size:** ~127K train / 5K val images
- **Resolution:** 224×224 (pre-resized to 256px for faster loading)

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-Base |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dim | 768 |
| MLP ratio | 4.0 |
| Patch size | 16×16 |
| Input resolution | 224×224 |
| Sequence length | 197 (196 patches + CLS) |
| Parameters | 85.9M |

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
| Seeds | 42, 123, 456 |

## Quick Start

### Training a single model

```bash
python full_scale_experiment.py \
    --data_dir /path/to/imagenet100_resized \
    --output_dir /path/to/results \
    --pe_type rope \
    --seed 42 \
    --mode train \
    --epochs 300 \
    --batch_size 128 \
    --num_workers 12 \
    --num_classes 100
```

### Running analysis

```bash
python full_scale_experiment.py \
    --data_dir /path/to/imagenet100_resized \
    --output_dir /path/to/results \
    --mode analyze \
    --num_classes 100
```

### Extracting table data (noise ablation + probes)

```bash
python extract_tables_data.py
```

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

## Analyses Included

1. **Training dynamics** — Accuracy/loss curves with per-seed variance bands
2. **Cosine similarity matrices** — Pairwise PE vector similarity
3. **PCA/t-SNE projections** — Information geometry visualization
4. **Shannon entropy per dimension** — Information capacity allocation
5. **Variance per dimension** — Signal amplitude profiles
6. **Mutual information** — Position-attention dependency per layer
7. **Attention entropy** — Focus patterns per layer
8. **Noise ablation** — Robustness under PE perturbation (0.1×–5×σ) and PE removal
9. **Linear probe analysis** — Row/column/position decodability from PE vectors
10. **Layer-wise activation entropy** — Information bottleneck dynamics

## Citation

```bibtex
@article{author2026pe_analysis,
  title={Information-Theoretic Analysis of Positional Encoding Strategies 
         in Vision Transformers: A Comparative Study of Four Approaches},
  author={[Author Name]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ImageNet-100 subset defined by [Tian et al. (2020)](https://arxiv.org/abs/1906.05849)
- Training recipe adapted from [DeiT](https://github.com/facebookresearch/deit) (Touvron et al., 2021)
- RoPE implementation inspired by [RoFormer](https://arxiv.org/abs/2104.09864) (Su et al., 2024)
