# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSAMBA (Self-Supervised Audio Mamba) is a PyTorch implementation of self-supervised audio representation learning using Mamba State Space Models. It pretrains on unlabeled audio (AudioSet + LibriSpeech) via masked patch prediction, then fine-tunes on downstream audio classification/recognition tasks. It is ~92.7% faster and ~95.4% more memory-efficient than the SSAST baseline (Vision Transformer equivalent).

## Setup

```bash
pip install -r requirements.txt
# Also required: clone Vision Mamba inside the repo root (ssamba/)
# Vim is NOT currently cloned — must be done manually before running any model code
git clone https://github.com/hustvl/Vim.git
```

For VoxCeleb1/IEMOCAP tasks, also install s3prl:
```bash
pip install s3prl
```

## Running Training / Fine-tuning

All training is invoked via `src/run_amba.py`. The shell scripts under `src/pretrain/` and `src/finetune/*/` are SLURM job wrappers — they can be run directly after removing `#SBATCH` directives. Note: scripts still contain hardcoded cluster paths (`/engram/naplab/...`, `/home/ss6928/...`) that must be updated for your environment.

**Pretraining:**
```bash
cd src/pretrain && ./run_mask_patch_amba.sh
```

**Fine-tuning (e.g., ESC-50):**
```bash
cd src/finetune/esc50 && ./run_esc_patch_amba.sh ssamba_base_300
```

**Inference benchmark:**
```bash
cd src/inference && python inference_amba.py --model_size base
```

## Architecture

### Data flow
```
Raw Audio → Mel-Spectrogram (128 mel bins × 1024 frames)
          → 16×16 patch embedding → token sequence
          → Vision Mamba encoder (bidirectional Mamba blocks)
          → task head (classification / masked reconstruction)
```

### Key files
| File | Role |
|------|------|
| `src/models/both_models.py` | `AMBAModel` (Mamba-based) and `ASTModel` (ViT-based) — core model classes |
| `src/run_amba.py` | CLI entry point; parses all hyperparameters and launches training |
| `src/traintest.py` | Supervised fine-tuning train/eval loops |
| `src/traintest_mask.py` | Masked pre-training loop |
| `src/dataloader.py` | `AudioDataset`: loads JSON manifests, generates log-mel spectrograms, applies SpecAugment and mix-up |
| `src/utilities/util.py` | `AverageMeter`, normalization helpers, metric utilities |

### Model sizes
- **base**: embed_dim=768, depth=24
- **small**: embed_dim=384, depth=24
- **tiny**: embed_dim=192, depth=24

Pretrained weight naming: `ssamba_{size}_{masked_patches}` (e.g., `ssamba_base_400`).

### Key CLI flags (run_amba.py)
- Audio: `--num_mel_bins`, `--target_length`, `--dataset_mean`, `--dataset_std`
- Model: `--model_size`, `--embed_dim`, `--depth`, `--bimamba_type`, `--drop_path_rate`
- Training: `--lr`, `--batch-size`, `--n-epochs`, `--warmup`, `--mixup`
- Augmentation: `--freqm` (frequency mask), `--timem` (time mask)
- Mamba-specific: `--if_rope`, `--rms_norm`, `--final_pool_type`

### Data format
Datasets are referenced via JSON manifests listing audio file paths and labels. Data preparation scripts are in `src/prep_data/`.

## Downstream Tasks
AudioSet (527-class tagging), ESC-50 (5-fold CV), Speech Commands v1/v2, Urban8k (1-min scenes), VoxCeleb1 (speaker ID, via SUPERB), IEMOCAP (emotion recognition, via SUPERB).
