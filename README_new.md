# SSAMBA: Self-Supervised Audio Mamba

[![arXiv](https://img.shields.io/badge/arXiv-2405.11831-b31b1b.svg)](https://arxiv.org/abs/2405.11831)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface&logoColor=yellow)](https://huggingface.co/attentionisallyouneed369/ssamba)

## Summary

SSAMBA is a self-supervised audio representation learning model that replaces the Vision Transformer encoder in [SSAST](https://arxiv.org/abs/2110.09784) with a bidirectional [Vision Mamba](https://arxiv.org/abs/2401.09417) (state space model) backbone. It is pretrained on unlabeled audio via masked patch prediction and fine-tuned on downstream classification tasks. Compared to SSAST, SSAMBA is ~92.7% faster in batch inference and ~95.4% more memory-efficient at the tiny model size with 22k input tokens.

---

## Repo Overview

```
ssamba/
├── Vim/                          # Vision Mamba (git submodule)
├── src/
│   ├── run.py                    # Unified training entrypoint (--model amba | ssast)
│   ├── dataloader.py             # AudioDataset: JSON manifests → log-mel → SpecAugment
│   ├── traintest.py              # Fine-tuning train/eval loop
│   ├── traintest_mask.py         # Masked pretraining loop
│   ├── models/
│   │   ├── amba_model.py         # AMBAModel — Mamba-based encoder
│   │   ├── ast_model.py          # ASTModel  — ViT-based encoder (SSAST baseline)
│   │   └── ssast/                # Shared s3prl upstream expert (VoxCeleb1 / IEMOCAP)
│   ├── pretrain/                 # Pretraining shell scripts
│   ├── finetune/                 # Per-task fine-tuning scripts
│   │   ├── audioset/
│   │   ├── esc50/
│   │   ├── speechcommands_v1/
│   │   ├── speechcommands_v2/
│   │   ├── urban8k/
│   │   ├── voxceleb1/
│   │   └── iemocap/
│   ├── inference/                # Inference speed / memory benchmarks
│   ├── prep_data/                # Dataset download and preprocessing scripts
│   │   ├── audioset/
│   │   ├── librispeech/
│   │   ├── esc50/
│   │   ├── urban8k/
│   │   └── pretraining/
│   └── model_weights/            # Place downloaded .pth files here
```

---

## Setup

### 1. Clone and install

```bash
git clone --recurse-submodules https://github.com/Maryada-GS/ssamba.git
cd ssamba
uv sync
```

If you already cloned without `--recurse-submodules`, initialize the Vision Mamba submodule:

```bash
git submodule update --init
```

> If you encounter errors related to `bimamba_type`, see this [GitHub issue comment](https://github.com/hustvl/Vim/issues/14#issuecomment-1964685563).

For VoxCeleb1 / IEMOCAP tasks, install the optional s3prl extra:

```bash
uv sync --extra superb
```

### 2. Download pretrained weights

Pretrained SSAMBA weights (base/small/tiny × 250/300/400 masked patches):

[Google Drive — Pretrained Weights](https://drive.google.com/drive/u/1/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX)

Place downloaded `.pth` files under `src/model_weights/`. Naming convention used by fine-tuning scripts: `ssamba_{size}_{patches}.pth` (e.g. `ssamba_base_400.pth`).

---

## Datasets

| Dataset | Used for | Auto-download? | How |
|---|---|---|---|
| **AudioSet** | Pretraining + fine-tuning | No | HuggingFace [`confit/audioset-16khz-wds`](https://huggingface.co/datasets/confit/audioset-16khz-wds) (requires format conversion) or manual YouTube crawler |
| **LibriSpeech** | Pretraining | Yes | `download_librispeech.py` fetches from openslr.org (~300 GB) |
| **ESC-50** | Fine-tuning | Yes | `prep_esc50.py` downloads from GitHub via wget |
| **Speech Commands v1/v2** | Fine-tuning | Yes | Run script calls `prep_sc.py` automatically on first run |
| **UrbanSound8K** | Fine-tuning | No | Manual download from [urbansounddataset.weebly.com](https://urbansounddataset.weebly.com/urbansound8k.html) |
| **VoxCeleb1** | Fine-tuning (SUPERB) | No | License agreement required at [voxceleb.net](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) |
| **IEMOCAP** | Fine-tuning (SUPERB) | No | License request required at [sail.usc.edu/iemocap](https://sail.usc.edu/iemocap/) |

---

## Pretraining

SSAMBA is pretrained on a mixture of AudioSet and LibriSpeech.

### Step 1 — Prepare AudioSet

AudioSet must be downloaded manually (requires a YouTube crawler). Once you have the audio files, place them under:

```
src/dataset/audioset/audio/unbal_train/   ← unbalanced training audio (.flac)
src/dataset/audioset/audio/eval/          ← evaluation audio (.flac)
```

Then preprocess (converts stereo → mono, resamples to 16 kHz) and generate JSON manifests:

```bash
cd src/prep_data/audioset
python preprocess_audioset.py   # fix sample rate / channels in-place
python prep_audioset.py         # generate datafiles/unbal_train_data.json and eval_data.json
```

Output: `src/dataset/audioset/datafiles/unbal_train_data.json` and `eval_data.json`.

> For AudioSet balanced fine-tuning, you will also need `bal_train_data.json` — a filtered subset of the unbalanced set containing only the ~20k balanced samples.

### Step 2 — Prepare LibriSpeech

```bash
cd src/prep_data/librispeech
python download_librispeech.py      # downloads ~300 GB to src/dataset/librispeech/LibriSpeech/
python preprocess_librispeech.py    # stereo → mono, resample → writes ProcessedLibriSpeech/
python prep_librispeech.py          # walks .flac files, writes librispeech_tr960_cut.json
```

Output: `src/dataset/librispeech/librispeech_tr960_cut.json`.

### Step 3 — Mix datasets

```bash
cd src/prep_data/pretraining
python mix_pretraining_data.py
```

Reads `src/dataset/audioset/datafiles/unbal_train_data.json` and `src/dataset/librispeech/librispeech_tr960_cut.json`.
Output: `src/dataset/pretraining/audioset_librispeech.json`.

### Step 4 — Run pretraining

```bash
cd src/pretrain
./run_mask_patch_amba.sh      # SSAMBA (Mamba backbone)
./run_mask_patch.sh           # SSAST  (ViT backbone, for comparison)
```

Key hyperparameters in `run_mask_patch_amba.sh`:

| Variable | Default | Description |
|---|---|---|
| `mask_patch` | 300 | Patches masked per sample |
| `batch_size` | 64 | Batch size |
| `lr` | 1e-4 | Learning rate |
| `epoch` | 10 | Training epochs |
| `model_size` | base | `base` / `small` / `tiny` |

---

## Fine-tuning

All fine-tuning scripts take the pretrained model name (without `.pth`) as `$1`:

```bash
./run_*.sh ssamba_base_400
```

### AudioSet (balanced 20k, 527-class tagging)

Requires `bal_train_data.json` and `eval_modified_labeled_with_data_key.json` under `src/dataset/audioset/datafiles/`.

```bash
cd src/finetune/audioset
./run_as_amba.sh ssamba_base_400     # SSAMBA
./run_as.sh      ssamba_base_400     # SSAST baseline
```

Metric: mAP.

### ESC-50 (5-fold cross-validation, 50-class environmental sound)

```bash
cd src/prep_data/esc50
python prep_esc50.py     # downloads ESC-50 and writes datafiles to src/dataset/esc50/datafiles/
```

```bash
cd src/finetune/esc50
./run_esc_patch_amba.sh ssamba_base_400
./run_esc_patch.sh      ssamba_base_400
```

Results are averaged across 5 folds at the end. Metric: accuracy.

### Speech Commands v1 (35-class keyword spotting)

The run script downloads and prepares the dataset automatically on first run.

```bash
cd src/finetune/speechcommands_v1
./run_sc_amba.sh ssamba_base_400
```

### Speech Commands v2 (35-class keyword spotting)

```bash
cd src/finetune/speechcommands_v2
./run_sc_amba.sh ssamba_base_400
```

Metric: accuracy.

### UrbanSound8K (10-class 1-minute audio scenes)

UrbanSound8K must be downloaded manually from [urbansounddataset.weebly.com](https://urbansounddataset.weebly.com/urbansound8k.html) and extracted to `src/dataset/urban8k/`.

```bash
cd src/prep_data/urban8k
python prep_urban8k.py     # concatenates clips into 60-second segments per fold
```

```bash
cd src/finetune/urban8k
./urban_amba.sh ssamba_base_400
```

> Note: `urban_amba.sh` references `run_amba_1sec.py` which is not yet implemented.

### VoxCeleb1 — Speaker Identification (via SUPERB)

Requires `uv sync --extra superb` and the VoxCeleb1 dataset from [voxceleb.net](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).

The upstream expert is registered in `src/finetune/voxceleb1/ssast/hubconf.py`. Available function names follow the pattern `amba_patch{250,300,400}_{base,small,tiny}`.

```bash
cd src/finetune/voxceleb1
./run_sid.sh amba_patch400_base
```

Metric: accuracy (speaker ID).

### IEMOCAP — Emotion Recognition (via SUPERB)

Requires `uv sync --extra superb` and the IEMOCAP dataset (license request at [sail.usc.edu/iemocap](https://sail.usc.edu/iemocap/)).

```bash
cd src/finetune/iemocap
./run_er.sh amba_patch400_base
```

Runs cross-validation. Metric: accuracy.

---

## Inference Benchmark

The inference scripts measure latency and peak GPU memory across a range of input token sizes. They must be run from `src/` so that the `models/` package is on the Python path:

```bash
cd src
python inference/inference_amba.py  --model_size base    # SSAMBA
python inference/inference_ssast.py --model_size base    # SSAST
```

Results are written to `inference_times_{size}_amba_batch2.csv`.

---

## Architecture

![architecture](figures/ssamba.png)

**Data flow:**
```
Raw Audio → log-mel spectrogram (128 bins × T frames)
          → 16×16 patch embedding → token sequence
          → bidirectional Vision Mamba encoder
          → task head (masked reconstruction | classification)
```

**Model sizes:**

| Size  | embed_dim | depth |
|-------|-----------|-------|
| base  | 768       | 24    |
| small | 384       | 24    |
| tiny  | 192       | 24    |

---

## Citation

```bibtex
@inproceedings{Shams_2024,
   title={SSAMBA: Self-Supervised Audio Representation Learning With Mamba State Space Model},
   url={http://dx.doi.org/10.1109/SLT61566.2024.10832304},
   DOI={10.1109/slt61566.2024.10832304},
   booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)},
   publisher={IEEE},
   author={Shams, Siavash and Dindar, Sukru Samet and Jiang, Xilin and Mesgarani, Nima},
   year={2024},
   month=dec, pages={1053--1059}
}
```
