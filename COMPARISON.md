# SSAMBA vs Audio-Mamba (AuM) — Comparison Summary

## TL;DR

Both use bidirectional Mamba to encode audio spectrograms, but they solve different problems: **SSAMBA is a self-supervised pretraining framework**, while **AuM is a supervised classifier that refines how bidirectional Mamba should be structured**.

---

## 1. Core Objective

| | SSAMBA | Audio-Mamba (AuM) |
|---|---|---|
| **Goal** | Learn general audio representations without labels | Supervised audio classification |
| **Training paradigm** | Self-supervised pretrain → fine-tune | Supervised only (pretrain on ImageNet/AudioSet as transfer, not SSL) |
| **Paper venue** | IEEE SLT 2024 | IEEE SPL 2024 |

---

## 2. Architecture

| | SSAMBA | Audio-Mamba (AuM) |
|---|---|---|
| **Backbone** | VisionMamba (from Vim repo) | Forks `models_mamba.py` from Vim (`# This file modifies ...`) |
| **Depth** | 24 layers (base/small/tiny) | 24 layers |
| **Bidirectional variants** | One (`bimamba_type=v2`) | Three: Fo-Fo (unidirectional), Fo-Bi (v1, shared weights), Bi-Bi (v2, fully independent) |
| **CLS token position** | Front (0), hardcoded in finetuning | Middle (N//2) default — well-motivated for bidirectional SSMs |
| **Patch embedding** | Fixed patch size (16×16) | **FlexiPatch** — variable patch sizes during training |
| **Positional embedding** | Fixed learnable, manual bilinear interp at load | **FlexiPosEmbed** — auto bilinear interp every forward pass |
| **RoPE** | Inherited flag, not well-integrated | 2D Vision RoPE, bilinear interpolated |
| **Model sizes** | base (768d), small (384d), tiny (192d) | base (768d), small (384d), tiny (192d) |

---

## 3. Bidirectional Mamba — How It's Done

Both repos use the **same external pairing strategy**: allocate 2× the number of blocks, pair them up, run one forward and one backward, then sum:

```python
for i in range(len(self.layers) // 2):
    hidden_states_f, residual_f = self.layers[i*2](hidden_states, residual)
    hidden_states_b, residual_b = self.layers[i*2+1](hidden_states.flip([1]), ...)
    hidden_states = hidden_states_f + hidden_states_b.flip([1])
    residual     = residual_f     + residual_b.flip([1])
```

`depth=24` → **12 bidirectional pairs**. AuM additionally supports `v1` (bidirectional within a single block, shared weights, separate backward A matrix).

| Bidirectionality | SSAMBA | AuM |
|---|---|---|
| External pairing (v2, separate blocks) | Yes (only option) | Yes |
| Internal within-block (v1, shared weights) | No | Yes |
| Unidirectional baseline (Fo-Fo) | Flag only | Yes |

---

## 4. Pretraining / Weight Initialization

| | SSAMBA | Audio-Mamba (AuM) |
|---|---|---|
| **Self-supervised pretraining** | Yes — masked patch modeling (MPC + MPG) | No |
| **Pretraining objective** | NCE (discriminative) + MSE (generative) over masked patches | N/A |
| **Transfer source** | Pretrained from scratch on AudioSet + LibriSpeech | ImageNet ViM weights OR AudioSet AuM weights |
| **Pretrain cluster targets** | K-means on raw spectrogram patches (no teacher needed) | N/A |

---

## 5. Data Pipeline

| | SSAMBA | Audio-Mamba (AuM) |
|---|---|---|
| **Input format** | Custom JSON manifests | Custom JSON manifests (same format) |
| **Mel-spectrogram** | torchaudio kaldi compat, 128 bins | torchaudio, 128 bins |
| **Augmentation** | SpecAugment (freq+time mask), mix-up, optional noise | SpecAugment, mix-up |
| **Normalization** | `(x - mean) / (std * 2)` | `(x - mean) / (std * 2)` (same) |

---

## 6. Training Infrastructure

| | SSAMBA | Audio-Mamba (AuM) |
|---|---|---|
| **Multi-GPU** | `torch.nn.DataParallel` | HuggingFace Accelerate (fp16, cleaner DDP) |
| **Optimizer (pretrain)** | AdamW | N/A |
| **Optimizer (finetune)** | Adam with split LR (backbone vs MLP head) | Adam |
| **LR scheduling** | MultiStepLR or ReduceLROnPlateau | Cosine decay |
| **Weight averaging** | Optional (`--wa`, `--wa_start`, `--wa_end`) | Not present |
| **Logging** | Weights & Biases (optional) | Weights & Biases |

---

## 7. Downstream Tasks

| | SSAMBA | Audio-Mamba (AuM) |
|---|---|---|
| **Datasets** | AudioSet, ESC-50, Speech Commands v1/v2, UrbanSound8k, VoxCeleb1, IEMOCAP | AudioSet (full + balanced), VGGSound, Speech Commands v2, VoxCeleb, EPIC-SOUNDS |
| **Task types** | Classification, tagging, speaker ID, emotion recognition | Classification, tagging, action recognition (EPIC) |
| **Metrics** | mAP, accuracy, AUC | mAP, accuracy |

---

## 8. Key Differences in Design Philosophy

**SSAMBA** asks: *"Can Mamba replace ViT in self-supervised audio pretraining?"*
- Efficiency is the headline claim (~93% faster, ~95% less memory vs SSAST)
- SSL heads (MPC + MPG) are the novel contribution
- Architecture is largely borrowed from VisionMamba

**AuM** asks: *"What is the best way to make Mamba bidirectional for audio?"*
- Systematically compares three bidirectional scanning strategies
- Introduces FlexiPatch for patch-size robustness
- Motivated middle CLS token placement for SSMs
- Cleaner engineering (Accelerate, modular bidirectional variants)
- Requires patching the installed `mamba_ssm` package — a significant setup friction point

---

## 9. Architectural Summary Table

| Aspect | SSAMBA | AuM |
|---|---|---|
| Backbone source | External VisionMamba (Vim repo) | Fork of Vim's `models_mamba.py` |
| Selective SSM | Yes | Yes |
| Layers / bidir pairs | 24 blocks / 12 pairs | Same |
| embed_dim (base/small/tiny) | 768 / 384 / 192 | Same |
| Bidirectional variants | 1 (v2 only) | 3 (Fo-Fo, v1, v2) |
| Patch embedding | Fixed size | FlexiPatch (variable at runtime) |
| Positional embedding | Fixed learnable, manual interp | FlexiPos (auto interp per forward) |
| CLS token default position | 0 (front) | N//2 (middle) — better for bidir SSMs |
| RoPE | Inherited flag, not well-integrated | 2D Vision RoPE, bilinear interp |
| Normalization | RMSNorm + fused_add_norm | Same |
| Initialization | GPT-2 scheme + trunc_normal_ | Same |
| drop_path_rate default | 0.1 | 0 |
| SSL pretraining heads | MPC + MPG (core contribution) | None |
| Classification head | LayerNorm → Linear | Linear only |
| Token sequence transpose | No | Yes (`transpose_token_sequence`) |
| Random token rank aug | No | Yes (`if_random_token_rank`) |
| Layer-wise output access | `model.module.v.layers` | `model.layers` |

---

## When to use which

- Use **SSAMBA** if you have large amounts of **unlabeled audio** and want to pretrain a general-purpose encoder
- Use **AuM** if you have labeled data and want a **drop-in supervised classifier** with more architectural flexibility and cleaner training code
