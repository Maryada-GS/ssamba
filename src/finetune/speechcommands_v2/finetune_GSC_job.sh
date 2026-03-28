#!/bin/bash

# End-to-end launcher for Speech Commands v2 fine-tuning.
# Run from anywhere — paths are resolved from this script's location.
# Steps:
#   1. Prepare dataset (skipped if already done)
#   2. Download pretrained weights (skipped if already present)
#   3. Submit fine-tuning SLURM job

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Prepare dataset
# ---------------------------------------------------------------------------
python prep_sc.py

# ---------------------------------------------------------------------------
# Step 2: Download model weights
# ---------------------------------------------------------------------------
declare -a models=("ssamba_tiny_250")

for model in "${models[@]}"; do
    python download_weights.py "${model}"
done

# ---------------------------------------------------------------------------
# Step 3: Submit fine-tuning job(s)
# ---------------------------------------------------------------------------
for model in "${models[@]}"; do
    sbatch run_sc_amba.sh "${model}"
done
