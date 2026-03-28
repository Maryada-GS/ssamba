#!/bin/bash

#SBATCH --job-name=amba_sid
#SBATCH --gres=gpu:a40:1         # Request an A40 GPU
#SBATCH --cpus-per-task=12        # Number of CPUs
#SBATCH --mem=32G                # Amount of memory
#SBATCH --output=job_%j.out      # Standard output and error log

set -x
export TORCH_HOME=../../pretrained_models

# Default parameters
model_function=${1:-ssast_patch400_base}  # default to 'ssast_patch400_base'
lr=${2:-1e-4}

expname=sid_${model_function}_${lr}
expdir=./exp/$expname
mkdir -p $expdir

python3 -m s3prl.run_downstream --expdir $expdir -m train -u $model_function -d voxceleb1 -s hidden_states -o config.optimizer.lr=$lr -f
