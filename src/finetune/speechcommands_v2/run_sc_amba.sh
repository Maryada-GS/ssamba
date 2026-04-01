#!/bin/bash
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --job-name=tiny-250
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.log

source ../../../.venv/bin/activate
echo "[$(date)] Python: $(which python)"
echo "[$(date)] Python version: $(python --version)"

set -e
set -x


if [ -z "$1" ]; then
    echo "Usage: $0 <pretrain_model>  (e.g. ssamba_tiny_250)"
    exit 1
fi

export TORCH_HOME=../../pretrained_models

# Prep speechcommands dataset if not already done
echo "[$(date)] Stage 1/3: preparing dataset..."
if [ -e data/datafiles ]; then
    echo "[$(date)] Stage 1/3: dataset already prepared, skipping."
else
    python prep_sc.py
    echo "[$(date)] Stage 1/3: dataset preparation done."
fi

# ---------------------------------------------------------------------------
# Pretrained model
# ---------------------------------------------------------------------------
pretrain_exp="amba"
pretrain_model=$1

# Download pretrained weights if not already present (skips if .pth exists)
echo "[$(date)] Stage 2/3: downloading weights for ${pretrain_model}..."
python download_weights.py "${pretrain_model}"
echo "[$(date)] Stage 2/3: weights ready."
pretrain_path="../../model_weights/${pretrain_model}.pth"

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
dataset=speechcommands
dataset_mean=-6.845978
dataset_std=5.5654526
target_length=128
noise=True

tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
bal=none
lr=2.5e-4
freqm=48
timem=48
mixup=0.6
epoch=30
batch_size=128
fshape=16
tshape=16
fstride=10
tstride=10
head_lr=1
task=ft_avgtok

# ---------------------------------------------------------------------------
# Model size (inferred from pretrain_model name)
# ---------------------------------------------------------------------------
if [[ $pretrain_model == *"tiny"* ]]; then
    model_size="tiny"
    embed_dim=192
elif [[ $pretrain_model == *"small"* ]]; then
    model_size="small"
    embed_dim=384
else
    model_size="base"
    embed_dim=768
fi

# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------
exp_dir=./experiments/test01-${dataset}\
-f${fstride}-t${tstride}\
-b${batch_size}-lr${lr}\
-${task}-${model_size}\
-${pretrain_exp}-${pretrain_model}\
-${head_lr}x-noise${noise}

# ---------------------------------------------------------------------------
# Run fine-tuning
# ---------------------------------------------------------------------------
echo "[$(date)] Stage 3/3: starting fine-tuning (model=${pretrain_model}, size=${model_size})..."
# Original command commented out — Mamba config vars were undefined in this script;
# removed them below to let run.py argparse defaults apply.
# CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --model amba \
#     --use_wandb \
#     --dataset           ${dataset} \
#     --data-train        ${tr_data} \
#     --data-val          ${val_data} \
#     --data-eval         ${eval_data} \
#     --exp-dir           ${exp_dir} \
#     --label-csv         ./data/speechcommands_class_labels_indices.csv \
#     --n_class           35 \
#     --pretrained_mdl_path ${pretrain_path} \
#     --model_size        ${model_size} \
#     --embed_dim         ${embed_dim} \
#     --depth             ${depth} \
#     --task              ${task} \
#     --lr                ${lr} \
#     --n-epochs          ${epoch} \
#     --batch-size        ${batch_size} \
#     --save_model        False \
#     --freqm             ${freqm} \
#     --timem             ${timem} \
#     --mixup             ${mixup} \
#     --bal               ${bal} \
#     --fshape            ${fshape} \
#     --tshape            ${tshape} \
#     --fstride           ${fstride} \
#     --tstride           ${tstride} \
#     --warmup            True \
#     --adaptschedule     False \
#     --head_lr           ${head_lr} \
#     --noise             ${noise} \
#     --dataset_mean      ${dataset_mean} \
#     --dataset_std       ${dataset_std} \
#     --target_length     ${target_length} \
#     --num_mel_bins      128 \
#     --lrscheduler_start 5 \
#     --lrscheduler_step  1 \
#     --lrscheduler_decay 0.85 \
#     --loss              BCE \
#     --metrics           acc \
#     --wa                False \
#     --rms_norm          ${rms_norm} \
#     --residual_in_fp32  ${residual_in_fp32} \
#     --fused_add_norm    ${fused_add_norm} \
#     --if_rope           ${if_rope} \
#     --if_rope_residual  ${if_rope_residual} \
#     --bimamba_type      ${bimamba_type} \
#     --drop_path_rate    ${drop_path_rate} \
#     --stride            ${stride} \
#     --channels          ${channels} \
#     --num_classes       ${num_classes} \
#     --drop_rate         ${drop_rate} \
#     --norm_epsilon      ${norm_epsilon} \
#     --if_bidirectional  ${if_bidirectional} \
#     --final_pool_type   ${final_pool_type} \
#     --if_abs_pos_embed  ${if_abs_pos_embed} \
#     --if_bimamba        ${if_bimamba} \
#     --if_cls_token      ${if_cls_token} \
#     --if_devide_out     ${if_devide_out} \
#     --use_double_cls_token   ${use_double_cls_token} \
#     --use_middle_cls_token   ${use_middle_cls_token}

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --model amba \
    --dataset           ${dataset} \
    --data-train        ${tr_data} \
    --data-val          ${val_data} \
    --data-eval         ${eval_data} \
    --exp-dir           ${exp_dir} \
    --label-csv         ./data/speechcommands_class_labels_indices.csv \
    --n_class           35 \
    --pretrained_mdl_path ${pretrain_path} \
    --model_size        ${model_size} \
    --embed_dim         ${embed_dim} \
    --task              ${task} \
    --lr                ${lr} \
    --n-epochs          ${epoch} \
    --batch-size        ${batch_size} \
    --save_model        False \
    --freqm             ${freqm} \
    --timem             ${timem} \
    --mixup             ${mixup} \
    --bal               ${bal} \
    --fshape            ${fshape} \
    --tshape            ${tshape} \
    --fstride           ${fstride} \
    --tstride           ${tstride} \
    --warmup            True \
    --adaptschedule     False \
    --head_lr           ${head_lr} \
    --noise             ${noise} \
    --dataset_mean      ${dataset_mean} \
    --dataset_std       ${dataset_std} \
    --target_length     ${target_length} \
    --num_mel_bins      128 \
    --lrscheduler_start 5 \
    --lrscheduler_step  1 \
    --lrscheduler_decay 0.85 \
    --loss              BCE \
    --metrics           acc \
    --wa                False \
    --if_abs_pos_embed  'true' \
    --if_cls_token      'true' \
    --if_devide_out     'true' \
    --use_middle_cls_token 'true' \
    --final_pool_type   'mean' \
    --num-workers       2 \
    --use_mlflow
echo "[$(date)] Stage 3/3: fine-tuning finished."
