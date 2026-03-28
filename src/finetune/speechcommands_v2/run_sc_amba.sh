#!/bin/bash
#SBATCH --job-name=sc_amba
#SBATCH --gres=gpu:l40:1          # Request an L40 GPU
#SBATCH --cpus-per-task=8         # Number of CPUs
#SBATCH --mem=32G                 # Amount of memory
#SBATCH --output=job_%j.out       # Standard output and error log

set -x
export TORCH_HOME=../../pretrained_models
mkdir -p exp

# Prep speechcommands dataset if not already done
if [ -e data/datafiles ]; then
    echo "speechcommands already downloaded and processed."
else
    python prep_sc.py
fi

# ---------------------------------------------------------------------------
# Pretrained model
# ---------------------------------------------------------------------------
pretrain_exp="amba"
pretrain_model=$1
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
exp_dir=./exp/test01-${dataset}\
-f${fstride}-t${tstride}\
-b${batch_size}-lr${lr}\
-${task}-${model_size}\
-${pretrain_exp}-${pretrain_model}\
-${head_lr}x-noise${noise}

# ---------------------------------------------------------------------------
# Run fine-tuning
# ---------------------------------------------------------------------------
CUDA_CACHE_DISABLE=1 python -W ignore ../../run_amba.py \
    --use_wandb \
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
    --depth             ${depth} \
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
    --rms_norm          ${rms_norm} \
    --residual_in_fp32  ${residual_in_fp32} \
    --fused_add_norm    ${fused_add_norm} \
    --if_rope           ${if_rope} \
    --if_rope_residual  ${if_rope_residual} \
    --bimamba_type      ${bimamba_type} \
    --drop_path_rate    ${drop_path_rate} \
    --stride            ${stride} \
    --channels          ${channels} \
    --num_classes       ${num_classes} \
    --drop_rate         ${drop_rate} \
    --norm_epsilon      ${norm_epsilon} \
    --if_bidirectional  ${if_bidirectional} \
    --final_pool_type   ${final_pool_type} \
    --if_abs_pos_embed  ${if_abs_pos_embed} \
    --if_bimamba        ${if_bimamba} \
    --if_cls_token      ${if_cls_token} \
    --if_devide_out     ${if_devide_out} \
    --use_double_cls_token   ${use_double_cls_token} \
    --use_middle_cls_token   ${use_middle_cls_token}
