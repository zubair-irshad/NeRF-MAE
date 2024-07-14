#!/usr/bin/env bash

set -x
set -e

# DATA_ROOT=/wild6d_data/zubair/nerf_rpn/front3d_rpn_data
# DATA_ROOT=/wild6d_data/zubair/FRONT3D_MAE

DATA_ROOT="/wild6d_data/zubair/MAE_complete_data"
# DATA_ROOT="/wild6d_data/zubair/FRONT3D_MAE"

# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/front3d_rpn_data"
# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/hm3d_rpn_data"
# DATA_ROOT="/wild6d_data/zubair/arkitscenes_rpn_data"
# dataset_name="scannet"
resolution=160

# dataset_name="front3d"
# dataset_name="scannet"
# dataset_name="arkitscenes"
dataset_name="front3d"


if [ "$dataset_name" == "hypersim" ]; then
    resolution=200
fi

# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"


python3 -u run_swin_mae3d.py \
--mode train \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--num_epochs 2000 \
--wandb \
--lr 1e-4 \
--weight_decay 1e-3 \
--log_interval 30 \
--eval_interval 200 \
--normalize_density \
--log_to_file \
--batch_size 32 \
--resolution $resolution \
--masking_prob 0.75 \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_all" \
--gpus 0,2,3,4,5,6,7 \
--percent_train 1.0 \
--tags "${dataset_name}_all" \
# --masking_strategy random \
# --flip_prob 0.0 \
# --rot_scale_prob 0.0 \
# --rotate_prob 0.0 \
# --checkpoint /wild6d_data/zubair/nerf_mae/results/front3d_FT_rgb_0.75_ptmae1.0_aug_recon_all_trainscenes/epoch_2000.pt
# --flip_prob 0.0 \
# --rot_scale_prob 0.0 \
# --rotate_prob 0.0 \