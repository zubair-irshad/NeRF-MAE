#!/usr/bin/env bash

set -x
set -e

DATA_ROOT=/wild6d_data/zubair/nerf_rpn/hypersim_rpn_data

python3 -u run_swin_mae3d.py \
--mode train \
--dataset hypersim \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--dataset_split ${DATA_ROOT}/hypersim_split.npz \
--num_epochs 4000 \
--wandb \
--lr 3e-4 \
--weight_decay 1e-3 \
--log_interval 10 \
--eval_interval 500 \
--log_to_file \
--batch_size 8 \
--resolution 200 \
--flip_prob 0.0 \
--rot_scale_prob 0.0 \
--rotate_prob 0.0 \
--masking_prob 0.75 \
--save_path /wild6d_data/zubair/nerf_mae/results/hypersim_mae_0.75_color_only \
--gpus 5,6,7
