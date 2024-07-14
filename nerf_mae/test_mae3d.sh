#!/usr/bin/env bash

set -x
set -e

# DATA_ROOT=/wild6d_data/zubair/nerf_rpn/front3d_rpn_data
# DATA_ROOT=/wild6d_data/zubair/FRONT3D_MAE

dataset_name="front3d"
# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"

DATA_ROOT="/arkit_data/zubair/front3d_rpn_160_sparse4"


python3 -u run_swin_mae3d.py \
--mode eval \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--normalize_density \
--log_to_file \
--batch_size 8 \
--resolution 160 \
--masking_prob 0.75 \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "/wild6d_data/zubair/nerf_mae/results/front3d_sparse4" \
--gpus 5 \
--percent_train 1.0 \
--checkpoint /wild6d_data/zubair/nerf_mae/results/front3d_sparse4/epoch_1000.pt
# --flip_prob 0.0 \
# --rot_scale_prob 0.0 \
# --rotate_prob 0.0 \