#!/usr/bin/env bash

set -x
set -e

dataset_name="front3d"
split_name="3dfront"

DATA_ROOT="NeRF-MAE/dataset/${dataset_name}_rpn_data"


python3 -u run_swin_mae3d.py \
--mode eval \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--normalize_density \
--log_to_file \
--batch_size 4 \
--resolution 160 \
--masking_prob 0.75 \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${split_name}_split.npz" \
--save_path "../otput/nerf_mae/results//nerfmae_all" \
--gpus 0 \
--percent_train 1.0 \
--checkpoint NeRF-MAE/checkpoints/mae_pretrained/nerf_mae_pretrained.pt