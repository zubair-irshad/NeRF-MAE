#!/usr/bin/env bash

set -x
set -e

dataset_name="nerfmae"

DATA_ROOT="../dataset/pretrain"


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
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "../otput/nerf_mae/results//nerfmae_all" \
--gpus 0 \
--percent_train 1.0 \
--checkpoint ../output/nerf_mae/results/nerfmae_all/epoch_1200.pt