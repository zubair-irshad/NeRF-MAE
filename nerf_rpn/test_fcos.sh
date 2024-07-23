#!/usr/bin/env bash

set -x
set -e

resolution=160
dataset_name="front3d"

DATA_ROOT="/wild6d_data/zubair/nerf_rpn/front3d"

python3 -u run_fcos.py \
--mode eval \
--resolution 160 \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--norm_reg_targets \
--centerness_on_reg \
--nms_thresh 0.3 \
--output_proposals \
--save_level_index \
--rotated_bbox \
--batch_size 1 \
--gpus 0 \
--normalize_density \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "/wild6d_data/zubair/nerf_mae/front3d_fcos_swin_1000" \
--checkpoint "/wild6d_data/zubair/nerf_mae/front3d_fcos_swin_1000/model_best.pt" \
