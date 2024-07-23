#!/usr/bin/env bash

set -x
set -e

resolution=160
dataset_name="front3d"
split_name= "3dfront"

DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"

python3 -u run_fcos_pretrained.py \
--mode eval \
--resolution 160 \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--boxes_path ${DATA_ROOT}/obb \
--norm_reg_targets \
--normalize_density \
--centerness_on_reg \
--rotated_bbox \
--output_proposals \
--save_level_index \
--nms_thresh 0.3 \
--batch_size 2 \
--gpus 7 \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${split_name}_split.npz" \
--save_path "output/nerf_mae/results/${dataset_name}_finetune" \
--checkpoint "output/nerf_mae/results/${dataset_name}_finetune/model_best_ap50_ap25_0.6498397588729858_0.8356480598449707.pt" \