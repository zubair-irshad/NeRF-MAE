#!/usr/bin/env bash

set -x
set -e

resolution=160
dataset_name="front3d"
split_name="3dfront"
DATA_ROOT="../dataset/finetune/${dataset_name}_rpn_data"

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
--dataset_split "${DATA_ROOT}/${split_name}_split.npz" \
--save_path "../output/nerf_mae/results/${dataset_name}_scratch" \
--checkpoint "../output/nerf_mae/results/${dataset_name}_scratch/model_best.pt" \
