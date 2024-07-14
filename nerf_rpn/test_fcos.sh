#!/usr/bin/env bash

set -x
set -e

# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/front3d_rpn_data"

resolution=160
dataset_name="front3d"
# if [ "$dataset_name" == "hypersim" ]; then
#     resolution=200
# fi
# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"

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
# --boxes_path None
# --save_path "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_fcos_swin_1k_pt1.0_addl1_wonorm" \
# --checkpoint "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_fcos_swin_1k_pt1.0_addl1_wonorm/model_best.pt" \

# --boxes_path ${DATA_ROOT}/obb \
# python3 -u run_fcos.py \
# --mode eval \
# --dataset front3d \
# --resolution 160 \
# --backbone_type swin_s \
# --features_path ${DATA_ROOT}/features \
# --boxes_path ${DATA_ROOT}/obb \
# --dataset_split ${DATA_ROOT}/3dfront_split.npz \
# --save_path /wild6d_data/zubair/nerf_mae/results/front3d_fcos_swin \
# --checkpoint /wild6d_data/zubair/nerf_mae/results/front3d_fcos_swin/model_best.pt \
# --norm_reg_targets \
# --centerness_on_reg \
# --nms_thresh 0.3 \
# --output_proposals \
# --save_level_index \
# --normalize_density \
# --rotated_bbox \
# --batch_size 2 \
# --gpus 0
