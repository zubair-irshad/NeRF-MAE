#!/usr/bin/env bash

set -x
set -e

#DATA_ROOT=/wild6d_data/zubair/nerf_rpn/front3d_rpn_data

resolution=160
dataset_name="front3d"
# if [ "$dataset_name" == "hypersim" ]; then
#     resolution=200
# fi

DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"

# DATA_ROOT="/arkit_data/zubair/front3d_rpn_160_sparse1"

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
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_all_2k_3e-4" \
--checkpoint "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_all_2k_3e-4/model_best_ap50_ap25_0.6498397588729858_0.8356480598449707.pt" \

#FRONT3D testing
# --dataset front3d \
# --dataset_split ${DATA_ROOT}/3dfront_split.npz \
# --save_path /wild6d_data/zubair/nerf_mae/results/front3d_fcos_swinmae_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation \
# --checkpoint /wild6d_data/zubair/nerf_mae/results/front3d_fcos_swinmae_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation/model_best.pt \
