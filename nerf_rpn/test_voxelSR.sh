#!/usr/bin/env bash

set -x
set -e

# DATA_ROOT=nerf_rpn/front3d_rpn_data

# DATA_ROOT=nerf_rpn/scannet_rpn_data

resolution=160
out_resolution=384
dataset_name="hm3d"
# if [ "$dataset_name" == "hypersim" ]; then
#     resolution=200
# fi

# DATA_ROOT="nerf_rpn/${dataset_name}_rpn_data_allres"

DATA_ROOT="nerf_rpn/hm3d_rpn_data"

python3 -u run_voxelSR.py \
--mode eval \
--resolution $resolution \
--out_resolution $out_resolution \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--out_feat_path ${DATA_ROOT}/features_384 \
--norm_reg_targets \
--centerness_on_reg \
--rotated_bbox \
--nms_thresh 0.3 \
--batch_size 2 \
--gpus 3 \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "nerf_mae/results/${dataset_name}_voxelSR_pretrain_384_old" \
--checkpoint "nerf_mae/results/${dataset_name}_voxelSR_pretrain_384_old/model_best.pt" \
--use_pretrained_model
# --save_path "nerf_mae/results/${dataset_name}_voxelSR_swinmae_1k_1515_0.75_clr_mrmvrgb_384_pretrain_500" \


