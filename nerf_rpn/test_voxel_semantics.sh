#!/usr/bin/env bash

set -x
set -e

# DATA_ROOT=/wild6d_data/zubair/nerf_rpn/front3d_rpn_data

# DATA_ROOT=/wild6d_data/zubair/nerf_rpn/scannet_rpn_data

resolution=160
dataset_name="hm3d"
# if [ "$dataset_name" == "hypersim" ]; then
#     resolution=200
# fi

# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"
# SEM_VOXLEL_PATH="/wild6d_data/zubair/nerf_rpn/FRONT3D_render_seg_all/voxel"


DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"
SEM_VOXLEL_PATH="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data/voxel_${dataset_name}"

python3 -u run_voxel_semantics.py \
--mode eval \
--resolution $resolution \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--sem_feat_path ${SEM_VOXLEL_PATH} \
--norm_reg_targets \
--centerness_on_reg \
--rotated_bbox \
--nms_thresh 0.3 \
--batch_size 1 \
--normalize_density \
--gpus 5 \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--flip_prob 0.0 \
--rot_scale_prob 0.0 \
--rotate_prob 0.0 \
--save_path "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_voxelSem_3.5k_0.75_skip_weightedce_NOPT" \
--checkpoint "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_voxelSem_3.5k_0.75_skip_weightedce_NOPT/model_best.pt" \
# --use_pretrained_model
# --mae_checkpoint "/wild6d_data/zubair/nerf_mae/results/front3d_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation/epoch_2000.pt" \

# --lr 3e-4 \