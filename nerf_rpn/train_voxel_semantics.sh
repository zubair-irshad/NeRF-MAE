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

# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data_allres"
# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"
# SEM_VOXLEL_PATH="/wild6d_data/zubair/nerf_rpn/FRONT3D_render_seg_all/voxel"

DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"
SEM_VOXLEL_PATH="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data/voxel_${dataset_name}"

python3 -u run_voxel_semantics.py \
--mode train \
--resolution $resolution \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--sem_feat_path ${SEM_VOXLEL_PATH} \
--num_epochs 1000 \
--wandb \
--lr 3e-4 \
--weight_decay 1e-3 \
--clip_grad_norm 0.1 \
--log_interval 10 \
--eval_interval 20 \
--keep_checkpoints 3 \
--norm_reg_targets \
--centerness_on_reg \
--center_sampling_radius 1.5 \
--iou_loss_type iou \
--rotated_bbox \
--log_to_file \
--nms_thresh 0.3 \
--batch_size 8 \
--normalize_density \
--gpus 4,5,6,7 \
--percent_train 1.0 \
--tags "${dataset_name}_voxelSem_3.5k_0.75_skip_weightedce" \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_voxelSem_3.5k_0.75_skip_weightedce" \
--flip_prob 0.0 \
--rot_scale_prob 0.0 \
--rotate_prob 0.0 \
--use_pretrained_model \
--mae_checkpoint "/wild6d_data/zubair/nerf_mae/results/front3d_hm3d_hs_3.5k_0.75_ptmae1.0_aug_loss_mask/epoch_1200.pt"
# --mae_checkpoint "/wild6d_data/zubair/nerf_mae/results/front3d_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation/epoch_2000.pt" \
# --use_pretrained_model
# --lr 3e-4 \