#!/usr/bin/env bash

set -x
set -e

#choices="hypersim", "front3d", "scannet"

# DATA_ROOT=/wild6d_data/zubair/nerf_rpn/front3d_rpn_data
# DATA_ROOT=/wild6d_data/zubair/nerf_rpn/scannet_rpn_data

resolution=160
dataset_name="front3d"
if [ "$dataset_name" == "hypersim" ]; then
    resolution=200
fi
# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"
DATA_ROOT="/datasets/nerf_rpn/${dataset_name}_rpn_data"
# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data_160"

python3 -u run_fcos.py \
--mode train \
--resolution $resolution \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--boxes_path ${DATA_ROOT}/obb \
--num_epochs 1000 \
--wandb \
--lr 3e-4 \
--weight_decay 1e-3 \
--clip_grad_norm 0.1 \
--log_interval 10 \
--eval_interval 10 \
--keep_checkpoints 3 \
--norm_reg_targets \
--centerness_on_reg \
--center_sampling_radius 1.5 \
--iou_loss_type iou \
--rotated_bbox \
--log_to_file \
--nms_thresh 0.3 \
--batch_size 8 \
--gpus 0,1,2 \
--percent_train 1.0 \
--normalize_density \
--tags "${dataset_name}_fcos_part4" \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "/datasets/nerf_mae/results/${dataset_name}_reproduce_dgx"

# --checkpoint "/wild6d_data/zubair/nerf_mae/results/nerfrpn_checkpoints/front3d_fcos_swinS.pt"
# --checkpoint "/wild6d_data/zubair/nerf_mae/results/front3d_fcos_swinmae_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation/model_best.pt"



#Scannet
# --tags scannet_fcos_swin_2k_pt1.0_wo_normdensity \
# --dataset scannet \
# --dataset_split ${DATA_ROOT}/scannet_split.npz \
# --save_path /wild6d_data/zubair/nerf_mae/results/scannet_fcos_swin_2k_pt1.0_wo_normdensity \