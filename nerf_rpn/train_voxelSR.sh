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
--mode train \
--resolution $resolution \
--out_resolution $out_resolution \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--out_feat_path ${DATA_ROOT}/features_384 \
--num_epochs 500 \
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
--gpus 5,6,7 \
--percent_train 1.0 \
--tags "${dataset_name}_voxelSR_pretrain_384_old_NOPT" \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "nerf_mae/results/${dataset_name}_voxelSR_pretrain_384_old_NOPT" \
--flip_prob 0.0 \
--rot_scale_prob 0.0 \
--rotate_prob 0.0 \
# --use_pretrained_model \
# --mae_checkpoint "nerf_mae/results/front3d_hm3d_hs_3.5k_0.75_ptmae1.0_aug_loss_mask/epoch_1200.pt"
# --mae_checkpoint "nerf_mae/results/front3d_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation/epoch_2000.pt" \
# --use_pretrained_model

#FRONT3D training
# --normalize_density \
# --tags front3d_1515_fcos_swin_mae0.75_colormaskremovergb_ptmae0.1 \
# --dataset front3d \
# --dataset_split ${DATA_ROOT}/3dfront_split.npz \
# --save_path nerf_mae/results/front3d_fcos_swinmae_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation \
# --mae_checkpoint nerf_mae/results/front3d_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation/epoch_2000.pt \
