#!/usr/bin/env bash

set -x
set -e

# DATA_ROOT=/wild6d_data/zubair/nerf_rpn/front3d_rpn_data

# DATA_ROOT=/wild6d_data/zubair/nerf_rpn/scannet_rpn_data

resolution=160
dataset_name="front3d"
if [ "$dataset_name" == "hypersim" ]; then
    resolution=200
fi
DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data"

# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/front3d_rpn_data"

#only for hypersim
# DATA_ROOT="/wild6d_data/zubair/nerf_rpn/${dataset_name}_rpn_data_160"

python3 -u run_fcos_pretrained.py \
--mode train \
--resolution $resolution \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--boxes_path ${DATA_ROOT}/obb \
--num_epochs 1000 \
--wandb \
--lr 1e-4 \
--weight_decay 1e-3 \
--clip_grad_norm 0.1 \
--log_interval 10 \
--eval_interval 10 \
--norm_reg_targets \
--centerness_on_reg \
--center_sampling_radius 1.5 \
--iou_loss_type iou \
--rotated_bbox \
--log_to_file \
--nms_thresh 0.3 \
--batch_size 8 \
--gpus 4,5,6 \
--percent_train 1.0 \
--normalize_density \
--tags "${dataset_name}_all_2k_3e-4_2" \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_all_2k_1e-4_2" \
--mae_checkpoint "/wild6d_data/zubair/nerf_mae/results/front3d_all/epoch_2000.pt"
# --mae_checkpoint "/wild6d_data/zubair/nerf_mae/results/front3d_sparse1_noskip/epoch_1000.pt"


# --mae_checkpoint "/wild6d_data/zubair/nerf_mae/results/front3d_hm3d_hs_3.5k_0.75_ptmae1.0_aug_loss_mask/epoch_1200.pt" \
# --mae_checkpoint "/wild6d_data/zubair/nerf_mae/results/front3d_HS_HM3D_3.5k_0.75_ptmae1.0_aug_clronly/epoch_2000.pt" \
#--mae_checkpoint /wild6d_data/zubair/nerf_mae/results/${dataset_name}_FT_front3drgb_0.75_ptmae1.0_aug_recon_train_scenes/epoch_2000.pt \
#--mae_checkpoint "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_FT_rgb_0.75_ptmae1.0_aug_recon_all_trainscenes/epoch_2000.pt" \
# --checkpoint "/wild6d_data/zubair/nerf_mae/results/${dataset_name}_fcos_swinmae_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation/model_best.pt"
# --eval_interval 20 \
# --lr 1e-3 \
#FRONT3D training
# --normalize_density \
# --tags front3d_1515_fcos_swin_mae0.75_colormaskremovergb_ptmae0.1 \
# --dataset front3d \
# --dataset_split ${DATA_ROOT}/3dfront_split.npz \
# --save_path /wild6d_data/zubair/nerf_mae/results/front3d_fcos_swinmae_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation \
# --mae_checkpoint /wild6d_data/zubair/nerf_mae/results/front3d_1515_0.75_color_only_maskremovergb_ptmae1.0_augmentation/epoch_2000.pt \
