#!/usr/bin/env bash

set -x
set -e

resolution=160
dataset_name="front3d"
split_name = "3dfront"
if [ "$dataset_name" == "hypersim" ]; then
    resolution=200
fi
DATA_ROOT="../dataset/finetune/${dataset_name}_rpn_data"

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
--tags "${dataset_name}_finetune" \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${split_name}_split.npz" \
--save_path "output/nerf_mae/results/${dataset_name}_finetune" \
--mae_checkpoint "/nerf_mae/results/front3d_all/epoch_1200.pt"