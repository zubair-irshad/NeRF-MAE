#!/usr/bin/env bash

set -x
set -e

DATA_ROOT="../dataset/pretrain"
resolution=160

dataset_name="nerfmae"


if [ "$dataset_name" == "hypersim" ]; then
    resolution=200
fi


python3 -u run_swin_mae3d.py \
--mode train \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--num_epochs 2000 \
--wandb \
--lr 1e-4 \
--weight_decay 1e-3 \
--log_interval 30 \
--eval_interval 200 \
--normalize_density \
--log_to_file \
--batch_size 8 \
--resolution $resolution \
--masking_prob 0.75 \
--dataset "${dataset_name}" \
--dataset_split "${DATA_ROOT}/${dataset_name}_split.npz" \
--save_path "output/nerf_mae/results/${dataset_name}_all" \
--gpus 0,1,2,3 \
--percent_train 1.0 \
--tags "${dataset_name}_all" \