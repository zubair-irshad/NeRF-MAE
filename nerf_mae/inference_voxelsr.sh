#!/usr/bin/env bash

set -x
set -e

python3 -u model/mae/inference_voxel_sr.py \
--folder_name "/home/zubairirshad/Downloads/front3d_rpn_data" \
--filename "3dfront_0117_00" \
--normalize_density \
--resolution 160 \
--masking_prob 0.75 \
--masking_strategy "random" \
--checkpoint "/home/zubairirshad/Downloads/ckpts/nerf_mae/results/front3d_voxelSR_swinmae_1k_1515_0.75_clr_mrmvrgb_256_pretrain_500/model_best.pt" \
--out_resolution 256 \
--is_eval 

# --folder_name "/home/zubairirshad/Downloads/front3d_rpn_data" \
# --filename "3dfront_0030_00" \
# --checkpoint "/home/zubairirshad/Downloads/ckpts_nerf_mae/grid_2/epoch_4000.pt" \
