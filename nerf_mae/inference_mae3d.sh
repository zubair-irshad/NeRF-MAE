#!/usr/bin/env bash

set -x
set -e

python3 -u model/mae/inference.py \
--backbone_type swin_s \
--folder_name "/home/zubairirshad/Downloads/rlbench_raw" \
--filename "turn_tap" \
--normalize_density \
--resolution 256 \
--masking_prob 0.75 \
--masking_strategy "random"

# --folder_name "/home/zubairirshad/Downloads/front3d_rpn_data" \
# --filename "3dfront_0030_00" \
# --checkpoint "/home/zubairirshad/Downloads/ckpts_nerf_mae/grid_2/epoch_4000.pt" \
