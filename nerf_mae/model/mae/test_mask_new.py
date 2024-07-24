import open3d as o3d
import numpy as np

import torch
from einops import rearrange
import torch.nn as nn
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("/home/zubairirshad/NeRF_MAE_internal/nerf_mae")
from model.mae.torch_utils import *
from model.mae.viz_utils import *


def patchify_3d(x, mask):
    """
    rgbsimga: (N, 4, H, W, D)
    x: (N, L, L, L, patch_size**3 *4)
    """
    patch_size=(4, 4, 4)

    print("x.shape", x.shape)

    p = patch_size[0]
    assert x.shape[2] == x.shape[3] == x.shape[4] and x.shape[2] % p == 0
    h = w = l = int(round(x.shape[2] // p))
    x = x.reshape(shape=(x.shape[0], 4, h, p, w, p, l, p))
    print("x.shape", x.shape)
    x = rearrange(x, "n c h p w q l r -> n h w l p q r c")
    x = rearrange(x, "n h w l p q r c -> n h w l (p q r) c")
    
    print("x.shape here", x.shape)
    mask = mask.reshape(shape=(x.shape[0], 4, h, p, w, p, l, p))
    mask = rearrange(mask, "n c h p w q l r -> n h w l p q r c")
    mask = rearrange(mask, "n h w l p q r c -> n h w l (p q r) c")
    mask = mask[:, :, :, :, :, 0]

    return x, mask

def unpatchify_3d_full(x, patch_size=None, resolution=None, channel_dims=3):
    """
    rgbsimga: (N, 4, H, W, D)
    x: (N, L, L, L, patch_size**3 *4)
    """
    p = patch_size[0]
    h = w = l = int(round(resolution / p))

    x = x.reshape(shape=(x.shape[0], h, w, l, p, p, p, channel_dims))
    x = rearrange(x, "n h w l p q r c -> n h p w q l r c")
    x = x.reshape(x.shape[0], h * p, w * p, l * p, channel_dims)

    return x


if __name__ == "__main__":
    dataset = "front3d"
    folder_name = "/home/zubairirshad/Downloads/hm3d_rpn_data"
    filename = "00009-vLpv2VX547B_0"
    hypersim = False
    resolution = 160

    new_res = np.array([resolution, resolution, resolution])
    features_path = os.path.join(folder_name, "features", filename + ".npz")
    feature = np.load(features_path, allow_pickle=True)

    res = feature["resolution"]
    rgbsigma = feature["rgbsigma"]
    grid_original = construct_grid(res)

    rgbsigma_original = rgbsigma.copy()
    rgbsigma = rgbsigma.reshape(-1, 4)
    alpha = rgbsigma[..., -1]

    if dataset == "scannet":
        alpha = density_to_alpha_scannet(alpha)
    elif dataset == "front3d":
        alpha = density_to_alpha(alpha)

    mask = alpha > 0.01
    mask = mask.reshape(-1)
    grid_original = grid_original[mask, :]
    grid_np = rgbsigma[:, :3][mask, :]
    pcd_pts = torch.from_numpy(rgbsigma_original).permute(3, 0, 1, 2)

    # new_res = np.array([200, 200, 200])
    pcd_pts_padded, pad_mask = pad_tensor(pcd_pts, target_shape=new_res, pad_value=0)
    print("non zero pad mask", torch.count_nonzero(pad_mask))
    # print("pad_mask", pad_mask.shape)
    pcd_padded_np = pcd_pts_padded.reshape(4, -1).T[:, :3].numpy()

    print("pad_mask", pad_mask.shape)
    # pad_mask = pad_mask.permute(0, 2, 3, 4, 1)[:, :, :, :, 0]
    print("pad_mask", pad_mask.shape)

    # pad_mask = pad_mask.reshape(-1).bool()
    grid = construct_grid(new_res)

    # print("pad_mask", pad_mask.shape, grid.shape, pcd_padded_np.shape)
    # grid_inter = grid[pad_mask, :]
    # pcd_padded_np_inter = pcd_padded_np[pad_mask, :]

    print("pcd_pts_padded", pcd_pts_padded.shape)
    rgbsigma_padded = pcd_pts_padded.permute(0, 2, 3, 4, 1)
    print("rgbsigma_padded", rgbsigma_padded.shape)


    mask_token = torch.nn.Parameter(torch.zeros(4))

    # mask_token = torch.nn.Parameter(torch.FloatTensor([1.0, 0.804, 0.824, 0.851]))

    rgbsigma_padded = rgbsigma_padded.permute(0, 4, 1, 2, 3)
    target, mask = patchify_3d(rgbsigma_padded, pad_mask)


    print("rgbsigma_padded", rgbsigma_padded.shape)
    rgbsigma_padded = rgbsigma_padded.permute(0, 2, 3, 4, 1)

    grid_mask, mask_patches = window_masking_3d(
        rgbsigma_padded, mask_token=mask_token, p_remove=0.75
    )

    print("mask_patches", mask_patches.shape)
    print("mask", mask.shape)
    mask_keep = mask.squeeze(-1).int() * (1- mask_patches)
    mask_remove = mask.squeeze(-1).int() * (mask_patches)
    

    patch_size=(4, 4, 4)


    mask_remove_unroll = unpatchify_3d_full(
        mask,
        patch_size=patch_size,
        resolution=resolution,
        channel_dims=1,
    )

    mask_keep_patch_unroll = unpatchify_3d_full(
        mask_keep.unsqueeze(-1),
        patch_size=patch_size,
        resolution=resolution,
        channel_dims=1,
    )

    mask_remove_patch_unroll = unpatchify_3d_full(
        mask_remove.unsqueeze(-1),
        patch_size=patch_size,
        resolution=resolution,
        channel_dims=1,
    )


    print("=================Visualizing GT=======================\n\n\n")
    grid_vis_original = grid.reshape(-1, 3) * mask_remove_unroll.reshape(-1, 1)
    target_rgb_vis_original = target.reshape(-1, 3) * mask_remove_unroll.reshape(
        -1, 1
    )
    # a = np.array(target_rgb_vis_original[:, 0])[::-1]
    # print("a.shape", a.shape)
    # target_rgb_vis_original[:, 0] = torch.FloatTensor(a)
    #mask
    # print("grid_vis_original", grid_vis_original.shape)
    # grid_vis_original = grid_vis_original[viz_mask, :]
    # print("grid vis mask", grid_vis_original.shape)
    # target_rgb_vis_original = target_rgb_vis_original[viz_mask, :]
    
    draw_grid_colors(grid_vis_original, target_rgb_vis_original.numpy(), coordinate_system=True)


    # print("grid_np.shape, grid_mask", grid_mask.shape, grid_mask.shape)
    # grid_np_mask = grid_mask.detach().cpu().numpy()
    # grid_np = grid_np_mask.reshape((-1, 4))[:, :3]

    # alpha_padded = pcd_pts_padded.reshape(4, -1).T[:, 3]


    # if dataset == "scannet":
    #     alpha = density_to_alpha_scannet(alpha_padded)
    # elif dataset == "front3d":
    #     alpha = density_to_alpha(alpha_padded)

    # mask = alpha_padded > 0.01
    # print("mask_patches", mask_patches.shape, mask.shape)
    # mask_patches = mask_patches.reshape(-1)
    # mask = mask.reshape(-1)

    # # mask = mask * ~(mask_patches.bool())

    # mask_remove = mask * mask_patches.bool()
    # mask_keep = mask * ~mask_patches.bool()

    # print("mask", mask.shape)
    # grid = construct_grid(new_res)

    # grid_remove = grid[mask_remove, :]
    # grid_np_remove = pcd_padded_np[mask_remove, :]


    # grid_keep = grid[mask_keep, :]
    # grid_np_keep = pcd_padded_np[mask_keep, :]

    # draw_grid_colors(grid_remove, grid_np_remove)
    # draw_grid_colors(grid_keep, grid_np_keep)
