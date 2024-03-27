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


if __name__ == "__main__":
    dataset = "front3d"
    folder_name = "/home/zubairirshad/Downloads/front3d_rpn_data"
    filename = "3dfront_0117_00"

    # dataset = "scannet"
    # folder_name = '/home/zubairirshad/Downloads/scannet_rpn_data'
    # filename = 'scene0000_00'

    # folder_name = "/home/zubairirshad/Downloads/front3d_rpn_data/vis_scenes"
    # filename = "3dfront_0023_00"

    # folder_name = "/home/zubairirshad/Downloads/hypersim_rpn_data"
    # filename = "ai_018_001"
    hypersim = False
    resolution = 160

    new_res = np.array([resolution, resolution, resolution])
    features_path = os.path.join(folder_name, "features", filename + ".npz")
    # box_path = os.path.join(folder_name, "obb", filename + ".npy")

    # boxes = np.load(box_path, allow_pickle=True)
    feature = np.load(features_path, allow_pickle=True)

    res = feature["resolution"]
    rgbsigma = feature["rgbsigma"]
    print("res", res)
    print("rgbsigma", rgbsigma.shape)
    grid_original = construct_grid(res)

    rgbsigma_original = rgbsigma.copy()
    print("rgbsigma_original", rgbsigma_original.shape)

    # rgbsigma = np.transpose(rgbsigma_original, (1, 2, 0, 3)).reshape(-1, 4)
    rgbsigma = rgbsigma.reshape(-1, 4)
    print("rgbsigma", rgbsigma.shape)
    # rgbsigma = rgbsigma.reshape(-1, 4)

    alpha = rgbsigma[..., -1]

    if dataset == "scannet":
        alpha = density_to_alpha_scannet(alpha)
    elif dataset == "front3d":
        alpha = density_to_alpha(alpha)

    print("rgb max min", np.max(rgbsigma[..., :3]), np.min(rgbsigma[..., :3]))
    print("alpha max min", np.max(alpha), np.min(alpha))

    mask = alpha > 0.01
    mask = mask.reshape(-1)
    grid_original = grid_original[mask, :]
    grid_np = rgbsigma[:, :3][mask, :]

    draw_grid_colors(grid_original, grid_np)

    pcd_pts = torch.from_numpy(rgbsigma_original).permute(3, 0, 1, 2)

    # new_res = np.array([200, 200, 200])
    pcd_pts_padded, pad_mask = pad_tensor(pcd_pts, target_shape=new_res, pad_value=0)
    print("non zero pad mask", torch.count_nonzero(pad_mask))
    # print("pad_mask", pad_mask.shape)
    pcd_padded_np = pcd_pts_padded.reshape(4, -1).T[:, :3].numpy()

    print("pad_mask", pad_mask.shape)
    pad_mask = pad_mask.permute(0, 2, 3, 4, 1)[:, :, :, :, 0]
    print("pad_mask", pad_mask.shape)

    pad_mask = pad_mask.reshape(-1).bool()
    grid = construct_grid(new_res)

    print("pad_mask", pad_mask.shape, grid.shape, pcd_padded_np.shape)
    grid_inter = grid[pad_mask, :]
    pcd_padded_np_inter = pcd_padded_np[pad_mask, :]

    draw_grid_colors(grid_inter, pcd_padded_np_inter)

    test_2d = False
    if test_2d:
        img = Image.open("/home/zubairirshad/Downloads/dog-cat.jpg")
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(224)]
        )

        img = transform(img).unsqueeze(0)
        plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.show()
        img = img.permute(0, 2, 3, 1)
        print(img.shape)
        x_masked, mask = window_masking(img)
        plt.imshow(x_masked[0].detach().cpu().numpy())
        plt.show()
    else:
        # X, Y, Z = np.mgrid[0:160, 0:160, 0:160]
        # grid_np = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)
        # grid_np = np.expand_dims(grid_np, axis=0)
        # print("grid_np", grid_np.shape)
        # grid = torch.from_numpy(grid_np)

        print("pcd_pts_padded", pcd_pts_padded.shape)
        rgbsigma_padded = pcd_pts_padded.permute(0, 2, 3, 4, 1)
        print("rgbsigma_padded", rgbsigma_padded.shape)


        mask_token = torch.nn.Parameter(torch.zeros(4))

        # mask_token = torch.nn.Parameter(torch.FloatTensor([1.0, 0.804, 0.824, 0.851]))

        grid_mask, mask_patches = window_masking_3d(
            rgbsigma_padded, mask_token=mask_token, p_remove=0.75
        )

        print("grid_np.shape, grid_mask", grid_mask.shape, grid_mask.shape)
        grid_np_mask = grid_mask.detach().cpu().numpy()
        grid_np = grid_np_mask.reshape((-1, 4))[:, :3]

        alpha_padded = pcd_pts_padded.reshape(4, -1).T[:, 3]

        # if not hypersim:
        #     alpha_padded = density_to_alpha(alpha_padded)

        if dataset == "scannet":
            alpha = density_to_alpha_scannet(alpha_padded)
        elif dataset == "front3d":
            alpha = density_to_alpha(alpha_padded)

        mask = alpha_padded > 0.01
        print("mask_patches", mask_patches.shape, mask.shape)
        mask_patches = mask_patches.reshape(-1)
        mask = mask.reshape(-1)

        # mask = mask * ~(mask_patches.bool())

        mask_remove = mask * mask_patches.bool()
        mask_keep = mask * ~mask_patches.bool()

        # mask_patch_remove = pad_mask.bool() & mask_patches.bool()
        # mask_patch_keep = pad_mask.bool() & ~mask_patches.bool()

        print("mask", mask.shape)
        grid = construct_grid(new_res)
        # grid = grid * mask.unsqueeze(-1).detach().cpu().numpy()

        grid_remove = grid[mask_remove, :]
        grid_np_remove = pcd_padded_np[mask_remove, :]

        # grid_remove = grid.numpy().copy()
        # grid_np_remove = pcd_padded_np.copy()
        # grid_np_remove[mask_remove, :] = 0

        # grid_keep = grid.numpy().copy()
        # grid_np_keep = pcd_padded_np.copy()
        # grid_np_keep[mask_keep, :] = 0

        grid_keep = grid[mask_keep, :]
        grid_np_keep = pcd_padded_np[mask_keep, :]

        # grid_patch_remove = grid[mask_patch_remove, :]
        # grid_np_patch_remove = pcd_padded_np[mask_patch_remove, :]

        # grid_patch_keep = grid[mask_patch_keep, :]
        # grid_np_patch_keep = pcd_padded_np[mask_patch_keep, :]

        # mask = ~mask.bool()
        # grid_np[mask, :] = 0
        # draw_grid_colors(grid_original)

        # draw_grid_colors(grid_patch_keep)
        # draw_grid_colors(grid_patch_keep, grid_np_patch_keep)

        # draw_grid_colors(grid_patch_remove)
        # draw_grid_colors(grid_patch_remove, grid_np_patch_remove)

        # draw_grid_colors(grid_remove)
        # draw_grid_colors(grid_keep)

        draw_grid_colors(grid_remove, grid_np_remove)
        draw_grid_colors(grid_keep, grid_np_keep)
