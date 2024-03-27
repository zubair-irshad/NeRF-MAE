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
import random
sys.path.append("/home/zubairirshad/NeRF_MAE_internal/nerf_mae")
from model.mae.torch_utils import *
# from model.mae.viz_utils import *
import trimesh

def window_masking_3d(
    x,
    patch_size=(4, 4, 4),
    p_remove=0.50,
    mask_token=None,
    sampling_strategy="random",
):
    batch_size, height, width, depth, channels = x.shape
    mask = torch.zeros((batch_size, height, width, depth, 1)).to(x.device)
    patch_h, patch_w, patch_d = patch_size

    patch_h, patch_w, patch_d = patch_size
    # num_patches = (height // patch_h) * (width // patch_w) * (depth // patch_d)
    # num_masked_patches = 0

    if sampling_strategy == "grid":
        # Calculate the number of patches in each dimension
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        num_patches_d = depth // patch_d

        # Calculate the total number of patches
        num_patches = num_patches_h * num_patches_w * num_patches_d

        # Create a list of all patch indices
        all_patch_indices = [
            (h, w, d)
            for h in range(num_patches_h)
            for w in range(num_patches_w)
            for d in range(num_patches_d)
        ]

        # Iterate through the shuffled patch indices and keep only 'num_to_keep' patches
        num_masked_patches = 0
        count = 0
        for h, w, d in all_patch_indices:
            # we want to mask one out of every four patches

            if count == 0 or count == 1 or count == 2:
                h_start, h_end = h * patch_h, (h + 1) * patch_h
                w_start, w_end = w * patch_w, (w + 1) * patch_w
                d_start, d_end = d * patch_d, (d + 1) * patch_d
                mask[:, h_start:h_end, w_start:w_end, d_start:d_end, :] = 1
                num_masked_patches += 1
                count += 1
            else:
                count += 1
                if count == 4:
                    count = 0

    elif sampling_strategy == "block":
        # mask our 25% of the volume 3 times at random starting and ending locations
        # Calculate the number of patches in each dimension
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        num_patches_d = depth // patch_d

        # Calculate the total number of patches
        num_patches = num_patches_h * num_patches_w * num_patches_d

        # Create a list of all patch indices
        all_patch_indices = [
            (h, w, d)
            for h in range(num_patches_h)
            for w in range(num_patches_w)
            for d in range(num_patches_d)
        ]

        num_to_keep = num_patches // 4  # Keep one out of every four patches
        total_masked_patches = 0
        for i in range(3):
            num_masked_patches = 0
            h_start = height // patch_h
            w_start = width // patch_w
            d_start = depth // patch_d

            h_start = np.random.randint(0, h_start - 0.25 * h_start)

            for h, w, d in all_patch_indices:
                # select random strting locations for the 25% of the volume
                if h > h_start:
                    h_start_new, h_end = h * patch_h, (h + 1) * patch_h
                    w_start_new, w_end = w * patch_w, (w + 1) * patch_w
                    d_start_new, d_end = d * patch_d, (d + 1) * patch_d

                    # check if the patch is already masked, if not mask it
                    if (
                        mask[
                            :,
                            h_start_new:h_end,
                            w_start_new:w_end,
                            d_start_new:d_end,
                            :,
                        ].sum()
                        == 0
                    ):
                        mask[
                            :,
                            h_start_new:h_end,
                            w_start_new:w_end,
                            d_start_new:d_end,
                            :,
                        ] = 1
                        num_masked_patches += 1

                if num_masked_patches >= num_to_keep:
                    total_masked_patches += num_masked_patches
                    num_masked_patches = 0

                    break  # Stop masking patches once we've reached the desired number
        num_masked_patches = total_masked_patches

    elif sampling_strategy == "random":
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        num_patches_d = depth // patch_d

        # Calculate the total number of patches
        num_patches = num_patches_h * num_patches_w * num_patches_d

        num_masked_patches = 0
        for h in range(0, height - patch_h + 1, patch_h):
            for w in range(0, width - patch_w + 1, patch_w):
                for d in range(0, depth - patch_d + 1, patch_d):
                    if random.random() < p_remove:
                        num_masked_patches += 1
                        mask[
                            :, h : h + patch_h, w : w + patch_w, d : d + patch_d, :
                        ] = 1

    masked_x = x.clone()
    masked_x[mask.bool().expand(-1, -1, -1, -1, channels)] = 0
    if mask_token is not None:
        mask_token = mask_token.to(masked_x.device)
        index_mask = (mask.bool()).squeeze(-1)
        masked_x[index_mask, :] = mask_token

    # percent_masked = 100 * num_masked_patches / num_patches
    # print(f"Total number of patches: {num_patches}")
    # print(f"Total number of masked patches: {num_masked_patches}")
    # print(f"Percentage of patches that are masked: {percent_masked:.2f}%")

    return masked_x, mask

if __name__ == "__main__":
    # dataset = "front3d"
    # folder_name = "/home/zubairirshad/Downloads/front3d_rpn_data"
    # filename = "3dfront_0117_00"

    filename = '/home/zubairirshad/Downloads/voxel_grid_cups.npy'

    voxel_grid = np.load(filename)
    print("voxel_grid", voxel_grid.shape)

    resolution = 100
    voxel_size = 0.5

    alpha = 0.5

    # vis_voxel_grid = voxel_grid.transpose(0, 4, 1, 2, 3)

    print("vis_voxel_grid", voxel_grid.shape)

    points = voxel_grid[:, :, :, :, 0:3]

    rgb_voxel = torch.FloatTensor(voxel_grid[:, :, :, :, 3:6])
    rgb = (voxel_grid[:, :, :, :, 3:6] + 1) / 2

    # #convert rgb to bgr 
    # rgb = rgb[..., ::-1]

    points = points.reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # draw = o3d.visualization.EV.draw
    # i = 1
    # name = 'pcd'+str(i)
    # draw({'geometry': pcd, 'name':name })

    # dataset = "scannet"
    # folder_name = '/home/zubairirshad/Downloads/scannet_rpn_data'
    # filename = 'scene0000_00'

    # folder_name = "/home/zubairirshad/Downloads/front3d_rpn_data/vis_scenes"
    # filename = "3dfront_0023_00"

    # folder_name = "/home/zubairirshad/Downloads/hypersim_rpn_data"
    # filename = "ai_018_001"
    # hypersim = False
    # resolution = 160

    # new_res = np.array([resolution, resolution, resolution])
    # features_path = os.path.join(folder_name, "features", filename + ".npz")
    # # box_path = os.path.join(folder_name, "obb", filename + ".npy")

    # # boxes = np.load(box_path, allow_pickle=True)
    # feature = np.load(features_path, allow_pickle=True)

    # res = feature["resolution"]
    # rgbsigma = feature["rgbsigma"]
    # print("res", res)
    # print("rgbsigma", rgbsigma.shape)
    # grid_original = construct_grid(res)

    # rgbsigma_original = rgbsigma.copy()
    # print("rgbsigma_original", rgbsigma_original.shape)

    # # rgbsigma = np.transpose(rgbsigma_original, (1, 2, 0, 3)).reshape(-1, 4)
    # rgbsigma = rgbsigma.reshape(-1, 4)
    # print("rgbsigma", rgbsigma.shape)
    # # rgbsigma = rgbsigma.reshape(-1, 4)

    # alpha = rgbsigma[..., -1]

    # if dataset == "scannet":
    #     alpha = density_to_alpha_scannet(alpha)
    # elif dataset == "front3d":
    #     alpha = density_to_alpha(alpha)

    # print("rgb max min", np.max(rgbsigma[..., :3]), np.min(rgbsigma[..., :3]))
    # print("alpha max min", np.max(alpha), np.min(alpha))

    # mask = alpha > 0.01
    # mask = mask.reshape(-1)
    # grid_original = grid_original[mask, :]
    # grid_np = rgbsigma[:, :3][mask, :]

    # draw_grid_colors(grid_original, grid_np)

    # pcd_pts = torch.from_numpy(rgbsigma_original).permute(3, 0, 1, 2)

    # # new_res = np.array([200, 200, 200])
    # pcd_pts_padded, pad_mask = pad_tensor(pcd_pts, target_shape=new_res, pad_value=0)
    # print("non zero pad mask", torch.count_nonzero(pad_mask))
    # # print("pad_mask", pad_mask.shape)
    # pcd_padded_np = pcd_pts_padded.reshape(4, -1).T[:, :3].numpy()

    # print("pad_mask", pad_mask.shape)
    # pad_mask = pad_mask.permute(0, 2, 3, 4, 1)[:, :, :, :, 0]
    # print("pad_mask", pad_mask.shape)

    # pad_mask = pad_mask.reshape(-1).bool()
    # grid = construct_grid(new_res)

    # print("pad_mask", pad_mask.shape, grid.shape, pcd_padded_np.shape)
    # grid_inter = grid[pad_mask, :]
    # pcd_padded_np_inter = pcd_padded_np[pad_mask, :]

    # draw_grid_colors(grid_inter, pcd_padded_np_inter)

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

        # print("pcd_pts_padded", pcd_pts_padded.shape)
        # rgbsigma_padded = pcd_pts_padded.permute(0, 2, 3, 4, 1)
        # print("rgbsigma_padded", rgbsigma_padded.shape)


        mask_token = torch.nn.Parameter(torch.zeros(3))

        # mask_token = torch.nn.Parameter(torch.FloatTensor([1.0, 0.804, 0.824, 0.851]))

        grid_mask, mask_patches = window_masking_3d(
            rgb_voxel, mask_token=mask_token, p_remove=0.75
        )

        # print("grid_np.shape, grid_mask", grid_mask.shape, grid_mask.shape)
        grid_np_mask = grid_mask.detach().cpu().numpy()
        grid_np = grid_np_mask.reshape((-1, 4))[:, :3]

        # alpha_padded = pcd_pts_padded.reshape(4, -1).T[:, 3]

        # # if not hypersim:
        # #     alpha_padded = density_to_alpha(alpha_padded)

        # if dataset == "scannet":
        #     alpha = density_to_alpha_scannet(alpha_padded)
        # elif dataset == "front3d":
        #     alpha = density_to_alpha(alpha_padded)

        # mask = alpha_padded > 0.01
        # print("mask_patches", mask_patches.shape, mask.shape)
        mask_patches = mask_patches.reshape(-1)
        # mask = mask.reshape(-1)

        # mask = mask * ~(mask_patches.bool())

        mask_remove = mask_patches.bool()
        mask_keep = ~mask_patches.bool()

        # mask_patch_remove = pad_mask.bool() & mask_patches.bool()
        # mask_patch_keep = pad_mask.bool() & ~mask_patches.bool()

        # print("mask", mask.shape)
        # grid = construct_grid(new_res)

        # print("points", points.shape)
        # print("rgb", rgb.shape)
        # pts_remove = points.copy()
        # pts_remove = points[mask_remove, :]

        # rgb_remove = colors.copy()
        # rgb_remove = rgb_remove[mask_remove, :]

        # pcd_remove = o3d.geometry.PointCloud()
        # pcd_remove.points = o3d.utility.Vector3dVector(pts_remove)
        # pcd_remove.colors = o3d.utility.Vector3dVector(rgb_remove)
        # draw = o3d.visualization.EV.draw

        # draw({'geometry': pcd_remove, 'name':'pcd_remove' })


        print("points", points.shape)
        print("rgb", rgb.shape)
        pts_remove = points.copy()
        pts_remove = points[mask_keep, :]

        rgb_remove = colors.copy()
        rgb_remove = rgb_remove[mask_keep, :]

        pcd_remove = o3d.geometry.PointCloud()
        pcd_remove.points = o3d.utility.Vector3dVector(pts_remove)
        pcd_remove.colors = o3d.utility.Vector3dVector(rgb_remove)
        draw = o3d.visualization.EV.draw

        draw({'geometry': pcd_remove, 'name':'pcd_remove' })


        # grid = grid * mask.unsqueeze(-1).detach().cpu().numpy()

        # grid_remove = grid[mask_remove, :]
        # grid_np_remove = pcd_padded_np[mask_remove, :]

        # grid_remove = grid.numpy().copy()
        # grid_np_remove = pcd_padded_np.copy()
        # grid_np_remove[mask_remove, :] = 0

        # grid_keep = grid.numpy().copy()
        # grid_np_keep = pcd_padded_np.copy()
        # grid_np_keep[mask_keep, :] = 0

        # grid_keep = grid[mask_keep, :]
        # grid_np_keep = pcd_padded_np[mask_keep, :]

        # # grid_patch_remove = grid[mask_patch_remove, :]
        # # grid_np_patch_remove = pcd_padded_np[mask_patch_remove, :]

        # # grid_patch_keep = grid[mask_patch_keep, :]
        # # grid_np_patch_keep = pcd_padded_np[mask_patch_keep, :]

        # # mask = ~mask.bool()
        # # grid_np[mask, :] = 0
        # # draw_grid_colors(grid_original)

        # # draw_grid_colors(grid_patch_keep)
        # # draw_grid_colors(grid_patch_keep, grid_np_patch_keep)

        # # draw_grid_colors(grid_patch_remove)
        # # draw_grid_colors(grid_patch_remove, grid_np_patch_remove)

        # # draw_grid_colors(grid_remove)
        # # draw_grid_colors(grid_keep)

        # draw_grid_colors(grid_remove, grid_np_remove)
        # draw_grid_colors(grid_keep, grid_np_keep)
