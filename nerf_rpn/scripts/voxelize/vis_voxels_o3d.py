import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch

sys.path.append("NeRF_MAE_internal/nerf_mae")

from model.mae.torch_utils import *
from model.mae.viz_utils import *

rgb_colors = [
    (38, 70, 83),    # Dark Slate Blue
    (42, 157, 143),  # Jungle Green
    (233, 196, 106), # Gold
    (244, 162, 97),  # Tangerine
    (231, 111, 81),  # Terra Cotta
    (173, 216, 230), # Light Blue
    (77, 121, 255),  # Royal Blue
    (82, 82, 122),   # Raisin Black
    (147, 204, 234), # Light Steel Blue
    (105, 142, 121), # Sage Green
    (61, 68, 109),   # Independence
    (163, 120, 103), # Ferra
    (224, 153, 150), # Cameo
    (158, 197, 68),  # Granny Smith Apple
    (66, 115, 119),  # Topaz
    (132, 168, 182), # Blue Chalk
    (140, 129, 136), # Mamba
    (119, 139, 165), # Light Steel Blue
    (199, 67, 117),  # Fandango
    (190, 143, 143)  # Light Coral
]



def write_ply(voxel, path, colors):
    # colors = np.multiply([
    #         plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    #     ], 255).astype(np.uint8)

    # I want to get the same
    # colors = np.multiply(
    #     [plt.cm.get_cmap("gist_ncar", 41)((i * 7 + 5) % 41)[:3] for i in range(41)], 255
    # ).astype(np.uint8)

    num_points = np.sum(voxel != 0)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        for i in range(voxel.shape[0]):
            for j in range(voxel.shape[1]):
                for k in range(voxel.shape[2]):
                    if voxel[i, j, k] != 0:
                        color = colors[voxel[i, j, k]]
                        f.write("{:.4f} ".format(i))
                        f.write("{:.4f} ".format(j))
                        f.write("{:.4f} ".format(k))
                        f.write("{:d} ".format(color[0]))
                        f.write("{:d} ".format(color[1]))
                        f.write("{:d}".format(color[2]))
                        f.write("\n")

scene_name = '3dfront_0037_00'
# folder_dir = 'Downloads/ckpts_nerf_mae/front3d_voxelSem_3.5k_0.75_aug_loss_mask_skip_normden/voxel_outputs'

folder_dir = 'Downloads/ckpts_nerf_mae/front3d_voxelSem_3.5k_0.75_aug_loss_mask_skip_normden_NOPT/voxel_outputs'
os.makedirs(os.path.join(folder_dir,scene_name),exist_ok=True)
voxel_feat_path = os.path.join(folder_dir, scene_name+'.npy')

voxel_gt_path = os.path.join(folder_dir, scene_name+'_gt.npy')


feat = np.load(voxel_feat_path)
feat_gt = np.load(voxel_gt_path)

print("unique labels in feat_gt", np.unique(feat_gt))
print("unique labels in feat", np.unique(feat))


nerf_rpn_masks = 'Downloads/masks'
nerf_rpn_path = os.path.join(nerf_rpn_masks, scene_name+'.npy')

nerf_rpn_mask = np.load(nerf_rpn_path)

print("np.unique(nerf_rpn_mask)", np.unique(nerf_rpn_mask))
write_ply(nerf_rpn_mask, os.path.join(folder_dir, scene_name + "_nerf_rpn_mask.ply"), rgb_colors)
# unique_labels = np.unique(feat)
# print(unique_labels)

mesh =  o3d.geometry.TriangleMesh.create_coordinate_frame(size = 10.0)

nerf_rpn_ply_path = os.path.join(folder_dir, scene_name + "_nerf_rpn_mask.ply")
pcd_nerf_rpn = o3d.io.read_point_cloud(nerf_rpn_ply_path)
o3d.visualization.draw_geometries([pcd_nerf_rpn, mesh])


# mask out index >0

# mask = feat >0

#Celing id is 12 mask it out 

mask = feat_gt ==12

feat[mask] = 0

mask_gt = feat_gt ==12
feat_gt[mask_gt] = 0

# feat = feat*mask

#what are the unique labels in voxels


out_dir = os.path.join(folder_dir, scene_name)

write_ply(feat, os.path.join(out_dir, scene_name + ".ply"), rgb_colors)

write_ply(feat_gt, os.path.join(out_dir, scene_name + "_gt.ply"), rgb_colors)



voxel_ply_path = os.path.join(out_dir, scene_name + ".ply")
pcd = o3d.io.read_point_cloud(voxel_ply_path)

voxel_gt_path = os.path.join(out_dir, scene_name + "_gt.ply")
pcd_gt = o3d.io.read_point_cloud(voxel_gt_path)

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# viz_mask = generate_point_cloud_mask(points)
# viz_mask = viz_mask.reshape(-1)

# grid_inter = points[viz_mask, :]
# pcd_padded_np_inter = colors[viz_mask, :]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(grid_inter)
# pcd.colors = o3d.utility.Vector3dVector(pcd_padded_np_inter)



# pcds.append(mesh)
    
o3d.visualization.draw_geometries([pcd_gt, mesh])
o3d.visualization.draw_geometries([pcd, mesh])


input_feat_dir = 'Downloads/front3d_rpn_data/features'
resolution = 160
new_res = np.array([resolution, resolution, resolution])
features_path = os.path.join(input_feat_dir, scene_name + ".npz")
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

# if dataset == "scannet":
#     alpha = density_to_alpha_scannet(alpha)
# elif dataset == "front3d":
alpha = density_to_alpha(alpha)

print("rgb max min", np.max(rgbsigma[..., :3]), np.min(rgbsigma[..., :3]))
print("alpha max min", np.max(alpha), np.min(alpha))

mask = alpha > 0.01
# mask = mask.reshape(-1)
# grid_original = grid_original[mask, :]
# grid_np = rgbsigma[:, :3][mask, :]

print("mask", mask.shape)

#mask has dimension mask (2031360,) in numpy i want to make in mask (2031360,3) by repeating

mask = np.tile(mask[:, np.newaxis], 3)


#I am getting
#mask = mask[:, None].repeat(1, 3)
#numpy.AxisError: axis 3 is out of bounds for array of dimension 2

print("grid_original", grid_original.shape, rgbsigma.shape)
grid_original = grid_original * mask
grid_np = rgbsigma[:, :3] * mask

print("grid_original", grid_original.shape, grid_np.shape)


viz_mask = generate_point_cloud_mask(grid_original)
viz_mask = viz_mask.reshape(-1)

grid_original = grid_original[viz_mask, :]
grid_np = grid_np[viz_mask, :]

draw_grid_colors(grid_original, grid_np)

# pcd_pts = torch.from_numpy(rgbsigma_original).permute(3, 0, 1, 2)

# # new_res = np.array([200, 200, 200])
# pcd_pts_padded, pad_mask = pad_tensor(pcd_pts, target_shape=new_res, pad_value=0)
# print("non zero pad mask", torch.count_nonzero(pad_mask))
# # print("pad_mask", pad_mask.shape)
# pcd_padded_np = pcd_pts_padded.reshape(4, -1).T[:, :3].numpy()

# print("pad_mask", pad_mask.shape)
# pad_mask = pad_mask.permute(0, 2, 3, 4, 1)[:, :, :, :, 0]
# print("pad_mask", pad_mask.shape)


# print("pad_mask", pad_mask.shape, mask_gt.shape)

# print("pad_mask", pad_mask, mask_gt)


# #convert mask_gt to tensor and from bool to float32

# mask_gt = torch.from_numpy(mask_gt).float()
# print("pad mask gt mask dtype", pad_mask.dtype, mask_gt.dtype)

# #Now combine pad_mask and mask_gt
# pad_mask = pad_mask*(1-mask_gt)

# alpha_padded = pcd_pts_padded.reshape(4, -1).T[:, 3]

# # pad_mask 

# mask = alpha_padded > 0.01
# pad_mask = pad_mask.reshape(-1).bool()
# grid = construct_grid(new_res)

# #combine pad mask and mask_gt


# pad_mask = pad_mask*mask
# print("pad_mask", pad_mask.shape, grid.shape, pcd_padded_np.shape)
# grid_inter = grid[pad_mask, :]
# pcd_padded_np_inter = pcd_padded_np[pad_mask, :]


# print("grid_inter", grid_inter.shape, pcd_padded_np_inter.shape)

# #mask the celing of this grid based on mask_gt

# # mask_gt = mask_gt.reshape(-1)
# # grid_inter = grid_inter[mask_gt, :]
# # pcd_padded_np_inter = pcd_padded_np_inter[mask_gt, :]


# viz_mask = generate_point_cloud_mask(grid_inter)
# viz_mask = viz_mask.reshape(-1)

# grid_inter = grid_inter[viz_mask, :]
# pcd_padded_np_inter = pcd_padded_np_inter[viz_mask, :]

# draw_grid_colors(grid_inter, pcd_padded_np_inter, coordinate_system=True)
