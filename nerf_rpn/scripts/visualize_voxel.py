import numpy as np
import os
import open3d as o3d
import sys
sys.path.append("/home/zubairirshad/NeRF_MAE_internal/nerf_mae")
from model.mae.viz_utils import *
from model.mae.torch_utils import *

voxel_folder = '/home/zubairirshad/Downloads/FRONT3D_render_seg/voxel'
voxel_files = os.listdir(voxel_folder)


import matplotlib.pyplot as plt


def write_ply(voxel, path):
    colors = np.multiply([
            plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
        ], 255).astype(np.uint8)

    num_points = np.sum(voxel != 0)

    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {num_points}\n'
                'property float x\n'
                'property float y\n'
                'property float z\n' 
                'property uchar red\n'
                'property uchar green\n' 
                'property uchar blue\n'
                'end_header\n')
        
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
                        f.write("{:d}". format(color[2]))
                        f.write("\n")

for voxel_file in voxel_files:
    voxel_path = os.path.join(voxel_folder, voxel_file)
    voxel = np.load(voxel_path)

    write_ply(voxel, voxel_path.replace('.npy', '.ply'))