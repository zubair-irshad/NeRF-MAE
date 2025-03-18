import numpy as np
import os
import open3d as o3d
import sys
sys.path.append("NeRF_MAE_internal/nerf_mae")
from model.mae.viz_utils import *
from model.mae.torch_utils import *
import matplotlib.pyplot as plt

pcd_folder = 'Downloads/FRONT3D_render_seg/pcd'
pcd_files = os.listdir(pcd_folder)

def write_ply(pcd, path):
    colors = np.multiply([
            plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
        ], 255).astype(np.uint8)

    num_points = np.sum([p.shape[0] for p in pcd.values()])

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
        
        for obj_id, p in pcd.items():
            print("obj_id", obj_id).shape
            print("p", p.shape)
            color = colors[obj_id]
            for num in range(p.shape[0]):
                f.write("{:.4f} ".format(p[num][0]))
                f.write("{:.4f} ".format(p[num][1]))
                f.write("{:.4f} ".format(p[num][2]))
                f.write("{:d} ".format(color[0]))
                f.write("{:d} ".format(color[1]))
                f.write("{:d}". format(color[2]))
                f.write("\n")

for pcd_file in pcd_files:
    pcd_path = os.path.join(pcd_folder, pcd_file)
    pcd = np.load(pcd_path)

    pcd = {'points': pcd['points'], 'ids': pcd['ids']}
    # print("pcd", pcd.files)
    # print("np array pcd", np.array(pcd).shape)

    write_ply(pcd, pcd_path.replace('.npz', '.ply'))