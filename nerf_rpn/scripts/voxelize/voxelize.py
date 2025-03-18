import numpy as np
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


def write_ply(voxel, path):
    # colors = np.multiply([
    #         plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    #     ], 255).astype(np.uint8)

    colors = np.multiply(
        [plt.cm.get_cmap("gist_ncar", 41)((i * 7 + 5) % 41)[:3] for i in range(41)], 255
    ).astype(np.uint8)

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


def voxelize(points, ids, room_bbox, width, length, height):
    max_id = np.max(ids)
    bin = np.zeros((width, length, height, max_id + 1), dtype=np.int32)
    bbox_min, bbox_max = room_bbox[0], room_bbox[1]

    x = np.clip(
        (points[:, 0] - bbox_min[0]) / (bbox_max[0] - bbox_min[0]) * width, 0, width - 1
    ).astype(np.int32)
    y = np.clip(
        (points[:, 1] - bbox_min[1]) / (bbox_max[1] - bbox_min[1]) * length,
        0,
        length - 1,
    ).astype(np.int32)
    z = np.clip(
        (points[:, 2] - bbox_min[2]) / (bbox_max[2] - bbox_min[2]) * height,
        0,
        height - 1,
    ).astype(np.int32)

    for i in range(points.shape[0]):
        bin[x[i], y[i], z[i], ids[i]] += 1

    voxel = np.argmax(bin, axis=-1)

    return voxel


if __name__ == "__main__":
    dir = "Downloads/FRONT3D_render_seg_all"

    scenes = os.listdir(dir)

    for s in tqdm(scenes):
        if s == "00179-MVVzj944atG_3":
            continue
        pcd_dir = os.path.join(dir, s, "pcd")
        feat_dir = "Downloads/front3d_rpn_data/features"
        xform_dir = "Downloads/front3d_nerf_data"
        out_dir = os.path.join(dir, s, "voxel")
        os.makedirs(out_dir, exist_ok=True)

        scene_name = s.split(".")[0]
        pcd = np.load(os.path.join(pcd_dir, s + ".npz"))
        points = pcd["points"]
        ids = pcd["ids"]

        feat = np.load(os.path.join(feat_dir, scene_name + ".npz"))
        res = feat["resolution"]
        # res = res[[2, 0, 1]]
        # print("res", res)
        width, length, height = res

        # height, length, width = res

        with open(
            os.path.join(xform_dir, scene_name, "train", "transforms.json"), "r"
        ) as f:
            data = json.load(f)
            room_bbox = np.array(data["room_bbox"])

        voxel = voxelize(points, ids, room_bbox, width, length, height)

        print("voxel shape:", voxel.shape)
        print("voxel unique:", np.unique(voxel))
        print("voxel max:", np.max(voxel))
        print("voxel", voxel)
        np.save(os.path.join(out_dir, scene_name + ".npy"), voxel)
        write_ply(voxel, os.path.join(out_dir, scene_name + ".ply"))
