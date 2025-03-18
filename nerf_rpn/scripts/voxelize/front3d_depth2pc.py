import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import h5py
import matplotlib.pyplot as plt
from typing import Union, Dict, List
import pandas as pd
from test_category_mapping import category_mapping, num_to_category, category_to_number
import sys

np.set_printoptions(threshold=sys.maxsize)


def modify_mask_img(mask_img, mapping_file, category_mapping):
    mapping_df = pd.read_csv(mapping_file)
    # Create a dictionary to map ids to names
    id_to_name = dict(zip(mapping_df["id"], mapping_df["name"]))

    # print("id_to_name: ", id_to_name)

    # unique_categories = sorted(set(category_mapping.values()))
    # unique_categories.remove('void')  # Remove 'void' from the list of unique categories
    # unique_categories = ['void'] + unique_categories  # Add 'void' back to the beginning of the list

    # category_to_number = {category: index for index, category in enumerate(unique_categories)}

    # Convert the mask_img to category names using the id_to_name dictionary
    mask_img_names = np.vectorize(id_to_name.get)(mask_img)

    # tea_table_mask = mask_img_names == 'tea table'
    # print("tea table masks",np.unique(mask_img_names[tea_table_mask]))
    # Map the category names to the new category names using category_mapping
    mask_img_names = mask_img_names.astype(object)
    for original_category, new_category in category_mapping.items():
        # if original_category == 'tea table':
        #     print("new_category", new_category)
        mask_img_names[mask_img_names == original_category] = new_category
        # if original_category == 'tea table':
        #     coffee_table_mask = mask_img_names == 'coffee_table'
        # print("coffee table masks",mask_img_names[coffee_table_mask])

    # coffee_table_mask = mask_img_names == 'coffee_table'
    # print("coffee table masks",mask_img_names[coffee_table_mask])
    print("mask_img_names", np.unique(mask_img_names))
    # Convert the category names back to IDs using the category_to_number dictionary
    modified_mask_img = np.vectorize(category_to_number.get)(mask_img_names)
    return modified_mask_img


def plot_mask_with_id_names(mask_img, id_to_name):
    """
    Plot the mask_img with relevant id names at the center of the mask regions.

    Parameters:
        mask_img (numpy.ndarray): The 2D array representing the mask image with ids of different classes.
        mapping_file (str): Path to the CSV file containing the mapping of ids to names.

    Returns:
        None (displays the plot).
    """
    # Load the mapping file

    # print("id_to_name: ", id_to_name)

    # Get unique ids in the mask_img
    ids = np.unique(mask_img)

    # Plot the mask_img with id names at the center of the mask
    plt.imshow(mask_img)

    for id_val in ids:
        # Find the coordinates where the current id appears in the mask_img
        y, x = np.where(mask_img == id_val)

        # Calculate the center of the mask region
        center_y = int(np.mean(y))
        center_x = int(np.mean(x))

        # Get the corresponding name for the current id from the mapping dictionary
        id_name = id_to_name.get(
            id_val, f"ID {id_val}"
        )  # If the id is not found in the mapping, use "ID {id_val}"

        # Display the id name at the center of the mask region
        plt.text(
            center_x,
            center_y,
            id_name,
            color="red",
            ha="center",
            va="center",
            fontsize=12,
        )

    plt.show()


def label_mapping_2D(img: np.ndarray, mapping_dict: Dict):
    """To map the labels in img following the rule in mapping_dict."""
    out_img = np.zeros_like(img)
    existing_labels = np.unique(img)
    for label in existing_labels:
        out_img[img == label] = mapping_dict[label]
    return out_img


def write_ply(pcd, path):
    # colors = np.multiply([
    #         plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    #     ], 255).astype(np.uint8)

    # colors = np.multiply([
    #         plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    #     ], 255).astype(np.uint8)

    colors = np.multiply(
        [plt.cm.get_cmap("gist_ncar", 100)((i * 7 + 5) % 100)[:3] for i in range(100)],
        255,
    ).astype(np.uint8)

    num_points = np.sum([p.shape[0] for p in pcd.values()])

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

        for obj_id, p in pcd.items():
            color = colors[obj_id]
            for num in range(p.shape[0]):
                f.write("{:.4f} ".format(p[num][0]))
                f.write("{:.4f} ".format(p[num][1]))
                f.write("{:.4f} ".format(p[num][2]))
                f.write("{:d} ".format(color[0]))
                f.write("{:d} ".format(color[1]))
                f.write("{:d}".format(color[2]))
                f.write("\n")


def depth2pc(depth_dir, mask_dir, scene_dir, points_per_obj=100000):
    scene_name = os.path.basename(scene_dir)

    with open(os.path.join(scene_dir, "train", "transforms.json"), "r") as f:
        json_file = json.load(f)

    pcd = defaultdict(list)

    depth_files = sorted(os.listdir(depth_dir), key=lambda x: int(x.split(".")[0]))
    mask_files = sorted(os.listdir(mask_dir), key=lambda x: int(x.split(".")[0]))

    assert len(depth_files) == len(mask_files), "depth and mask files are not matched"
    assert len(depth_files) == len(
        json_file["frames"]
    ), "depth and json files are not matched"

    fx, fy, cx, cy = (
        json_file["fl_x"],
        json_file["fl_y"],
        json_file["cx"],
        json_file["cy"],
    )
    for i in range(len(depth_files)):
        frame = json_file["frames"][i]

        depth_img = h5py.File(os.path.join(depth_dir, depth_files[i]), "r")
        depth_img = np.array(depth_img["depth"][:])

        mask_img = h5py.File(os.path.join(mask_dir, mask_files[i]), "r")
        # print("mask_img: ", mask_img.keys())
        # mask_img = np.array(mask_img['cp_instance_id_segmaps'][:])
        mask_img = np.array(mask_img["class_segmaps"][:])

        mapping_file = (
            "NeRF_MAE_internal/nerf_rpn/scripts/voxelize/3D_front_mapping.csv"
        )
        mask_img = modify_mask_img(
            mask_img, mapping_file=mapping_file, category_mapping=category_mapping
        )

        # plt.imshow(mask_img)
        # plt.show()

        # mapping_df = pd.read_csv(mapping_file)
        # # Create a dictionary to map ids to names
        # id_to_name = dict(zip(mapping_df['id'], mapping_df['name']))

        # plot_mask_with_id_names(mask_img, id_to_name=num_to_category)
        # plt.imshow(mask_img)
        # plt.show()

        H, W = mask_img.shape
        assert (
            H == depth_img.shape[0] and W == depth_img.shape[1]
        ), "depth and mask shapes not matched"

        x = np.linspace(0, H - 1, H, endpoint=True)
        y = np.linspace(0, W - 1, W, endpoint=True)
        j, i = np.meshgrid(x, y, indexing="ij")

        c_x = (i + 0.5 - cx) / fx * depth_img
        c_y = (H - j - 0.5 - cy) / fy * depth_img
        c_z = -depth_img
        c_coord = np.stack([c_x, c_y, c_z], axis=-1)

        c2w = np.array(frame["transform_matrix"])

        c_coord = c_coord.reshape([-1, 3])
        w_coord = c2w[:3, :3] @ c_coord.T + c2w[:3, 3][:, np.newaxis]
        w_coord = w_coord.T
        valid_depth = (depth_img.reshape(-1) > 0) & (depth_img.reshape(-1) < 15)

        ids = np.unique(mask_img)
        # print("ids: ", ids)
        for id in ids:
            # if id == 0:
            #     continue
            mask = mask_img == id
            mask = mask.reshape(-1)
            mask = mask & valid_depth
            pcd[id].append(w_coord[mask, :])

    for id in pcd.keys():
        pcd[id] = np.concatenate(pcd[id], axis=0)
        if pcd[id].shape[0] > points_per_obj:
            pcd[id] = pcd[id][
                np.random.choice(pcd[id].shape[0], points_per_obj, replace=False), :
            ]

    return pcd


def write_npz(pcd, path):
    ids = []
    points = []
    for id, p in pcd.items():
        ids.extend([id] * p.shape[0])
        points.append(p)

    points = np.concatenate(points, axis=0)
    ids = np.array(ids)

    np.savez(path, points=points, ids=ids)


if __name__ == "__main__":
    # dir = 'Downloads/FRONT3D_render_seg_2'
    dir = "Downloads/FRONT3D_render_seg_all"
    scenes = os.listdir("Downloads/front3d_nerf_data")
    all_depth_folders = os.listdir(dir)
    for s in tqdm(scenes):
        if s not in all_depth_folders:
            continue
        depth_dir = os.path.join(dir, s, "depth")
        mask_dir = os.path.join(dir, s, "segmaps")
        out_dir = os.path.join(dir, s, "pcd")
        os.makedirs(out_dir, exist_ok=True)

        pcd = depth2pc(
            os.path.join(depth_dir),
            os.path.join(mask_dir),
            os.path.join("Downloads/front3d_nerf_data", s),
        )
        write_npz(pcd, os.path.join(out_dir, s + ".npz"))
        write_ply(pcd, os.path.join(out_dir, s + ".ply"))
