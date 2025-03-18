import os
import argparse
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import zoom

import json
import open3d as o3d
import torch
import sys

sys.path.append("NeRF_MAE_internal/nerf_mae")

from model.mae.viz_utils import *

def nerf_matrix_to_ngp(nerf_matrix, scale, offset, from_mitsuba):
    '''
    Converts a nerf matrix to an ngp matrix. Follows the implementation in
    nerf_loader.h
    '''
    result = nerf_matrix.copy()
    result[:, [1, 2]] *= -1
    result[:, 3] = result[:, 3] * scale + offset

    if from_mitsuba:
        result[:, [0, 2]] *= -1
    else:
        # Cycle axes xyz<-yzx
        result = result[[1, 2, 0], :]

    return result

def transform_to_ngp_bbox(bbox_raw, scale, offset, from_mitsuba):
    """ 
        Transform a bounding box from the raw dataset to the ngp coordinate. (for 3dfront)
        Input:
            bbox_raw: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            dataset: ngp.Dataset
        Return:
            min_pt: [min_x, min_y, min_z]
            max_pt: [max_x, max_y, max_z]
    """
    bbox_raw = np.array(bbox_raw)
    min_pt_raw, max_pt_raw = bbox_raw[0], bbox_raw[1]
    extent = max_pt_raw - min_pt_raw
    position = (min_pt_raw+max_pt_raw)/2

    xform = np.hstack([np.eye(3, 3), np.expand_dims(position, 1)])
    # xform = dataset.nerf_matrix_to_ngp(xform)

    print("xform_before", xform)
    xform = nerf_matrix_to_ngp(xform, scale, offset, from_mitsuba)

    print("xform_after", xform)
    # extent *= dataset.scale
    extent *= scale
    min_pt, max_pt = get_ngp_obj_bounding_box(xform, extent)

    return min_pt, max_pt

def get_ngp_obj_bounding_box(xform, extent):
    corners = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, 1],
    ], dtype=float).T

    corners *= np.expand_dims(extent, 1) * 0.5
    corners = xform[:, :3] @ corners + xform[:, 3, None]
    return np.min(corners, axis=1), np.max(corners, axis=1)


def filter_pointcloud(pointcloud, colors, bounding_box):
    min_bound, max_bound = bounding_box
    filtered_points = []
    filtered_colors = []

    for point, color in zip(pointcloud, colors):
        if (
            (min_bound[0] <= point[0] <= max_bound[0])
            and (min_bound[1] <= point[1] <= max_bound[1])
            and (min_bound[2] <= point[2] <= max_bound[2])
        ):
            filtered_points.append(point)
            filtered_colors.append(color)

    return np.array(filtered_points), np.array(filtered_colors)


def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def depth_nerf_density_to_alpha(density):
    activation = np.clip(density, a_min=0, a_max=None)  # original nerf uses relu
    return np.clip(1.0 - np.exp(-activation / 100.0), 0.0, 1.0)


def construct_grid(res):
    res_x, res_y, res_z = res
    x = torch.linspace(0, res_x, res_x)
    y = torch.linspace(0, res_y, res_y)
    z = torch.linspace(0, res_z, res_z)

    scale = torch.tensor(res).max()
    x /= scale
    y /= scale
    z /= scale

    # Shift by 0.5 voxel
    x += 0.5 * (1.0 / scale)
    y += 0.5 * (1.0 / scale)
    z += 0.5 * (1.0 / scale)

    grid = []
    for i in range(res_z):
        for j in range(res_y):
            for k in range(res_x):
                grid.append([x[k], y[j], z[i]])

    return torch.tensor(grid)


# def construct_grid(res):
#     res_x, res_y, res_z = res
#     x = np.linspace(0, res_x, res_x)
#     y = np.linspace(0, res_y, res_y)
#     z = np.linspace(0, res_z, res_z)

#     scale = res.max()
#     x /= scale
#     y /= scale
#     z /= scale

#     # Shift by 0.5 voxel
#     x += 0.5 * (1.0 / scale)
#     y += 0.5 * (1.0 / scale)
#     z += 0.5 * (1.0 / scale)

#     grid = []
#     for i in range(res_z):
#         for j in range(res_y):
#             for k in range(res_x):
#                 grid.append([x[k], y[j], z[i]])

#     return np.array(grid)


def write_box_vertex_to_ply(f, box):
    f.write(f"{box[0]} {box[1]} {box[2]} 255 255 255\n")
    f.write(f"{box[0]} {box[4]} {box[2]} 255 255 255\n")
    f.write(f"{box[3]} {box[4]} {box[2]} 255 255 255\n")
    f.write(f"{box[3]} {box[1]} {box[2]} 255 255 255\n")
    f.write(f"{box[0]} {box[1]} {box[5]} 255 255 255\n")
    f.write(f"{box[0]} {box[4]} {box[5]} 255 255 255\n")
    f.write(f"{box[3]} {box[4]} {box[5]} 255 255 255\n")
    f.write(f"{box[3]} {box[1]} {box[5]} 255 255 255\n")


def get_obb_corners(xform, extent):
    corners = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, -1],
            [-1, -1, 1],
        ],
        dtype=float,
    ).T

    corners *= np.expand_dims(extent, 1) * 0.5
    corners = xform[:, :3] @ corners + xform[:, 3, None]

    return corners


def write_obb_vertex_to_ply(f, obb, needs_y_up=False):
    rot = obb[-1]
    xform = np.array(
        [
            [np.cos(rot), -np.sin(rot), 0, obb[0]],
            [np.sin(rot), np.cos(rot), 0, obb[1]],
            [0, 0, 1, obb[2]],
        ]
    )

    corners = get_obb_corners(xform, obb[3:6])

    if needs_y_up:
        # From z up to y up
        perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T

        corners = perm @ corners

    for i in range(8):
        f.write(
            f"{corners[0][i]:4f} {corners[1][i]:4f} {corners[2][i]:4f} 255 255 255\n"
        )

def plot_obb_vertex_open3d(all_lines, obb, needs_y_up=False):
    rot = obb[-1]
    xform = np.array(
        [
            [np.cos(rot), -np.sin(rot), 0, obb[0]],
            [np.sin(rot), np.cos(rot), 0, obb[1]],
            [0, 0, 1, obb[2]],
        ]
    )

    corners = get_obb_corners(xform, obb[3:6])

    if needs_y_up:
        # From z up to y up
        perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T

        corners = perm @ corners

    points = np.transpose(corners)



    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ], dtype=np.int32)  # Define the line segments based on corner indices
    colors = np.tile([1, 0, 0], (len(lines), 1))  # Red color for all lines

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    all_lines.append(line_set)

    return all_lines

def obb_vertex_json(obb, needs_y_up=False):
    rot = obb[-1]
    xform = np.array(
        [
            [np.cos(rot), -np.sin(rot), 0, obb[0]],
            [np.sin(rot), np.cos(rot), 0, obb[1]],
            [0, 0, 1, obb[2]],
        ]
    )

    corners = get_obb_corners(xform, obb[3:6])

    if needs_y_up:
        # From z up to y up
        perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T

        corners = perm @ corners

    vertices = []
    for i in range(8):
        vertices.append((corners[0][i], corners[1][i], corners[2][i]))
        # f.write(
        #     f"{corners[0][i]:4f} {corners[1][i]:4f} {corners[2][i]:4f} 255 255 255\n"
        # )
    return vertices


def write_box_edge_to_ply(f, idx):
    for i in range(3):
        f.write(f"{idx + i} {idx + i + 1}\n")
        f.write(f"{idx + i + 4} {idx + i + 5}\n")
        f.write(f"{idx + i} {idx + i + 4}\n")

    f.write(f"{idx} {idx + 3}\n")
    f.write(f"{idx + 4} {idx + 7}\n")
    f.write(f"{idx + 3} {idx + 7}\n")


def write_alpha_grid_to_ply(f, alpha, grid, threshold=0):
    for i in range(len(grid)):
        if alpha[i] > threshold:
            f.write(
                f"{grid[i][0]:4f} {grid[i][1]:4f} {grid[i][2]:4f} {alpha[i]} {alpha[i]} {alpha[i]}\n"
            )


def write_colormapped_alpha_grid_to_ply(f, alpha, grid, threshold=0):
    """
    alpha expected to be in 0-1 range
    """
    colormap = cm.get_cmap("plasma")
    rgb = (colormap(alpha) * 255).astype(np.uint8)
    for i in range(len(grid)):
        if alpha[i] > threshold:
            f.write(
                f"{grid[i][0]:4f} {grid[i][1]:4f} {grid[i][2]:4f} {rgb[i][0]} {rgb[i][1]} {rgb[i][2]}\n"
            )


def write_objectness_heatmap_to_ply(f, alpha, score, grid, threshold=0):
    """
    alpha expected to be in 0-1 range
    """
    colormap = cm.get_cmap("turbo")
    rgb = (colormap(score) * 255).astype(np.uint8)
    for i in range(len(grid)):
        if alpha[i] > threshold:
            f.write(
                f"{grid[i][0]:4f} {grid[i][1]:4f} {grid[i][2]:4f} {rgb[i][0]} {rgb[i][1]} {rgb[i][2]}\n"
            )


def write_rgb_to_ply(f, rgb, alpha, grid, threshold=0):
    rgb = (rgb * 255).astype(np.uint8)
    for i in range(len(grid)):
        if alpha[i] > threshold:
            f.write(
                f"{grid[i][0]:4f} {grid[i][1]:4f} {grid[i][2]:4f} {rgb[i][0]} {rgb[i][1]} {rgb[i][2]}\n"
            )


def get_objectness_grid(data, res):
    acc = np.zeros(res)
    for level in ["0", "1", "2", "3"]:
        score = data[level][0]
        score = zoom(score, res / np.array(score.shape), order=3)
        # score = np.clip(score, 0, None)
        # score = np.sqrt(score)  # sqrt to make it more visible
        acc += score

    acc = np.transpose(acc, (2, 1, 0)).reshape(-1)
    acc /= acc.max()
    return acc


def get_num_points_in_box(rgbsigma, alpha_threshold):
    rgbsigma = np.transpose(rgbsigma, (2, 1, 0, 3))
    alpha = rgbsigma[..., -1]
    alpha = density_to_alpha(alpha)
    print("alpha", alpha.shape)
    print("alpha[0, :]", alpha[0, ...].shape)

    num_grid_points = np.nonzero(alpha > alpha_threshold)

    print(
        "num_grid_points",
        num_grid_points[0].shape,
        num_grid_points[1].shape,
        num_grid_points[2].shape,
    )

    print("len(num_grid_points)", len(num_grid_points))

    print(num_grid_points.shape)
    num_grid_points = [
        (alpha[0, :] > alpha_threshold).sum(),
        alpha[1, :] > alpha_threshold.sum(),
        alpha[2, :] > alpha_threshold.sum(),
    ]
    print("num_grod_points", num_grid_points)

    return num_grid_points


def visualize_scene(
    scene_name,
    output_dir,
    feature_dir,
    box_dir=None,
    box_format="obb",
    objectness_dir=None,
    alpha_threshold=0.01,
    transpose_yz=False,
    write_all=True,
    dataset="scannet",
    json_dict = None
):
    
    # room_box_original = json_dict["room_bbox"]
    # print("room_box_original", room_box_original)
    boxes = None
    if box_dir is not None:
        boxes = np.load(os.path.join(box_dir, scene_name + ".npy"), allow_pickle=True)
    feature = np.load(os.path.join(feature_dir, scene_name + ".npz"), allow_pickle=True)

    print("feature", feature["resolution"])

    res = feature["resolution"]
    rgbsigma = feature["rgbsigma"]
    scale_features = feature["scale"]
    offset = feature["offset"]
    from_mitsuba = feature["from_mitsuba"]

    print("res", res)
    print("rgbsigma original", rgbsigma.shape)

    # Current ones
    # First reshape from (H * W * D, C) to (D, H, W, C)
    # rgbsigma = rgbsigma.reshape(res[2], res[1], res[0], -1)
    # rgbsigma = np.transpose(rgbsigma, (2, 1, 0, 3)) # to (W, L, H, 4)

    print("rgbsigma reshape", rgbsigma.shape)

    # classic
    rgbsigma_original = rgbsigma.copy()
    rgbsigma = np.transpose(rgbsigma, (2, 1, 0, 3)).reshape(-1, 4)
    # rgbsimga = rgbsigma.reshape(-1, 4)

    print("res", res)
    print("rgb_sigma", rgbsigma.shape)
    # num_grid_points = get_num_points_in_box(rgbsigma, alpha_threshold)

    # transpose_yz = True
    # if transpose_yz:
    #     rgbsigma = rgbsigma.reshape((res[2], res[1], res[0], -1))
    #     rgbsigma = np.transpose(rgbsigma, (1, 2, 0, 3))
    #     rgbsigma = rgbsigma.reshape((res[0] * res[1] * res[2], -1))
    #     res = res[[2, 0, 1]]

    rgbsigma_original = rgbsigma.copy()

    # rgbsigma = np.transpose(rgbsigma, (2, 1, 0, 3)).reshape(-1, 4)
    scene_box = np.concatenate((np.zeros(3), res))
    scale = 1.0 / res.max()

    alpha = rgbsigma[:, -1]
    print("alpha max min before", np.max(alpha), np.min(alpha))
    # alpha = depth_nerf_density_to_alpha(alpha)

    if dataset == "scannet":
        alpha = depth_nerf_density_to_alpha(alpha)
    elif dataset == "front3d" or dataset =="arkitscenes" or dataset=="hm3d" or dataset =="rlbench":
        alpha = density_to_alpha(alpha)
    # alpha = (alpha * 255).astype(np.uint8)

    print("rgb max min", np.max(rgbsigma[..., :3]), np.min(rgbsigma[..., :3]))
    print("alpha max min", np.max(alpha), np.min(alpha))

    if write_all:
        num_grid_points = alpha.shape[0]

    else:
        num_grid_points = (alpha > alpha_threshold).sum()

    if objectness_dir is not None:
        score = np.load(
            os.path.join(objectness_dir, scene_name + "_objectness.npz"),
            allow_pickle=True,
        )
        score = get_objectness_grid(score, res)

    grid = construct_grid(res)

    mask = alpha > 0.01
    grid = grid[mask, :]
    rgbsigma = rgbsigma[mask, :]

    # print("grid", grid.shape)
    # viz_mask = generate_point_cloud_mask(grid)

    # grid = grid[viz_mask, :]
    # rgbsigma = rgbsigma[viz_mask, :]

    pcds = []


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    pcd.colors = o3d.utility.Vector3dVector(rgbsigma[:, :3])

    draw = o3d.visualization.EV.draw

    name = 'pcd'+str(1)
    draw({'geometry': pcd, 'name':name })

    # o3d.visualization.draw_geometries([pcd, mesh])

    grid_pts = o3d.utility.Vector3dVector(grid)
    o3d_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        grid_pts
    ).get_box_points()

    print("o3d_bbox", np.array(o3d_bbox))

    # bbox_min = feature["bbox_min"]
    # bbox_max = feature["bbox_max"]

    # print("bbox_min", bbox_min)
    # print("bbox_max", bbox_max)
    # print("====================================\n")

    # print("scale_features, offset, from_mitsuba", scale_features, offset, from_mitsuba)

    # min_transformed, max_transfored = transform_to_ngp_bbox(room_box_original, scale_features, offset, from_mitsuba)

    # print("min_transformed", min_transformed)
    # print("max_transfored", max_transfored)

    # # room_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)

    # room_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_transformed, max_bound=max_transfored)
    # o3d_bbox = room_aabb.get_box_points()

    # print("bbox_min: ", bbox_min)
    # print("bbox_max: ", bbox_max)

    # cylinder_segments = line_set(np.array(o3d_bbox))
    # for i in range(len(cylinder_segments)):
    #     pcds.append(cylinder_segments[i])
    
    # coords = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # # pcds.append(coords)
    # # unitcube = unit_cube()
    # pcds.append(pcd)
    # # pcds.append(mesh)
    # # o3d.visualization.draw_geometries([pcd, mesh, *unitcube])
    # if boxes is not None:
    #     lines = []
    #     for i in range(boxes.shape[0]):
    #         if box_format == "obb":
    #             box = boxes[i]
    #             # print("box", box)
    #             box[:6] *= scale
    #             lines = plot_obb_vertex_open3d(lines, box)
    #     geometries = pcds + lines

    # else:
    #     geometries = pcds
    

    o3d.visualization.draw_geometries(geometries)

    obb = o3d.geometry.OrientedBoundingBox.create_from_points(vertices)

    with open(os.path.join(output_dir, scene_name + ".ply"), "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        if write_all:
            f.write(
                f"element vertex {num_grid_points}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n\n"
            )

        else:
            if boxes is not None:
                f.write(
                    f"element vertex {8 * boxes.shape[0] + 8 + num_grid_points}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    f"element edge {12 * boxes.shape[0] + 12}\n"
                    "property int vertex1\n"
                    "property int vertex2\n"
                    "end_header\n\n"
                )
            else:
                f.write(
                    f"element vertex {num_grid_points}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    "end_header\n\n"
                )

        all_vertices = {}
        if not write_all:
            if boxes is not None:
                write_box_vertex_to_ply(f, scene_box * scale)
                for i in range(boxes.shape[0]):
                    if box_format == "obb":
                        box = boxes[i]
                        box[:6] *= scale
                        write_obb_vertex_to_ply(f, box)
                        vertices = obb_vertex_json(box)
                        all_vertices[i] = []
                        all_vertices[i] = vertices
                    else:
                        write_box_vertex_to_ply(f, boxes[i] * scale)

        # write_alpha_grid_to_ply(f, alpha, grid, threshold=alpha_threshold)

        a = res
        np.savetxt(
            os.path.join(output_dir, scene_name + "_dimensions.txt"), a
        )  # X is an array

        with open(os.path.join(output_dir, scene_name + "_vertices.json"), "w") as fp:
            json.dump(all_vertices, fp)

        if objectness_dir is None:
            # write_colormapped_alpha_grid_to_ply(f, alpha, grid, threshold=alpha_threshold)
            # mask = alpha > 0.01
            # grid = grid[mask, :]
            # rgbsigma = rgbsigma[mask, :]
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(grid)

            # pcd.colors = o3d.utility.Vector3dVector(rgbsigma[:, :3])
            # o3d.visualization.draw_geometries([pcd])
            write_rgb_to_ply(f, rgbsigma[:, :3], alpha, grid, threshold=alpha_threshold)
        else:
            write_objectness_heatmap_to_ply(
                f, alpha, score, grid, threshold=alpha_threshold
            )

        if boxes is not None:
            f.write("\n")
            write_box_edge_to_ply(f, 0)
            for i in range(boxes.shape[0]):
                write_box_edge_to_ply(f, (i + 1) * 8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ply files of NeRF RPN input features and boxes for visualization."
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Path to the directory to save the ply files.",
    )
    parser.add_argument(
        "--feature_dir",
        "-f",
        type=str,
        required=True,
        help="Path to the directory containing the NeRF RPN input features.",
    )

    # parser.add_argument(
    #     "--dataset_dir",
    #     "-d",
    #     type=str,
    #     help="Path to the directory containing the NeRF RPN transforms with original room boxes.",
    # )

    parser.add_argument(
        "--box_dir",
        "-b",
        type=str,
        default=None,
        help="Path to the directory containing the boxes.",
    )
    parser.add_argument(
        "--box_format",
        "-bf",
        type=str,
        default="obb",
        help='Format of the boxes. Can be either "obb" or "aabb".',
    )
    parser.add_argument(
        "--objectness_dir",
        type=str,
        default=None,
        help="Path to the directory containing the objectness scores.",
    )
    parser.add_argument(
        "--alpha_threshold", type=float, default=0.01, help="Threshold for alpha."
    )
    parser.add_argument(
        "--transpose_yz",
        "-tr",
        action="store_true",
        help="Whether to transpose the y and z axes.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    npz_files = os.listdir(args.feature_dir)
    npy_files = os.listdir(args.box_dir) if args.box_dir is not None else None

    npz_files = [
        f
        for f in npz_files
        if f.endswith(".npz") and os.path.isfile(os.path.join(args.feature_dir, f))
    ]

    scenes = [f.split(".")[0] for f in npz_files]

    os.makedirs(args.output_dir, exist_ok=True)

    print("npz_files", npz_files)

    dataset = "rlbench"

    # dataset = "front3d"
    for scene in scenes:
        print("scene", scene)

        # if scene != "scene0000_00":
        #     continue

        # if scene != "00009-vLpv2VX547B_0":
        #     continue


        #OPTIONAL for this script
        # json_path = os.path.join(
        #     args.dataset_dir, scene, "train", "transforms.json"
        # )
        # with open(json_path, "r") as f:
        #     json_dict = json.load(f)

        # if scene != "3dfront_0117_00":
        # if scene != "00009-vLpv2VX547B_0":
        # if scene != "00135-HeSYRw7eMtG_9":
        # if scene != "00141-iigzG1rtanx_12":
        # if scene != "00680-YmWinf3mhb5_0":
        # if scene != "00255-NGyoyh91xXJ_2":

        print("before scene")
        if scene != "open_drawer":    
            continue

        print("after scene")

        visualize_scene(
            scene,
            args.output_dir,
            args.feature_dir,
            args.box_dir,
            args.box_format,
            args.objectness_dir,
            args.alpha_threshold,
            args.transpose_yz,
            dataset=dataset,
            # json_dict = json_dict
            
        )

    # fn = partial(
    #     visualize_scene,
    #     output_dir=args.output_dir,
    #     feature_dir=args.feature_dir,
    #     box_dir=args.box_dir,
    #     box_format=args.box_format,
    #     objectness_dir=args.objectness_dir,
    #     alpha_threshold=args.alpha_threshold,
    #     transpose_yz=args.transpose_yz,
    # )

    # process_map(fn, scenes, max_workers=8)
