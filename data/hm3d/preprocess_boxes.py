import os
import json
import numpy as np
import argparse
import h5py
import pandas as pd

from copy import deepcopy
from tqdm import tqdm


# # Excluding problematic objects with the following NYU40 labels
# excluded_labels = [1, 2, 8, 9, 11, 13, 16, 19, 20, 21, 22, 23, 25, 26, 27, 28, 30, 34]


# def nerf_matrix_to_ngp(nerf_matrix, scale, offset, from_mitsuba):
#     '''
#     Converts a nerf matrix to an ngp matrix. Follows the implementation in
#     nerf_loader.h
#     '''
#     result = nerf_matrix.copy()
#     result[:, [1, 2]] *= -1
#     result[:, 3] = result[:, 3] * scale + offset

#     if from_mitsuba:
#         result[:, [0, 2]] *= -1
#     else:
#         # Cycle axes xyz<-yzx
#         result = result[[1, 2, 0], :]

#     return result


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
    position = (min_pt_raw + max_pt_raw) / 2

    xform = np.hstack([np.eye(3, 3), np.expand_dims(position, 1)])
    # xform = dataset.nerf_matrix_to_ngp(xform)

    print("xform_before", xform)
    xform = nerf_matrix_to_ngp(xform, scale, offset, from_mitsuba)

    print("xform_after", xform)
    # extent *= dataset.scale
    extent *= scale
    min_pt, max_pt = get_ngp_obj_bounding_box(xform, extent)

    return min_pt, max_pt


# def get_ngp_obj_bounding_box(xform, extent):
#     corners = np.array([
#         [1, 1, 1],
#         [1, 1, -1],
#         [1, -1, -1],
#         [1, -1, 1],
#         [-1, 1, 1],
#         [-1, 1, -1],
#         [-1, -1, -1],
#         [-1, -1, 1],
#     ], dtype=float).T

#     corners *= np.expand_dims(extent, 1) * 0.5
#     corners = xform[:, :3] @ corners + xform[:, 3, None]
#     return np.min(corners, axis=1), np.max(corners, axis=1)


def nerf_matrix_to_ngp(nerf_matrix, scale, offset, from_mitsuba):
    """
    Converts a nerf matrix to an ngp matrix. Follows the implementation in
    nerf_loader.h
    """
    result = nerf_matrix.copy()
    result[:, [1, 2]] *= -1
    result[:, 3] = result[:, 3] * scale + offset

    if from_mitsuba:
        result[:, [0, 2]] *= -1
    else:
        # Cycle axes xyz<-yzx
        # don't do it if we don't permute
        # result = result

        print("result before", result)
        print("result.shape", result.shape)

        # only permute 1,2,0 for the rotation component i.e. fist 3 rows of the first 3 columns but don't touch the positions i.e. last column

        # result[:3, :3] = result[:3, [1, 2, 0]]

        # print("result", result)
        result = result[[1, 2, 0], :]

        print("result after 1 2 0", result)

        # z-up to y-up onlt for rotation part

        # 2,0,1 is what perm has currently
        # perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        # we basically want to go from z-up to y-up and design perm that way

        # perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        # this perm doesn't work let's try a different one

        # we have already tried this one

        # perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        # perm = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # perm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # we have tried these combinations but they don't work

        # perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        # zxy, yzx, yxz, zyx, xzy, xyz

        # write perm of xzy
        # perm = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0] ])

        # permute z and y axis
        # perm = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0] ])

        # result[:3, :3] = result[:3, [0, 2, 1]]
        # result[:3, :3] = result[:3, [0, 1, 2]]
        # result[:3, :3] = result[:3, [1, 0, 2]]
        # result[:3, :3] = result[:3, [1, 2, 0]]
        # result[:3, :3] = result[:3, [2, 0, 1]]
        # result[:3, :3] = result[:3, [2, 1, 0]]

        # potential
        # result[:3, :3] = result[:3, :3][[2, 0, 1]]
        # result[:3, :3] = result[:3, :3][[2, 1, 0]]
        # result[:3, :3] = result[:3, :3][[0, 2, 1]]
        # result[:3, :3] = result[:3, :3][[1, 2, 0]]
        # result[:3, :3] = result[:3, :3][[1, 0, 2]]
        # # result[:3, :3] = perm @ result[:3, :3]

        # print("result after", result)
        # result = result[[1, 0, 2], :]

    return result


def get_ngp_obj_bounding_box(xform, extent):
    """
    Get AABB from the OBB of an object in ngp coordinates.
    """
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

    return np.min(corners, axis=1), np.max(corners, axis=1)


def clip_boxes_to_mesh(boxes, size):
    """
    Clip boxes so that they lie inside a mesh of size `size`.
    """
    dim = boxes.ndim
    boxes_x = boxes[..., 0::3]
    boxes_y = boxes[..., 1::3]
    boxes_z = boxes[..., 2::3]
    width, height, depth = size

    boxes_x = np.clip(boxes_x, 0, width)
    boxes_y = np.clip(boxes_y, 0, height)
    boxes_z = np.clip(boxes_z, 0, depth)

    clipped_boxes = np.stack((boxes_x, boxes_y, boxes_z), axis=dim)
    return clipped_boxes.reshape(boxes.shape)


def process_obbs(json_dict, numpy_dict):
    """
    Handle the xyzwhdtheta format of OBBs.
    """

    room_box_original = json_dict["room_bbox"]

    grid_res = numpy_dict["resolution"]

    if any(dim < 100 for dim in grid_res):
        invalid_scene = True
    else:
        invalid_scene = False

    bbox_min = numpy_dict["bbox_min"]
    bbox_max = numpy_dict["bbox_max"]
    scale = numpy_dict["scale"]
    offset = numpy_dict["offset"]
    from_mitsuba = numpy_dict["from_mitsuba"]

    # since we are not permuting y and z axis and instead saving it in res[2], res[1] and res[0], do it like this here as well

    # grid_res = np.array([grid_res[2], grid_res[1], grid_res[0]])

    # min_transformed, max_transfored = transform_to_ngp_bbox(room_box_original, scale, offset, from_mitsuba)

    # bbox_min = min_transformed
    # bbox_max = max_transfored

    # print("grid_res: ", grid_res)
    # print("bbox_min: ", bbox_min)
    # print("bbox_max: ", bbox_max)
    # print("scale: ", scale)
    # print("offset: ", offset)
    # print("from_mitsuba: ", from_mitsuba)

    # z up to y-up
    # perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T

    # From y up to z up
    perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    grid_res = perm @ grid_res
    bbox_min = perm @ bbox_min
    bbox_max = perm @ bbox_max

    diag = bbox_max - bbox_min

    # print("diag: ", diag)
    # print("grid_res: ", grid_res)
    # print("bbox_min: ", bbox_min)
    # print("bbox_max: ", bbox_max)

    # try this
    grid_res = np.array([grid_res[1], grid_res[2], grid_res[0]])

    boxes = []
    for obj in json_dict["bounding_boxes"]:
        # print("obj: ", obj)
        # if filter_by_label and (
        #     obj["label"] in excluded_labels or obj["manually_filtered"]
        # ):
        #     continue

        extent = np.array(obj["extents"])
        orientation = np.array(obj["orientation"])
        position = np.array(obj["position"])

        xform = np.hstack([orientation, np.expand_dims(position, 1)])
        xform = nerf_matrix_to_ngp(xform, scale, offset, from_mitsuba)
        extent *= scale

        # From y up to z up

        xform = perm @ xform
        position = xform[:, 3]

        if xform[0, 0] == 0:
            theta = np.pi / 2
        else:
            # print("xform[0, 0]: ", xform[0, 0], xform[1, 0])
            theta = np.arctan(xform[1, 0] / xform[0, 0])

        # print("position: ", position)

        # print("theta: ", theta)

        # if (position < bbox_min).any() or (position > bbox_max).any():
        #     continue

        position = (position - bbox_min) / diag * grid_res

        # extent = np.array([extent[2], extent[0], extent[1]])

        extent = (extent / diag) * grid_res

        # extents z-up to y-up
        # extent = np.array([extent[2], extent[0], extent[1]])

        # Do not round to int
        boxes.append(np.concatenate((position, extent, np.expand_dims(theta, 0))))

    if len(boxes) > 2:
        is_valid_boxes = True
    else:
        is_valid_boxes = False
    boxes = np.array(boxes)
    # print("boxes: ", boxes)

    return boxes, is_valid_boxes, invalid_scene


def process_ngp_transforms(json_dict, numpy_dict, filter_by_size, min_size):
    """
    Handle the xyzxyz format of AABBs.
    """
    grid_res = numpy_dict["resolution"]
    bbox_min = numpy_dict["bbox_min"]
    bbox_max = numpy_dict["bbox_max"]
    scale = numpy_dict["scale"]
    offset = numpy_dict["offset"]
    from_mitsuba = numpy_dict["from_mitsuba"]

    diag = bbox_max - bbox_min
    boxes = []

    for obj in json_dict["bounding_boxes"]:
        # if filter_by_label and (
        #     obj["label"] in excluded_labels or obj["manually_filtered"]
        # ):
        #     continue

        extent = np.array(obj["extents"])
        orientation = np.array(obj["orientation"])
        position = np.array(obj["position"])

        xform = np.hstack([orientation, np.expand_dims(position, 1)])
        xform = nerf_matrix_to_ngp(xform, scale, offset, from_mitsuba)
        extent *= scale

        min_pt, max_pt = get_ngp_obj_bounding_box(xform, extent)
        min_pt = (min_pt - bbox_min) / diag * grid_res
        max_pt = (max_pt - bbox_min) / diag * grid_res
        min_pt = np.around(min_pt).astype(int)
        max_pt = np.around(max_pt).astype(int)

        boxes.append(np.concatenate((min_pt, max_pt)))

    boxes = np.array(boxes)
    boxes = clip_boxes_to_mesh(boxes, grid_res)

    # Filter out degenerated boxes, i.e. boxes with zero volume
    # This filters out boxes that are outside the grid and also boxes that
    # are too small to be interesting (i.e. < 1 voxel)
    degenerate_boxes = boxes[:, 3:] <= boxes[:, :3]
    boxes = boxes[~degenerate_boxes.any(axis=1)]

    if filter_by_size:
        keep = boxes[:, 3:] - boxes[:, :3] >= min_size
        boxes = boxes[keep.all(axis=1)]

    return boxes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the Hypersim ngp bounding boxes transforms.json to npy files as RPN input."
    )

    parser.add_argument("--format", choices=["aabb", "obb"], required=True)
    parser.add_argument(
        "--dataset_dir",
        default="",
        help="Path to the folder containing the NeRF scenes.",
    )
    parser.add_argument(
        "--feature_dir", default="", help="Path to the extracted features .npz files."
    )
    parser.add_argument(
        "--output_dir", default="", help="Path to the output directory."
    )
    # parser.add_argument(
    #     "--manual_label_path",
    #     default="",
    #     help="Path to the manual label files. Only needed if you want to manually filter out objects.",
    # )

    # parser.add_argument(
    #     "--filter_by_label",
    #     action="store_true",
    #     help="Filter out objects by semantics.",
    # )
    # parser.add_argument(
    #     "--label_descs",
    #     default="",
    #     help="Path to the NYU40 label descriptions given in Hypersim.",
    # )
    # parser.add_argument(
    #     "--hypersim_path", default="", help="Path to the original Hypersim dataset."
    # )
    # parser.add_argument(
    #     "--semantics", default="", help="Path to the Hypersim semantics."
    # )

    # parser.add_argument(
    #     "--filter_by_size", action="store_true", help="Filter out objects by size."
    # )
    # parser.add_argument(
    #     "--min_size",
    #     default=2,
    #     type=int,
    #     help="The minimum size " "(of the shortest dim) for an object to be included.",
    # )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scene_names = os.listdir(args.dataset_dir)
    scene_names = [
        f for f in scene_names if os.path.isdir(os.path.join(args.dataset_dir, f))
    ]

    npz_files = os.listdir(args.feature_dir)
    npz_files = [
        f for f in npz_files if os.path.isfile(os.path.join(args.feature_dir, f))
    ]
    scene_names = [f.split(".")[0] for f in npz_files]

    scene_count = 0
    total_scene_count = 0
    for scene_name in tqdm(scene_names):
        # if scene_name != "3dfront_0117_00":
        # if scene_name != "00009-vLpv2VX547B_0":
        #     continue
        if not os.path.isdir(os.path.join(args.dataset_dir, scene_name)):
            continue
        json_path = os.path.join(
            args.dataset_dir, scene_name, "train", "transforms.json"
        )
        npz_path = os.path.join(args.feature_dir, scene_name + ".npz")
        assert os.path.isfile(json_path)
        assert os.path.isfile(npz_path)

        with open(json_path, "r") as f:
            json_dict = json.load(f)
            numpy_dict = np.load(npz_path)

            # We are not filtering by anything
            # if args.filter_by_label:
            #     mesh_path = os.path.join(args.hypersim_path, scene_name, '_detail', 'mesh')
            #     sem_path = os.path.join(args.semantics, scene_name, '_detail', 'mesh')
            #     load_and_add_labels(json_dict, mesh_path, sem_path)

            # if args.manual_label_path:
            #     apply_manual_filters(json_dict, os.path.join(args.manual_label_path, scene_name + '.csv'))

            # if args.format == "aabb":
            #     boxes = process_ngp_transforms(
            #         json_dict,
            #         numpy_dict,
            #         args.filter_by_label,
            #         args.filter_by_size,
            #         args.min_size,
            #     )
            # elif args.format == "obb":
            #     boxes, is_valid_boxes = process_obbs(
            #         json_dict,
            #         numpy_dict,
            #     )

            # format is always obb for hm3d

            boxes, is_valid_boxes, invalid_scene = process_obbs(
                json_dict,
                numpy_dict,
            )

            # only save a box if len of boxes is greater than 2
            if is_valid_boxes and not invalid_scene:
                scene_count += 1
                output_path = os.path.join(args.output_dir, scene_name + ".npy")
                np.save(output_path, boxes)

            # # if is_valid_boxes:
            # scene_count += 1
            # output_path = os.path.join(args.output_dir, scene_name + ".npy")
            # np.save(output_path, boxes)

            if scene_count == 160:
                break
            total_scene_count += 1
            print("Processed scene: ", scene_name)
            print(
                "total number of scenes processed out of: ",
                scene_count,
                "/",
                total_scene_count,
            )
