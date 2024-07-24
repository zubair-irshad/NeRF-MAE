import os
import json
import numpy as np
import argparse
import h5py
import pandas as pd

from copy import deepcopy
from tqdm import tqdm


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




def clip_boxes_to_mesh(boxes, size):
    '''
    Clip boxes so that they lie inside a mesh of size `size`.
    '''
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


def process_obbs(json_dict, numpy_dict, filter_by_size, min_size):
    '''
    Handle the xyzwhdtheta format of OBBs.
    '''
    grid_res = numpy_dict['resolution']
    bbox_min = numpy_dict['bbox_min']
    bbox_max = numpy_dict['bbox_max']
    scale = numpy_dict['scale']
    offset = numpy_dict['offset']
    from_mitsuba = numpy_dict['from_mitsuba']

    print("grid_res", grid_res)
    print("bbox_min", bbox_min)
    print("bbox_max", bbox_max)

    # From y up to z up
    perm = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])

    grid_res = perm @ grid_res
    bbox_min = perm @ bbox_min
    bbox_max = perm @ bbox_max
    diag = bbox_max - bbox_min

    boxes = []
    for obj in json_dict['bounding_boxes']:
        # if filter_by_label and (obj['label'] in excluded_labels or obj['manually_filtered']):
        #     continue

        extent = np.array(obj['extents'])
        orientation = np.array(obj['orientation'])
        position = np.array(obj['position'])

        xform = np.hstack([orientation, np.expand_dims(position, 1)])
        xform = nerf_matrix_to_ngp(xform, scale, offset, from_mitsuba)
        extent *= scale

        # From y up to z up
        xform = perm @ xform
        position = xform[:, 3]

        if xform[0, 0] == 0:
            theta = np.pi / 2
        else:
            theta = np.arctan(xform[1, 0] / xform[0, 0])

        if (position < bbox_min).any() or (position > bbox_max).any():
            continue

        position = (position - bbox_min) / diag * grid_res
        extent = extent / diag * grid_res

        if filter_by_size and (extent < min_size).any():
            continue

        # Do not round to int
        boxes.append(np.concatenate((position, extent, np.expand_dims(theta, 0))))

    boxes = np.array(boxes)

    return boxes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the Hypersim ngp bounding boxes transforms.json to npy files as RPN input.')

    parser.add_argument('--format', choices=['aabb', 'obb'], required=True)
    parser.add_argument('--dataset_dir', default='', help='Path to the folder containing the NeRF scenes.')
    parser.add_argument('--feature_dir', default='', help='Path to the extracted features .npz files.')
    parser.add_argument('--output_dir', default='', help='Path to the output directory.')
    parser.add_argument('--manual_label_path', default='', 
                        help='Path to the manual label files. Only needed if you want to manually filter out objects.')

    parser.add_argument('--filter_by_label', action='store_true', help='Filter out objects by semantics.')
    parser.add_argument('--label_descs', default='', help='Path to the NYU40 label descriptions given in Hypersim.')
    parser.add_argument('--hypersim_path', default='', help='Path to the original Hypersim dataset.')
    parser.add_argument('--semantics', default='', help='Path to the Hypersim semantics.')

    parser.add_argument('--filter_by_size', action='store_true', help='Filter out objects by size.')
    parser.add_argument('--min_size', default=2, type=int, help='The minimum size '
                        '(of the shortest dim) for an object to be included.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    scene_names = os.listdir(args.dataset_dir)
    scene_names = [f for f in scene_names if os.path.isdir(os.path.join(args.dataset_dir, f))]

    npz_files = os.listdir(args.feature_dir)
    npz_files = [f for f in npz_files if os.path.isfile(os.path.join(args.feature_dir, f))]
    scene_names = [f.split('.')[0] for f in npz_files]

    for scene_name in tqdm(scene_names):
        if not os.path.isdir(os.path.join(args.dataset_dir, scene_name)):
            continue
        json_path = os.path.join(args.dataset_dir, scene_name, 'train', 'transforms.json')
        npz_path = os.path.join(args.feature_dir, scene_name + '.npz')
        assert os.path.isfile(json_path)
        assert os.path.isfile(npz_path)

        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            numpy_dict = np.load(npz_path)

            # if args.filter_by_label:
            #     mesh_path = os.path.join(args.hypersim_path, scene_name, '_detail', 'mesh')
            #     sem_path = os.path.join(args.semantics, scene_name, '_detail', 'mesh')
            #     load_and_add_labels(json_dict, mesh_path, sem_path)

            # if args.manual_label_path:
            #     apply_manual_filters(json_dict, os.path.join(args.manual_label_path, scene_name + '.csv'))

            # if args.format == 'aabb':
            #     boxes = process_ngp_transforms(json_dict, numpy_dict,
            #                                    args.filter_by_size, args.min_size)
            # elif args.format == 'obb':
            boxes = process_obbs(json_dict, numpy_dict,
                                    args.filter_by_size, args.min_size)

            output_path = os.path.join(args.output_dir, scene_name + '.npy')
            np.save(output_path, boxes)
