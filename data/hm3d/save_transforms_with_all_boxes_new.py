import os
import json
import numpy as np
import glob
import shutil
import open3d as o3d
import pandas as pd
import numpy as np

# Excluding problematic objects with the following NYU40 labels
excluded_labels = [1, 2, 8, 9, 11, 13, 16, 19, 20, 21, 22, 23, 25, 26, 27, 28, 30, 34, 40]
excluded_labels_nyu40 = [
    'wall',
    'floor',
    'door',
    'window',
    'counter',
    'shelves',
    'curtain',
    'books',
    'refrigerator',
    'television',
    'paper',
    'towel',
    'box',
    'whiteboard',
    'person',
    'night stand',
    'sink',
    'bag',
    'bookshelf',
    'otherprop',
    'pillow',
    'otherstructure'
]

local_folder = "/home/zubairirshad"
remote_folder = "/home/mirshad7"

# remote_folder = local_folder

# hm3d_to_mp3d_path = "/home/zubairirshad/NeRF_MAE_internal/data/hm3d/matterport_category_mappings.tsv"
hm3d_to_mp3d_path = os.path.join(remote_folder, "NeRF_MAE/data/hm3d/matterport_category_mappings.tsv")
# hm3d_to_mp3d_path = "/home/zubairirshad/NeRF_MAE_internal/data/hm3d/matterport_category_mappings.tsv"

df = pd.read_csv(hm3d_to_mp3d_path, sep="    ", header=0, engine="python")
hm3d_to_nyu40 = {row["category"]: row["nyu40id"] for _, row in df.iterrows()}
nyu40_id_label = {1: 'wall', 8: 'door', 22: 'ceiling', 2: 'floor', 11: 'picture', 9: 'window', 5: 'chair', 0: 'void', 18: 'pillow', 40: 'otherprop', 35: 'lamp', 3: 'cabinet', 16: 'curtain', 7: 'table', 19: 'mirror', 27: 'towel', 34: 'sink', 15: 'shelves', 6: 'sofa', 4: 'bed', 32: 'night stand', 33: 'toilet', 38: 'otherstructure', 25: 'television', 14: 'desk', 29: 'box', 39: 'otherfurniture', 12: 'counter', 21: 'clothes', 36: 'bathtub', 23: 'books', 17: 'dresser', 24: 'refridgerator', 10: 'bookshelf', 28: 'shower curtain', 13: 'blinds', 20: 'floor mat', 37: 'bag', 30: 'whiteboard', 26: 'paper', 31: 'person'}


def calculate_scene_obb_from_objects(bounding_boxes):
    # Calculate the minimum and maximum coordinates of all object OBBs

    extents, orientations, positions = bounding_boxes
    min_coords = np.inf
    max_coords = -np.inf

    for extent, orientation, position in zip(extents, orientations, positions):
        obb = o3d.geometry.OrientedBoundingBox(center=position, R=orientation, extent=extent)
        obb_points = np.asarray(obb.get_box_points())

        min_coords = np.minimum(min_coords, obb_points.min(axis=0))
        max_coords = np.maximum(max_coords, obb_points.max(axis=0))

    # print("min_coords", min_coords)
    # print("max_coords", max_coords)
    # aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_coords, max_bound=max_coords)
    return min_coords, max_coords

def transform_bounding_boxes(bounding_boxes, transform_matrix):
    extents, orientations, positions = bounding_boxes
    new_positions = []
    new_extents = []
    new_orientations = []
    for extent, orientation, position in zip(extents, orientations, positions):
        bbox = o3d.geometry.OrientedBoundingBox(center = position, R = orientation, extent=extent)
        # bbox_transformed = bbox.transform(transform_matrix)

        print("bbox center extent orientation", bbox.center, bbox.extent, bbox.R)
        # bbox.rotate(transform_matrix[:3, :3], center=(0, 0, 0))
        bbox.translate(transform_matrix[:3, 3])
        
        print("transform_matrix[:3, 3]", transform_matrix[:3, 3])
        print("bbox center extent orientation after translate", bbox.center, bbox.extent, bbox.R)

        new_positions.append(bbox.center)
        new_extents.append(bbox.extent)
        new_orientations.append(bbox.R)

    bounding_boxes = (new_extents, new_orientations, new_positions)

    return bounding_boxes



def get_boxes(box_file_path):
    extents = []
    rotations = []
    translations = []
    # Load the JSON data from the bounding box file
    with open(box_file_path, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)
        
    bounding_boxes = []
    # excluded_classes = ['mirror', 'door', 'pillow', 'fan', 'light', 'closet']
    # excluded_classes = ['bed', 'pillow']
    for bbox_info in bbox_data:
        class_name = bbox_info['class_name']
        # if class_name not in excluded_classes:
        #     continue
        bbox = bbox_info['bbox']
        
        min_pt = bbox[0]
        max_pt = bbox[1]
        
        #Let's do xzy instead of xyz
        min_pt[1], min_pt[2] = min_pt[2], min_pt[1]
        max_pt[1], max_pt[2] = max_pt[2], max_pt[1]
        bbox[0] = min_pt
        bbox[1] = max_pt
        # obj_aabb = obj_dict['aabb']

        extents.append(np.array(bbox[1])-np.array(bbox[0]))
        rotations.append(np.eye(3))
        translations.append((np.array(bbox[0])+np.array(bbox[1]))/2.0)

    return extents, rotations, translations

def get_filtered_boxes(box_file_path, filter_by_size=True, min_size = 6.5, grid_res=160, folder = None):
    extents = []
    rotations = []
    translations = []
    # Load the JSON data from the bounding box file
    with open(box_file_path, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)
        
    # bounding_boxes = []
    # excluded_classes = ['mirror', 'door', 'pillow', 'fan', 'light', 'closet']
    # excluded_classes = ['bed', 'pillow']
    print("==============================================\n")
    for bbox_info in bbox_data:
        
        class_name = bbox_info['class_name']
        bbox = bbox_info['bbox']
        
        # print("hm3d_to_nyu40.get(class_name, None)", hm3d_to_nyu40.get(class_name, 40))
        if class_name == 'sofa':
            class_name = 'couch'
        class_name_nyu40  = hm3d_to_nyu40.get(class_name.lower().strip(), 40)

        # if folder == '00009-vLpv2VX547B_14':
        #     print("---------------------------------")
        #     print("class_name", class_name)
        #     print("class_name_nyu40", class_name_nyu40)
        nyu40_label = nyu40_id_label.get(int(class_name_nyu40), None)

        if folder == '00009-vLpv2VX547B_14':
            print("nyu40_label original", nyu40_label)

        if nyu40_label in excluded_labels_nyu40:
            continue
        # else:
        #     print("nyu40_label", nyu40_label)



        min_pt = bbox[0]
        max_pt = bbox[1]

        #filter by size
        bbox_min = np.min(min_pt, axis=0)
        bbox_max = np.max(max_pt, axis=0)
        diag = bbox_max - bbox_min
        extent = np.array(bbox[1])-np.array(bbox[0])
        extent = extent / diag * grid_res

        if sum(any_extent > 35 for any_extent in extent) ==3:
            # print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n\n\n")
            # print("one of the bounding boxes is too large, skipping it")
            continue
        
        # if folder == '00009-vLpv2VX547B_14':
        #     print("extent", extent)
        #     print("---------------------------------")

        if filter_by_size and (extent < min_size).any():
            continue
        
        #Let's do xzy instead of xyz
        min_pt[1], min_pt[2] = min_pt[2], min_pt[1]
        max_pt[1], max_pt[2] = max_pt[2], max_pt[1]
        bbox[0] = min_pt
        bbox[1] = max_pt
        # obj_aabb = obj_dict['aabb']

        # extent = np.array(bbox[1])-np.array(bbox[0])
        # rotation = np.eye(3)
        # translation = (np.array(bbox[0])+np.array(bbox[1]))/2.0

        # bounding_box = {
        #     "extents": extent.tolist(),
        #     "orientation": rotation.tolist(),
        #     "position": translation.tolist()
        # }
        # bounding_boxes.append(bounding_box)
        extents.append(np.array(bbox[1])-np.array(bbox[0]))
        rotations.append(np.eye(3))
        translations.append((np.array(bbox[0])+np.array(bbox[1]))/2.0)
    return extents, rotations, translations
    # return bounding_boxes

def get_obb_dict(bounding_boxes):
    extents, orientations, positions = bounding_boxes
    obb_dict = []
    for extent, orientation, position in zip(extents, orientations, positions):
        obb_dict.append({
            "extents": extent.tolist(),
            "orientation": orientation.tolist(),
            "position": position.tolist()
        })
    return obb_dict

def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses: np.ndarray):
    """Transforms poses so principal components lie on XYZ axes.

    Args:
      poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    # t = poses[:, :3, 3]
    # t_mean = t.mean(axis=0)
    # t = t - t_mean

    # eigval, eigvec = np.linalg.eig(t.T @ t)
    # # Sort eigenvectors in order of largest to smallest eigenvalue.
    # inds = np.argsort(eigval)[::-1]
    # eigvec = eigvec[:, inds]
    # rot = eigvec.T
    # if np.linalg.det(rot) < 0:
    #     rot = np.diag(np.array([1, 1, -1])) @ rot

    # transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    # poses_recentered = unpad_poses(transform @ pad_poses(poses))
    # transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # # Flip coordinate system if z component of y-axis is negative
    # if poses_recentered.mean(axis=0)[2, 1] < 0:
    #     poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    #     transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # # Just make sure it's it in the [-1, 1]^3 cube
    # scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    # poses_recentered[:, :3, 3] *= scale_factor
    # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    translation = poses[..., :3, 3]

    mean_translation = np.mean(translation, axis=0)

    translation = mean_translation
    # translation_diff = translation - mean_translation


    transform = np.eye(4)
    transform[:3, 3] = -translation
    transform = transform[:3, :]
    poses_recentered = transform @ poses

    print("transform", transform)

    return poses_recentered, transform


# Define folder paths and camera parameters

# dir = "/arkit_data/masked_rdp_2"

#raw data dir
# dir = "/home/mirshad7/Downloads/hm3d_raw"
# dir = "/home/zubairirshad/Downloads/hm3d_raw"

dir = os.path.join(remote_folder, "Downloads/hm3d_raw")
folders = os.listdir(dir)

# out_dir = "/home/mirshad7/Downloads/hm3d_transforms_with_boxes"
# out_dir = "/home/zubairirshad/Downloads/hm3d_transforms_with_boxes"

# out_dir = os.path.join(remote_folder, "Downloads/hm3d_transforms_with_allboxes")
out_dir = os.path.join(remote_folder, "Downloads/hm3d_transforms_with_allboxes_new")

os.makedirs(out_dir, exist_ok=True)

print("total folders", len(folders))

# obj_obb_folder_path = '/home/mirshad7/Downloads/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis'
# obj_obb_folder_path = '/home/zubairirshad/Downloads/objects_bboxes_per_room/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis'

obj_obb_folder_path = os.path.join(remote_folder, "Downloads/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis")
# obj_obb_folder_path = os.path.join(remote_folder, "Downloads/objects_bboxes_per_room/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis")

count = 0
for folder in folders:
    input_folder = os.path.join(dir, folder)

    out_transforms_folder = os.path.join(out_dir, folder, "train")
    os.makedirs(out_transforms_folder, exist_ok=True)
    # input_folder = '/home/zubairirshad/Downloads/masked_rdp_2/00891-cvZr5TUy5C5_6'

    output_folder = os.path.join(input_folder, "train", "images")
    pose_folder = os.path.join(
        input_folder, "train", "poses"
    )  # New directory for pose files

    width = 512
    height = 512  # Set both width and height to 512

    # Create the output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(pose_folder, exist_ok=True)

    # Calculate camera parameters
    focal_length_x = 256.0 / np.tan(np.deg2rad(90.0) / 2)
    camera_angle_x = 2 * np.arctan(width / (2 * focal_length_x))

    # bbox_file_path = "/home/zubairirshad/Downloads/objects_bboxes_per_room/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis/00891-cvZr5TUy5C5_6.json"

    bbox_file_path = os.path.join(obj_obb_folder_path, folder + '.json')
    # excluded_classes = []
    # # Load the JSON data from the bounding box file
    # with open(bbox_file_path, 'r') as bbox_file:
    #     bbox_data = json.load(bbox_file)
    # bounding_boxes = ['mirror', 'door', 'pillow', 'fan', 'light']
    # for bbox_info in bbox_data:
    #     class_name = bbox_info['class_name']
    #     if class_name in excluded_classes:
    #         continue
    #     bbox = bbox_info['bbox']
    #     # obj_aabb = obj_dict['aabb']
    #     obj_bbox_ngp = {
    #         "extents": (np.array(bbox[1])-np.array(bbox[0])).tolist(),
    #         "orientation": np.eye(3).tolist(),
    #         "position": ((np.array(bbox[0])+np.array(bbox[1]))/2.0).tolist(),
    #     }
    #     bounding_boxes.append(obj_bbox_ngp)
    #     # bounding_boxes.append([bbox[0], bbox[1]])

    # Initialize the transforms dictionary
    transforms = {
        "camera_angle_x": float(camera_angle_x),
        "fl_x": float(focal_length_x),
        "fl_y": float(focal_length_x),
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "scale": 1.0,
        "aabb_scale": 2.0,
        "cx": float(width / 2),
        "cy": float(height / 2),
        "w": float(width),
        "h": float(height),
        "frames": [],
    }

    # Add the bounding boxes to the transforms data under a new key
    # transforms['bounding_boxes'] = bounding_boxes

    # List all files in the input folder
    all_files = os.listdir(input_folder)

    # Process each file and move them to the appropriate directory
    for file_name in all_files:
        if file_name.endswith(".png"):
            # Move PNG files to the "images" directory
            src_path = os.path.join(input_folder, file_name)
            dst_path = os.path.join(output_folder, file_name)
            shutil.move(src_path, dst_path)
        elif file_name.endswith(".json"):
            # Move JSON pose files to the "poses" directory
            src_path = os.path.join(input_folder, file_name)
            dst_path = os.path.join(pose_folder, file_name)
            shutil.move(src_path, dst_path)

    # List all JSON pose files
    pose_list = glob.glob(os.path.join(pose_folder, "*.json"))

    # Process each pose file and corresponding image

    all_frames_path = []
    all_poses = []
    for pos_file in pose_list:
        image_index = os.path.splitext(os.path.basename(pos_file))[0]
        image_path = os.path.join(output_folder, f"{image_index}.png")
        transform_matrix = np.array(json.load(open(pos_file))["pose"]).astype(
            np.float32
        )
        all_poses.append(transform_matrix)

        # Compute the relative path
        relative_path = os.path.relpath(image_path, input_folder)

        # Split the relative path using the directory separator
        relative_parts = relative_path.split(os.path.sep)

        # Join the relevant parts to get "images/713.png"
        desired_relative_path = os.path.join(*relative_parts[1:])
        
        all_frames_path.append(desired_relative_path)
        # frame = {
        #     "file_path": os.path.relpath(image_path, input_folder),
        #     "transform_matrix": transform_matrix.tolist()
        # }

        # transforms["frames"].append(frame)

    all_poses = np.array(all_poses)

    sizes = all_poses.size

    if sizes ==0:
        print("removing folder", input_folder, "since no poses found")
        shutil.rmtree(input_folder)
        continue
    num_poses = all_poses.shape[0]



    # min_coords, max_coords = calculate_scene_obb_from_objects(bounding_boxes)

    #we will scale all poses by min_ccords

    # do pca on the poses so they all lie inside the unit bounding box i.e. [-1,1]^3
    all_poses, transform_matrix = transform_poses_pca(all_poses)
    all_poses_homogeneous = np.zeros((num_poses, 4, 4))
    all_poses_homogeneous[:, :3, :4] = all_poses
    all_poses_homogeneous[:, 3, 3] = 1.0
    all_poses = all_poses_homogeneous

    extents, rotations, translations = get_boxes(bbox_file_path)
    bounding_boxes = (extents, rotations, translations)

    print("transform_matrix", transform_matrix.shape)
    bounding_boxes = transform_bounding_boxes(bounding_boxes, transform_matrix)
    min_coords, max_coords = calculate_scene_obb_from_objects(bounding_boxes)

    print("min_coords, max_coords", min_coords, max_coords)

    for filepath, pose in zip(all_frames_path, all_poses):
        frame = {"file_path": filepath, "transform_matrix": pose.tolist()}
        transforms["frames"].append(frame)

    if not isinstance(min_coords, np.ndarray):
        min_coords = np.array([-1, -1, -1])
        max_coords = np.array([1, 1, 1])
    transforms["room_bbox"] = [min_coords.tolist(), max_coords.tolist()]


    room_bbox = [np.array(min_coords), np.array(max_coords)]
    scale = 1.5 / np.max(room_bbox[1] - room_bbox[0])
    cent_after_scale = scale * (room_bbox[0] + room_bbox[1])/2.0
    offset = np.array([0.5, 0.5, 0.5]) - cent_after_scale

    transforms["scale"] = scale
    transforms["offset"] = offset.tolist()


    # object_obbs = get_filtered_object_obbs()

    object_bounding_boxes = get_filtered_boxes(bbox_file_path, folder = folder)
    object_bounding_boxes_transformed = transform_bounding_boxes(object_bounding_boxes, transform_matrix)
    object_bounding_boxes_dict = get_obb_dict(object_bounding_boxes_transformed)

    transforms["bounding_boxes"] = object_bounding_boxes_dict
    # Write the transforms dictionary to a JSON file
    # with open(os.path.join(input_folder, "train", "transforms.json"), "w") as json_file:
    #     json.dump(transforms, json_file, indent=4)

    with open(os.path.join(out_transforms_folder, "transforms.json"), "w") as json_file:
        json.dump(transforms, json_file, indent=4)

    print("transforms.json file created successfully for folder: ", folder)
    print("total folders processed: ", count)
    count += 1


# # List all JSON pose files
# pose_list = glob.glob(os.path.join(pose_folder, "*.json"))

# # Process each pose file and corresponding image
# for pos_file in pose_list:
#     image_index = os.path.splitext(os.path.basename(pos_file))[0]
#     image_path = os.path.join(output_folder, f"{image_index}.png")
#     transform_matrix = np.array(json.load(open(pos_file))["pose"]).astype(np.float32)

#     frame = {
#         "file_path": os.path.relpath(image_path, input_folder),
#         "transform_matrix": transform_matrix.tolist()
#     }

#     transforms["frames"].append(frame)

# # Write the transforms dictionary to a JSON file
# with open(os.path.join(input_folder, 'transforms.json'), 'w') as json_file:
#     json.dump(transforms, json_file, indent=4)

# print("Files moved and transforms.json file created successfully.")
