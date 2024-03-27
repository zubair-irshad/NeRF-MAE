import os
import json
import numpy as np
import glob
import shutil


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
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform


# Define folder paths and camera parameters

dir = "/arkit_data/masked_rdp_2"

folders = os.listdir(dir)

print("total folders", len(folders))

count = 0
for folder in folders:
    input_folder = os.path.join(dir, folder)
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
    # do pca on the poses so they all lie inside the unit bounding box i.e. [-1,1]^3
    all_poses, _ = transform_poses_pca(all_poses)
    all_poses_homogeneous = np.zeros((num_poses, 4, 4))
    all_poses_homogeneous[:, :3, :4] = all_poses
    all_poses_homogeneous[:, 3, 3] = 1.0
    all_poses = all_poses_homogeneous

    for filepath, pose in zip(all_frames_path, all_poses):
        frame = {"file_path": filepath, "transform_matrix": pose.tolist()}

        transforms["frames"].append(frame)

    # Write the transforms dictionary to a JSON file
    with open(os.path.join(input_folder, "train", "transforms.json"), "w") as json_file:
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
