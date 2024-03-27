import os
import shutil
import random
import json
import pandas as pd
import sys
import numpy as np
import cv2

# sys.path.append("/home/ubuntu/zubair/NeRF_MAE")
# sys.path.append("..")
from utils import TrajStringToMatrix


def load_json(js_path):
    with open(js_path, "r") as f:
        json_data = json.load(f)
    return json_data


def get_boxes(path):
    extents = []
    translations = []
    rotations = []
    json_data = load_json(path)
    bbox_list = []
    for label_info in json_data["data"]:
        rotation = np.array(
            label_info["segments"]["obbAligned"]["normalizedAxes"]
        ).reshape(3, 3)
        transform = (
            np.array(label_info["segments"]["obbAligned"]["centroid"])
            .reshape(-1, 3)
            .reshape(3)
        )
        scale = (
            np.array(label_info["segments"]["obbAligned"]["axesLengths"])
            .reshape(-1, 3)
            .reshape(3)
        )

        extents.append(scale)
        translations.append(transform)
        rotations.append(rotation)
        # box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
        # bbox_list.append(box3d)
    # bbox_list = np.asarray(bbox_list)
    bounding_boxes = (extents, rotations, translations)
    return bounding_boxes


def image_hwc_to_chw(img):
    """
    transpose the image from height, width, channel -> channel, height, width
    (pytorch format)
    """
    return img.transpose((2, 0, 1))


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def arkit_get_pose(frame_pose):
    frame_pose[0:3, 1:3] *= -1
    # frame_pose = frame_pose[np.array([1, 0, 2, 3]), :]
    # frame_pose[2, :] *= -1
    return frame_pose


def rotate_image(img, direction):
    if direction == "Up":
        pass
    elif direction == "Left":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif direction == "Right":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == "Down":
        img = cv2.rotate(img, cv2.ROTATE_180)
    else:
        raise Exception(f"No such direction (={direction}) rotation")
    return img


def filter_images(traj_file_path, image_folder_path, filtered_folder_path):
    # Create the filtered folder if it doesn't exist
    os.makedirs(filtered_folder_path, exist_ok=True)

    # Read the trajectory file and extract the timestamps
    timestamps = []
    with open(traj_file_path, "r") as traj_file:
        for i, line in enumerate(traj_file):
            ts, RT = TrajStringToMatrix(line)
            ts = round(float(ts), 3)
            timestamps.append(str(ts))

    if len(timestamps) > 900:
        print("too long of a trajectory length for instant ngp")
        return 0
    elif len(timestamps) > 500 and len(timestamps) < 900:
        print("subsampling trajectory every 2  frames")
        timestamps = np.array(timestamps)
        timestamps = timestamps[::2].tolist()

    # Copy the corresponding images to the filtered folder

    all_files = sorted(os.listdir(image_folder_path))
    img_count = 0
    for i, filename in enumerate(all_files):
        if filename.endswith(".png"):
            timestamp = filename.split("_")[1].split(".png")[0]
            # print("timestamp", timestamp)
            # print("rounded_timestamp", rounded_timestamp)
            if timestamp in timestamps:
                image_path = os.path.join(image_folder_path, filename)
                img_count += 1

                img = cv2.imread(image_path)
                # img = rotate_image(img, sky_direction)
                new_image_path = os.path.join(filtered_folder_path, timestamp + ".png")
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = image_hwc_to_chw(np.asarray(img, np.float32))
                # print("img", img.shape)

                cv2.imwrite(new_image_path, img)
                # shutil.copy(image_path, new_image_path)

    return img_count


def save_transforms_train(folder_path, output_folder, bounding_boxes):
    # Create the filtered folder if it doesn't exist
    train_folder = os.path.join(folder_path, "train")
    os.makedirs(train_folder, exist_ok=True)

    traj_file_path = os.path.join(folder_path, "lowres_wide.traj")
    intrinsics_folder = os.path.join(folder_path, "lowres_wide_intrinsics")
    image_folder_path = os.path.join(output_folder, "train/images")

    # os.makedirs(image_folder_path, exist_ok=True)

    shutil.copytree(
        os.path.join(folder_path, "filtered_lowres_wide"), image_folder_path
    )

    all_images = os.listdir(image_folder_path)

    intrinsic_file = random.choice(os.listdir(intrinsics_folder))

    intrinsic_file_path = os.path.join(intrinsics_folder, intrinsic_file)

    with open(intrinsic_file_path, "r") as file:
        line = file.readline().strip()
        (
            width,
            height,
            focal_length_x,
            focal_length_y,
            principal_point_x,
            principal_point_y,
        ) = map(float, line.split())

    # Calculate camera_angle_x from focal_length_x
    camera_angle_x = 2 * np.arctan(width / (2 * focal_length_x))

    # Create the transforms dictionary
    transforms = {
        "camera_angle_x": camera_angle_x,
        "fl_x": focal_length_x,
        "fl_y": focal_length_y,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "cx": principal_point_x,
        "cy": principal_point_y,
        "w": width,
        "h": height,
        "frames": [],
        "scale": 1.0,
        "aabb_scale": 8.0,
    }

    # Read the trajectory file and extract the timestamps
    timestamps = []
    RTs = []
    with open(traj_file_path, "r") as traj_file:
        for i, line in enumerate(traj_file):
            ts, RT = TrajStringToMatrix(line)

            # #convert opencv to openGL
            # RT = convert_pose(RT)
            RT = RT.reshape((4, 4))
            RT = arkit_get_pose(RT)
            # RT = convert_pose(RT)
            RTs.append(RT)
            ts = round(float(ts), 3)
            timestamps.append(str(ts))

    # Copy the corresponding images to the filtered folder

    frames = []
    for i, timestamp in enumerate(timestamps):
        frame = {}
        image_name = timestamp + ".png"
        if image_name in all_images:
            frame["file_path"] = "images/" + timestamp + ".png"
            frame["transform_matrix"] = RTs[i].tolist()
            frames.append(frame)

    transforms["frames"] = frames

    (extents, rotations, translations) = bounding_boxes

    # Create a new dictionary for the bounding boxes
    bounding_boxes = []

    # Iterate over each extent, translation, and rotation
    for extent, translation, rotation in zip(extents, translations, rotations):
        # Create a dictionary for each bounding box
        bounding_box = {
            "extents": extent.tolist(),
            "orientation": rotation.tolist(),
            "position": translation.tolist(),
        }
        # Add the bounding box dictionary to the list
        bounding_boxes.append(bounding_box)

    # Add the bounding_boxes list to the transforms dictionary
    transforms["bounding_boxes"] = bounding_boxes

    # Save the transforms dictionary to a json file
    transforms_file_path = os.path.join(output_folder, "train", "transforms.json")
    with open(transforms_file_path, "w") as file:
        json.dump(transforms, file)


if __name__ == "__main__":
    # data_folder = "/arkit_data/3dod"
    # output_path = "/arkit_data/ngp_data"

    data_folder = "/home/zubairirshad/ARKitScenes/data/3dod/test"
    output_path = "/home/zubairirshad/ARKitScenes/data/3dod/test_ngp_data"

    os.makedirs(output_path, exist_ok=True)

    META_DATA_CSV_FILE = os.path.join(data_folder, "metadata.csv")
    # Read the CSV file
    meta_data = pd.read_csv(META_DATA_CSV_FILE)
    folder = os.path.join(data_folder, "Training")

    all_ids = os.listdir(folder)

    all_img_count = 0
    for i, id in enumerate(all_ids):
        if i > 2000:
            break
        # video_id = 47333462
        video_id = id

        output_folder = os.path.join(output_path, str(video_id))
        if os.path.exists(output_folder):
            print("path exists,", output_folder, "skipping...")
            continue
        else:
            os.makedirs(output_folder, exist_ok=True)

        box_file_path = os.path.join(
            folder, video_id, video_id + "_3dod_annotation.json"
        )
        bounding_boxes = get_boxes(box_file_path)

        # folder_path = '/home/zubairirshad/ARKitScenes/data/raw/Training/40777073'
        folder_path = os.path.join(folder, str(video_id), str(video_id)+'_frames')
        traj_file_path = os.path.join(folder_path, "lowres_wide.traj")
        image_folder_path = os.path.join(folder_path, "lowres_wide")
        filtered_folder_path = os.path.join(folder_path, "filtered_lowres_wide")

        img_count = filter_images(
            traj_file_path, image_folder_path, filtered_folder_path
        )
        if img_count ==0:
            continue
        all_img_count += img_count
        save_transforms_train(folder_path, output_folder, bounding_boxes)

        print("Done with id", id, "img_count", img_count)

    print("all image count", all_img_count)
