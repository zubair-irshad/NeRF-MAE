import os
import json
import shutil
import random
import numpy as np

# Input and output directory paths

folder = "/arkit_data/hm3d_raw_random20"
downsample_num = 1
all_dirs = os.listdir(folder)

val_downsample_num = 10

for num_dir, dir in enumerate(all_dirs):
    # print("processing", dir)
    # if dir == "3dfront_2000_00":
    #     continue
    input_dir = os.path.join(folder, dir)

    # input_dir = "Downloads/single_scene_front3d/3dfront_2001_00/train"
    # output_dir = "Downloads/single_scene_front3d_downsample"

    # Create the output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # Load the original transforms.json file
    with open(os.path.join(input_dir, "train", "transforms.json"), "r") as json_file:
        transforms_data = json.load(json_file)

    # List all image file paths
    image_dir = os.path.join(input_dir, "train", "images")
    image_files = [
        f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")
    ]

    random.seed(42)  # Set a seed for reproducibility

    def get_numeric_part(file_name):
        return int("".join(filter(str.isdigit, file_name)))

    random.seed(42)  # Set a seed for reproducibility

    # Sort image files based on the numeric part
    image_files = sorted(image_files, key=get_numeric_part)

    selected_image_files = image_files[::downsample_num]

    # if dir == "3dfront_0000_00":
    #     print("image_files", image_files)
    #     print("=============$$$$$$$$$$$$$$$$$$==========================\n\n\n")
    #     print("selected_image_files", selected_image_files)
    #     print("=============$$$$$$$$$$$$$$$$$$$$$==========================\n\n\n")
    indices_select = [j for j in range(len(image_files)) if j % val_downsample_num == 0]

    val_num = np.array(indices_select) + 2

    # print("val_num", val_num)

    # cap the maximum val_num to len(image_files)
    val_num = val_num[:-2]

    # select val every 12th image
    # selected_image_files_val = sorted(image_files)

    # select fir`st 10 images

    # print("Number of images:", len(selected_image_files))

    # print("Number of total images:", len(image_files))

    selected_image_files_val = np.array(image_files)[val_num][:10].tolist()

    # print("Number of val images:", len(selected_image_files_val))

    # Create a new transforms.json file for the downsampled images
    downsampled_transforms = transforms_data.copy()

    downsampled_transforms_val = transforms_data.copy()

    # Update paths in downsampled_transforms and remove frames for non-downsampled images
    downsampled_frames = []
    downsampled_frames_val = []
    for frame in downsampled_transforms["frames"]:
        file_path = frame["file_path"]
        file_name = os.path.basename(file_path)

        if (
            file_name in selected_image_files
            and file_name not in selected_image_files_val
        ):
            new_file_path = os.path.join("images", file_name)
            frame["file_path"] = new_file_path
            downsampled_frames.append(frame)

        if file_name in selected_image_files_val:
            new_file_path = os.path.join("images", file_name)
            frame["file_path"] = new_file_path
            downsampled_frames_val.append(frame)

    downsampled_transforms["frames"] = downsampled_frames
    downsampled_transforms_val["frames"] = downsampled_frames_val

    with open(os.path.join(input_dir, "train", "transforms.json"), "w") as json_file:
        json.dump(downsampled_transforms, json_file, indent=4)

    val_dir = os.path.join(input_dir, "val")
    os.makedirs(val_dir, exist_ok=True)

    with open(os.path.join(val_dir, "transforms.json"), "w") as json_file:
        json.dump(downsampled_transforms_val, json_file, indent=4)

    # # Save the downsampled transforms.json file
    # with open(os.path.join(output_dir, "train", "transforms.json"), "w") as json_file:
    #     json.dump(downsampled_transforms, json_file, indent=4)

    # print("Downsampling completed. Images and transforms.json are saved in:", input_dir)
    # print("completed", num_dir)
