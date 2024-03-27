import os
import shutil

source_dir = "/arkit_data/hm3d_transforms_with_allboxes_new"
destination_dir = "/arkit_data/masked_rdp_2"

# Walk through the source directory

all_dir = os.listdir(source_dir)

for dir in all_dir:
    source_file_path = os.path.join(source_dir, dir, "train", "transforms.json")
    # for file in files:
    # if file == "transforms.json":
    #     source_file_path = os.path.join(root, file)

    # Determine the relative path from the source to the destination

    # relative_path = os.path.relpath(source_file_path, source_dir)
    destination_file_path = os.path.join(destination_dir, dir, "train")
    # destination_file_path = os.path.join(destination_dir, relative_path)

    # Ensure the destination directory exists
    # os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

    # Copy the file to the destination directory
    shutil.copy(source_file_path, destination_file_path)
    print(f"Copied {source_file_path} to {destination_file_path}")

# for root, _, files in os.walk(source_dir):
