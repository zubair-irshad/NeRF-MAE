import os
import shutil

source_dir = "/arkit_data/masked_rdp_2/"
# destination_dir = "/arkit_data/masked_rdp_2/"
old_destination_dir = "/arkit_data/hm3d_transforms_with_allboxes_old/"

os.makedirs(old_destination_dir, exist_ok=True)
all_dir = os.listdir(source_dir)

for dir in all_dir:
    source_file_path = os.path.join(source_dir, dir, "train", "transforms_old.json")

    # # Walk through the source directory
    # for root, _, files in os.walk(source_dir):
    #     for file in files:
    #         if file == "transforms.json":
    #             source_file_path = os.path.join(root, file)

    destination_file_path = os.path.join(old_destination_dir, dir, "train")
    os.makedirs(destination_file_path, exist_ok=True)
    # destination_file_path = os.path.join(destination_dir, relative_path)

    # Ensure the destination directory exists
    # os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

    # Copy the file to the destination directory
    shutil.move(source_file_path, destination_file_path)
    print(f"Copied {source_file_path} to {destination_file_path}")

    # # Determine the relative path from the source to the destination
    # relative_path = os.path.relpath(source_file_path, source_dir)
    # destination_file_path = os.path.join(destination_dir, relative_path)
    # old_destination_file_path = os.path.join(old_destination_dir, relative_path)

    # # Ensure the destination directory exists
    # os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
    # os.makedirs(os.path.dirname(old_destination_file_path), exist_ok=True)

    # # Copy the file to the destination directory and rename it to transforms_old.json
    # shutil.copy(source_file_path, destination_file_path)
    # os.rename(destination_file_path, old_destination_file_path)

    # print(f"Moved {source_file_path} to {old_destination_file_path}")
