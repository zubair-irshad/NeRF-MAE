import os
import json

# Define the paths to the original and new directories
original_dir = "/arkit_data/masked_rdp_2/"
new_dir = "/arkit_data/hm3d_transforms_with_boxes"  # Replace with the actual path

count = 0
# Iterate through the subfolders of the original directory
for subfolder in os.listdir(original_dir):
    subfolder_path = os.path.join(original_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Check if the subfolder exists in the new directory
        new_subfolder_path = os.path.join(new_dir, subfolder)
        if os.path.exists(new_subfolder_path):
            # Read the original transforms.json file
            original_json_path = os.path.join(
                subfolder_path, "train", "transforms.json"
            )
            with open(original_json_path, "r") as original_json_file:
                original_data = json.load(original_json_file)

            # Read the new transforms.json file
            new_json_path = os.path.join(new_subfolder_path, "train", "transforms.json")
            with open(new_json_path, "r") as new_json_file:
                new_data = json.load(new_json_file)

            # Copy the "room_bbox" key from new_data to original_data
            original_data["room_bbox"] = new_data.get("room_bbox")

            # Write the updated data back to the original transforms.json file
            with open(original_json_path, "w") as original_json_file:
                json.dump(original_data, original_json_file, indent=4)

            print(f"Updated transforms.json in {subfolder}")
            count += 1
            print(count)

print("Done updating transforms.json files.")
