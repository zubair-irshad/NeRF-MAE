import os
import shutil

folder_path = "NeRF_MAE/data/scannet/scannet_processed/scans_test"

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file does not have the '_00' suffix
    if not filename.endswith("_00"):
        # Create the absolute file path
        file_path = os.path.join(folder_path, filename)

        # Delete the file
        shutil.rmtree(file_path)
        print(f"Deleted: {filename}")
