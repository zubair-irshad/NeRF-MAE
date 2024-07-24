import os
import numpy as np
import json

data_folder = "/home/zubairirshad/Downloads/arkitscenes_raw"

# Iterate over subfolders
for subfolder in os.listdir(data_folder):
    subfolder_path = os.path.join(data_folder, subfolder)
    if os.path.isdir(subfolder_path):
        transforms_file_path = os.path.join(subfolder_path, "train", "transforms.json")
        
        # Read the existing transforms.json file
        with open(transforms_file_path, "r") as file:
            transforms = json.load(file)
        
        # Correct the rotation
        bounding_boxes = transforms.get("bounding_boxes", [])
        for bbox in bounding_boxes:
            rotation = np.array(bbox.get("orientation"))
            corrected_rotation = np.transpose(rotation).tolist()
            bbox["orientation"] = corrected_rotation
        
        transforms["bounding_boxes"] = bounding_boxes
        
        # Save the corrected transforms back to the file
        with open(transforms_file_path, "w") as file:
            json.dump(transforms, file, indent=4)
            
        print(f"Corrected rotation in {transforms_file_path}")
