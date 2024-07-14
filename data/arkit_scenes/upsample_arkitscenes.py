import os
import json
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Path to the image folder
image_folder = "/home/zubairirshad/ARKitScenes/data/3dod/test_ngp_data/40573679/train/images"

# Path to the transforms.json file
json_path = "/home/zubairirshad/ARKitScenes/data/3dod/test_ngp_data/40573679/train/transforms.json"

# Define transformation parameters
upscale_factor = 4
new_fl_x = 212.027 * upscale_factor
new_fl_y = 212.027 * upscale_factor
new_cx = 127.933 * upscale_factor
new_cy = 95.9333 * upscale_factor
new_w = 256.0 * upscale_factor
new_h = 192.0 * upscale_factor

# Load and apply transformations to images
image_transform = transforms.Compose([
    transforms.Resize((int(new_h), int(new_w))),
    transforms.ToTensor()
])

# Load and modify transforms.json
with open(json_path, 'r') as json_file:
    transforms_data = json.load(json_file)

transforms_data["fl_x"] = new_fl_x
transforms_data["fl_y"] = new_fl_y
transforms_data["cx"] = new_cx
transforms_data["cy"] = new_cy
transforms_data["w"] = new_w
transforms_data["h"] = new_h

# Calculate sharpness and save it in the transforms.json
sharpness_values = {}

count=0
for filename in os.listdir(image_folder):
    print("Processing", filename)
    print("Count", count)
    count+=1
    if filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        
        # Calculate sharpness using the provided function
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_values[filename] = fm
        
        # Upscale and save the image
        image = Image.open(image_path)
        upscaled_image = image_transform(image)
        new_image_path = os.path.join(image_folder, f"{filename}")
        transforms.functional.to_pil_image(upscaled_image).save(new_image_path)

# Add sharpness values to the frames in transforms.json
for frame in transforms_data["frames"]:
    file_path = frame["file_path"]
    filename = os.path.basename(file_path)
    sharpness = sharpness_values.get(filename, 0)  # Default to 0 if not found
    frame["sharpness"] = sharpness

# Save the modified transforms.json
with open(json_path, 'w') as json_file:
    json.dump(transforms_data, json_file, indent=4)
