import json
import numpy as np

# included_labels = 

# lamp
# decoration
# table
# sink
# toilet 
# lamp
# wardrobe
# tv
# decorative
# radiator
# couch
# trashcan 
# device
# nightstand
# dining chair
# support beam
# heater
# kitchen
# faucet
# table lamp
# dresser
# refrigerator
# microwave
# bathroom
# shelving 
# washbasin
# bench
# coffee machine
# countertop
# chandelier
# printer

# Define the paths to your JSON files

transforms_file_path = "Downloads/masked_rdp_2/00891-cvZr5TUy5C5_6/transforms.json"



# Load the JSON data from the transforms file
with open(transforms_file_path, 'r') as transforms_file:
    transforms_data = json.load(transforms_file)

# Extract the bounding boxes

bbox_file_path = "Downloads/objects_bboxes_per_room/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis/00891-cvZr5TUy5C5_6.json"

# Load the JSON data from the bounding box file
with open(bbox_file_path, 'r') as bbox_file:
    bbox_data = json.load(bbox_file)
bounding_boxes = []
for bbox_info in bbox_data:
    bbox = bbox_info['bbox']
    # obj_aabb = obj_dict['aabb']
    obj_bbox_ngp = {
        "extents": (np.array(bbox[1])-np.array(bbox[0])).tolist(),
        "orientation": np.eye(3).tolist(),
        "position": ((np.array(bbox[0])+np.array(bbox[1]))/2.0).tolist(),
    }
    bounding_boxes.append(obj_bbox_ngp)
    # bounding_boxes.append([bbox[0], bbox[1]])

# Add the bounding boxes to the transforms data under a new key
transforms_data['bounding_boxes'] = bounding_boxes



# Save the modified transforms data back to the transforms.json file
output_file_path = "transformed_transforms.json"  # Specify the desired output file path
with open(output_file_path, 'w') as output_file:
    json.dump(transforms_data, output_file, indent=4)

print(f"Bounding boxes saved as a new key in {output_file_path}")
