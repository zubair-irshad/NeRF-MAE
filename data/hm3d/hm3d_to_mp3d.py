import pandas as pd
import os
import numpy as np
import json

excluded_labels_nyu40 = [
    'wall',
    'floor',
    'door',
    'window',
    'counter',
    'shelves',
    'curtain',
    'books',
    'refrigerator',
    'television',
    'paper',
    'towel',
    'box',
    'whiteboard',
    'person',
    'night stand',
    'sink',
    'bag',
    'otherprop'
]

def get_boxes(box_file_path):
    extents = []
    rotations = []
    translations = []
    # Load the JSON data from the bounding box file
    with open(box_file_path, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)
        
    bounding_boxes = []
    # excluded_classes = ['mirror', 'door', 'pillow', 'fan', 'light', 'closet']
    # excluded_classes = ['bed', 'pillow']
    for bbox_info in bbox_data:
        class_name = bbox_info['class_name']
        print("class_name", class_name)
        # if class_name not in excluded_classes:
        #     continue
        bbox = bbox_info['bbox']
        
        min_pt = bbox[0]
        max_pt = bbox[1]
        
        #Let's do xzy instead of xyz
        min_pt[1], min_pt[2] = min_pt[2], min_pt[1]
        max_pt[1], max_pt[2] = max_pt[2], max_pt[1]
        bbox[0] = min_pt
        bbox[1] = max_pt
        # obj_aabb = obj_dict['aabb']

        extents.append(np.array(bbox[1])-np.array(bbox[0]))
        rotations.append(np.eye(3))
        translations.append((np.array(bbox[0])+np.array(bbox[1]))/2.0)

    return extents, rotations, translations

hm3d_to_mp3d_path = "NeRF_MAE_internal/data/hm3d/matterport_category_mappings.tsv"
df = pd.read_csv(hm3d_to_mp3d_path, sep="    ", header=0, engine="python")
# hm3d_to_mp3d = {row["category"]: row["mpcat40index"] for _, row in df.iterrows()}
hm3d_to_nyu40 = {row["category"]: row["nyu40id"] for _, row in df.iterrows()}
# id_to_category = {}
# # # Iterate through the DataFrame rows and populate the dictionary
# for _, row in df.iterrows():
#     if row["nyu40id"] == np.nan or row["nyu40class"] == np.nan:
#         continue
#     # id_to_category[row["mpcat40index"]] = row["mpcat40"]
#     id_to_category[row["nyu40id"]] = row["nyu40class"]

# # id_to_category = {row["category"]: row["mpcat40index"] for _, row in df.iterrows()}
# # Now id_to_category contains all unique ID-to-Category mappings
# print(id_to_category)


nyu40_id_label = {1: 'wall', 8: 'door', 22: 'ceiling', 2: 'floor', 11: 'picture', 9: 'window', 5: 'chair', 0: 'void', 18: 'pillow', 40: 'otherprop', 35: 'lamp', 3: 'cabinet', 16: 'curtain', 7: 'table', 19: 'mirror', 27: 'towel', 34: 'sink', 15: 'shelves', 6: 'sofa', 4: 'bed', 32: 'night stand', 33: 'toilet', 38: 'otherstructure', 25: 'television', 14: 'desk', 29: 'box', 39: 'otherfurniture', 12: 'counter', 21: 'clothes', 36: 'bathtub', 23: 'books', 17: 'dresser', 24: 'refridgerator', 10: 'bookshelf', 28: 'shower curtain', 13: 'blinds', 20: 'floor mat', 37: 'bag', 30: 'whiteboard', 26: 'paper', 31: 'person'}

# import math
# original_dict = {1.0: 'wall', 8.0: 'door', 22.0: 'ceiling', 2.0: 'floor', 11.0: 'picture', 9.0: 'window', 5.0: 'chair', 0.0: 'void', 18.0: 'pillow', 40.0: 'otherprop', 35.0: 'lamp', 3.0: 'cabinet', 16.0: 'curtain', 7.0: 'table', 19.0: 'mirror', 27.0: 'towel', 34.0: 'sink', 15.0: 'shelves', 6.0: 'sofa', 4.0: 'bed', 32.0: 'night stand', 33.0: 'toilet', 38.0: 'otherstructure', 25.0: 'television', 14.0: 'desk', 29.0: 'box', 39.0: 'otherfurniture', 12.0: 'counter', 21.0: 'clothes', 36.0: 'bathtub', 23.0: 'books', 17.0: 'dresser', 24.0: 'refridgerator', 10.0: 'bookshelf', 28.0: 'shower curtain', 13.0: 'blinds', 20.0: 'floor mat', 37.0: 'bag', 30.0: 'whiteboard', 26.0: 'paper', 31.0: 'person', float('nan'): float('nan')}

# # Remove NaN key and convert float keys to integers
# new_dict = {int(key): value for key, value in original_dict.items() if not math.isnan(key)}

# print(new_dict)

# mp3d_id_to_category = {1: 'wall', 4: 'door', 17: 'ceiling', 2: 'floor', 6: 'picture', 9: 'window', 3: 'chair', 0: 'void', 8: 'cushion', 39: 'objects', 28: 'lighting', 7: 'cabinet', 12: 'curtain', 5: 'table', 14: 'plant', 21: 'mirror', 20: 'towel', 15: 'sink', 31: 'shelving', 10: 'sofa', 11: 'bed', 13: 'chest_of_drawers', 18: 'toilet', 24: 'column', 30: 'railing', 16: 'stairs', 19: 'stool', 22: 'tv_monitor', 41: 'unlabeled', 23: 'shower', 26: 'counter', 34: 'seating', 27: 'fireplace', 38: 'clothes', 25: 'bathtub', 29: 'beam', 40: 'misc', 37: 'appliances', 36: 'furniture', 32: 'blinds', 35: 'board_panel', 33: 'gym_equipment'}

instance_name = '00009-vLpv2VX547B_10'
box_file_path = os.path.join('Downloads/objects_bboxes_per_room/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis', instance_name + '.json')

with open(box_file_path, 'r') as bbox_file:
    bbox_data = json.load(bbox_file)
    
bounding_boxes = []
# excluded_classes = ['mirror', 'door', 'pillow', 'fan', 'light', 'closet']
# excluded_classes = ['bed', 'pillow']
for bbox_info in bbox_data:
    class_name = bbox_info['class_name']
    print("class_name", class_name)
    class_name_nyu40  = hm3d_to_nyu40.get(class_name, None)
    print("class_name_nyu40", class_name_nyu40)
    nyu40_label = nyu40_id_label.get(int(class_name_nyu40), None)
    print("nyu40_label", nyu40_label)
    
# # bbox_file_path = "Downloads/objects_bboxes_per_room/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis/00891-cvZr5TUy5C5_6.json"
# extents, rotations, translations = get_boxes(box_file_path)