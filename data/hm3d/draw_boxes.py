import open3d as o3d
import numpy as np

# Load your transforms.json file and extract the bounding box data
# Replace this with your actual file loading code
# transforms_json = {
#     "bounding_boxes": [
#         {
#             "extents": [
#                 1.6610000133514404,
#                 0.869782030582428,
#                 0.7623017430305481
#             ],
#             "orientation": [
#                 [1.0, -1.4353919652876357e-07, 0.0],
#                 [1.4353919652876357e-07, 1.0, 0.0],
#                 [0.0, 0.0, 1.0]
#             ],
#             "position": [
#                 -1.7376667261123657,
#                 1.3474738597869873,
#                 0.3800032436847687
#             ]
#         }
#     ],
#     "room_bbox": [
#         [-3.3, 0.0, -0.10000000149011612],
#         [0.5, 4.2, 3.0]
#     ]
# }
import os
import json
transforms_folder = '/home/zubairirshad/Downloads/hm3d_transforms_with_allboxes_translate/00009-vLpv2VX547B_0'
new_json_path = os.path.join(transforms_folder, "train", "transforms.json")
with open(new_json_path, "r") as new_json_file:
    transforms_json = json.load(new_json_file)


# Extract room bounding box (AABB) and object bounding box (OBB) data
room_min = transforms_json["room_bbox"][0]
room_max = transforms_json["room_bbox"][1]

print("room_min: ", room_min)
print("room_max: ", room_max)

obb_positions = [bbox["position"] for bbox in transforms_json["bounding_boxes"]]
obb_orientations = [bbox["orientation"] for bbox in transforms_json["bounding_boxes"]]
obb_extents = [bbox["extents"] for bbox in transforms_json["bounding_boxes"]]

# Create Open3D geometry for room AABB
room_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=room_min, max_bound=room_max)
room_aabb.color = (1, 0, 0)  # Set the color to red


boxes = []
boxes.append(room_aabb)
# Create Open3D geometry for object OBBs
obb_list = []
for position, orientation, extent in zip(obb_positions, obb_orientations, obb_extents):
    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = position
    obb.R = np.array(orientation)
    obb.extent = extent
    obb.color = (0, 1, 0)  # Set the color to green
    # obb_list.append(obb)
    boxes.append(obb)

mesh =  o3d.geometry.TriangleMesh.create_coordinate_frame()
boxes.append(mesh)

o3d.visualization.draw_geometries(boxes)
# Create a visualization window
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Add the room AABB and object OBBs to the scene
# vis.add_geometry(room_aabb)
# vis.add_geometries(obb_list)

# # Run the visualization loop
# vis.run()
# vis.destroy_window()
