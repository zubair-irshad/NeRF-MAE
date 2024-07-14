import os
import numpy as np

split_file = (
    f"/wild6d_data/zubair/nerf_rpn/front3d/front3d_split.npz"
)

out_name = '3dfront_new'
split = np.load(split_file)
# Get the list of scenes from the features directory

out_dir = "/wild6d_data/zubair/nerf_rpn/front3d"
features_dir = os.path.join(out_dir, "features")

out_file = os.path.join(out_dir, out_name + "_split.npz")
# out_file = "/wild6d_data/zubair/MAE_complete_data/front3d_split.npz"
scenes = []
for file_name in os.listdir(features_dir):
    if file_name.endswith(".npz"):
        scene_name = os.path.splitext(file_name)[0]
        scenes.append(scene_name)


# original_feature_folder = '/wild6d_data/zubair/nerf_rpn/front3d_rpn_data/features'

# original_scenes = []
# for file_name in os.listdir(original_feature_folder):
#     if file_name.endswith(".npz"):
#         scene_name = os.path.splitext(file_name)[0]
#         original_scenes.append(scene_name)

# print("len scenes", len(scenes))

# print("scenes", scenes)
# print("=====================================\n\n\n")
# print("original scenes", original_scenes)
#Now remove original scenes from the scenes list
        
# scenes = list(set(scenes) - set(original_scenes))

# scenes = [x for x in scenes if x not in original_scenes]

# print("len original scenes", len(original_scenes))
# print("len scenes after", len(scenes))


# Create an array of indices representing all scenes
all_indices = np.arange(len(scenes))

# # Shuffle the indices randomly
# np.random.shuffle(all_indices)

# # Select the first 20 indices for validation
# val_indices = all_indices[:20]

# # Select the next 18 indices for testing
# test_indices = all_indices[20:38]

# The remaining indices are for training
# train_indices = all_indices[38:]
train_indices = all_indices



# Use the selected indices to create the split
# selected_val_scenes = [scenes[i] for i in val_indices]
# selected_test_scenes = [scenes[i] for i in test_indices]
# selected_train_scenes = [scenes[i] for i in train_indices]

val_indices = all_indices
test_indices = all_indices

print("len train val test", len(train_indices), len(val_indices), len(test_indices))

# # Now you can update your modified_split dictionary
# modified_split = dict(split)
# modified_split["train_scenes"] = np.array(selected_train_scenes)
# modified_split["val_scenes"] = selected_val_scenes
# modified_split["test_scenes"] = selected_test_scenes

# Now you can update your modified_split dictionary
modified_split = dict(split)
modified_split["train_scenes"] = scenes
modified_split["val_scenes"] = scenes
modified_split["test_scenes"] = scenes

# Save the modified split file
np.savez(out_file, **modified_split)