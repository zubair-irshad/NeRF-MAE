import os
import numpy as np

split_file = (
    f"/home/ubuntu/NeRF-MAE/dataset/pretrain/nerfmae_split.npz"
)

out_name = 'nerfmae_split_new'
split = np.load(split_file)
# Get the list of scenes from the features directory

out_dir = "/home/ubuntu/NeRF-MAE/dataset/pretrain"
features_dir = os.path.join(out_dir, "features")

out_file = os.path.join(out_dir, out_name + "_split.npz")
scenes = []
for file_name in os.listdir(features_dir):
    if file_name.endswith(".npz"):
        scene_name = os.path.splitext(file_name)[0]
        scenes.append(scene_name)


all_indices = np.arange(len(scenes))
train_indices = all_indices

val_indices = all_indices
test_indices = all_indices

print("len train val test", len(train_indices), len(val_indices), len(test_indices))
modified_split = dict(split)
modified_split["train_scenes"] = scenes
modified_split["val_scenes"] = scenes
modified_split["test_scenes"] = scenes

# Save the modified split file
np.savez(out_file, **modified_split)