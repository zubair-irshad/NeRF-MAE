#!/bin/bash

# Specify the output directory where you want to download the files
output_dir="scannet_meshes"

# Loop through the scenes and download the specified files
for scene_number in {300..500}; do
    scene_suffix=$(printf "%04d" $scene_number)
    scene_name="scene${scene_suffix}_00"

    # Download _vh_clean.aggregation.json file
    python download_scannet.py -o "$output_dir" --id "$scene_name" --type _vh_clean.aggregation.json

    # Download _vh_clean_2.ply file
    python download_scannet.py -o "$output_dir" --id "$scene_name" --type _vh_clean_2.ply

    # Download .txt file (modify the filename if necessary)
    python download_scannet.py -o "$output_dir" --id "$scene_name" --type .txt

    #Dowload _vh_clean_2.0.010000.segs.json file
    python download_scannet.py -o "$output_dir" --id "$scene_name" --type _vh_clean_2.0.010000.segs.json
done