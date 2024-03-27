#!/bin/bash

# Set the source and destination directories
source_bucket="tri-ml-datasets"
source_folder="fm_datasets/ScanNet/scans/"
destination_folder="/data/zubair/scannet_processed/scans_test"

# Loop through the scenes and variants
for scene in $(aws s3 ls "s3://${source_bucket}/${source_folder}" | awk '{print $2}' | sed 's,/$,,'); do
    # Skip the root directory
    echo "Processing Scene: ${scene}"
    if [[ "${scene}" == "train" ]]; then
        continue
    fi

    variant=$(echo "${scene}" | awk -F_ '{print $2}')
    # scene_number=$(echo "${scene}" | awk -F_ '{print $1}' | awk -Fscene '{print $2}')

    scene_number=$(echo "${scene}" | awk -F_ '{print $1}' | awk -Fscene '{print $2}' | sed 's/^0*//')

    # Print variant and scene number
    echo "Processing Scene: ${scene}, Variant: ${variant}, Scene Number: ${scene_number}"

    # Check if the variant is '00' and scene_number is between 72 and 200
    if [[ "${variant}" == "00" ]] && ((300 <= ${scene_number} && ${scene_number} <= 500)); then
        # Download the scene
        aws s3 sync "s3://${source_bucket}/${source_folder}${scene}/color" "${destination_folder}/${scene}/color"
        aws s3 sync "s3://${source_bucket}/${source_folder}${scene}/depth" "${destination_folder}/${scene}/depth"
        aws s3 sync "s3://${source_bucket}/${source_folder}${scene}/intrinsic" "${destination_folder}/${scene}/intrinsic"
        aws s3 sync "s3://${source_bucket}/${source_folder}${scene}/pose" "${destination_folder}/${scene}/pose"
    fi
done