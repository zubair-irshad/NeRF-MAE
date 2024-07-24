import argparse
import os, sys

sys.path.append('os.path.join(os.path.dirname(__file__), "..")')
from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument(
    "--input_folder", required=True, help="path to folder containing .sens files"
)
parser.add_argument("--output_folder", required=True, help="path to output folder")
parser.add_argument(
    "--export_depth_images", dest="export_depth_images", action="store_true"
)
parser.add_argument(
    "--export_color_images", dest="export_color_images", action="store_true"
)
parser.add_argument("--export_poses", dest="export_poses", action="store_true")
parser.add_argument(
    "--export_intrinsics", dest="export_intrinsics", action="store_true"
)
parser.set_defaults(
    export_depth_images=False,
    export_color_images=False,
    export_poses=False,
    export_intrinsics=False,
)

opt = parser.parse_args()
print(opt)


def process_sens_file(sens_file, folder):
    out_path = os.path.join(opt.output_folder, folder)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load the data
    sys.stdout.write("loading %s..." % sens_file)
    sd = SensorData(sens_file)
    sys.stdout.write("loaded!\n")

    if opt.export_depth_images:
        sd.export_depth_images(os.path.join(out_path, "depth"))
    if opt.export_color_images:
        sd.export_color_images(os.path.join(out_path, "color"))
    if opt.export_poses:
        sd.export_poses(os.path.join(out_path, "pose"))
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(out_path, "intrinsic"))


def main():
    # if not os.path.exists(opt.input_folder):
    #     print(f"Error: Input folder '{opt.input_folder}' not found.")
    #     return

    all_folders = os.listdir(opt.input_folder)
    # sens_files = [f for f in os.listdir(opt.input_folder) if f.endswith(".sens")]

    for folder in all_folders:
        # if os.path.exists(os.path.join(opt.output_folder, folder)):
        #     print("Skipping:", folder, "...as it already exists.")
        #     continue
        sense_file_path = os.path.join(opt.input_folder, folder, folder + ".sens")
        process_sens_file(sense_file_path, folder)


# def main():
#     if not os.path.exists(opt.output_path):
#         os.makedirs(opt.output_path)
#     # load the data
#     print("loading %s..." % opt.filename)
#     sd = SensorData(opt.filename)
#     print("loaded!\n")
#     if opt.export_depth_images:
#         sd.export_depth_images(os.path.join(opt.output_path, "depth"))
#     if opt.export_color_images:
#         sd.export_color_images(os.path.join(opt.output_path, "color"))
#     if opt.export_poses:
#         sd.export_poses(os.path.join(opt.output_path, "pose"))
#     if opt.export_intrinsics:
#         sd.export_intrinsics(os.path.join(opt.output_path, "intrinsic"))


if __name__ == "__main__":
    main()
