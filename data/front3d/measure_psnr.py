# import os
# from PIL import Image
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

# val_folder_dir = '/data/zubair/front3d_nerf_data_sparse1'

# train_folder_dir = '/data/zubair/FRONT3D_render_sparse1'
# all_folders = os.listdir(val_folder_dir)

# overall_psnr = []
# overall_ssim = []


# for folder in all_folders:
#     val_folder = os.path.join(val_folder_dir, folder, 'val', 'screenshots')
#     train_folder = os.path.join(train_folder_dir, folder, 'train', 'images')
#     if os.path.exists(val_folder):
#         # read all images in val folder
#         all_images = os.listdir(val_folder)

#         mean_psnr = []
#         mean_ssim = []

#         for image in all_images:
#             # read images using Pillow
#             pred_image = Image.open(os.path.join(val_folder, image)).convert('RGB')
#             gt_image = Image.open(os.path.join(train_folder, image)).convert('RGB')

#             # convert images to numpy arrays
#             pred_array = np.array(pred_image)
#             gt_array = np.array(gt_image)
#             psnr_value = peak_signal_noise_ratio(gt_array, pred_array)
#             pred_array = pred_array.astype(np.float32) / 255.0
#             gt_array = gt_array.astype(np.float32) / 255.0
#             # compute SSIM using scikit-image

#             ssim_value = ssim(
#                 pred_array, gt_array, data_range=1.0, channel_axis=-1
#             )
#             mean_psnr.append(psnr_value)
#             mean_ssim.append(ssim_value)
#         print(f'Folder: {folder}')
#         print(f'PSNR: {np.mean(mean_psnr)}')
#         print(f'SSIM: {np.mean(mean_ssim)}')

#         overall_psnr.append(np.mean(mean_psnr))
#         overall_ssim.append(np.mean(mean_ssim))


# print(f'Overall PSNR: {np.mean(overall_psnr)}')
# print(f'Overall SSIM: {np.mean(overall_ssim)}')

import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor

val_folder_dir = "/data/zubair/front3d_nerf_data_sparse3"
train_folder_dir = "/data/zubair/FRONT3D_render_sparse3"
all_folders = os.listdir(val_folder_dir)

overall_psnr = []
overall_ssim = []


def process_images(folder):
    val_folder = os.path.join(val_folder_dir, folder, "val", "screenshots")
    train_folder = os.path.join(train_folder_dir, folder, "train", "images")

    if os.path.exists(val_folder):
        all_images = os.listdir(val_folder)
        mean_psnr = []
        mean_ssim = []

        for image in all_images:
            pred_image = Image.open(os.path.join(val_folder, image)).convert("RGB")
            gt_image = Image.open(os.path.join(train_folder, image)).convert("RGB")

            pred_array = np.array(pred_image)
            gt_array = np.array(gt_image)

            psnr_value = peak_signal_noise_ratio(gt_array, pred_array)
            pred_array = pred_array.astype(np.float32) / 255.0
            gt_array = gt_array.astype(np.float32) / 255.0

            ssim_value = ssim(pred_array, gt_array, data_range=1.0, channel_axis=-1)
            mean_psnr.append(psnr_value)
            mean_ssim.append(ssim_value)

        # print(f'Folder: {folder}')
        # print(f'PSNR: {np.mean(mean_psnr)}')
        # print(f'SSIM: {np.mean(mean_ssim)}')

        return np.mean(mean_psnr), np.mean(mean_ssim)

    else:
        print(f"Folder {folder} does not have both validation and training folders.")
        return 0, 0


with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_images, all_folders))

overall_psnr, overall_ssim = zip(*results)

print(f"Overall PSNR: {np.mean(overall_psnr)}")
print(f"Overall SSIM: {np.mean(overall_ssim)}")
