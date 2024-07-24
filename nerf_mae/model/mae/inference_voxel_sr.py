import torch
import numpy as np
import os
import argparse
import sys

sys.path.append("/home/zubairirshad/NeRF_MAE_internal")
from einops import rearrange

from nerf_rpn.model.feature_extractor import (
    SwinTransformer_VoxelSR_Pretrained as SwinTransformer_VoxelSR_Pretrained,
    SwinTransformer_VoxelSR as SwinTransformer_VoxelSR,
)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and eval the NeRF RPN baseline using FCOS."
    )

    parser.add_argument(
        "--mode", default="train", choices=["train", "eval", "benchmark"]
    )
    parser.add_argument(
        "--dataset",
        "--dataset_name",
        default="hypersim",
        choices=["hypersim", "front3d", "general", "scannet"],
    )

    parser.add_argument("--folder_name", default="", help="The path to the features.")
    parser.add_argument("--filename", default=None, help="The path to the boxes.")
    parser.add_argument("--save_path", default="", help="The path to save the model.")
    parser.add_argument("--dataset_split", default="", help="The dataset split to use.")
    parser.add_argument(
        "--checkpoint", default="", help="The path to the checkpoint to load."
    )
    parser.add_argument(
        "--load_backbone_only",
        action="store_true",
        help="Only load the backbone weights.",
    )
    parser.add_argument(
        "--preload", action="store_true", help="Preload the features and boxes."
    )

    # General dataset csv files
    parser.add_argument("--train_csv", default="", help="The path to the train csv.")
    parser.add_argument("--val_csv", default="", help="The path to the val csv.")
    parser.add_argument("--test_csv", default="", help="The path to the test csv.")

    parser.add_argument(
        "--backbone_type",
        type=str,
        default="swin_s",
        choices=["resnet", "vgg_AF", "vgg_EF", "swin_t", "swin_s", "swin_b", "swin_l"],
    )
    parser.add_argument(
        "--input_dim", type=int, default=4, help="Input dimension for backbone."
    )

    parser.add_argument(
        "--masking_prob", type=float, default=0.5, help="Input dimension for backbone."
    )

    parser.add_argument(
        "--rotated_bbox",
        action="store_true",
        help="If true, bbox: (N, 7), [x, y, z, w, h, d, theta] \
                              If false, bbox: (N, 6), [xmin, ymin, zmin, xmax, ymax, zmax]",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=160,
        help="The max resolution of the input features.",
    )
    parser.add_argument(
        "--normalize_density",
        action="store_true",
        help="Whether to normalize the density.",
    )
    parser.add_argument(
        "--output_proposals",
        action="store_true",
        help="Whether to output proposals during evaluation.",
    )
    parser.add_argument(
        "--save_level_index", action="store_true", help="Whether to save level indices"
    )
    parser.add_argument(
        "--filter",
        choices=["none", "tp", "fp"],
        default="none",
        help="Filter the proposal output for visualization and debugging.",
    )
    parser.add_argument(
        "--masking_strategy",
        default="none",
        help="maksing strategy random, grid, block",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=0.7,
        help="The IoU threshold for the proposal filter, only used if --output_proposals is True "
        'and --filter is not "none".',
    )
    parser.add_argument(
        "--output_voxel_scores",
        action="store_true",
        help="Whether to output per-voxel objectness scores during evaluation, by default output "
        "to save_path/voxel_scores dir.",
    )

    # Training parameters
    parser.add_argument("--batch_size", default=1, type=int, help="The batch size.")
    parser.add_argument(
        "--num_epochs", default=100, type=int, help="The number of epochs to train."
    )
    parser.add_argument("--lr", default=5e-3, type=float, help="The learning rate.")
    parser.add_argument(
        "--reg_loss_weight",
        default=1.0,
        type=float,
        help="The weight for balancing the regression loss.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="The weight decay coefficient of AdamW.",
    )
    parser.add_argument(
        "--clip_grad_norm", default=0.1, type=float, help="The gradient clipping norm."
    )

    parser.add_argument(
        "--log_interval",
        default=20,
        type=int,
        help="The number of iterations to print the loss.",
    )
    parser.add_argument(
        "--log_to_file", action="store_true", help="Whether to log to a file."
    )
    parser.add_argument(
        "--eval_interval", default=1, type=int, help="The number of epochs to evaluate."
    )
    parser.add_argument(
        "--keep_checkpoints",
        default=1,
        type=int,
        help="The number of latest checkpoints to keep.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Whether to use wandb for logging."
    )

    parser.add_argument(
        "--rotate_prob",
        default=0.5,
        type=float,
        help="The probability of rotating the scene by 90 degrees.",
    )
    parser.add_argument(
        "--flip_prob",
        default=0.5,
        type=float,
        help="The probability of flipping the scene.",
    )
    parser.add_argument(
        "--rot_scale_prob",
        default=0.5,
        type=float,
        help="The probability of extra scaling and rotation.",
    )

    # Distributed training parameters
    parser.add_argument(
        "--gpus",
        default="",
        help="The gpus to use for distributed training. If empty, "
        "uses the first available gpu. DDP is only enabled if this is greater than one.",
    )

    # RPN head and loss parameters
    parser.add_argument(
        "--num_convs",
        default=4,
        type=int,
        help="The number of common convolutional layers in the RPN head.",
    )
    parser.add_argument(
        "--norm_reg_targets",
        action="store_true",
        help="Whether to normalize the regression targets for each feature map level.",
    )
    parser.add_argument(
        "--centerness_on_reg",
        action="store_true",
        help="Whether to append the centerness layer to the regression tower.",
    )
    parser.add_argument(
        "--center_sampling_radius",
        default=1.5,
        type=float,
        help="The radius for center sampling of the bounding box.",
    )
    parser.add_argument(
        "--iou_loss_type",
        choices=["iou", "linear_iou", "giou", "diou", "smooth_l1"],
        default="iou",
        help="The type of IOU loss to use for the RPN.",
    )
    parser.add_argument(
        "--use_additional_l1_loss",
        action="store_true",
        help="Whether to use additional smooth L1 loss for the OBB midpoint offsets.",
    )
    parser.add_argument(
        "--conv_at_start",
        action="store_true",
        help="Whether to add convolutional layers at the start of the backbone.",
    )
    parser.add_argument(
        "--proj2d_loss_weight",
        default=0.0,
        type=float,
        help="The weight for balancing the 2D projection loss.",
    )

    # RPN testing parameters (FCOSPostProcessor)
    parser.add_argument(
        "--pre_nms_top_n",
        default=2500,
        type=int,
        help="The number of top proposals to keep before applying NMS.",
    )
    parser.add_argument(
        "--fpn_post_nms_top_n",
        default=2500,
        type=int,
        help="The number of top proposals to keep after applying NMS.",
    )
    parser.add_argument(
        "--nms_thresh", default=0.3, type=float, help="The NMS threshold."
    )
    parser.add_argument(
        "--pre_nms_thresh", default=0.0, type=float, help="The score threshold."
    )
    parser.add_argument(
        "--min_size", default=0.0, type=float, help="The minimum size of a proposal."
    )
    parser.add_argument(
        "--ap_top_n",
        default=None,
        type=int,
        help="The number of top proposals to use for AP evaluation.",
    )

    parser.add_argument(
        "--output_all",
        action="store_true",
        help="Output proposals of all sets in evaluation.",
    )

    parser.add_argument(
        "--mae_checkpoint",
        default=None,
        help="The path to the checkpoint to load for the MAE model.",
    )
    parser.add_argument(
        "--out_resolution",
        type=int,
        default=160,
        help="The resolution of the output.",
    )

    parser.add_argument(
        "--is_eval",
        action="store_true",
        help="Whether to evaluate the model.",
    )


    args = parser.parse_args()
    return args


def construct_grid(res):
    res_x, res_y, res_z = res
    x = torch.linspace(0, res_x - 1, res_x) / max(res)
    y = torch.linspace(0, res_y - 1, res_y) / max(res)
    z = torch.linspace(0, res_z - 1, res_z) / max(res)

    # Shift by 0.5 voxel
    x += 0.5 / max(res)
    y += 0.5 / max(res)
    z += 0.5 / max(res)

    # Construct grid using broadcasting
    xx = x.view(res_x, 1, 1).repeat(1, res_y, res_z).flatten()
    yy = y.view(1, res_y, 1).repeat(res_x, 1, res_z).flatten()
    zz = z.view(1, 1, res_z).repeat(res_x, res_y, 1).flatten()
    grid = torch.stack((xx, yy, zz), dim=1)

    return grid

def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def unpatchify_3d_full(x, patch_size=None, resolution=None, channel_dims=4):
    """
    rgbsimga: (N, 4, H, W, D)
    x: (N, L, L, L, patch_size**3 *4)
    """
    patch_size = [4, 4, 4]
    p = patch_size[0]
    h = w = l = int(round(resolution / p))

    print("h, w, l", h, w, l, p)

    x = x.reshape(shape=(x.shape[0], h, w, l, p, p, p, channel_dims))

    x = rearrange(x, "n h w l p q r c -> n h p w q l r c")
    x = x.reshape(x.shape[0], h * p, w * p, l * p, channel_dims)

    return x



def load_data(folder_name, filename, resolution, normalize_density=True):
    # folder_name = "/home/zubairirshad/Downloads/hypersim_rpn_data/vis_scenes"
    # filename = "ai_006_007"

    # resolution = 160
    new_res = np.array([resolution, resolution, resolution])
    features_path = os.path.join(folder_name, "features", filename + ".npz")
    # box_path = os.path.join(folder_name, "obb", filename + ".npy")

    # boxes = np.load(box_path, allow_pickle=True)
    feature = np.load(features_path, allow_pickle=True)

    res = feature["resolution"]
    rgbsigma = feature["rgbsigma"]

    if normalize_density:
        alpha = density_to_alpha(rgbsigma[..., -1])
        rgbsigma[..., -1] = alpha

    # From (W, L, H, C) to (C, W, L, H)
    rgbsigma = np.transpose(rgbsigma, (3, 0, 1, 2))
    rgbsigma = torch.from_numpy(rgbsigma)

    if rgbsigma.dtype == torch.uint8:
        print("==================here=======================\n\n\n")
        # normalize rgbsigma to [0, 1]
        rgbsigma = rgbsigma.float() / 255.0
    return res, rgbsigma

def unpatchify_3d(x):
    """
    rgbsimga: (N, 4, H, W, D)
    x: (N, L, L, L, patch_size**3 *4)
    """
    patch_size = [4, 4, 4]
    p = patch_size[0]
    _, h, w, l, _ = x.shape

    x = x.reshape(shape=(x.shape[0], h, w, l, p**3, 4))
    return x

def pad_tensor(tensor, target_shape, pad_value):
    """
    Pad pytorch tensor of shpae 1, H,W,L, 3 pad it to a size of 1, target_shape[0],target_shape[1],target_shape[2],3 with zeros in pytorch and give me the resulting mask of non-padded values
    """
    mask = torch.ones(tensor.shape)
    # tensor = tensor.permute(3, 0, 1, 2)
    tensor = torch.nn.functional.pad(
        tensor,
        (
            0,
            target_shape[0] - tensor.shape[3],
            0,
            target_shape[1] - tensor.shape[2],
            0,
            target_shape[2] - tensor.shape[1],
        ),
        mode="constant",
        value=pad_value,
    )

    mask = torch.nn.functional.pad(
        mask,
        (
            0,
            target_shape[0] - mask.shape[3],
            0,
            target_shape[1] - mask.shape[2],
            0,
            target_shape[2] - mask.shape[1],
        ),
        mode="constant",
        value=0,
    )
    # mask = mask == 1
    return tensor.unsqueeze(0), mask.unsqueeze(0)

def transform(x, resolution=160):
    # new_res = [160, 160, 160]

    new_res = [
        resolution,
        resolution,
        resolution,
    ]
    # new_res = [200, 200, 200]
    masks = []
    padded_sem = []
    for rgbsimga in x:
        sem_padded, mask = pad_tensor(rgbsimga, target_shape=new_res, pad_value=0)
        padded_sem.append(sem_padded)
        masks.append(mask)
    return padded_sem, masks
    
def main():
    args = parse_args()
    folder_name = args.folder_name
    filename = args.filename

    res, rgbsigma = load_data(folder_name, filename, resolution=args.resolution)

    target_path = os.path.join(folder_name, "features_256", filename + ".npz")
    target = np.load(target_path, allow_pickle=True)

    target = target["rgbsigma"]

    print("target", target.shape)
    target = np.transpose(target, (3, 0, 1, 2))
    target = torch.from_numpy(target)

    target, _ = transform([target], resolution=args.out_resolution)
    

    print("target, target", target[0].shape)
    target = target[0].permute(0, 2, 3, 4, 1)

    print("target", target.shape)

    is_eval = True
    model = SwinTransformer_VoxelSR_Pretrained(
        resolution=args.resolution,
        checkpoint_path=args.mae_checkpoint,
        is_eval=is_eval,
        out_resolution=args.out_resolution,
    )

    print("args.checkpoint", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    

    print("rgbsigma", rgbsigma.shape)




    rgbsigma, _ = transform([rgbsigma], resolution=args.resolution)

    print("rgbsigma", rgbsigma[0].shape)
    rgbsigma = rgbsigma[0].squeeze(0)
    pred = model([rgbsigma])

    print("pred", pred.shape)
    
    pred = unpatchify_3d_full(pred, resolution=args.out_resolution)

    print("pred", pred.shape)

    print("target", target.shape)
    target_alpha = target[:, :, :, :, 3]
    print("target_alpha", target_alpha.shape)
    mask_remove = target_alpha > 0.01

    print("mask_remove", mask_remove.shape)

    # new_res = np.array([model.resolution, model.resolution, model.resolution])
    new_res = np.array([args.out_resolution, args.out_resolution, args.out_resolution])
    grid = construct_grid(new_res)

    print("grid", grid.shape)

    print("=================Visualizing GT=======================\n\n\n")
    grid_vis_original = grid.reshape(-1, 3) * mask_remove.reshape(-1, 1)
    grid_vis_original = grid_vis_original.detach().cpu().numpy()
    target_rgb_vis_original = pred[:, :,:,:,:3].reshape(-1, 3) * mask_remove.reshape(-1, 1)
    target_rgb_vis_original = target_rgb_vis_original.detach().cpu().numpy()
    

    output_folder = os.path.join(folder_name, "output_voxel_sr")
    os.makedirs(output_folder, exist_ok=True)
    grid_vis_original_path = os.path.join(output_folder, filename + "_grid_vis_original.npy")
    np.save(grid_vis_original_path, grid_vis_original)

    target_rgb_vis_original_path = os.path.join(output_folder, filename + "_target_rgb_vis_original.npy")
    np.save(target_rgb_vis_original_path, target_rgb_vis_original)



if __name__ == "__main__":
    main()