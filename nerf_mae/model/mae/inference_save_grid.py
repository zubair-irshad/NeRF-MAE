import torch
import numpy as np
import os
import argparse
import sys

sys.path.append("/home/zubairirshad/NeRF_MAE_internal/nerf_mae")
from einops import rearrange
from model.mae.swin_mae3d import SwinTransformer_MAE3D
from model.mae.torch_utils import *
from model.mae.viz_utils import *
import time


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

    args = parser.parse_args()
    return args


def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def unpatchify_3d_full(x, patch_size=None, resolution=None, channel_dims=3):
    """
    rgbsimga: (N, 4, H, W, D)
    x: (N, L, L, L, patch_size**3 *4)
    """
    p = patch_size[0]
    h = w = l = int(round(resolution / p))

    print("x", x.shape)
    print("h", h, "w", w, "l", l, "p", p, "channel_dims", channel_dims)
    x = x.reshape(shape=(x.shape[0], h, w, l, p, p, p, channel_dims))
    x = rearrange(x, "n h w l p q r c -> n h p w q l r c")
    x = x.reshape(x.shape[0], h * p, w * p, l * p, channel_dims)

    return x


def build_model(args):
    backbone_type = args.backbone_type
    swin = {
        "swin_t": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "swin_s": {
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "swin_b": {
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "swin_l": {
            "embed_dim": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
        },
    }
    model = SwinTransformer_MAE3D(
        patch_size=[4, 4, 4],
        # patch_size=[5, 5, 5],
        embed_dim=swin[backbone_type]["embed_dim"],
        depths=swin[backbone_type]["depths"],
        num_heads=swin[backbone_type]["num_heads"],
        window_size=[4, 4, 4],
        # window_size=[5, 5, 5],
        stochastic_depth_prob=0.1,
        expand_dim=True,
        masking_prob=args.masking_prob,
        resolution=args.resolution,
        masking_strategy=args.masking_strategy,
        # is_eval=True
    )
    return model


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


def transform(x, resolution=256):
    # new_res = [160, 160, 160]
    new_res = [
        resolution,
        resolution,
        resolution,
    ]
    # new_res = [200, 200, 200]
    masks = []
    padded_rgbsigma = []
    for rgbsimga in x:
        rgb_sigma_padded, mask_rgbsigma = pad_tensor(
            rgbsimga, target_shape=new_res, pad_value=0
        )
        padded_rgbsigma.append(rgb_sigma_padded)
        masks.append(mask_rgbsigma)
    return padded_rgbsigma, masks

from torchvision.ops.misc import MLP, Permute

def patchify_3d(x, mask=None):
    """
    rgbsimga: (N, 4, H, W, D)
    x: (N, L, L, L, patch_size**3 *4)
    """
    patch_size = [4, 4, 4]
    p = patch_size[0]
    print("x", x.shape)
    print("p", p)
    print("x.shape[2]", x.shape[2], x.shape[3], x.shape[4], p)
    assert x.shape[2] == x.shape[3] == x.shape[4] and x.shape[2] % p == 0
    h = w = l = int(round(x.shape[2] // p))
    x = x.reshape(shape=(x.shape[0], 4, h, p, w, p, l, p))
    x = rearrange(x, "n c h p w q l r -> n h w l p q r c")
    x = rearrange(x, "n h w l p q r c -> n h w l (p q r) c")

    if mask is not None:
        mask = mask.reshape(shape=(x.shape[0], 4, h, p, w, p, l, p))
        mask = rearrange(mask, "n c h p w q l r -> n h w l p q r c")
        mask = rearrange(mask, "n h w l p q r c -> n h w l (p q r) c")
        mask = mask[:, :, :, :, :, 0].unsqueeze(-1).int()
        return x, mask

    else:
        return x


def main():
    args = parse_args()
    folder_name = args.folder_name
    filename = args.filename

    res, rgbsigma = load_data(folder_name, filename, resolution=args.resolution)
    model = build_model(args)

    patch_size = model.patch_size

    print("patch_size", patch_size)

    if args.checkpoint:
        assert os.path.exists(args.checkpoint), "The checkpoint does not exist."
        print(f"Loading checkpoint from {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()


    x = [rgbsigma]

    print("x", x[0].shape)

    patch_partition = nn.Sequential(
        nn.Conv3d(
            4,
            96,
            kernel_size=(patch_size[0], patch_size[1], patch_size[2]),
            stride=(patch_size[0], patch_size[1], patch_size[2]),
        ),
        Permute([0, 2, 3, 4, 1]),
        nn.LayerNorm(96),
    )
    x, mask = transform(x)
    x = torch.cat((x), dim=0)
    mask = torch.cat((mask), dim=0).to(x.device)

    latent = patch_partition(x)
    # x = x + pos_embed.type_as(x).to(x.device).clone().detach()
    mask_token = torch.zeros(96)
    latent, mask_patches = window_masking_3d(
        latent,
        p_remove=0.75,
        mask_token=mask_token,
    )

    target, mask = patchify_3d(x, mask)

    target_alpha = target[..., 3].unsqueeze(-1)
    mask = target_alpha > 0.01

    # latent, mask_patches = self.forward_encoder(x)

    # with torch.no_grad():
    #     (
    #         loss,
    #         loss_rgb,
    #         loss_alpha,
    #         pred,
    #         mask,
    #         mask_patches,
    #         target_rgb,
    #     ) = model([rgbsigma], is_eval=True)


    # pred_rgb = pred[..., :3]
    target_rgb = target
    print("target_rgb", target_rgb.shape)
    target_rgb = target_rgb[..., :3]
    print("target_rgb", target_rgb.shape)

    # print("pred_rgb", pred_rgb.shape, "target_rgb", target_rgb.shape, "mask", mask.shape, "mask_patches", mask_patches.shape)
    
    # pred_rgb = unpatchify_3d_full(
    #     pred_rgb,
    #     patch_size=model.patch_size,
    #     resolution=model.resolution,
    #     channel_dims=3,
    # )

    target_rgb = unpatchify_3d_full(
        target_rgb,
        patch_size=[4, 4, 4],
        resolution=256,
        channel_dims=3,
    )

    mask_keep = mask.squeeze(-1).int() * (1- mask_patches)
    mask_remove = mask.squeeze(-1).int() * (mask_patches)

    # mask_keep_patch = mask_remove.squeeze(-1).int() * (1 - mask_patches)
    # mask_remove_patch = mask_remove.squeeze(-1).int() * mask_patches

    # test_mask_keep_remove = mask_keep_patch + mask_remove_patch
    # print(
    #     "test_mask_keep_remove",
    #     mask_patches.dtype,
    #     mask_keep_patch.dtype,
    #     mask_remove_patch.dtype,
    #     test_mask_keep_remove.dtype,
    #     mask_remove.squeeze(-1).int().dtype,
    # )
    # print(
    #     "test mask keep remove",
    #     torch.equal(test_mask_keep_remove, mask_remove.squeeze(-1).int().float()),
    # )

    mask_remove_unroll = unpatchify_3d_full(
        mask,
        patch_size=[4, 4, 4],
        resolution=256,
        channel_dims=1,
    )

    mask_keep_patch_unroll = unpatchify_3d_full(
        mask_keep.unsqueeze(-1),
        patch_size=[4, 4, 4],
        resolution=256,
        channel_dims=1,
    )

    mask_remove_patch_unroll = unpatchify_3d_full(
        mask_remove.unsqueeze(-1),
        patch_size=model.patch_size,
        resolution=model.resolution,
        channel_dims=1,
    )


    # print("pred_rgb, target rgb, mask_remove unroll", pred_rgb.shape, target_rgb.shape, mask_remove_unroll.shape)

    # metrics = model.output_metrics(pred_rgb, target_rgb, mask_remove_unroll)

    # print("MSE", metrics["MSE"])
    # print("PSNR", metrics["PSNR"])


    # print("mask", mask_keep_patch_unroll.shape)
    # print("mask_remove", mask_remove.shape)

    # print("mask_patches", mask_patches.shape)



    # grid = grid[viz_mask, :]
    # rgbsigma = rgbsigma[viz_mask, :]

    new_res = np.array([model.resolution, model.resolution, model.resolution])
    grid = construct_grid(new_res)

    # viz_mask = generate_point_cloud_mask(grid)
    # viz_mask = viz_mask.reshape(-1)

    print("=================Visualizing GT=======================\n\n\n")
    grid_vis_original = grid.reshape(-1, 3) * mask_remove_unroll.reshape(-1, 1)
    target_rgb_vis_original = target_rgb.reshape(-1, 3) * mask_remove_unroll.reshape(
        -1, 1
    )
    
    # a = np.array(target_rgb_vis_original[:, 0])[::-1]
    # print("a.shape", a.shape)
    # target_rgb_vis_original[:, 0] = torch.FloatTensor(a)
    #mask
    # print("grid_vis_original", grid_vis_original.shape)
    # grid_vis_original = grid_vis_original[viz_mask, :]
    # print("grid vis mask", grid_vis_original.shape)
    # target_rgb_vis_original = target_rgb_vis_original[viz_mask, :]

    #save grid_vis_original and target_rgb_vis_original as npy

    output_folder = os.path.join(folder_name, "outputs_turn_tap")
    os.makedirs(output_folder, exist_ok=True)
    grid_vis_original_path = os.path.join(output_folder, filename + "_grid_vis_original.npy")
    np.save(grid_vis_original_path, grid_vis_original)

    target_rgb_vis_original_path = os.path.join(output_folder, filename + "_target_rgb_vis_original.npy")
    np.save(target_rgb_vis_original_path, target_rgb_vis_original)
    
    # draw_grid_colors(grid_vis_original, target_rgb_vis_original.numpy(), coordinate_system=False)

    print("Visualizing GT RGB with masked opacities >0.01\n")
    grid_vis_patches = grid.reshape(-1, 3) * mask_remove_patch_unroll.reshape(-1, 1)
    target_rgb_vis_patches = target_rgb.reshape(-1, 3) * mask_remove_patch_unroll.reshape(
        -1, 1
    )
    # grid_vis_patches = grid_vis_patches[viz_mask, :]
    # target_rgb_vis_patches = target_rgb_vis_patches[viz_mask, :]

    grid_vis_patches_path = os.path.join(output_folder, filename + "_grid_vis_patches.npy")
    np.save(grid_vis_patches_path, grid_vis_patches)

    target_rgb_vis_patches_path = os.path.join(output_folder, filename + "_target_rgb_vis_patches.npy")
    np.save(target_rgb_vis_patches_path, target_rgb_vis_patches)
    # draw_grid_colors(grid_vis_patches, target_rgb_vis_patches.numpy())



    # print("=================Visualizing Predictions=======================\n\n\n")
    # print("Visualizing predicted RGB with masked opacities >0.01\n")
    # grid_vis_original = grid.reshape(-1, 3) * mask_remove_unroll.reshape(-1, 1)
    # pred_rgb_vis_original = pred_rgb.reshape(-1, 3) * mask_remove_unroll.reshape(-1, 1)

    # #mask
    # grid_vis_original = grid_vis_original[viz_mask, :]
    # pred_rgb_vis_original = pred_rgb_vis_original[viz_mask, :]
    # draw_grid_colors(grid_vis_original, pred_rgb_vis_original.numpy())

    # print(
    #     "Visualizing blend of predicted rgb with mask remove patch + original rgb with mask keep patch both with GT opacitities >0.01\n"
    # )
    # grid_vis_original = grid.reshape(-1, 3) * mask_remove_unroll.reshape(-1, 1)
    # pred_rgb_vis_blend = pred_rgb.reshape(-1, 3) * mask_remove_patch_unroll.reshape(
    #     -1, 1
    # ) + target_rgb.reshape(-1, 3) * mask_keep_patch_unroll.reshape(-1, 1)

    # grid_vis_original = grid_vis_original[viz_mask, :]
    # pred_rgb_vis_blend = pred_rgb_vis_blend[viz_mask, :]

    # draw_grid_colors(grid_vis_original, pred_rgb_vis_blend.numpy())


if __name__ == "__main__":
    main()
