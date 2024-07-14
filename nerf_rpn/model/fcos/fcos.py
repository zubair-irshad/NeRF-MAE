#
# 3D FCOS model over NeRF, adapted from
# https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/modeling/rpn/fcos/fcos.py
#

import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from .inference import FCOSPostProcessor
from .loss import FCOSLossComputation
from .utils import print_shape


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOSHead(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_convs,
        fpn_strides,
        norm_reg_targets=True,
        centerness_on_reg=True,
        use_obb=False,
    ):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = 1
        self.fpn_strides = fpn_strides
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.num_convs = num_convs
        self.use_obb = use_obb

        cls_tower = []
        bbox_tower = []
        for i in range(self.num_convs):
            conv_func = nn.Conv3d
            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module("cls_tower", nn.Sequential(*cls_tower))
        self.add_module("bbox_tower", nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv3d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        bbox_pred_dim = 8 if self.use_obb else 6  # theta for heading angle if use_obb
        self.bbox_pred = nn.Conv3d(
            in_channels, bbox_pred_dim, kernel_size=3, stride=1, padding=1
        )
        self.centerness = nn.Conv3d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [
            self.cls_tower,
            self.bbox_tower,
            self.cls_logits,
            self.bbox_pred,
            self.centerness,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv3d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                # (N, 6/8, W, L, H)
                bbox_pred[:, :6, ...] = F.relu(
                    bbox_pred[:, :6, ...]
                )  # midpoint offsets for OBB shouldn't go through relu
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_pred[:, :6, ...] *= self.fpn_strides[l]
                    bbox_reg.append(bbox_pred)
            else:
                bbox_reg.append(
                    torch.exp(bbox_pred)
                )  # this part might be wrong, so always use norm_reg_targets

        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, args, in_channels, fpn_strides, resolution=None, world_size=1):
        super(FCOSModule, self).__init__()

        head = FCOSHead(
            in_channels,
            args.num_convs,
            fpn_strides,
            norm_reg_targets=args.norm_reg_targets,
            centerness_on_reg=args.centerness_on_reg,
            use_obb=args.rotated_bbox,
        )

        box_selector_test = FCOSPostProcessor(
            args.pre_nms_thresh,
            args.pre_nms_top_n,
            args.nms_thresh,
            args.fpn_post_nms_top_n,
            args.min_size,
            1,
            use_obb=args.rotated_bbox,
        )

        loss_evaluator = FCOSLossComputation(
            fpn_strides,
            args.center_sampling_radius,
            args.iou_loss_type,
            args.norm_reg_targets,
            world_size=world_size,
            use_obb=args.rotated_bbox,
            use_additional_l1_loss=args.use_additional_l1_loss,
            proj2d_loss_weight=args.proj2d_loss_weight,
        )

        self.resolution = resolution
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = fpn_strides
        self.world_size = world_size

    def forward(self, grid_sizes, features, targets=None, objectness_output_paths=None):
        """
        Arguments:
            grid_sizes (list[tuple[int, int, int]]): grid sizes for each scene
            features (list[Tensor]): features computed from the NeRF that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[Tensor): ground-truth boxes present in the scene (optional)

        Returns:
            boxes (list[Tensor]): the predicted boxes from the RPN, one Tensor per
                scene.
            scores (list[Tensor]): the scores for the predictions, one Tensor per scene.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)

        if self.resolution is not None:
            padding_masks = self.compute_padding_masks(locations, grid_sizes)
        else:
            padding_masks = (
                self.compute_padding_masks(locations, grid_sizes)
                if features[0].shape[0] > 1
                else None
            )

        if objectness_output_paths is not None:
            self.output_objectness(
                box_cls, centerness, grid_sizes, objectness_output_paths
            )

        if self.training:
            return self._forward_train(
                locations,
                box_cls,
                box_regression,
                centerness,
                targets,
                padding_masks=padding_masks,
            )
        else:
            return self._forward_test(
                locations,
                box_cls,
                box_regression,
                centerness,
                grid_sizes,
                padding_masks=padding_masks,
            )

    def _forward_train(
        self, locations, box_cls, box_regression, centerness, targets, padding_masks
    ):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations,
            box_cls,
            box_regression,
            centerness,
            targets,
            padding_masks=padding_masks,
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
        }
        return None, None, losses

    def _forward_test(
        self, locations, box_cls, box_regression, centerness, grid_sizes, padding_masks
    ):
        boxes, scores = self.box_selector_test(
            locations,
            box_cls,
            box_regression,
            centerness,
            grid_sizes,
            padding_masks=padding_masks,
        )

        return boxes, scores, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            w, l, h = feature.size()[-3:]
            locations_per_level = self.compute_locations_per_level(
                w, l, h, self.fpn_strides[level], feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, w, l, h, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride, dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, l * stride, step=stride, dtype=torch.float32, device=device
        )
        shifts_z = torch.arange(
            0, h * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_x, shift_y, shift_z = torch.meshgrid(
            shifts_x, shifts_y, shifts_z, indexing="ij"
        )
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shift_z = shift_z.reshape(-1)
        locations = torch.stack((shift_x, shift_y, shift_z), dim=1) + stride // 2
        return locations

    def compute_padding_masks(self, locations, ori_sizes):
        masks = []
        for locations_per_level in locations:
            masks.append(
                self.compute_padding_masks_per_level(locations_per_level, ori_sizes)
            )
        return masks

    def compute_padding_masks_per_level(self, locations, ori_sizes):
        masks_per_level = []
        for ori_size in ori_sizes:
            w_ori, l_ori, h_ori = ori_size
            mask = (
                (locations[:, 0] < w_ori)
                & (locations[:, 1] < l_ori)
                & (locations[:, 2] < h_ori)
            )
            masks_per_level.append(mask)

        return torch.stack(masks_per_level, dim=0)

    def output_objectness(self, box_cls, centerness, ori_sizes, output_paths):
        for i in range(len(ori_sizes)):
            all_levels = {}
            for level in range(len(box_cls)):
                score = box_cls[level][i].sigmoid() * centerness[level][i].sigmoid()
                w, l, h = np.ceil(
                    np.array(ori_sizes[i]) / self.fpn_strides[level]
                ).astype(np.int)
                score = score[0, :w, :l, :h]
                score = torch.sqrt(score)
                score = score.cpu().numpy()
                all_levels[str(level)] = score

            output_path = output_paths[i]
            np.savez_compressed(output_path, **all_levels)


class FCOSOverNeRF(torch.nn.Module):
    """
    Combining backbone and FCOS head
    """

    def __init__(
        self, args, backbone, fpn_strides, resolution=None, world_size=1
    ) -> None:
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        super().__init__()
        self.args = args
        self.world_size = world_size
        self.backbone = backbone
        self.fcos_module = FCOSModule(
            args,
            backbone.out_channels,
            fpn_strides,
            resolution=resolution,
            world_size=world_size,
        )
        self.resoluton = resolution

    def transform(self, meshes, target_shape=None):
        """
        Do 0-padding to support different sizes of input meshes if
        batch size > 1.

        Args:
            meshes (list[Tensor]): list of meshes to be transformed.
        """
        if target_shape is None:
            shapes = [mesh.shape for mesh in meshes]
            target_shape = np.max(shapes, axis=0)
        else:
            target_shape = np.array(target_shape)
        # print(f'Padding to {target_shape}')
        for i, mesh in enumerate(meshes):
            meshes[i] = F.pad(
                mesh,
                (
                    0,
                    target_shape[-1] - mesh.shape[-1],
                    0,
                    target_shape[-2] - mesh.shape[-2],
                    0,
                    target_shape[-3] - mesh.shape[-3],
                ),
                mode="constant",
                value=0,
            )

        return meshes

    def check_bbox_degeneration(self, targets):
        for target_idx, boxes in enumerate(targets):
            degenerate_boxes = boxes[:, 3:] <= boxes[:, :3]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive width, length and height."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    def forward(self, meshes, targets=None, objectness_output_paths=None):
        """
        Args:
            meshes (list[Tensor]): meshes to be processed
            targets (list[Tensor]): ground-truth boxes present in the scenes (optional)

        Returns:
            result (tuple(list[Tensor], dict[Tensor], list[Tensor])): the output from the model.
                The first element is a list of tensors, each of which contains the proposals of
                shape [N, 6] or [N, 7] for each scene. The second element is a dictionary of losses.
                The third element is a list of tensors, each of which contains the scores for each proposal.

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for boxes in targets:
                    if isinstance(boxes, torch.Tensor):
                        if self.args.rotated_bbox:
                            torch._assert(
                                len(boxes.shape) == 2 and boxes.shape[-1] == 7,
                                f"Expected target boxes to be a tensor of shape [N, 7], got {boxes.shape}.",
                            )
                        else:
                            torch._assert(
                                len(boxes.shape) == 2 and boxes.shape[-1] == 6,
                                f"Expected target boxes to be a tensor of shape [N, 6], got {boxes.shape}.",
                            )
                    else:
                        torch._assert(
                            False,
                            f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                        )

        original_mesh_sizes = []
        for mesh in meshes:
            val = mesh.shape[-3:]
            torch._assert(
                len(val) == 3,
                f"expecting the last three dimensions of the Tensor to be W, L and H instead got {mesh.shape[-3:]}",
            )
            original_mesh_sizes.append((val[0], val[1], val[2]))

        # Transformation (0-padding) for batch size > 1
        if self.resoluton is not None:
            target_shape = [self.resoluton, self.resoluton, self.resoluton]
            meshes = self.transform(meshes, target_shape)
        else:
            if len(meshes) > 1:
                meshes = self.transform(meshes)
        # Check for degenerate boxes (with negative side length)
        # Do not check degenerate boxes right now due to rotation
        # if targets is not None:
        #     self.check_bbox_degeneration(targets)

        # Here we assume either the batch size is 1 or the grids of the same size
        mesh_tensors = torch.stack(meshes, dim=0)
        features = list(self.backbone(mesh_tensors))
        boxes, scores, losses = self.fcos_module(
            original_mesh_sizes, features, targets, objectness_output_paths
        )

        return boxes, losses, scores
    

if __name__ == "__main__":

    import argparse
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
            default="front3d",
            choices=["hypersim", "front3d", "general", "scannet", "hm3d"],
        )

        parser.add_argument("--features_path", default="", help="The path to the features.")
        parser.add_argument("--boxes_path", default="", help="The path to the boxes.")
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
        parser.add_argument(
            "--percent_train", type=float, default=1.0, help="Input dimension for backbone."
        )
        # Training parameters
        parser.add_argument("--batch_size", default=1, type=int, help="The batch size.")
        parser.add_argument(
            "--num_epochs", default=100, type=int, help="The number of epochs to train."
        )
        parser.add_argument("--lr", default=1e-4, type=float, help="The learning rate.")
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
            "--tags",
            default="",
            help="wandb tags "
            "uses the first available gpu. DDP is only enabled if this is greater than one.",
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

    from model.mae.swin_mae3d import SwinTransformer_MAE3D_New
    backbone_type = "swin_s"
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
    model = SwinTransformer_MAE3D_New(
        patch_size=[4, 4, 4],
        embed_dim=swin[backbone_type]["embed_dim"],
        depths=swin[backbone_type]["depths"],
        num_heads=swin[backbone_type]["num_heads"],
        window_size=[4, 4, 4],
        stochastic_depth_prob=0.1,
        expand_dim=True,
        resolution=160,
        masking_prob=0.75,
    )
