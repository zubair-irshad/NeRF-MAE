import json
import os
import glob
import torch
import numpy as np
import argparse
import logging
import importlib.util
from copy import deepcopy
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# from model.fcos.fcos import FCOSOverNeRF

import sys

sys.path.append("..")

# from nerf_rpn.model.feature_extractor import (
#     SwinTransformer_VoxelSR_Pretrained_Skip as SwinTransformer_VoxelSR_Pretrained,
#     SwinTransformer_VoxelSR_Skip as SwinTransformer_VoxelSR,
# )

from nerf_rpn.model.feature_extractor import (
    SwinTransformer_VoxelSR_Pretrained as SwinTransformer_VoxelSR_Pretrained,
    SwinTransformer_VoxelSR as SwinTransformer_VoxelSR,
)


# from model.feature_extractor import Bottleneck, ResNet_FPN_256, ResNet_FPN_64
# from model.feature_extractor import VGG_FPN, SwinTransformer_FPN
from nerf_rpn.datasets import (
    Front3DSuperResolutionDataset,
    HypersimRPNDataset,
    Front3DRPNDataset,
    GeneralRPNDataset,
    ScanNetRPNDataset,
)
from tqdm import tqdm
import wandb


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
        choices=["hypersim", "front3d", "general", "scannet", "hm3d"],
    )

    parser.add_argument("--features_path", default="", help="The path to the features.")
    parser.add_argument("--boxes_path", default=None, help="The path to the boxes.")
    parser.add_argument(
        "--out_feat_path", default=None, help="The path to the output features."
    )
    parser.add_argument("--save_path", default="", help="The path to save the model.")
    parser.add_argument("--dataset_split", default="", help="The dataset split to use.")
    parser.add_argument(
        "--checkpoint", default=None, help="The path to the checkpoint to load."
    )

    parser.add_argument(
        "--mae_checkpoint", default="", help="The path to the checkpoint to load."
    )
    parser.add_argument(
        "--load_backbone_only",
        action="store_true",
        help="Only load the backbone weights.",
    )
    parser.add_argument(
        "--preload", action="store_true", help="Preload the features and boxes."
    )

    parser.add_argument(
        "--use_pretrained_model",
        action="store_true",
        default=False,
        help="Preload the features and boxes.",
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
        "--out_resolution",
        type=int,
        default=256,
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
        "--percent_train", type=float, default=1.0, help="Input dimension for backbone."
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
        "--tags",
        default="",
        help="wandb tags "
        "uses the first available gpu. DDP is only enabled if this is greater than one.",
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


class Trainer:
    def __init__(self, args, rank=0, world_size=1, device_id=None, logger=None):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device_id = device_id
        self.logger = logger if logger is not None else logging.getLogger()
        torch.multiprocessing.set_sharing_strategy("file_system")

        if args.wandb and rank == 0:
            wandb.login(key="c4779119ee9d0aea91b4afb315bafb0bac03be91")
            wandb.init(
                project="nerf-mae",
                tags=[self.args.tags],
                config=deepcopy(args),
            )

        self.logger.info("Constructing model...")

        self.build_model(args)

        if args.checkpoint is not None:
            assert os.path.exists(args.checkpoint), "The checkpoint does not exist."
            self.logger.info(f"Loading checkpoint from {args.checkpoint}.")
            # if args.load_backbone_only:
            #     self.logger.info("Loading backbone only.")
            # checkpoint = torch.load(args.checkpoint, map_location="cpu")

            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["state_dict"])

            # print('Training args from checkpoint:')
            # print(checkpoint['train_args'])

            # self.model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
            # if not args.load_backbone_only:
            #     self.model.fcos_module.load_state_dict(checkpoint["fcos_state_dict"])

        if torch.cuda.is_available():
            self.model.cuda()

        if self.world_size > 1:
            self.model = DDP(
                self.model, device_ids=[self.device_id], find_unused_parameters=False
            )

        if args.wandb and rank == 0:
            wandb.watch(self.model, log_freq=50)

        spec = importlib.util.find_spec("torchinfo")
        if spec is not None:
            import torchinfo

            input_rgbsigma = [torch.rand(4, 200, 200, 130)] * 2
            input_boxes = torch.rand(32, 6)
            input_boxes[:, 3:] += 1.0
            input_boxes = [input_boxes] * 2
            # torchinfo.summary(self.model, input_data=(input_rgbsigma, input_boxes))
        # else:
        #     self.logger.info(self.model)

        self.init_datasets()

    def build_model(self, args):
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

        if self.args.checkpoint is not None:
            is_eval = True
        else:
            is_eval = False

        if self.args.use_pretrained_model:
            print(
                "==================Using pretrained Model========================\n\n"
            )
            self.model = SwinTransformer_VoxelSR_Pretrained(
                resolution=self.args.resolution,
                checkpoint_path=self.args.mae_checkpoint,
                is_eval=is_eval,
                out_resolution=self.args.out_resolution,
            )
        else:
            print(
                "==================Using Start from Scratch Model========================\n\n"
            )
            self.model = SwinTransformer_VoxelSR(
                patch_size=[4, 4, 4],
                embed_dim=swin[backbone_type]["embed_dim"],
                depths=swin[backbone_type]["depths"],
                num_heads=swin[backbone_type]["num_heads"],
                window_size=[4, 4, 4],
                stochastic_depth_prob=0,
                expand_dim=True,
                input_dim=self.args.input_dim,
                out_resolution=self.args.out_resolution,
            )

    def init_datasets(self):
        if not self.args.dataset_split and self.args.dataset != "general":
            raise ValueError(
                "The dataset split must be specified if the dataset type is not general."
            )

        self.logger.info(f"Loading dataset split from {self.args.dataset_split}.")

        if self.args.dataset != "general":
            with np.load(self.args.dataset_split) as split:
                self.train_scenes = split["train_scenes"]
                self.test_scenes = split["test_scenes"]
                self.val_scenes = split["val_scenes"]

                if self.args.output_all:
                    self.test_scenes = np.concatenate(
                        [self.train_scenes, self.test_scenes, self.val_scenes]
                    )

        if self.args.mode == "eval":
            if self.args.dataset == "hypersim":
                self.test_set = HypersimRPNDataset(
                    self.args.features_path,
                    self.args.boxes_path,
                    normalize_density=self.args.normalize_density,
                    scene_list=self.test_scenes,
                    preload=self.args.preload,
                )
            elif self.args.dataset == "front3d" or self.args.dataset == "hm3d":
                self.test_set = Front3DSuperResolutionDataset(
                    self.args.features_path,
                    out_feat_path=self.args.out_feat_path,
                    normalize_density=self.args.normalize_density,
                    scene_list=self.test_scenes,
                    preload=self.args.preload,
                )
            elif self.args.dataset == "scannet":
                self.test_set = ScanNetRPNDataset(
                    self.test_scenes, self.args.features_path, self.args.boxes_path
                )
            else:
                self.test_set = GeneralRPNDataset(
                    self.args.test_csv, self.args.normalize_density
                )

            if self.rank == 0:
                self.logger.info(f"Loaded {len(self.test_set)} scenes for evaluation")

    def save_checkpoint(self, epoch, path):
        if self.world_size == 1:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "train_args": self.args.__dict__,
                },
                path,
            )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": self.model.module.state_dict(),
                    "train_args": self.args.__dict__,
                },
                path,
            )

    def delete_old_checkpoints(self, path, keep_latest=5):
        files = glob.glob(f"{path}/epoch_*.pt")
        files.sort(key=os.path.getmtime)
        if len(files) > keep_latest:
            for file in files[:-keep_latest]:
                logging.info(f"Deleting old checkpoint {file}.")
                os.remove(file)

    def train_loop(self):
        if self.args.dataset == "hypersim":
            self.train_set = HypersimRPNDataset(
                self.args.features_path,
                self.args.boxes_path,
                scene_list=self.train_scenes,
                normalize_density=self.args.normalize_density,
                flip_prob=self.args.flip_prob,
                rotate_prob=self.args.rotate_prob,
                rot_scale_prob=self.args.rot_scale_prob,
                preload=self.args.preload,
            )

            if self.rank == 0:
                self.val_set = HypersimRPNDataset(
                    self.args.features_path,
                    self.args.boxes_path,
                    scene_list=self.val_scenes,
                    normalize_density=self.args.normalize_density,
                    preload=self.args.preload,
                )
        elif self.args.dataset == "front3d" or self.args.dataset == "hm3d":
            # print("self.train_scenes heree", self.train_scenes)
            self.train_set = Front3DSuperResolutionDataset(
                self.args.features_path,
                out_feat_path=self.args.out_feat_path,
                scene_list=self.train_scenes,
                normalize_density=self.args.normalize_density,
                flip_prob=self.args.flip_prob,
                rotate_prob=self.args.rotate_prob,
                rot_scale_prob=self.args.rot_scale_prob,
                preload=self.args.preload,
                percent_train=self.args.percent_train,
            )

            if self.rank == 0:
                self.val_set = Front3DSuperResolutionDataset(
                    self.args.features_path,
                    out_feat_path=self.args.out_feat_path,
                    scene_list=self.val_scenes,
                    normalize_density=self.args.normalize_density,
                    preload=self.args.preload,
                )
        elif self.args.dataset == "scannet":
            self.train_set = ScanNetRPNDataset(
                self.train_scenes,
                self.args.features_path,
                self.args.boxes_path,
                flip_prob=self.args.flip_prob,
                rotate_prob=self.args.rotate_prob,
                rot_scale_prob=self.args.rot_scale_prob,
            )
            if self.rank == 0:
                self.val_set = ScanNetRPNDataset(
                    self.val_scenes, self.args.features_path, self.args.boxes_path
                )

        collate_fn = self.train_set.collate_fn

        if self.rank == 0:
            self.logger.info(
                f"Loaded {len(self.train_set)} training scenes, "
                f"{len(self.val_set)} validation scenes"
            )

        if self.world_size == 1:
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=self.args.batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
        else:
            self.train_sampler = DistributedSampler(self.train_set)
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=self.args.batch_size // self.world_size,
                collate_fn=collate_fn,
                sampler=self.train_sampler,
                num_workers=2,
                pin_memory=True,
            )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.args.lr,
            total_steps=self.args.num_epochs * len(self.train_loader),
        )

        self.best_metric = None
        os.makedirs(self.args.save_path, exist_ok=True)
        # self.eval(self.val_set)

        for epoch in range(1, self.args.num_epochs + 1):
            if self.world_size > 1:
                self.train_sampler.set_epoch(epoch)

            self.train_epoch(epoch)
            if self.rank != 0:
                continue

            if epoch % self.args.eval_interval == 0 or epoch == self.args.num_epochs:
                # recalls, APs = self.eval(self.val_set)
                psnr_mean, mse_mean = self.eval(self.val_set)
                metric = psnr_mean
                # metric = recalls[-1]
                if self.best_metric is None or metric > self.best_metric:
                    self.best_metric = metric
                    self.save_checkpoint(
                        epoch, os.path.join(self.args.save_path, "model_best.pt")
                    )

                # self.save_checkpoint(
                #     epoch, os.path.join(self.args.save_path, f"epoch_{epoch}.pt")
                # )
                # self.delete_old_checkpoints(
                #     self.args.save_path, keep_latest=self.args.keep_checkpoints
                # )

    def train_epoch(self, epoch):
        # torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(self.train_loader):
            # self.logger.debug(f'GPU {self.device_id} Epoch {epoch} Iter {i} {batch[-1]} '
            #                   f'Grid size: {[x.shape for x in batch[0]]}, GT boxes: {[x.shape for x in batch[1]]}')

            self.model.train()
            self.optimizer.zero_grad()

            rgbsigma, out_rgbsigma, _ = batch

            if torch.cuda.is_available():
                rgbsigma = [item.cuda() for item in rgbsigma]
                out_rgbsigma = [item.cuda() for item in out_rgbsigma]

                # boxes = [item.cuda() for item in boxes]

            pred = self.model(rgbsigma)

            loss = self.model.module.loss_fn(out_rgbsigma, pred)

            # losses["loss_reg"] *= self.args.reg_loss_weight
            # loss = losses["loss_cls"] + losses["loss_reg"] + losses["loss_centerness"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.clip_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            self.logger.debug(
                f"GPU {self.device_id} Epoch {epoch} Iter {i} {batch[-1]} "
                f"loss: {loss.item():.6f}"
            )

            if self.world_size > 1:
                dist.barrier()
                dist.all_reduce(loss)
                loss /= self.world_size

                # dist.all_reduce(loss_rgb)
                # loss_rgb /= self.world_size

                # if torch.is_tensor(loss_alpha):
                #     dist.all_reduce(loss_alpha)
                #     loss_alpha /= self.world_size

            if i % self.args.log_interval == 0 and self.rank == 0:
                self.logger.info(
                    f"epoch {epoch} [{i}/{len(self.train_loader)}]  "
                    f"lr: {self.scheduler.get_last_lr()[0]:.6f}  "
                    f"loss: {loss.item():.4f}"
                )

            if self.args.wandb and self.rank == 0:
                wandb.log(
                    {
                        "lr": self.scheduler.get_last_lr()[0],
                        "loss_recon": loss.item(),
                        # "loss_rgb": loss_rgb.item(),
                        "epoch": epoch,
                        "iter": i,
                    }
                )

                # if torch.is_tensor(loss_alpha):
                #     wandb.log({"loss_alpha": loss_alpha.item()})
                # else:
                #     wandb.log({"loss_alpha": 0.0})

    @torch.no_grad()
    def eval(self, dataset):
        print("inside eval............")
        self.model.eval()
        collate_fn = dataset.collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size // self.world_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

        self.logger.info(f"Evaluating...")
        psnr_list = []
        all_mse = []

        for batch in tqdm(dataloader):
            rgbsigma, out_rgbsigma, scenes = batch
            if torch.cuda.is_available():
                rgbsigma = [item.cuda() for item in rgbsigma]
                out_rgbsigma = [item.cuda() for item in out_rgbsigma]

            pred = self.model(rgbsigma)
            if self.args.mode == "eval":
                metrics = self.model.output_metrics(out_rgbsigma, pred)
            else:
                metrics = self.model.module.output_metrics(out_rgbsigma, pred)

            psnr_list.append(metrics["PSNR"])
            all_mse.append(metrics["MSE"])

            print(f"\n Eval MSE/PSNR:")
            print(f'PSNR: {metrics["PSNR"]}, MSR: {metrics["MSE"]}')

            if self.args.wandb:
                wandb.log(
                    {
                        f"Val PSNR": metrics["PSNR"],
                        f"Val MSE": metrics["MSE"],
                    },
                    commit=False,
                )

        psnr_mean = np.array(psnr_list).mean()
        mse_mean = np.array(all_mse).mean()

        if self.args.wandb:
            wandb.log(
                {
                    f"Val Mean PSNR": psnr_mean,
                    f"Val Mean MSE": mse_mean,
                },
                commit=False,
            )

        print("===================MEAN PSRN/MSE====================\n\n")
        print(f"PSNR: {psnr_mean}, MSE: {mse_mean}")
        print("===================================\n\n")

        if self.args.mode == "eval":
            data = {"psnr_mean": psnr_mean, "mse_mean": mse_mean}
            os.makedirs(self.args.save_path, exist_ok=True)
            with open(os.path.join(self.args.save_path, "eval.json"), "w") as outfile:
                json.dump(data, outfile)

        return psnr_mean, mse_mean


def main_worker(proc, nprocs, args, gpu_ids, init_method):
    """
    Main worker function for multiprocessing.
    """

    # port = random.randint(10000, 20000)
    # init_method = f"tcp://127.0.0.1:{port}"
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        # init_method="tcp://127.0.0.1:17660",
        world_size=nprocs,
        rank=proc,
    )
    torch.cuda.set_device(gpu_ids[proc])

    logger = logging.getLogger(f"worker_{proc}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if args.log_to_file:
        log_dir = os.path.join(args.save_path, "log")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"worker_{proc}.log"))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    trainer = Trainer(args, proc, nprocs, gpu_ids[proc], logger)
    dist.barrier()
    print("ehereeee")
    print("args.mode", args.mode)
    if args.mode == "train":
        trainer.train_loop()
    elif args.mode == "eval":
        print("hereeeeeeeeeeeeee")
        trainer.eval(trainer.test_set)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s"
    )

    gpu_ids = []
    if args.gpus:
        for token in args.gpus.split(","):
            if "-" in token:
                start, end = token.split("-")
                gpu_ids.extend(range(int(start), int(end) + 1))
            else:
                gpu_ids.append(int(token))

    port = random.randint(
        1024, 65535
    )  # Generate a random port number within the range of unassigned ports
    init_method = f"tcp://127.0.0.1:{port}"
    print("init_method", init_method)

    if len(gpu_ids) <= 1:
        if len(gpu_ids) == 1:
            torch.cuda.set_device(gpu_ids[0])

        logger = None
        if args.log_to_file:
            log_dir = os.path.join(args.save_path, "log")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, f"worker_0.log"))
            file_handler.setFormatter(
                logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
            )
            file_handler.setLevel(logging.DEBUG)
            logger = logging.getLogger("worker_0")
            logger.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        trainer = Trainer(args, logger=logger)

        if args.mode == "train":
            trainer.train_loop()
        elif args.mode == "eval":
            trainer.eval(trainer.test_set)
    else:
        nprocs = len(gpu_ids)
        logging.info(f"Using {nprocs} processes for DDP, GPUs: {gpu_ids}")
        mp.spawn(
            main_worker,
            nprocs=nprocs,
            args=(nprocs, args, gpu_ids, init_method),
            join=True,
        )


if __name__ == "__main__":
    main()
