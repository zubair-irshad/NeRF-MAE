import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import MLP, Permute
import math
from functools import partial
import random
import sys

sys.path.append("..")
from einops import rearrange


# # ===========================
# sys.path.append("NeRF_MAE_internal")
# # ===========================

from nerf_mae.model.mae.torch_utils import *
from nerf_mae.model.mae.unetr_block import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock


# Swin Transformer FPN


def shifted_window_attention(  # changed to 3D
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int] = [128, 128, 128],
    num_heads: int = 4,
    shift_size: List[int] = [64, 64, 64],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias for 3D.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, H, W, D, C]): The input tensor or 5-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
    Returns:
        Tensor[B, H, W, D, C]: The output tensor after shifted window attention.
    """
    B, H, W, D, C = input.shape
    # pad feature maps to multiples of window size -> 3D
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_d = (window_size[2] - D % window_size[2]) % window_size[2]
    x = F.pad(input, (0, 0, 0, pad_d, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, pad_D, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window -> 3D
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0
    if window_size[2] >= pad_D:
        shift_size[2] = 0

    # cyclic shift -> 3D
    if sum(shift_size) > 0:
        x = torch.roll(
            x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
        )

    # partition windows -> 3D
    num_windows = (
        (pad_H // window_size[0])
        * (pad_W // window_size[1])
        * (pad_D // window_size[2])
    )
    x = x.view(
        B,
        pad_H // window_size[0],
        window_size[0],
        pad_W // window_size[1],
        window_size[1],
        pad_D // window_size[2],
        window_size[2],
        C,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        B * num_windows, window_size[0] * window_size[1] * window_size[2], C
    )  # B*nW, Ws*Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask #TODO: change to 3d
        attn_mask = x.new_zeros((pad_H, pad_W, pad_D))
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift_size[0]),
            (-shift_size[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift_size[1]),
            (-shift_size[1], None),
        )
        d_slices = (
            (0, -window_size[2]),
            (-window_size[2], -shift_size[2]),
            (-shift_size[2], None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1], d[0] : d[1]] = count
                    count += 1
        attn_mask = attn_mask.view(
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
            pad_D // window_size[2],
            window_size[2],
        )
        attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(
            num_windows, window_size[0] * window_size[1] * window_size[2]
        )
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        attn = attn.view(
            x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1)
        )
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows -> 3D
    x = x.view(
        B,
        pad_H // window_size[0],
        pad_W // window_size[1],
        pad_D // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        C,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, pad_H, pad_W, pad_D, C)

    # reverse cyclic shift -> 3D
    if sum(shift_size) > 0:
        x = torch.roll(
            x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3)
        )

    # unpad features -> 3D
    x = x[:, :H, :W, :D, :].contiguous()
    return x


def _get_relative_position_bias(  # changed to 3D
    relative_position_bias_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: List[int],
) -> torch.Tensor:
    N = window_size[0] * window_size[1] * window_size[2]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = (
        relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    )
    return relative_position_bias


class ShiftedWindowAttention(nn.Module):  # changed to 3D
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 3 or len(shift_size) != 3:
            raise ValueError("window_size and shift_size must be of length 3")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias -> 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1)
                * (2 * self.window_size[1] - 1)
                * (2 * self.window_size[2] - 1),
                self.num_heads,
            )
        )  # 2*Wh-1 * 2*Ww-1 * 2*Wd-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window -> 3D
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij")
        )  # 3, Wh, Ww, Wd
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wh*Ww*Wd, Wh*Ww*Wd
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[2] - 1) * (
            2 * self.window_size[1] - 1
        )  # problematic
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1  # problematic
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wd*Wh*Ww*Wd
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, D, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, D, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )


class SwinTransformerBlock(nn.Module):  # changed to 3D
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            dim,
            [int(dim * mlp_ratio), dim],
            activation_layer=nn.GELU,
            inplace=None,
            dropout=dropout,
        )

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):  # changed to 3D
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        expand_dim: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, dim * 2 if expand_dim else dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def _patch_merging_pad(self, x: torch.Tensor) -> torch.Tensor:  # changed to 3D
        H, W, D, _ = x.shape[-4:]
        x = F.pad(x, (0, 0, 0, D % 2, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x1 = x[..., 1::2, 0::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x2 = x[..., 0::2, 1::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x3 = x[..., 1::2, 1::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x4 = x[..., 0::2, 0::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x5 = x[..., 1::2, 0::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x6 = x[..., 0::2, 1::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x7 = x[..., 1::2, 1::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # ... H/2 W/2 D/2 8*C
        return x

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, D, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, D/2, C]
        """
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 D/2 2*C
        return x


class SwinTransformer_MAE3D(nn.Module):  # TODO: change to 3D
    """
    Implements the 3D Swin Transformer FPN.
    Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-5
        ),
        block: Optional[Callable[..., nn.Module]] = SwinTransformerBlock,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        expand_dim: bool = True,
        out_channels: int = 256,
        input_dim: int = 4,
        decoder_embed_dim: int = 768,
        masking_prob=0.50,
        resolution=160,
        masking_strategy=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.sampling_strategy = masking_strategy
        # split image into non-overlapping patches
        self.patch_size = patch_size
        self.masking_prob = masking_prob
        self.resolution = resolution
        self.patch_partition = nn.Sequential(
            nn.Conv3d(
                input_dim,
                embed_dim,
                kernel_size=(patch_size[0], patch_size[1], patch_size[2]),
                stride=(patch_size[0], patch_size[1], patch_size[2]),
            ),
            Permute([0, 2, 3, 4, 1]),
            norm_layer(embed_dim),
        )

        self.stages = nn.ModuleList()
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        fpn_in_channels = []

        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = nn.ModuleList()
            dim = embed_dim * 2**i_stage if expand_dim else embed_dim
            fpn_in_channels.append(dim)

            # add patch merging layer
            if i_stage > 0:
                input_dim = (
                    fpn_in_channels[-2] if len(fpn_in_channels) > 1 else embed_dim
                )
                stage.append(downsample_layer(input_dim, norm_layer, expand_dim))

            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            self.stages.append(nn.Sequential(*stage))

        self.num_patches = int(round(self.resolution // patch_size[0]))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, self.num_patches, self.num_patches, self.num_patches, embed_dim
            ),
            requires_grad=False,
        )

        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(4))

        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # self.decoder_layers = nn.Sequential(
        #     nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Upsample(size=(10, 10, 10), mode="trilinear", align_corners=False),
        #     nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv3d(128, out_channels, kernel_size=3, stride=1, padding=1),
        # )

        # if self.resolution == 160:
        #     size = (40, 40, 40)
        # else:
        #     size = (50, 50, 50)
        size = (40, 40, 40)
        self.decoder_layers = nn.Sequential(
            nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            # nn.Upsample(scale_factor=2),
            nn.Upsample(size=(10, 10, 10), mode="trilinear", align_corners=False),
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Upsample(size=(20, 20, 20), mode="trilinear", align_corners=False),
            # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Upsample(size=size, mode="trilinear", align_corners=False),
            nn.Conv3d(128, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        torch.nn.init.normal_(self.mask_token, std=0.02)

    # def window_masking_3d(
    #     self, x, patch_size=(4, 4, 4), p_remove=0.50, mask_token=None
    # ):
    #     batch_size, height, width, depth, channels = x.shape
    #     mask = torch.zeros((batch_size, height, width, depth, 1)).to(x.device)
    #     patch_h, patch_w, patch_d = patch_size

    #     patch_h, patch_w, patch_d = patch_size
    #     num_patches = (height // patch_h) * (width // patch_w) * (depth // patch_d)
    #     num_masked_patches = 0

    #     # p_keep = 1 - p_remove
    #     for h in range(0, height - patch_h + 1, patch_h):
    #         for w in range(0, width - patch_w + 1, patch_w):
    #             for d in range(0, depth - patch_d + 1, patch_d):
    #                 if random.random() < p_remove:
    #                     num_masked_patches += 1
    #                     mask[
    #                         :, h : h + patch_h, w : w + patch_w, d : d + patch_d, :
    #                     ] = 1

    #     masked_x = x.clone()
    #     masked_x[mask.bool().expand(-1, -1, -1, -1, channels)] = 0
    #     if mask_token is not None:
    #         mask_token = mask_token.to(masked_x.device)
    #         index_mask = (mask.bool()).squeeze(-1)
    #         masked_x[index_mask, :] = mask_token

    #     percent_masked = 100 * num_masked_patches / num_patches
    #     print(f"Total number of patches: {num_patches}")
    #     print(f"Total number of masked patches: {num_masked_patches}")
    #     print(f"Percentage of patches that are masked: {percent_masked:.2f}%")

    #     return masked_x, mask

    def window_masking_3d(
        self,
        x,
        patch_size=(4, 4, 4),
        p_remove=0.50,
        mask_token=None,
        sampling_strategy="random",
    ):
        batch_size, height, width, depth, channels = x.shape
        mask = torch.zeros((batch_size, height, width, depth, 1)).to(x.device)
        patch_h, patch_w, patch_d = patch_size

        patch_h, patch_w, patch_d = patch_size
        # num_patches = (height // patch_h) * (width // patch_w) * (depth // patch_d)
        # num_masked_patches = 0

        if sampling_strategy == "grid":
            # Calculate the number of patches in each dimension
            num_patches_h = height // patch_h
            num_patches_w = width // patch_w
            num_patches_d = depth // patch_d

            # Calculate the total number of patches
            num_patches = num_patches_h * num_patches_w * num_patches_d

            # Create a list of all patch indices
            all_patch_indices = [
                (h, w, d)
                for h in range(num_patches_h)
                for w in range(num_patches_w)
                for d in range(num_patches_d)
            ]

            # Iterate through the shuffled patch indices and keep only 'num_to_keep' patches
            num_masked_patches = 0
            count = 0
            for h, w, d in all_patch_indices:
                # we want to mask one out of every four patches

                if count == 0 or count == 1 or count == 2:
                    h_start, h_end = h * patch_h, (h + 1) * patch_h
                    w_start, w_end = w * patch_w, (w + 1) * patch_w
                    d_start, d_end = d * patch_d, (d + 1) * patch_d
                    mask[:, h_start:h_end, w_start:w_end, d_start:d_end, :] = 1
                    num_masked_patches += 1
                    count += 1
                else:
                    count += 1
                    if count == 4:
                        count = 0

        elif sampling_strategy == "block":
            # mask our 25% of the volume 3 times at random starting and ending locations
            # Calculate the number of patches in each dimension
            num_patches_h = height // patch_h
            num_patches_w = width // patch_w
            num_patches_d = depth // patch_d

            # Calculate the total number of patches
            num_patches = num_patches_h * num_patches_w * num_patches_d

            # Create a list of all patch indices
            all_patch_indices = [
                (h, w, d)
                for h in range(num_patches_h)
                for w in range(num_patches_w)
                for d in range(num_patches_d)
            ]

            num_to_keep = num_patches // 4  # Keep one out of every four patches
            total_masked_patches = 0
            for i in range(3):
                num_masked_patches = 0
                h_start = height // patch_h
                w_start = width // patch_w
                d_start = depth // patch_d

                h_start = np.random.randint(0, h_start - 0.25 * h_start)

                for h, w, d in all_patch_indices:
                    # select random strting locations for the 25% of the volume
                    if h > h_start:
                        h_start_new, h_end = h * patch_h, (h + 1) * patch_h
                        w_start_new, w_end = w * patch_w, (w + 1) * patch_w
                        d_start_new, d_end = d * patch_d, (d + 1) * patch_d

                        # check if the patch is already masked, if not mask it
                        if (
                            mask[
                                :,
                                h_start_new:h_end,
                                w_start_new:w_end,
                                d_start_new:d_end,
                                :,
                            ].sum()
                            == 0
                        ):
                            mask[
                                :,
                                h_start_new:h_end,
                                w_start_new:w_end,
                                d_start_new:d_end,
                                :,
                            ] = 1
                            num_masked_patches += 1

                    if num_masked_patches >= num_to_keep:
                        total_masked_patches += num_masked_patches
                        num_masked_patches = 0

                        break  # Stop masking patches once we've reached the desired number
            num_masked_patches = total_masked_patches

        elif sampling_strategy == "random":
            num_patches_h = height // patch_h
            num_patches_w = width // patch_w
            num_patches_d = depth // patch_d

            # Calculate the total number of patches
            num_patches = num_patches_h * num_patches_w * num_patches_d

            num_masked_patches = 0
            for h in range(0, height - patch_h + 1, patch_h):
                for w in range(0, width - patch_w + 1, patch_w):
                    for d in range(0, depth - patch_d + 1, patch_d):
                        if random.random() < p_remove:
                            num_masked_patches += 1
                            mask[
                                :, h : h + patch_h, w : w + patch_w, d : d + patch_d, :
                            ] = 1

        masked_x = x.clone()
        masked_x[mask.bool().expand(-1, -1, -1, -1, channels)] = 0
        if mask_token is not None:
            mask_token = mask_token.to(masked_x.device)
            index_mask = (mask.bool()).squeeze(-1)
            masked_x[index_mask, :] = mask_token

        # percent_masked = 100 * num_masked_patches / num_patches
        # print(f"Total number of patches: {num_patches}")
        # print(f"Total number of masked patches: {num_masked_patches}")
        # print(f"Percentage of patches that are masked: {percent_masked:.2f}%")

        return masked_x, mask

    # def window_masking_3d(
    #     self, x, patch_size=(4, 4, 4), p_remove=0.50, mask_token=None
    # ):
    #     batch_size, height, width, depth, channels = x.shape
    #     mask = torch.zeros((batch_size, height, width, depth, 1)).to(x.device)
    #     patch_h, patch_w, patch_d = patch_size

    #     patch_h, patch_w, patch_d = patch_size
    #     # num_patches = (height // patch_h) * (width // patch_w) * (depth // patch_d)
    #     # num_masked_patches = 0

    #     # p_keep = 1-p_remove
    #     for h in range(0, height - patch_h + 1, patch_h):
    #         for w in range(0, width - patch_w + 1, patch_w):
    #             for d in range(0, depth - patch_d + 1, patch_d):
    #                 if random.random() < p_remove:
    #                     # num_masked_patches += 1
    #                     mask[
    #                         :, h : h + patch_h, w : w + patch_w, d : d + patch_d, :
    #                     ] = 1

    #     masked_x = x.clone()
    #     masked_x[mask.bool().expand(-1, -1, -1, -1, channels)] = 0
    #     if mask_token is not None:
    #         mask_token = mask_token.to(masked_x.device)
    #         index_mask = (mask.bool()).squeeze(-1)
    #         masked_x[index_mask, :] = mask_token

    #     return masked_x, mask

    # def patchify_3d(self, x, mask):
    #     """
    #     rgbsimga: (N, 4, H, W, D)
    #     x: (N, L, L, L, patch_size**3 *4)
    #     """
    #     p = self.patch_size[0]
    #     assert x.shape[2] == x.shape[3] == x.shape[4] and x.shape[2] % p == 0
    #     h = w = l = int(round(x.shape[2] // p))
    #     x = x.reshape(shape=(x.shape[0], 4, h, p, w, p, l, p))
    #     x = rearrange(x, "n c h p w q l r -> n h w l p q r c")
    #     x = rearrange(x, "n h w l p q r c -> n h w l (p q r) c")

    #     mask = mask.reshape(shape=(x.shape[0], 4, h, p, w, p, l, p))
    #     mask = rearrange(mask, "n c h p w q l r -> n h w l p q r c")
    #     mask = rearrange(mask, "n h w l p q r c -> n h w l (p q r) c")
    #     mask = mask[:, :, :, :, :, 0]

    #     # x = torch.einsum("nchpwqlr->nhwlpqrc", x)
    #     # x = x.reshape(x.shape[0], h, w, l, p**34)
    #     # mask = mask.reshape(shape=(x.shape[0], 4, h, p, w, p, l, p))
    #     # mask = torch.einsum("nchpwqlr->nhwlpqrc", mask)
    #     # mask = mask.reshape(x.shape[0], h, w, l, p**3, 4)
    #     return x, mask
    
    def patchify_3d(self, x, mask=None):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """
        p = self.patch_size[0]
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

    def unpatchify_3d(self, x):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """

        p = self.patch_size[0]
        _, h, w, l, _ = x.shape

        x = x.reshape(shape=(x.shape[0], h, w, l, p**3, 4))
        return x

    def unpatchify_3d_full(self, x):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """
        p = self.patch_size[0]
        h = w = l = self.resolution

        x = x.reshape(shape=(x.shape[0], h, w, l, p, p, p, 4))
        x = rearrange(x, "n h w l p q r c -> n c h p w q l r")
        x = x.reshape(x.shape[0], 4, h * p, w * p, l * p)

        return x

    def transform(self, x):
        # new_res = [160, 160, 160]
        new_res = [
            self.resolution,
            self.resolution,
            self.resolution,
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

    def forward_encoder(self, x):
        # print("x", x.shape)
        # x_masked, mask = self.window_masking_3d(
        #     x, p_remove=self.masking_prob, mask_token=self.mask_token
        # )
        # print("x_masked, mask", x_masked.shape, mask.shape)

        x = self.patch_partition(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x, mask_patches = self.window_masking_3d(
            x,
            p_remove=self.masking_prob,
            mask_token=self.mask_token,
            sampling_strategy=self.sampling_strategy,
        )
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            # print("x after stage", i + 1, x.shape)
        return x, mask_patches

    def ce_loss(self, pred_alpha, target_alpha, mask):
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        loss = criterion(pred_alpha, target_alpha)
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
        return loss

    def forward_loss(self, x, pred, mask_batch, mask_patches, is_eval=False):
        # for both RGB and alpha, loss is enforced everywhere, not just where alpha > 0.01 and where mask patches
        target, mask_batch = self.patchify_3d(x, mask_batch)
        pred = self.patchify_3d(pred)
        # pred = self.unpatchify_3d(pred)
        # pred = self.unpatchify_3d(pred)

        # print("mask_batch", mask_batch.shape)
        # print("mask_patches", mask_patches.shape)

        mask_remove = mask_batch.squeeze(-1).int() * (mask_patches)
        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        pred_alpha = pred[..., 3].unsqueeze(-1)

        mask = target_alpha > 0.01

        mask_remove = mask_remove.unsqueeze(-1).int()

        # print("pred_rgb", pred_rgb.shape, target_rgb.shape, mask_remove.shape)
        loss_rgb = (pred_rgb - target_rgb) ** 2
        loss_rgb = (loss_rgb * mask).sum() / mask.sum()
        # loss_rgb = (loss_rgb * mask_remove).sum() / mask_remove.sum()

        # apply alpha_mask everywhere not just where alpha > 0.01
        # print("mask_batch", mask_patches.shape)

        # print("pred_alpha", pred_alpha.shape, target_alpha.shape)
        pred_alpha = self.alpha_activation(pred_alpha)
        loss_alpha = (pred_alpha - target_alpha) ** 2
        # loss_alpha = (loss_alpha * mask_batch).sum() / mask_batch.sum()
        loss_alpha = (loss_alpha * mask_remove).sum() / mask_remove.sum()
        # loss_alpha = 0

        # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
        loss = loss_rgb + loss_alpha

        if is_eval:
            return (
                loss,
                loss_rgb,
                loss_alpha,
                pred,
                mask,
                # mask_patches_out,
                # mask_remove_out,
                target,
            )
        else:
            return loss, loss_rgb, loss_alpha
        
    # def forward_loss(self, x, pred, mask, mask_patches, is_eval=False):
    #     target, mask = self.patchify_3d(x, mask)
    #     pred = self.unpatchify_3d(pred)

    #     target_rgb = target[..., :3]
    #     target_alpha = target[..., 3].unsqueeze(-1)

    #     # mask = mask.int() * mask_patches
    #     # mask = mask.unsqueeze(-1).int()

    #     pred_rgb = pred[..., :3]
    #     # pred_alpha = pred[..., 3].unsqueeze(-1)

    #     mask = target_alpha > 0.01

    #     # if is_eval:
    #     #     # mask_patches_out = mask_patches.detach().clone()
    #     #     # mask_remove_out = mask_remove.detach().clone()
    #     #     mask = mask.detach().clone()

    #     # mask_remove = mask_remove.squeeze(-1).int() * mask_patches
    #     # mask_remove = mask_remove.squeeze(-1).int()

    #     loss_rgb = (pred_rgb - target_rgb) ** 2
    #     loss_rgb = (loss_rgb * mask).sum() / mask.sum()

    #     # loss_alpha = (pred_alpha - target_alpha) ** 2
    #     # loss_alpha = (loss_rgb * mask).sum() / mask.sum()
    #     loss_alpha = 0

    #     # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
    #     loss = loss_rgb + loss_alpha
    #     # loss = loss_rgb
    #     # loss_alpha = 0

    #     if is_eval:
    #         return (
    #             loss,
    #             loss_rgb,
    #             loss_alpha,
    #             pred,
    #             mask,
    #             # mask_patches_out,
    #             # mask_remove_out,
    #             target,
    #         )
    #     else:
    #         return loss, loss_rgb, loss_alpha

    def forward_decoder(self, latent):
        latent = latent.permute(0, 4, 1, 2, 3)
        # x = latent
        # for i, layer in enumerate(self.decoder_layers):
        #     x = layer(x)
        #     print(f"Output after layer {i+1}: {x.shape}")
        out = self.decoder_layers(latent)
        out = out.permute(0, 2, 3, 4, 1)
        return out

    def forward(self, x, is_eval=False):
        x, mask = self.transform(x)
        x = torch.cat((x), dim=0)
        mask = torch.cat((mask), dim=0).to(x.device)
        latent, mask_patches = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        if is_eval:
            (
                loss,
                loss_rgb,
                loss_alpha,
                pred_rgb,
                mask,
                target_rgb,
            ) = self.forward_loss(x, pred, mask, mask_patches, is_eval)
            return (
                loss,
                loss_rgb,
                loss_alpha,
                pred_rgb,
                mask,
                mask_patches,
                target_rgb,
            )
        else:
            loss, loss_rgb, loss_alpha = self.forward_loss(
                x, pred, mask, mask_patches, is_eval
            )
            return loss, loss_rgb, loss_alpha


class SwinTransformer_MAE3D_New(nn.Module):  # TODO: change to 3D
    """
    Implements the 3D Swin Transformer FPN.
    Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-5
        ),
        block: Optional[Callable[..., nn.Module]] = SwinTransformerBlock,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        expand_dim: bool = True,
        out_channels: int = 4,
        input_ch_dim: int = 4,
        decoder_embed_dim: int = 768,
        masking_prob=0.50,
        resolution=160,
        drop_rate=0.10,
        masking_strategy="random",
    ):
        super().__init__()
        self.out_channels = out_channels
        # split image into non-overlapping patches
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.masking_prob = masking_prob
        self.resolution = resolution
        self.patch_partition = nn.Sequential(
            nn.Conv3d(
                input_ch_dim,
                embed_dim,
                kernel_size=(patch_size[0], patch_size[1], patch_size[2]),
                stride=(patch_size[0], patch_size[1], patch_size[2]),
            ),
            Permute([0, 2, 3, 4, 1]),
            norm_layer(embed_dim),
        )

        self.stages = nn.ModuleList()
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        fpn_in_channels = []

        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = nn.ModuleList()
            dim = embed_dim * 2**i_stage if expand_dim else embed_dim
            fpn_in_channels.append(dim)

            # add patch merging layer
            if i_stage > 0:
                input_dim = (
                    fpn_in_channels[-2] if len(fpn_in_channels) > 1 else embed_dim
                )
                stage.append(downsample_layer(input_dim, norm_layer, expand_dim))

            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            self.stages.append(nn.Sequential(*stage))

        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.encoder2 = UnetrBasicBlock(
        #     in_channels=embed_dim,
        #     out_channels=embed_dim,
        #     kernel_size=3,
        #     stride=1,
        #     res_block=True,
        # )

        # print("input_dim", input_dim)
        # print("embed_dim", embed_dim)
        # self.encoder1 = UnetrBasicBlock(
        #     in_channels=input_ch_dim,
        #     out_channels=embed_dim // 2,
        #     kernel_size=3,
        #     stride=1,
        #     res_block=True,
        # )

        # self.encoder2 = UnetrBasicBlock(
        #     in_channels=embed_dim,
        #     out_channels=embed_dim,
        #     kernel_size=3,
        #     stride=1,
        #     res_block=True,
        # )

        # self.encoder3 = UnetrBasicBlock(
        #     in_channels=2 * embed_dim,
        #     out_channels=2 * embed_dim,
        #     kernel_size=3,
        #     stride=1,
        #     res_block=True,
        # )

        # self.encoder4 = UnetrBasicBlock(
        #     in_channels=4 * embed_dim,
        #     out_channels=4 * embed_dim,
        #     kernel_size=3,
        #     stride=1,
        #     res_block=True,
        # )
        # self.encoder5 = UnetrBasicBlock(
        #     in_channels=8 * embed_dim,
        #     out_channels=8 * embed_dim,
        #     kernel_size=3,
        #     stride=1,
        #     res_block=True,
        # )

        self.decoder4 = UnetrUpBlock(
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            in_channels=embed_dim * 1,
            out_channels=embed_dim // 2,
            kernel_size=3,
            upsample_kernel_size=4,
            res_block=True,
            use_skip=False,
        )

        self.out = UnetOutBlock(in_channels=embed_dim // 2, out_channels=out_channels)

        self.num_patches = int(round(self.resolution // patch_size[0]))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, self.num_patches, self.num_patches, self.num_patches, embed_dim
            ),
            requires_grad=False,
        )

        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(4))

        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.resolution == 160:
            size = (40, 40, 40)
        else:
            size = (50, 50, 50)

        self.alpha_activation = nn.Sigmoid()
        # self.decoder_layers = nn.Sequential(
        #     nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        #     # nn.Upsample(scale_factor=2),
        #     nn.Upsample(size=(10, 10, 10), mode="trilinear", align_corners=False),
        #     nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Upsample(scale_factor=2),
        #     nn.Upsample(size=(20, 20, 20), mode="trilinear", align_corners=False),
        #     # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        #     nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Upsample(scale_factor=2),
        #     nn.Upsample(size=size, mode="trilinear", align_corners=False),
        #     nn.Conv3d(128, out_channels, kernel_size=3, stride=1, padding=1),
        # )

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        torch.nn.init.normal_(self.mask_token, std=0.02)

    def window_masking_3d(
        self,
        x,
        patch_size=(4, 4, 4),
        p_remove=0.50,
        mask_token=None,
        sampling_strategy="random",
    ):
        batch_size, height, width, depth, channels = x.shape
        mask = torch.zeros((batch_size, height, width, depth, 1)).to(x.device)
        patch_h, patch_w, patch_d = patch_size

        patch_h, patch_w, patch_d = patch_size
        # num_patches = (height // patch_h) * (width // patch_w) * (depth // patch_d)
        # num_masked_patches = 0

        if sampling_strategy == "grid":
            # Calculate the number of patches in each dimension
            num_patches_h = height // patch_h
            num_patches_w = width // patch_w
            num_patches_d = depth // patch_d

            # Calculate the total number of patches
            num_patches = num_patches_h * num_patches_w * num_patches_d

            # Calculate the number of patches to keep
            num_to_keep = num_patches // 4  # Keep one out of every four patches

            # Create a list of all patch indices
            all_patch_indices = [
                (h, w, d)
                for h in range(num_patches_h)
                for w in range(num_patches_w)
                for d in range(num_patches_d)
            ]

            # Shuffle the list of patch indices randomly
            # random.shuffle(all_patch_indices)

            # Iterate through the shuffled patch indices and keep only 'num_to_keep' patches
            num_masked_patches = 0
            for h, w, d in all_patch_indices:
                if num_masked_patches >= num_to_keep:
                    break  # Stop masking patches once we've reached the desired number
                h_start, h_end = h * patch_h, (h + 1) * patch_h
                w_start, w_end = w * patch_w, (w + 1) * patch_w
                d_start, d_end = d * patch_d, (d + 1) * patch_d
                mask[:, h_start:h_end, w_start:w_end, d_start:d_end, :] = 1
                num_masked_patches += 1

        elif sampling_strategy == "random":
            # p_keep = 1-p_remove
            for h in range(0, height - patch_h + 1, patch_h):
                for w in range(0, width - patch_w + 1, patch_w):
                    for d in range(0, depth - patch_d + 1, patch_d):
                        if random.random() < p_remove:
                            # num_masked_patches += 1
                            mask[
                                :, h : h + patch_h, w : w + patch_w, d : d + patch_d, :
                            ] = 1

        masked_x = x.clone()
        masked_x[mask.bool().expand(-1, -1, -1, -1, channels)] = 0
        if mask_token is not None:
            mask_token = mask_token.to(masked_x.device)
            index_mask = (mask.bool()).squeeze(-1)
            masked_x[index_mask, :] = mask_token

        return masked_x, mask

    def patchify_3d(self, x, mask=None):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """
        p = self.patch_size[0]
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

    def unpatchify_3d(self, x):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """

        p = self.patch_size[0]
        _, h, w, l, _ = x.shape

        x = x.reshape(shape=(x.shape[0], h, w, l, p**3, 4))
        return x

    def unpatchify_3d_full(self, x):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """
        p = self.patch_size[0]
        h = w = l = self.resolution

        x = x.reshape(shape=(x.shape[0], h, w, l, p, p, p, 4))
        x = rearrange(x, "n h w l p q r c -> n c h p w q l r")
        x = x.reshape(x.shape[0], 4, h * p, w * p, l * p)

        return x

    def transform(self, x):
        # new_res = [160, 160, 160]
        new_res = [
            self.resolution,
            self.resolution,
            self.resolution,
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

    def forward_encoder_ecoder(self, x):
        # print("x", x.shape)

        # print("enc0", self.encoder1(x).shape)
        # enc0 = self.encoder1(x)
        x = self.patch_partition(x)
        # print("x after patch partition", x.shape)

        # x = self.pos_drop(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        # print("x", x.shape)
        x, mask_patches = self.window_masking_3d(
            x, p_remove=self.masking_prob, mask_token=self.mask_token
        )
        # print("x after masking", x.shape)
        features = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            # print("x after stage", i + 1, x.shape)

            features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous())
            # print("features permute", torch.permute(x, [0, 4, 1, 2, 3]).contiguous().shape)

        # print("features", features[0].shape, features[1].shape, features[2].shape, features[3].shape)

        # we need to output enc1, enc2, enc3, enc4 and
        # enc1 = self.encoder2(features[0])
        # enc2 = self.encoder3(features[1])
        # enc3 = self.encoder4(features[2])

        # dec4 = self.encoder5(features[3])
        # dec3 = self.decoder4(dec4, enc3)
        # dec2 = self.decoder3(dec3, enc2)
        # dec1 = self.decoder2(dec2, enc1)

        # dec4 = self.encoder5(features[3])
        dec3 = self.decoder4(features[3], features[2])
        dec2 = self.decoder3(dec3, features[1])
        dec1 = self.decoder2(dec2, features[0])

        # dec0 = self.decoder1(dec1, enc0)
        dec0 = self.decoder1(dec1)

        # print("dec0", dec0.shape)

        out = self.out(dec0)

        # print(
        #     "dec3 2 1 0 out", dec3.shape, dec2.shape, dec1.shape, dec0.shape, out.shape
        # )

        # print("out", out.shape)
        # output_features = [enc1, enc2, enc3, dec4, features[3]]
        # # print("encoder3", self.encoder3(features[2]).shape)
        # # return x, mask_patches
        return out, mask_patches

    # def ce_loss(self, pred_alpha, target_alpha, mask):
    #     criterion = nn.BCEWithLogitsLoss(reduction="none")
    #     loss = criterion(pred_alpha, target_alpha)
    #     loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
    #     return loss

    def forward_loss(self, x, pred, mask_batch, mask_patches, is_eval=False):
        # for both RGB and alpha, loss is enforced everywhere, not just where alpha > 0.01 and where mask patches
        target, mask_batch = self.patchify_3d(x, mask_batch)
        pred = self.patchify_3d(pred)
        # pred = self.unpatchify_3d(pred)

        # print("mask_batch", mask_batch.shape)
        # print("mask_patches", mask_patches.shape)

        mask_remove = mask_batch.squeeze(-1).int() * (mask_patches)
        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        pred_alpha = pred[..., 3].unsqueeze(-1)

        mask = target_alpha > 0.01

        mask_remove = mask_remove.unsqueeze(-1).int()

        # print("pred_rgb", pred_rgb.shape, target_rgb.shape, mask_remove.shape)
        loss_rgb = (pred_rgb - target_rgb) ** 2
        loss_rgb = (loss_rgb * mask).sum() / mask.sum()
        # loss_rgb = (loss_rgb * mask_remove).sum() / mask_remove.sum()

        # apply alpha_mask everywhere not just where alpha > 0.01
        # print("mask_batch", mask_patches.shape)

        # print("pred_alpha", pred_alpha.shape, target_alpha.shape)
        pred_alpha = self.alpha_activation(pred_alpha)
        loss_alpha = (pred_alpha - target_alpha) ** 2
        # loss_alpha = (loss_alpha * mask_batch).sum() / mask_batch.sum()
        loss_alpha = (loss_alpha * mask_remove).sum() / mask_remove.sum()
        # loss_alpha = 0

        # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
        loss = loss_rgb + loss_alpha

        if is_eval:
            return (
                loss,
                loss_rgb,
                loss_alpha,
                pred,
                mask,
                # mask_patches_out,
                # mask_remove_out,
                target,
            )
        else:
            return loss, loss_rgb, loss_alpha

    def forward_decoder(self, latent):
        latent = latent.permute(0, 4, 1, 2, 3)
        out = self.decoder_layers(latent)
        out = out.permute(0, 2, 3, 4, 1)
        return out

    def forward(self, x, is_eval=False):
        x, mask = self.transform(x)
        x = torch.cat((x), dim=0)
        mask = torch.cat((mask), dim=0).to(x.device)
        # out_feats = self.forward_encoder(x)
        # pred = self.forward_decoder(out_feats)
        pred, mask_patches = self.forward_encoder_ecoder(x)
        if is_eval:
            (
                loss,
                loss_rgb,
                loss_alpha,
                pred_rgb,
                mask,
                target_rgb,
            ) = self.forward_loss(x, pred, mask, mask_patches, is_eval)
            return (
                loss,
                loss_rgb,
                loss_alpha,
                pred_rgb,
                mask,
                target_rgb,
            )
        else:
            loss, loss_rgb, loss_alpha = self.forward_loss(
                x, pred, mask, mask_patches, is_eval
            )
            return loss, loss_rgb, loss_alpha

if __name__ == "__main__":
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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num_params", num_params)

    # num_decoder_params = sum(
    #     p.numel() for p in model.decoder_layers.parameters() if p.requires_grad
    # )
    # print("num_decoder_params", num_decoder_params)

    input_rgb_sigma = []
    input_rgb_sigma.append(torch.randn((4, 130, 130, 140)))
    input_rgb_sigma.append(torch.randn((4, 150, 150, 131)))
    # grid = torch.randn((2, 4, 160, 160, 160))

    loss, loss_rgb, loss_alpha = model(input_rgb_sigma)

    print("loss", loss_rgb, loss_alpha)
    # print("pred", pred.shape, mask.shape)
