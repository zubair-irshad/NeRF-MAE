import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import MLP, Permute
import math
from functools import partial
import sys
import os


# from metrics import *

from .fpn import FPN
from .metrics import *

import sys
from einops import rearrange

sys.path.append("..")


# from fpn import FPN
# from metrics import mse, psnr, pixel_accuracy, intersection_over_union
from nerf_mae.model.mae.swin_mae3d import SwinTransformer_MAE3D
from nerf_mae.model.mae.unetr_block import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

from nerf_mae.model.mae.swin_mae3d import SwinTransformer_MAE3D_New
from nerf_mae.model.mae.torch_utils import *

# from monai.losses import DiceCELoss, MaskedLoss

# from monai.transforms import AsDiscrete
# from monai.data import decollate_batch
# import segmentation_models_pytorch as smp


# Simplified 3D Residual Block
class ResidualBlockSimplified(nn.Module):
    """The simplified Basic Residual block of ResNet."""

    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            num_channels, num_channels, kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv3d(
            num_channels, num_channels, kernel_size=3, padding=1, stride=1
        )
        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y += X
        return F.relu(Y)


# ResNet Bottleneck for 3D convolution
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )  # change
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False  # change
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


# ResNet_FPN
class ResNet_FPN_64(nn.Module):
    """A smaller backbone for 64^3 inputs."""

    # block: the type of ResNet layer
    # layers: the depth of each size of layers, i.e. the num of layers before the next
    def __init__(self, block, layers, input_dim=4, use_fpn=True):
        super(ResNet_FPN_64, self).__init__()
        self.in_planes = 16
        self.out_channels = 64
        self.conv1 = nn.Conv3d(
            input_dim, 16, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(16)
        # Bottom-up layers
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        # Top layer
        self.toplayer = nn.Conv3d(
            512, 64, kernel_size=1, stride=1, padding=0
        )  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv3d(256, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv3d(
            64, self.out_channels, kernel_size=1, stride=1, padding=0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, X, Y, Z = y.size()
        return (
            F.interpolate(x, size=(X, Y, Z), mode="trilinear", align_corners=True) + y
        )

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        # c1 = F.max_pool3d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5


class ResNet_FPN_256(nn.Module):
    # block: the type of ResNet layer
    # layers: the depth of each size of layers, i.e. the num of layers before the next

    """
    Args:
        layers: list of int. Its size could be variable. The length will be the ouput
                length. The value is the depth of layers at that level
        is_max_pool: If it is False, the network will not use downsample

    Returns (of self.forward function):
        A feature list. Its size is equal to the size of self.layers.
    """

    def __init__(self, block, layers, input_dim=4, is_max_pool=False):
        super(ResNet_FPN_256, self).__init__()
        self.in_planes = 64
        self.out_channels = 256
        self.conv1 = nn.Conv3d(
            input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)

        # Bottom-up layers
        self.layers = nn.ModuleList()
        self.start_deep = self.in_planes
        self.is_max_pool = is_max_pool
        for i in range(len(layers)):
            self.layers.append(
                self._make_layer(
                    block,
                    self.start_deep * (2**i),
                    layers[i],
                    stride=1 if i == 0 else 2,
                )
            )

        # Smooth layers
        self.smooths = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.smooths.append(nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1))

        # Lateral layers
        self.latlayers = nn.ModuleList()
        for i in range(len(layers) - 1, -1, -1):
            self.latlayers.append(
                nn.Conv3d(
                    block.expansion * self.start_deep * (2**i),
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, X, Y, Z = y.size()
        return F.interpolate(x, size=(X, Y, Z), mode="nearest") + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        if self.is_max_pool:
            c1 = F.max_pool3d(c1, kernel_size=3, stride=2, padding=1)
        c_out = [c1]
        for i in range(len(self.layers)):
            c_out.append(self.layers[i](c_out[i]))

        # Top-down
        p5 = self.latlayers[0](c_out[-1])
        p_out = [p5]
        for i in range(len(self.latlayers) - 1):
            p_out.append(
                self._upsample_add(p_out[i], self.latlayers[i + 1](c_out[-2 - i]))
            )

        # Smooth
        for i in range(len(self.smooths)):
            p_out[i + 1] = self.smooths[i](p_out[i + 1])

        p_out.reverse()
        return p_out


# Simplified ResNet (for debug)
class ResNetSimplified_64(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=7, stride=1, padding=3
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.residuals = nn.ModuleList()
        for i in range(num_residuals):
            self.residuals.append(ResidualBlockSimplified(out_channels))

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        for i in range(len(self.residuals)):
            Y = self.residuals[i](Y)
        return (Y,)


class ResNetSimplified_256(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.residuals = nn.ModuleList()
        for i in range(num_residuals):
            self.residuals.append(ResidualBlockSimplified(out_channels))

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.pool1(Y)
        for i in range(len(self.residuals)):
            Y = self.residuals[i](Y)
        return (Y,)


# VGG_FPN
vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
    "AF": [64, 128, "F", 256, 256, "M", "F", 512, 512, "M", "F", 512, 512, "M", "F"],
    "DF": [
        64,
        64,
        128,
        128,
        "F",
        256,
        256,
        256,
        "M",
        "F",
        512,
        512,
        512,
        "M",
        "F",
        512,
        512,
        512,
        "M",
        "F",
    ],
    "EF": [
        64,
        64,
        128,
        128,
        "F",
        256,
        256,
        256,
        256,
        "M",
        "F",
        512,
        512,
        512,
        512,
        "M",
        "F",
        512,
        512,
        512,
        512,
        "M",
        "F",
    ],
}


class VGG_FPN(nn.Module):
    def __init__(
        self,
        cfg: str = "AF",
        in_channels: int = 4,
        batch_norm: bool = True,
        input_size: int = 256,
        conv_at_start: bool = False,
    ):
        """VGG-FPN backbone.
        Args:
            cfg (str): Config name of the VGG-FPN.
            in_channels (int): Number of input channels.
            batch_norm (bool): Use batch normalization.
            feature_size (int): The largest side length of input grid. If the input_size>=200, the network will downsmaple it.
            conv_at_start (bool): Use conv layer at the start of the network before first downsampling.
        """
        super().__init__()
        self.out_channels = 256
        _in_channels = in_channels if not conv_at_start else 32
        self.layers = self.make_layers(
            vgg_cfgs[cfg], _in_channels, batch_norm, input_size
        )
        self.fpn_neck = FPN([128, 256, 512, 512], self.out_channels, 4)

        self.conv_at_start = conv_at_start
        self.starting_layers = None
        self.ds_layers = None
        if self.conv_at_start:
            self.starting_layers = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
            )

            self.ds_layers = nn.Sequential(
                nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 128, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
            )

    def make_layers(
        self, cfg: List[Union[str, int]], in_channels, batch_norm, input_size
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        curr_layer: List[nn.Module] = []
        _in_channels = in_channels
        if input_size >= 160:
            layers += [
                nn.Conv3d(_in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            ]
        else:
            layers += [
                nn.Conv3d(_in_channels, 64, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
            ]
        _in_channels = 64
        for v in cfg:
            if v == "M":
                curr_layer += [nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)]
            elif v == "F":
                layers += [nn.Sequential(*curr_layer)]
                curr_layer = []
            else:
                v = cast(int, v)
                conv3d = nn.Conv3d(_in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    curr_layer += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    curr_layer += [conv3d, nn.ReLU(inplace=True)]
                _in_channels = v

        return nn.Sequential(*layers)

    def forward(self, X):
        features = []

        X_ds = None
        if self.conv_at_start:
            X = self.starting_layers(X)
            X_ds = self.ds_layers(X)

        for i in range(len(self.layers)):
            X = self.layers[i](X)
            features.append(X)

        if self.conv_at_start:
            features[-4] = features[-4] + X_ds

        return self.fpn_neck(features[-4:])


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


class SwinTransformer_FPN(nn.Module):  # TODO: change to 3D
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
    ):
        super().__init__()
        self.out_channels = out_channels
        # split image into non-overlapping patches
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

        self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features: List[torch.Tensor] = []
        x = self.patch_partition(x)
        # print("x after patch partition: ", x.shape, "\n")
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            features.append(
                torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            )  # [N, C, H, W, D]

        # print("==========================================Inside SwinTransformerFPN", "\n\n\n")
        # print("features: ", [f.shape for f in features], "\n")



        features = self.fpn_neck(features)

        # print("features after FPN: ", [f.shape for f in features], "\n")
        return features


class SwinTransformer_FPN_Pretrained_Skip(nn.Module):  # TODO: change to 3D
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
        expand_dim: bool = True,
        out_channels: int = 256,
        resolution=160,
        checkpoint_path=None,
        is_eval=False,
    ):
        super().__init__()

        # Load the checkpoint from the pretrained SwinTransformer_MAE3D model
        self.out_channels = out_channels
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

        depths = swin[backbone_type]["depths"]
        embed_dim = swin[backbone_type]["embed_dim"]

        model = SwinTransformer_MAE3D_New(
            patch_size=[4, 4, 4],
            embed_dim=swin[backbone_type]["embed_dim"],
            depths=swin[backbone_type]["depths"],
            num_heads=swin[backbone_type]["num_heads"],
            window_size=[4, 4, 4],
            stochastic_depth_prob=0.1,
            expand_dim=True,
            resolution=resolution,
        )

        print(
            "==============================is EVAL==================================\n\n\n\n"
        )
        print(f"is eval: {is_eval}.")

        if not is_eval:
            assert os.path.exists(checkpoint_path), "The checkpoint does not exist."

            print(
                "================================================================\n\n\n\n"
            )
            print(f"Loading MAE checkpoint from {checkpoint_path}.")
            print(
                "================================================================\n\n\n\n"
            )

            # print(f"Loading checkpoint from {checkpoint_path}.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        del model.decoder4
        del model.decoder3
        del model.decoder2
        del model.decoder1
        del model.out
        del model.mask_token

        fpn_in_channels = []

        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            dim = embed_dim * 2**i_stage if expand_dim else embed_dim
            fpn_in_channels.append(dim)

        # Add the FPN neck on top of the SwinTransformer base
        self.base = model
        self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

    def forward(self, x):
        # Forward pass through the SwinTransformer base
        features: List[torch.Tensor] = []
        x = self.base.patch_partition(x)
        x = x + self.base.pos_embed.type_as(x).to(x.device).clone().detach()
        for i in range(len(self.base.stages)):
            x = self.base.stages[i](x)
            features.append(
                torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            )  # [N, C, H, W, D]
        features = self.fpn_neck(features)
        return features


class SwinTransformer_FPN_Pretrained(nn.Module):  # TODO: change to 3D
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
        expand_dim: bool = True,
        out_channels: int = 256,
        resolution=160,
        checkpoint_path=None,
        is_eval=False,
    ):
        super().__init__()

        # Load the checkpoint from the pretrained SwinTransformer_MAE3D model
        self.out_channels = out_channels
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

        depths = swin[backbone_type]["depths"]
        embed_dim = swin[backbone_type]["embed_dim"]

        model = SwinTransformer_MAE3D(
            patch_size=[4, 4, 4],
            embed_dim=swin[backbone_type]["embed_dim"],
            depths=swin[backbone_type]["depths"],
            num_heads=swin[backbone_type]["num_heads"],
            window_size=[4, 4, 4],
            stochastic_depth_prob=0.1,
            expand_dim=True,
            resolution=resolution,
        )

        print(
            "==============================is EVAL==================================\n\n\n\n"
        )
        print(f"is eval: {is_eval}.")

        if not is_eval:
            assert os.path.exists(checkpoint_path), "The checkpoint does not exist."

            print(
                "================================================================\n\n\n\n"
            )
            print(f"Loading MAE checkpoint from {checkpoint_path}.")
            print(
                "================================================================\n\n\n\n"
            )

            # print(f"Loading checkpoint from {checkpoint_path}.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        # Remove the decoder_layers from the pretrained model
        del model.decoder_layers
        del model.mask_token

        fpn_in_channels = []

        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            dim = embed_dim * 2**i_stage if expand_dim else embed_dim
            fpn_in_channels.append(dim)

        # Add the FPN neck on top of the SwinTransformer base
        self.base = model
        self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

    def forward(self, x):
        # Forward pass through the SwinTransformer base
        features: List[torch.Tensor] = []
        x = self.base.patch_partition(x)
        x = x + self.base.pos_embed.type_as(x).to(x.device).clone().detach()
        for i in range(len(self.base.stages)):
            x = self.base.stages[i](x)
            features.append(
                torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            )  # [N, C, H, W, D]
        features = self.fpn_neck(features)
        return features


class SwinTransformer_VoxelSR(nn.Module):  # TODO: change to 3D
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
        resolution=160,
        out_resolution=256,
        decoder_embed_dim: int = 768,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.patch_size = patch_size
        self.input_resolution = resolution
        self.output_resolution = out_resolution
        # split image into non-overlapping patches
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

        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.output_resolution == 256:
            final_upsample_scale = 1.6
        elif self.output_resolution == 384:
            final_upsample_scale = 2.4

        self.super_resolution_decoder = nn.Sequential(
            # Decoder Layer 1
            nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 512, 10, 10, 10]
            # Decoder Layer 2
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 256, 20, 20, 20]
            # Decoder Layer 3
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 128, 40, 40, 40]
            # Additional Convolutional Layer
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(
                scale_factor=final_upsample_scale
            ),  # Output: [2, 64, 64, 64, 64]
            # Final Convolutional Layer
            nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def patchify_3d(self, x):
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

    def transform(self, x, resolution=160):
        # new_res = [160, 160, 160]

        new_res = [
            resolution,
            resolution,
            resolution,
        ]
        # new_res = [200, 200, 200]
        # masks = []
        padded_rgbsigma = []
        for rgbsimga in x:
            rgb_sigma_padded, _ = pad_tensor(
                rgbsimga, target_shape=new_res, pad_value=0
            )
            padded_rgbsigma.append(rgb_sigma_padded)
            # masks.append(mask_rgbsigma)
        return padded_rgbsigma

    def forward_loss(self, x, pred, is_eval=False):
        x = self.transform(x, resolution=self.output_resolution)
        x = torch.cat((x), dim=0)

        target = self.patchify_3d(x)
        pred = self.unpatchify_3d(pred)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        mask_remove = target_alpha > 0.01

        # mask_remove = mask_remove.squeeze(-1).int() * mask_patches
        mask_remove = mask_remove.squeeze(-1).int()
        loss_rgb = (pred_rgb - target_rgb) ** 2
        loss_rgb = (loss_rgb * mask_remove.unsqueeze(-1)).sum() / mask_remove.sum()
        # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
        # loss = loss_rgb + loss_alpha
        return loss_rgb

    def output_metrics(self, x, pred):
        x = self.transform(x, resolution=self.output_resolution)
        x = torch.cat((x), dim=0)
        target = self.patchify_3d(x)
        pred = self.unpatchify_3d(pred)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        mask_remove = target_alpha > 0.01

        value = mse(pred_rgb, target_rgb, valid_mask=mask_remove)
        psnr_value = psnr(pred_rgb, target_rgb, valid_mask=mask_remove)

        metrics = {}
        metrics["MSE"] = value.item()
        metrics["PSNR"] = psnr_value.item()

        return metrics

    def forward_decoder(self, latent):
        latent = latent.permute(0, 4, 1, 2, 3)
        out = self.super_resolution_decoder(latent)
        out = out.permute(0, 2, 3, 4, 1)
        return out

    def forward(self, x):
        # features: List[torch.Tensor] = []

        x = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        x = self.patch_partition(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            # features.append(
            #     torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            # )  # [N, C, H, W, D]

        pred = self.forward_decoder(x)
        # loss_rgb = self.forward_loss(original_x, pred)
        return pred
        # features = self.fpn_neck(features)
        # return features

    def loss_fn(self, x, pred):
        loss_rgb = self.forward_loss(x, pred)
        return loss_rgb




class SwinTransformer_VoxelSR_Skip(nn.Module):  # TODO: change to 3D
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
        resolution=160,
        out_resolution=256,
        decoder_embed_dim: int = 768,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.patch_size = patch_size
        self.input_resolution = resolution
        self.output_resolution = out_resolution
        # split image into non-overlapping patches
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

        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.output_resolution == 256:
            final_upsample_scale = 1.6
        elif self.output_resolution == 384:
            final_upsample_scale = 2.4

        self.encoder1 = UnetrBasicBlock(
            in_channels=4,
            out_channels=embed_dim // 2,
            kernel_size=3,
            stride=1,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            in_channels=embed_dim * 1,
            out_channels=embed_dim // 2,
            kernel_size=3,
            upsample_kernel_size=4,
            res_block=True,
            use_skip=True,
        )

        self.final_upsample = nn.Upsample(scale_factor=final_upsample_scale)
        self.voxel_out = UnetOutBlock(in_channels=embed_dim // 2, out_channels=4)
        # self.sem_out = UnetOutBlock(
        #     in_channels=embed_dim // 2, out_channels=self.out_channels
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

        # self.super_resolution_decoder = nn.Sequential(
        #     # Decoder Layer 1
        #     nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 512, 10, 10, 10]
        #     # Decoder Layer 2
        #     nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 256, 20, 20, 20]
        #     # Decoder Layer 3
        #     nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 128, 40, 40, 40]
        #     # Additional Convolutional Layer
        #     nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(
        #         scale_factor=final_upsample_scale
        #     ),  # Output: [2, 64, 64, 64, 64]
        #     # Final Convolutional Layer
        #     nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1),
        # )

    def patchify_3d(self, x):
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

    def transform(self, x, resolution=160):
        # new_res = [160, 160, 160]

        new_res = [
            resolution,
            resolution,
            resolution,
        ]
        # new_res = [200, 200, 200]
        # masks = []
        padded_rgbsigma = []
        for rgbsimga in x:
            rgb_sigma_padded, _ = pad_tensor(
                rgbsimga, target_shape=new_res, pad_value=0
            )
            padded_rgbsigma.append(rgb_sigma_padded)
            # masks.append(mask_rgbsigma)
        return padded_rgbsigma

    def forward_loss(self, x, pred, is_eval=False):
        x = self.transform(x, resolution=self.output_resolution)
        x = torch.cat((x), dim=0)

        target = x.permute(0, 2, 3, 4, 1)
        pred = pred.permute(0, 2, 3, 4, 1)

        # target = self.patchify_3d(x)
        # pred = self.unpatchify_3d(pred)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        mask_remove = target_alpha > 0.01

        # mask_remove = mask_remove.squeeze(-1).int() * mask_patches
        mask_remove = mask_remove.squeeze(-1).int()
        loss_rgb = (pred_rgb - target_rgb) ** 2
        loss_rgb = (loss_rgb * mask_remove.unsqueeze(-1)).sum() / mask_remove.sum()
        # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
        # loss = loss_rgb + loss_alpha
        return loss_rgb

    def output_metrics(self, x, pred):
        x = self.transform(x, resolution=self.output_resolution)
        x = torch.cat((x), dim=0)
        # target = self.patchify_3d(x)
        # pred = self.unpatchify_3d(pred)

        target = x.permute(0, 2, 3, 4, 1)
        pred = pred.permute(0, 2, 3, 4, 1)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        mask_remove = target_alpha > 0.01

        value = mse(pred_rgb, target_rgb, valid_mask=mask_remove)
        psnr_value = psnr(pred_rgb, target_rgb, valid_mask=mask_remove)

        metrics = {}
        metrics["MSE"] = value.item()
        metrics["PSNR"] = psnr_value.item()

        return metrics

    # def forward_decoder(self, latent):
    #     latent = latent.permute(0, 4, 1, 2, 3)
    #     out = self.super_resolution_decoder(latent)
    #     out = out.permute(0, 2, 3, 4, 1)
    #     return out

    def forward(self, x):
        # features: List[torch.Tensor] = []

        x = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)

        enc1 = self.encoder1(x)

        x = self.patch_partition(x)

        features = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous())
            # features.append(
            #     torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            # )  # [N, C, H, W, D]

        dec3 = self.decoder4(features[3], features[2])
        dec2 = self.decoder3(dec3, features[1])
        dec1 = self.decoder2(dec2, features[0])

        dec0 = self.decoder1(dec1, enc1)

        dec_upsample = self.final_upsample(dec0)
        pred = self.voxel_out(dec_upsample)

        # pred = self.forward_decoder(x)
        # loss_rgb = self.forward_loss(original_x, pred)
        return pred
        # features = self.fpn_neck(features)
        # return features

    def loss_fn(self, x, pred):
        loss_rgb = self.forward_loss(x, pred)
        return loss_rgb


class SwinTransformer_VoxelSR_Pretrained_Skip(nn.Module):  # TODO: change to 3D
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
        expand_dim: bool = True,
        out_channels: int = 256,
        resolution=160,
        out_resolution=256,
        decoder_embed_dim: int = 768,
        checkpoint_path=None,
        is_eval=False,
        patch_size=[4, 4, 4],
    ):
        super().__init__()

        # Load the checkpoint from the pretrained SwinTransformer_MAE3D model
        self.out_channels = out_channels
        backbone_type = "swin_s"
        self.patch_size = patch_size
        self.input_resolution = resolution
        self.output_resolution = out_resolution
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

        # model = SwinTransformer_MAE3D(
        #     patch_size=self.patch_size,
        #     embed_dim=swin[backbone_type]["embed_dim"],
        #     depths=swin[backbone_type]["depths"],
        #     num_heads=swin[backbone_type]["num_heads"],
        #     window_size=[4, 4, 4],
        #     stochastic_depth_prob=0.1,
        #     expand_dim=True,
        #     resolution=resolution,
        # )

        model = SwinTransformer_MAE3D_New(
            patch_size=self.patch_size,
            embed_dim=swin[backbone_type]["embed_dim"],
            depths=swin[backbone_type]["depths"],
            num_heads=swin[backbone_type]["num_heads"],
            window_size=[4, 4, 4],
            stochastic_depth_prob=0.1,
            expand_dim=True,
            resolution=resolution,
        )

        print(
            "==============================is EVAL==================================\n\n\n\n"
        )
        print(f"is eval: {is_eval}.")

        if not is_eval:
            assert os.path.exists(checkpoint_path), "The checkpoint does not exist."

            print(
                "================================================================\n\n\n\n"
            )
            print(f"Loading checkpoint from {checkpoint_path}")
            print(
                "================================================================\n\n\n\n"
            )

            # print(f"Loading checkpoint from {checkpoint_path}.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        # Remove the decoder_layers from the pretrained model
        # del model.decoder_layers

        del model.decoder1
        del model.out
        del model.mask_token
        # del model.decoder4
        # del model.decoder3
        # del model.decoder2

        if self.output_resolution == 256:
            final_upsample_scale = 1.6
        elif self.output_resolution == 384:
            final_upsample_scale = 2.4

        self.encoder1 = UnetrBasicBlock(
            in_channels=4,
            out_channels=model.embed_dim // 2,
            kernel_size=3,
            stride=1,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            in_channels=model.embed_dim * 1,
            out_channels=model.embed_dim // 2,
            kernel_size=3,
            upsample_kernel_size=4,
            res_block=True,
            use_skip=True,
        )

        self.final_upsample = nn.Upsample(scale_factor=final_upsample_scale)
        # Output: [2, 64, 64, 64, 64]
        # self.voxel_out = UnetrUpBlock(
        #     in_channels=int(model.embed_dim // 2),
        #     out_channels=4,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     res_block=True,
        #     use_skip=False,
        # )
        self.voxel_out = UnetOutBlock(in_channels=model.embed_dim // 2, out_channels=4)

        #     kernel_size=3,
        #     upsample_kernel_size=4,
        #     res_block=True,
        #     use_skip=True,
        # )

        # self.super_resolution_decoder = nn.Sequential(
        #     # Decoder Layer 1
        #     nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 512, 10, 10, 10]
        #     # Decoder Layer 2
        #     nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 256, 20, 20, 20]
        #     # Decoder Layer 3
        #     nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 128, 40, 40, 40]
        #     # Additional Convolutional Layer
        #     nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(
        #         scale_factor=final_upsample_scale
        #     ),  # Output: [2, 64, 64, 64, 64]
        #     # Final Convolutional Layer
        #     nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1),
        # )

        # fpn_in_channels = []

        # # build SwinTransformer blocks
        # for i_stage in range(len(depths)):
        #     dim = embed_dim * 2**i_stage if expand_dim else embed_dim
        #     fpn_in_channels.append(dim)

        # # Add the FPN neck on top of the SwinTransformer base
        self.base = model
        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

    def patchify_3d(self, x):
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

    def transform(self, x, resolution=160):
        # new_res = [160, 160, 160]

        new_res = [
            resolution,
            resolution,
            resolution,
        ]
        # new_res = [200, 200, 200]
        # masks = []
        padded_rgbsigma = []
        for rgbsimga in x:
            rgb_sigma_padded, _ = pad_tensor(
                rgbsimga, target_shape=new_res, pad_value=0
            )
            padded_rgbsigma.append(rgb_sigma_padded)
            # masks.append(mask_rgbsigma)
        return padded_rgbsigma

    def forward_loss(self, x, pred, is_eval=False):
        x = self.transform(x, resolution=self.output_resolution)
        x = torch.cat((x), dim=0)

        # target = self.patchify_3d(x)
        # pred = self.unpatchify_3d(pred)

        # print("x", x.shape)
        # print("pred", pred.shape)

        target = x.permute(0, 2, 3, 4, 1)
        pred = pred.permute(0, 2, 3, 4, 1)

        # print("target", target.shape)
        # print("pred", pred.shape)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        mask_remove = target_alpha > 0.01

        # mask_remove = mask_remove.squeeze(-1).int() * mask_patches
        mask_remove = mask_remove.squeeze(-1).int()
        loss_rgb = (pred_rgb - target_rgb) ** 2
        loss_rgb = (loss_rgb * mask_remove.unsqueeze(-1)).sum() / mask_remove.sum()
        # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
        # loss = loss_rgb + loss_alpha
        return loss_rgb

    def output_metrics(self, x, pred):
        x = self.transform(x, resolution=self.output_resolution)
        x = torch.cat((x), dim=0)

        target = x.permute(0, 2, 3, 4, 1)
        pred = pred.permute(0, 2, 3, 4, 1)

        # target = self.patchify_3d(x)
        # pred = self.unpatchify_3d(pred)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        mask_remove = target_alpha > 0.01

        value = mse(pred_rgb, target_rgb, valid_mask=mask_remove)
        psnr_value = psnr(pred_rgb, target_rgb, valid_mask=mask_remove)

        metrics = {}
        metrics["MSE"] = value.item()
        metrics["PSNR"] = psnr_value.item()

        return metrics

    # def forward_decoder(self, latent):
    #     latent = latent.permute(0, 4, 1, 2, 3)
    #     out = self.super_resolution_decoder(latent)
    #     out = out.permute(0, 2, 3, 4, 1)
    #     return out

    def forward(self, x):
        # original_x = x.clone().detach()
        # Forward pass through the SwinTransformer base

        # print("x", x.shape)

        # pad x to have same res for all voxels in the batch (masking is taken care of my output alpha>0.01 i think?)
        x = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)

        enc1 = self.encoder1(x)

        x = self.base.patch_partition(x)
        x = x + self.base.pos_embed.type_as(x).to(x.device).clone().detach()

        features = []
        for i in range(len(self.base.stages)):
            x = self.base.stages[i](x)
            features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous())
            # features.append(
            #     torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            # )  # [N, C, H, W, D]

        dec3 = self.base.decoder4(features[3], features[2])
        dec2 = self.base.decoder3(dec3, features[1])
        dec1 = self.base.decoder2(dec2, features[0])

        dec0 = self.decoder1(dec1, enc1)

        print("dec0", dec0.shape)

        dec_upsample = self.final_upsample(dec0)
        print("dec_upsample", dec_upsample.shape)
        pred = self.voxel_out(dec_upsample)

        print("pred", pred.shape)

        # print("pred", pred.shape)
        # print("dec0", dec0.shape)
        # print("dec1", dec1.shape)
        # print("dec_upsample", dec_upsample.shape)
        # pred = self.forward_decoder(x)
        # loss_rgb = self.forward_loss(original_x, pred)
        return pred

    def loss_fn(self, x, pred):
        loss_rgb = self.forward_loss(x, pred)
        return loss_rgb


class SwinTransformer_VoxelSR_Pretrained(nn.Module):  # TODO: change to 3D
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
        expand_dim: bool = True,
        out_channels: int = 256,
        resolution=160,
        out_resolution=256,
        decoder_embed_dim: int = 768,
        checkpoint_path=None,
        is_eval=False,
        patch_size=[4, 4, 4],
    ):
        super().__init__()

        # Load the checkpoint from the pretrained SwinTransformer_MAE3D model
        self.out_channels = out_channels
        backbone_type = "swin_s"
        self.patch_size = patch_size
        self.input_resolution = resolution
        self.output_resolution = out_resolution
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

        # model = SwinTransformer_MAE3D(
        #     patch_size=self.patch_size,
        #     embed_dim=swin[backbone_type]["embed_dim"],
        #     depths=swin[backbone_type]["depths"],
        #     num_heads=swin[backbone_type]["num_heads"],
        #     window_size=[4, 4, 4],
        #     stochastic_depth_prob=0.1,
        #     expand_dim=True,
        #     resolution=resolution,
        # )

        model = SwinTransformer_MAE3D_New(
            patch_size=self.patch_size,
            embed_dim=swin[backbone_type]["embed_dim"],
            depths=swin[backbone_type]["depths"],
            num_heads=swin[backbone_type]["num_heads"],
            window_size=[4, 4, 4],
            stochastic_depth_prob=0.1,
            expand_dim=True,
            resolution=resolution,
        )

        print(
            "==============================is EVAL==================================\n\n\n\n"
        )
        print(f"is eval: {is_eval}.")

        if not is_eval:
            assert os.path.exists(checkpoint_path), "The checkpoint does not exist."

            print(
                "================================================================\n\n\n\n"
            )
            print(f"Loading checkpoint from {checkpoint_path}")
            print(
                "================================================================\n\n\n\n"
            )

            # print(f"Loading checkpoint from {checkpoint_path}.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        # Remove the decoder_layers from the pretrained model
        # del model.decoder_layers

        del model.decoder1
        del model.out
        # del model.mask_token
        del model.decoder4
        del model.decoder3
        del model.decoder2

        # if self.output_resolution == 256:
        #     final_upsample_scale = 1.6
        # elif self.output_resolution == 384:
        #     final_upsample_scale = 2.4

        final_upsample_scale = 1.6

        self.super_resolution_decoder = nn.Sequential(
            # Decoder Layer 1
            nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 512, 10, 10, 10]
            # Decoder Layer 2
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 256, 20, 20, 20]
            # Decoder Layer 3
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 128, 40, 40, 40]
            # Additional Convolutional Layer
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(
                scale_factor=final_upsample_scale
            ),  # Output: [2, 64, 64, 64, 64]
            # Final Convolutional Layer
            nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # fpn_in_channels = []

        # # build SwinTransformer blocks
        # for i_stage in range(len(depths)):
        #     dim = embed_dim * 2**i_stage if expand_dim else embed_dim
        #     fpn_in_channels.append(dim)

        # # Add the FPN neck on top of the SwinTransformer base
        self.base = model
        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

    def patchify_3d(self, x):
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

    def transform(self, x, resolution=160):
        # new_res = [160, 160, 160]

        new_res = [
            resolution,
            resolution,
            resolution,
        ]
        # new_res = [200, 200, 200]
        # masks = []
        padded_rgbsigma = []
        for rgbsimga in x:
            rgb_sigma_padded, _ = pad_tensor(
                rgbsimga, target_shape=new_res, pad_value=0
            )
            padded_rgbsigma.append(rgb_sigma_padded)
            # masks.append(mask_rgbsigma)
        return padded_rgbsigma

    def forward_loss(self, x, pred, is_eval=False):
        x = self.transform(x, resolution=self.output_resolution)
        x = torch.cat((x), dim=0)

        target = self.patchify_3d(x)
        pred = self.unpatchify_3d(pred)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        mask_remove = target_alpha > 0.01

        # mask_remove = mask_remove.squeeze(-1).int() * mask_patches
        mask_remove = mask_remove.squeeze(-1).int()
        loss_rgb = (pred_rgb - target_rgb) ** 2
        loss_rgb = (loss_rgb * mask_remove.unsqueeze(-1)).sum() / mask_remove.sum()
        # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
        # loss = loss_rgb + loss_alpha
        return loss_rgb

    def output_metrics(self, x, pred):
        x = self.transform(x, resolution=self.output_resolution)
        x = torch.cat((x), dim=0)
        target = self.patchify_3d(x)
        pred = self.unpatchify_3d(pred)

        target_rgb = target[..., :3]
        target_alpha = target[..., 3].unsqueeze(-1)

        pred_rgb = pred[..., :3]
        mask_remove = target_alpha > 0.01

        value = mse(pred_rgb, target_rgb, valid_mask=mask_remove)
        psnr_value = psnr(pred_rgb, target_rgb, valid_mask=mask_remove)

        metrics = {}
        metrics["MSE"] = value.item()
        metrics["PSNR"] = psnr_value.item()

        return metrics

    def forward_decoder(self, latent):
        latent = latent.permute(0, 4, 1, 2, 3)
        out = self.super_resolution_decoder(latent)
        out = out.permute(0, 2, 3, 4, 1)
        return out

    def forward(self, x):
        # original_x = x.clone().detach()
        # Forward pass through the SwinTransformer base

        # print("x", x.shape)

        # pad x to have same res for all voxels in the batch (masking is taken care of my output alpha>0.01 i think?)
        x = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        x = self.base.patch_partition(x)
        x = x + self.base.pos_embed.type_as(x).to(x.device).clone().detach()
        for i in range(len(self.base.stages)):
            x = self.base.stages[i](x)
            # features.append(
            #     torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            # )  # [N, C, H, W, D]
        pred = self.forward_decoder(x)
        # loss_rgb = self.forward_loss(original_x, pred)
        return pred

    def loss_fn(self, x, pred):
        loss_rgb = self.forward_loss(x, pred)
        return loss_rgb


class SwinTransformer_VoxelSemantics_Pretrained_Skip(nn.Module):  # TODO: change to 3D
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
        expand_dim: bool = True,
        out_channels: int = 19,
        resolution=160,
        decoder_embed_dim: int = 768,
        checkpoint_path=None,
        is_eval=False,
        patch_size=[4, 4, 4],
        class_weights=None,
    ):
        super().__init__()

        # Load the checkpoint from the pretrained SwinTransformer_MAE3D model
        self.out_channels = out_channels
        backbone_type = "swin_s"
        self.patch_size = patch_size
        self.input_resolution = resolution
        self.iou_loss = mIoULoss_new(n_classes=out_channels)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # self.criterion = nn.CrossEntropyLoss()
        # self.class_weights = class_weights

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
            patch_size=self.patch_size,
            embed_dim=swin[backbone_type]["embed_dim"],
            depths=swin[backbone_type]["depths"],
            num_heads=swin[backbone_type]["num_heads"],
            window_size=[4, 4, 4],
            stochastic_depth_prob=0.1,
            expand_dim=True,
            resolution=resolution,
        )

        print(
            "==============================is EVAL==================================\n\n\n\n"
        )
        print(f"is eval: {is_eval}.")

        if not is_eval:
            assert os.path.exists(checkpoint_path), "The checkpoint does not exist."

            print(
                "================================================================\n\n\n\n"
            )
            print(f"Loading checkpoint from {checkpoint_path}")
            print(
                "================================================================\n\n\n\n"
            )

            # print(f"Loading checkpoint from {checkpoint_path}.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        # Remove the out_layer from the pretrained model

        del model.decoder1
        del model.out
        del model.mask_token

        self.encoder1 = UnetrBasicBlock(
            in_channels=4,
            out_channels=model.embed_dim // 2,
            kernel_size=3,
            stride=1,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            in_channels=model.embed_dim * 1,
            out_channels=model.embed_dim // 2,
            kernel_size=3,
            upsample_kernel_size=4,
            res_block=True,
            use_skip=True,
        )

        self.sem_out = UnetOutBlock(
            in_channels=model.embed_dim // 2, out_channels=self.out_channels
        )

        # # Add the FPN neck on top of the SwinTransformer base
        self.base = model
        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

    # def patchify_3d(self, x):
    #     """
    #     rgbsimga: (N, 4, H, W, D)
    #     x: (N, L, L, L, patch_size**3 *4)
    #     """
    #     p = self.patch_size[0]
    #     assert x.shape[2] == x.shape[3] == x.shape[4] and x.shape[2] % p == 0
    #     h = w = l = int(round(x.shape[2] // p))

    #     x = x.reshape(shape=(x.shape[0], 1, h, p, w, p, l, p))
    #     x = rearrange(x, "n c h p w q l r -> n h w l p q r c")
    #     x = rearrange(x, "n h w l p q r c -> n h w l (p q r) c")

    #     return x

    # def unpatchify_3d(self, x):
    #     """
    #     rgbsimga: (N, 4, H, W, D)
    #     x: (N, L, L, L, patch_size**3 *4)
    #     """
    #     p = self.patch_size[0]
    #     _, h, w, l, _ = x.shape

    #     x = x.reshape(shape=(x.shape[0], h, w, l, p**3, 19))
    #     return x

    def transform(self, x, resolution=160):
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

    def forward_loss(self, x, pred, is_eval=False):
        num_classes = self.out_channels
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        # masks = torch.cat((mask), dim=0).to(x.device)

        # print("x", x.shape)
        # print("pred", pred.shape)
        # target = self.patchify_3d(x)
        # pred = self.unpatchify_3d(pred)

        target = x.permute(0, 2, 3, 4, 1).long()
        pred = pred.permute(0, 2, 3, 4, 1)

        # B, H, W, D, L, _ = target.shape
        # # target = target.reshape(B * H * W * D * L).long()  # Flatten the target tensor
        # target = target.long()
        # pred = pred.reshape(
        #     B * H * W * D * L, num_classes
        # )  # Flatten and keep the number of classes

        # Now we only consider class >0, don't consider void class ##IMPORTANT, maybe change it later
        mask = target > 0
        mask = mask.long()
        # mask = mask.reshape(B * H * W * D * L)

        # loss = F.cross_entropy(
        #     pred, target, reduction="none"
        # )  # Use 'none' to prevent reduction

        # pred = pred.permute(0, 5, 1, 2, 3, 4)
        # target = target.squeeze(-1).unsqueeze(1)
        # mask = mask.squeeze(-1).unsqueeze(1)
        # sem_ce_loss = self.criterion(pred, target, mask)

        sem_ce_loss = masked_cross_entropy(
            self.criterion, target, pred, mask, num_classes=num_classes
        )
        _, iou = self.iou_loss(pred, target, mask)

        # class_weight = self.class_weights[1:].to(target.device)
        # weighted_loss = (1 - iou_all) * class_weight
        # weighted_loss = weighted_loss.mean()

        # iou_lambda = 0.1
        # loss = (1 - iou_lambda) * sem_ce_loss + weighted_loss * (iou_lambda)

        loss = sem_ce_loss
        return (
            loss,
            sem_ce_loss,
            iou,
        )

    def output_metrics(self, x, pred):
        num_classes = self.out_channels
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        # mask = torch.cat((mask), dim=0).to(x.device)

        target = x.permute(0, 2, 3, 4, 1).long()
        pred = pred.permute(0, 2, 3, 4, 1)

        # target = self.patchify_3d(x)
        # pred = self.unpatchify_3d(pred)

        # Now we only consider class >0, don't consider void class ##IMPORTANT, maybe change it later
        mask = target > 0
        mask = mask.long()

        probabilities_vis = F.softmax(pred, dim=-1)
        predicted_labels_vis = torch.argmax(probabilities_vis, dim=-1)
        pred_labels_vis = predicted_labels_vis * mask.squeeze(-1)

        target_vis = target.squeeze(-1) * mask.squeeze(-1)

        # print("predicted_labels", pred_labels_vis.shape)
        # print("mask", mask.shape)
        # mask = mask.reshape(B * H * W * D * L)

        sem_ce_loss_val = masked_cross_entropy(
            self.criterion, target, pred, mask, num_classes=num_classes
        )
        _, iou_val = self.iou_loss(pred, target, mask)

        # pred_out = pred.detach().cpu().numpy()
        # target_out = target.detach().cpu().numpy()
        # print("mask", mask.shape)

        # print("pred_out", pred_out.shape, "target_out", target_out.shape)
        B, H, W, D, _ = target.shape
        target = target.reshape(B * H * W * D).long()  # Flatten the target tensor

        pred = pred.reshape(
            B * H * W * D, num_classes
        )  # Flatten and keep the number of classes

        probabilities = F.softmax(pred, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        predicted_labels = predicted_labels[mask.view(-1) == 1]
        target = target[mask.view(-1) == 1]

        intersection, union, target = intersectionAndUnionGPU(
            predicted_labels, target, num_classes
        )
        metrics = {}
        metrics["intersection"] = intersection
        metrics["union"] = union
        metrics["target"] = target
        metrics["iou_val"] = iou_val

        return sem_ce_loss_val, metrics, pred_labels_vis, target_vis

    # def forward_decoder(self, latent):
    #     latent = latent.permute(0, 4, 1, 2, 3)
    #     out = self.semantic_decoder(latent)
    #     B = latent.shape[0]
    #     out = out.view(B, 1216, 40, 40, 40)
    #     out = out.permute(0, 2, 3, 4, 1)
    #     return out

    def forward(self, x):
        x, _ = self.transform(x, resolution=self.input_resolution)
        # original_padded_x = x.clone().detach()
        x = torch.cat((x), dim=0)
        enc1 = self.encoder1(x)
        x = self.base.patch_partition(x)
        # patch_partition_cache = x
        x = x + self.base.pos_embed.type_as(x).to(x.device).clone().detach()

        features = []
        for i in range(len(self.base.stages)):
            x = self.base.stages[i](x)
            features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous())
            # features.append(
            #     torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            # )  # [N, C, H, W, D]

        dec3 = self.base.decoder4(features[3], features[2])
        dec2 = self.base.decoder3(dec3, features[1])
        dec1 = self.base.decoder2(dec2, features[0])

        dec0 = self.decoder1(dec1, enc1)

        pred = self.sem_out(dec0)
        # pred = self.forward_decoder(x)
        # loss_rgb = self.forward_loss(original_x, pred)
        return pred

    def loss_fn(self, x, pred):
        loss_rgb, sem_ce_loss, iou = self.forward_loss(x, pred)
        return loss_rgb, sem_ce_loss, iou


class SwinTransformer_VoxelSemantics_Pretrained(nn.Module):  # TODO: change to 3D
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
        expand_dim: bool = True,
        out_channels: int = 256,
        resolution=160,
        decoder_embed_dim: int = 768,
        checkpoint_path=None,
        is_eval=False,
        patch_size=[4, 4, 4],
    ):
        super().__init__()

        # Load the checkpoint from the pretrained SwinTransformer_MAE3D model
        self.out_channels = out_channels
        backbone_type = "swin_s"
        self.patch_size = patch_size
        self.input_resolution = resolution
        self.iou_loss = mIoULoss(n_classes=19)

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
            patch_size=self.patch_size,
            embed_dim=swin[backbone_type]["embed_dim"],
            depths=swin[backbone_type]["depths"],
            num_heads=swin[backbone_type]["num_heads"],
            window_size=[4, 4, 4],
            stochastic_depth_prob=0.1,
            expand_dim=True,
            resolution=resolution,
        )

        print(
            "==============================is EVAL==================================\n\n\n\n"
        )
        print(f"is eval: {is_eval}.")

        if not is_eval:
            assert os.path.exists(checkpoint_path), "The checkpoint does not exist."

            print(
                "================================================================\n\n\n\n"
            )
            print(f"Loading checkpoint from {checkpoint_path}")
            print(
                "================================================================\n\n\n\n"
            )

            # print(f"Loading checkpoint from {checkpoint_path}.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        # Remove the decoder_layers from the pretrained model
        del model.decoder_layers

        self.semantic_decoder = nn.Sequential(
            # Decoder Layer 1
            nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 512, 10, 10, 10]
            # Decoder Layer 2
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 256, 20, 20, 20]
            # Decoder Layer 3
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 128, 40, 40, 40]
            # Decoder Layer 4
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 64, 80, 80, 80]
            # Decoder Layer 5
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 64, 80, 80, 80]
            # Final convolutional layer with 41 output channels
            nn.Conv3d(32, 19, kernel_size=3, stride=1, padding=1),
        )

        # # Add the FPN neck on top of the SwinTransformer base
        self.base = model
        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

    def patchify_3d(self, x):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """
        p = self.patch_size[0]
        assert x.shape[2] == x.shape[3] == x.shape[4] and x.shape[2] % p == 0
        h = w = l = int(round(x.shape[2] // p))

        x = x.reshape(shape=(x.shape[0], 1, h, p, w, p, l, p))
        x = rearrange(x, "n c h p w q l r -> n h w l p q r c")
        x = rearrange(x, "n h w l p q r c -> n h w l (p q r) c")

        return x

    def unpatchify_3d(self, x):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """
        p = self.patch_size[0]
        _, h, w, l, _ = x.shape

        x = x.reshape(shape=(x.shape[0], h, w, l, p**3, 19))
        return x

    def transform(self, x, resolution=160):
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

    def forward_loss(self, x, pred, is_eval=False):
        num_classes = 19
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        # masks = torch.cat((mask), dim=0).to(x.device)
        target = self.patchify_3d(x)
        pred = self.unpatchify_3d(pred)

        B, H, W, D, L, _ = target.shape
        # target = target.reshape(B * H * W * D * L).long()  # Flatten the target tensor
        target = target.long()
        # pred = pred.reshape(
        #     B * H * W * D * L, num_classes
        # )  # Flatten and keep the number of classes

        # Now we only consider class >0, don't consider void class ##IMPORTANT, maybe change it later
        mask = target > 0
        mask = mask.long()
        # mask = mask.reshape(B * H * W * D * L)

        # loss = F.cross_entropy(
        #     pred, target, reduction="none"
        # )  # Use 'none' to prevent reduction

        # pred = pred.permute(0, 5, 1, 2, 3, 4)
        # target = target.squeeze(-1).unsqueeze(1)
        # mask = mask.squeeze(-1).unsqueeze(1)
        # sem_ce_loss = self.criterion(pred, target, mask)

        sem_ce_loss = masked_cross_entropy(self.criterion, target, pred, mask)
        _, iou = self.iou_loss(pred, target, mask)

        # sem_ce_loss, iou = self.iou_loss(pred, target, mask)

        # outputs = [self.post_pred(i) for i in decollate_batch(pred)]
        # labels = [self.post_label(i) for i in decollate_batch(target)]

        # print("loss", loss.shape, "mask", mask.shape)

        # masked_loss = loss * mask

        # # Compute the mean of the masked loss
        # # Since the mask is binary (0s and 1s), we can use 'sum' and 'numel' to compute the mean
        # # by summing the masked elements and dividing by the total number of valid elements.
        # sem_ce_loss = masked_loss.sum() / mask.sum()

        # print("sem_ce_loss", sem_ce_loss.shape)
        # print("mask", mask.shape)

        # Apply the mask to the loss
        # masked_loss = loss * mask.view(-1)

        # # Compute the mean of the masked loss
        # sem_ce_loss = masked_loss.sum() / mask.sum()

        # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
        # loss = loss_rgb + loss_alpha
        return sem_ce_loss, iou

    def output_metrics(self, x, pred):
        num_classes = 19
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        # mask = torch.cat((mask), dim=0).to(x.device)

        target = self.patchify_3d(x)
        pred = self.unpatchify_3d(pred)

        # Now we only consider class >0, don't consider void class ##IMPORTANT, maybe change it later
        mask = target > 0
        mask = mask.long()
        # mask = mask.reshape(B * H * W * D * L)

        # DICELOSS CALCULATION==================
        # pred = pred.permute(0, 5, 1, 2, 3, 4)
        # target = target.squeeze(-1).unsqueeze(1)
        # mask = mask.squeeze(-1).unsqueeze(1)
        # sem_ce_loss_val = self.criterion(pred, target, mask)
        # DICELOSS CALCULATION==================

        sem_ce_loss_val = masked_cross_entropy(self.criterion, target, pred, mask)
        _, iou_val = self.iou_loss(pred, target, mask)

        # sem_ce_loss_val, iou_val = self.iou_loss(pred, target, mask)

        # loss = F.cross_entropy(
        #     pred, target, reduction="none"
        # )  # Use 'none' to prevent reduction
        # masked_loss = loss * mask.view(-1)
        # sem_ce_loss_val = masked_loss.sum() / mask.sum()

        # mask = mask.reshape(B * H * W * D * L)

        # probs = F.softmax(pred, dim=1)
        # predicted_labels = torch.argmax(probs, dim=1)

        # pred = pred.permute(0, 2, 3, 4, 5, 1)
        # target = target.squeeze(1)
        # mask = mask.squeeze(1)
        # print("pred", pred.shape, "target", target.shape, "mask", mask.shape)

        pred_out = pred.detach().cpu().numpy()
        target_out = target.detach().cpu().numpy()
        # print("mask", mask.shape)

        # print("pred_out", pred_out.shape, "target_out", target_out.shape)
        B, H, W, D, L, _ = target.shape
        target = target.reshape(B * H * W * D * L).long()  # Flatten the target tensor

        pred = pred.reshape(
            B * H * W * D * L, num_classes
        )  # Flatten and keep the number of classes

        probabilities = F.softmax(pred, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        predicted_labels = predicted_labels[mask.view(-1) == 1]
        target = target[mask.view(-1) == 1]

        # evaluator.update(
        #     predicted_labels.detach().cpu().numpy(), target.detach().cpu().numpy()
        # )

        intersection, union, target = intersectionAndUnionGPU(
            predicted_labels, target, num_classes
        )

        # iou_score = intersection_over_union(predicted_labels, target, mask)
        # accuracy_score = pixel_accuracy(predicted_labels, target, mask)

        metrics = {}
        metrics["intersection"] = intersection
        metrics["union"] = union
        metrics["target"] = target
        metrics["iou_val"] = iou_val

        return sem_ce_loss_val, metrics

    def forward_decoder(self, latent):
        latent = latent.permute(0, 4, 1, 2, 3)
        out = self.semantic_decoder(latent)
        B = latent.shape[0]
        out = out.view(B, 1216, 40, 40, 40)
        out = out.permute(0, 2, 3, 4, 1)
        return out

    def forward(self, x):
        # original_x = x.clone().detach()
        # Forward pass through the SwinTransformer base

        # print("x", x.shape)

        # pad x to have same res for all voxels in the batch (masking is taken care of my output alpha>0.01 i think?)
        x, _ = self.transform(x, resolution=self.input_resolution)
        # original_padded_x = x.clone().detach()
        x = torch.cat((x), dim=0)

        x = self.base.patch_partition(x)
        x = x + self.base.pos_embed.type_as(x).to(x.device).clone().detach()
        for i in range(len(self.base.stages)):
            x = self.base.stages[i](x)
            # features.append(
            #     torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            # )  # [N, C, H, W, D]
        pred = self.forward_decoder(x)
        # loss_rgb = self.forward_loss(original_x, pred)
        return pred

    def loss_fn(self, x, pred):
        loss_rgb, iou = self.forward_loss(x, pred)
        return loss_rgb, iou


class SwinTransformer_VoxelSemantics_Skip(nn.Module):  # TODO: change to 3D
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
        out_channels: int = 19,
        input_dim: int = 4,
        resolution=160,
        decoder_embed_dim: int = 768,
        class_weights=None,
    ):
        super().__init__()

        # Load the checkpoint from the pretrained SwinTransformer_MAE3D model
        self.out_channels = out_channels
        # backbone_type = "swin_s"
        # self.patch_size = patch_size
        # self.input_resolution = resolution
        self.iou_loss = mIoULoss_new(n_classes=out_channels)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # self.criterion = nn.CrossEntropyLoss()
        self.class_weights = class_weights

        self.patch_size = patch_size
        self.input_resolution = resolution
        # split image into non-overlapping patches
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

        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.encoder1 = UnetrBasicBlock(
            in_channels=4,
            out_channels=embed_dim // 2,
            kernel_size=3,
            stride=1,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            in_channels=embed_dim * 1,
            out_channels=embed_dim // 2,
            kernel_size=3,
            upsample_kernel_size=4,
            res_block=True,
            use_skip=True,
        )

        self.sem_out = UnetOutBlock(
            in_channels=embed_dim // 2, out_channels=self.out_channels
        )

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

        # self.decoder1 = UnetrUpBlock(
        #     in_channels=embed_dim * 1,
        #     out_channels=embed_dim // 2,
        #     kernel_size=3,
        #     upsample_kernel_size=4,
        #     res_block=True,
        #     use_skip=False,
        # )

        # self.sem_out = UnetOutBlock(
        #     in_channels=embed_dim // 2, out_channels=self.out_channels
        # )
        # self.sem_out = UnetOutBlock(
        #     in_channels=model.embed_dim // 2, out_channels=self.out_channels
        # )

        # # Add the FPN neck on top of the SwinTransformer base
        # self.base = model
        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

    # def patchify_3d(self, x):
    #     """
    #     rgbsimga: (N, 4, H, W, D)
    #     x: (N, L, L, L, patch_size**3 *4)
    #     """
    #     p = self.patch_size[0]
    #     assert x.shape[2] == x.shape[3] == x.shape[4] and x.shape[2] % p == 0
    #     h = w = l = int(round(x.shape[2] // p))

    #     x = x.reshape(shape=(x.shape[0], 1, h, p, w, p, l, p))
    #     x = rearrange(x, "n c h p w q l r -> n h w l p q r c")
    #     x = rearrange(x, "n h w l p q r c -> n h w l (p q r) c")

    #     return x

    # def unpatchify_3d(self, x):
    #     """
    #     rgbsimga: (N, 4, H, W, D)
    #     x: (N, L, L, L, patch_size**3 *4)
    #     """
    #     p = self.patch_size[0]
    #     _, h, w, l, _ = x.shape

    #     x = x.reshape(shape=(x.shape[0], h, w, l, p**3, 19))
    #     return x

    def transform(self, x, resolution=160):
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

    def forward_loss(self, x, pred, is_eval=False):
        num_classes = self.out_channels
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        # masks = torch.cat((mask), dim=0).to(x.device)

        # print("x", x.shape)
        # print("pred", pred.shape)
        # target = self.patchify_3d(x)
        # pred = self.unpatchify_3d(pred)

        # print("target before", x.shape)
        # print("pred before", pred.shape)

        target = x.permute(0, 2, 3, 4, 1).long()
        pred = pred.permute(0, 2, 3, 4, 1)

        # print("target", target.shape)
        # print("pred", pred.shape)

        # B, H, W, D, L, _ = target.shape
        # # target = target.reshape(B * H * W * D * L).long()  # Flatten the target tensor
        # target = target.long()
        # pred = pred.reshape(
        #     B * H * W * D * L, num_classes
        # )  # Flatten and keep the number of classes

        # Now we only consider class >0, don't consider void class ##IMPORTANT, maybe change it later
        mask = target > 0
        mask = mask.long()
        # mask = mask.reshape(B * H * W * D * L)

        # loss = F.cross_entropy(
        #     pred, target, reduction="none"
        # )  # Use 'none' to prevent reduction

        # pred = pred.permute(0, 5, 1, 2, 3, 4)
        # target = target.squeeze(-1).unsqueeze(1)
        # mask = mask.squeeze(-1).unsqueeze(1)
        # sem_ce_loss = self.criterion(pred, target, mask)

        sem_ce_loss = masked_cross_entropy(
            self.criterion, target, pred, mask, num_classes=num_classes
        )
        _, iou = self.iou_loss(pred, target, mask)

        # class_weight = self.class_weights[1:].to(target.device)
        # weighted_loss = (1 - iou_all) * class_weight
        # weighted_loss = weighted_loss.mean()

        # ce_lambda = 0.2
        # loss = ce_lambda * sem_ce_loss + weighted_loss * (1 - ce_lambda)

        # iou_lambda = 0.1
        # loss = (1 - iou_lambda) * sem_ce_loss + weighted_loss * (iou_lambda)

        loss = sem_ce_loss

        return loss, sem_ce_loss, iou

    def output_metrics(self, x, pred):
        num_classes = self.out_channels
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        # mask = torch.cat((mask), dim=0).to(x.device)

        target = x.permute(0, 2, 3, 4, 1).long()
        pred = pred.permute(0, 2, 3, 4, 1)

        # target = self.patchify_3d(x)
        # pred = self.unpatchify_3d(pred)
        print("target", target.shape)
        print("pred", pred.shape)
        # Now we only consider class >0, don't consider void class ##IMPORTANT, maybe change it later
        mask = target > 0
        mask = mask.long()
        # mask = mask.reshape(B * H * W * D * L)

        sem_ce_loss_val = masked_cross_entropy(
            self.criterion, target, pred, mask, num_classes=num_classes
        )
        _, iou_val = self.iou_loss(pred, target, mask)

        probabilities_vis = F.softmax(pred, dim=-1)
        predicted_labels_vis = torch.argmax(probabilities_vis, dim=-1)
        pred_labels_vis = predicted_labels_vis * mask.squeeze(-1)

        target_vis = target.squeeze(-1) * mask.squeeze(-1)

        # pred_out = pred.detach().cpu().numpy()
        # target_out = target.detach().cpu().numpy()
        # print("mask", mask.shape)

        # print("pred_out", pred_out.shape, "target_out", target_out.shape)
        # B, H, W, D, L, _ = target.shape
        # target = target.reshape(B * H * W * D * L).long()  # Flatten the target tensor

        # pred = pred.reshape(
        #     B * H * W * D * L, num_classes
        # )  # Flatten and keep the number of classes

        B, H, W, D, _ = target.shape
        target = target.reshape(B * H * W * D).long()  # Flatten the target tensor

        pred = pred.reshape(
            B * H * W * D, num_classes
        )  # Flatten and keep the number of classes

        probabilities = F.softmax(pred, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        predicted_labels = predicted_labels[mask.view(-1) == 1]
        target = target[mask.view(-1) == 1]

        intersection, union, target = intersectionAndUnionGPU(
            predicted_labels, target, num_classes
        )
        metrics = {}
        metrics["intersection"] = intersection
        metrics["union"] = union
        metrics["target"] = target
        metrics["iou_val"] = iou_val

        return sem_ce_loss_val, metrics, pred_labels_vis, target_vis

    # def forward_decoder(self, latent):
    #     latent = latent.permute(0, 4, 1, 2, 3)
    #     out = self.semantic_decoder(latent)
    #     B = latent.shape[0]
    #     out = out.view(B, 1216, 40, 40, 40)
    #     out = out.permute(0, 2, 3, 4, 1)
    #     return out

    def forward(self, x):
        x, _ = self.transform(x, resolution=self.input_resolution)
        # original_padded_x = x.clone().detach()
        x = torch.cat((x), dim=0)

        enc1 = self.encoder1(x)

        x = self.patch_partition(x)
        # x = x + self.base.pos_embed.type_as(x).to(x.device).clone().detach()
        features = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous())
            # features.append(
            #     torch.permute(x, [0, 4, 1, 2, 3]).contiguous()
            # )  # [N, C, H, W, D]

        dec3 = self.decoder4(features[3], features[2])
        dec2 = self.decoder3(dec3, features[1])
        # dec1 = self.decoder2(dec2, features[0])

        dec1 = self.decoder2(dec2, features[0])

        dec0 = self.decoder1(dec1, enc1)

        pred = self.sem_out(dec0)

        # # dec0 = self.decoder1(dec1, enc0)
        # dec0 = self.decoder1(dec1)

        # # print("dec0", dec0.shape)

        # pred = self.sem_out(dec0)
        # pred = self.forward_decoder(x)
        # loss_rgb = self.forward_loss(original_x, pred)
        return pred

    def loss_fn(self, x, pred):
        loss_rgb, sem_ce_loss, iou = self.forward_loss(x, pred)
        return loss_rgb, sem_ce_loss, iou


class SwinTransformer_VoxelSemantics(nn.Module):  # TODO: change to 3D
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
        resolution=160,
        decoder_embed_dim: int = 768,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.iou_loss = mIoULoss(n_classes=19)

        self.patch_size = patch_size
        self.input_resolution = resolution
        # split image into non-overlapping patches
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

        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.semantic_decoder = nn.Sequential(
            # Decoder Layer 1
            nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 512, 10, 10, 10]
            # Decoder Layer 2
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 256, 20, 20, 20]
            # Decoder Layer 3
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 128, 40, 40, 40]
            # Decoder Layer 4
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 64, 80, 80, 80]
            # Decoder Layer 5
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Output: [2, 64, 80, 80, 80]
            # Final convolutional layer with 41 output channels
            nn.Conv3d(32, 19, kernel_size=3, stride=1, padding=1),
        )

        # self.semantic_decoder = nn.Sequential(
        #     # Decoder Layer 1
        #     nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 512, 10, 10, 10]
        #     # Decoder Layer 2
        #     nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 256, 20, 20, 20]
        #     # Decoder Layer 3
        #     nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),  # Output: [2, 128, 40, 40, 40]
        #     nn.Conv3d(128, out_channels, kernel_size=3, stride=1, padding=1),
        # )

        # # Add the FPN neck on top of the SwinTransformer base
        # self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

    def patchify_3d(self, x):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """
        p = self.patch_size[0]
        assert x.shape[2] == x.shape[3] == x.shape[4] and x.shape[2] % p == 0
        h = w = l = int(round(x.shape[2] // p))

        x = x.reshape(shape=(x.shape[0], 1, h, p, w, p, l, p))
        x = rearrange(x, "n c h p w q l r -> n h w l p q r c")
        x = rearrange(x, "n h w l p q r c -> n h w l (p q r) c")

        # mask = mask.reshape(shape=(x.shape[0], 1, h, p, w, p, l, p))
        # mask = rearrange(mask, "n c h p w q l r -> n h w l p q r c")
        # mask = rearrange(mask, "n h w l p q r c -> n h w l (p q r) c")
        # mask = mask[:, :, :, :, :, 0]

        return x

    def unpatchify_3d(self, x):
        """
        rgbsimga: (N, 4, H, W, D)
        x: (N, L, L, L, patch_size**3 *4)
        """
        p = self.patch_size[0]
        _, h, w, l, _ = x.shape

        x = x.reshape(shape=(x.shape[0], h, w, l, p**3, 19))
        return x

    def transform(self, x, resolution=160):
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

    def forward_loss(self, x, pred, is_eval=False):
        num_classes = self.out_channels
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        # masks = torch.cat((mask), dim=0).to(x.device)
        target = self.patchify_3d(x)
        pred = self.unpatchify_3d(pred)

        B, H, W, D, L, _ = target.shape
        # target = target.reshape(B * H * W * D * L).long()  # Flatten the target tensor
        target = target.long()
        # pred = pred.reshape(
        #     B * H * W * D * L, num_classes
        # )  # Flatten and keep the number of classes

        # Now we only consider class >0, don't consider void class ##IMPORTANT, maybe change it later
        mask = target > 0
        mask = mask.long()
        # mask = mask.reshape(B * H * W * D * L)

        # loss = F.cross_entropy(
        #     pred, target, reduction="none"
        # )  # Use 'none' to prevent reduction

        # pred = pred.permute(0, 5, 1, 2, 3, 4)
        # target = target.squeeze(-1).unsqueeze(1)
        # mask = mask.squeeze(-1).unsqueeze(1)
        # sem_ce_loss = self.criterion(pred, target, mask)

        sem_ce_loss = masked_cross_entropy(self.criterion, target, pred, mask)
        _, iou = self.iou_loss(pred, target, mask)
        # sem_ce_loss, iou = self.iou_loss(pred, target, mask)

        # outputs = [self.post_pred(i) for i in decollate_batch(pred)]
        # labels = [self.post_label(i) for i in decollate_batch(target)]

        # print("loss", loss.shape, "mask", mask.shape)

        # masked_loss = loss * mask

        # # Compute the mean of the masked loss
        # # Since the mask is binary (0s and 1s), we can use 'sum' and 'numel' to compute the mean
        # # by summing the masked elements and dividing by the total number of valid elements.
        # sem_ce_loss = masked_loss.sum() / mask.sum()

        # print("sem_ce_loss", sem_ce_loss.shape)
        # print("mask", mask.shape)

        # Apply the mask to the loss
        # masked_loss = loss * mask.view(-1)

        # # Compute the mean of the masked loss
        # sem_ce_loss = masked_loss.sum() / mask.sum()

        # loss_alpha = self.ce_loss(pred_alpha, target_alpha, mask)
        # loss = loss_rgb + loss_alpha
        return sem_ce_loss, iou

    def output_metrics(self, x, pred):
        num_classes = self.out_channels
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        # mask = torch.cat((mask), dim=0).to(x.device)

        target = self.patchify_3d(x)
        pred = self.unpatchify_3d(pred)

        # Now we only consider class >0, don't consider void class ##IMPORTANT, maybe change it later
        mask = target > 0
        mask = mask.long()
        # mask = mask.reshape(B * H * W * D * L)

        # DICELOSS CALCULATION==================
        # pred = pred.permute(0, 5, 1, 2, 3, 4)
        # target = target.squeeze(-1).unsqueeze(1)
        # mask = mask.squeeze(-1).unsqueeze(1)
        # sem_ce_loss_val = self.criterion(pred, target, mask)
        # DICELOSS CALCULATION==================

        sem_ce_loss_val = masked_cross_entropy(target, pred, mask)
        _, iou_val = self.iou_loss(pred, target, mask)

        # sem_ce_loss_val, iou_val = self.iou_loss(pred, target, mask)

        # loss = F.cross_entropy(
        #     pred, target, reduction="none"
        # )  # Use 'none' to prevent reduction
        # masked_loss = loss * mask.view(-1)
        # sem_ce_loss_val = masked_loss.sum() / mask.sum()

        # mask = mask.reshape(B * H * W * D * L)

        # probs = F.softmax(pred, dim=1)
        # predicted_labels = torch.argmax(probs, dim=1)

        # pred = pred.permute(0, 2, 3, 4, 5, 1)
        # target = target.squeeze(1)
        # mask = mask.squeeze(1)
        # print("pred", pred.shape, "target", target.shape, "mask", mask.shape)

        B, H, W, D, L, _ = target.shape
        target = target.reshape(B * H * W * D * L).long()  # Flatten the target tensor

        pred = pred.reshape(
            B * H * W * D * L, num_classes
        )  # Flatten and keep the number of classes

        probabilities = F.softmax(pred, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        predicted_labels = predicted_labels[mask.view(-1) == 1]
        target = target[mask.view(-1) == 1]

        # evaluator.update(
        #     predicted_labels.detach().cpu().numpy(), target.detach().cpu().numpy()
        # )

        intersection, union, target = intersectionAndUnionGPU(
            predicted_labels, target, num_classes
        )

        # iou_score = intersection_over_union(predicted_labels, target, mask)
        # accuracy_score = pixel_accuracy(predicted_labels, target, mask)

        metrics = {}
        metrics["intersection"] = intersection
        metrics["union"] = union
        metrics["target"] = target
        metrics["iou_val"] = iou_val

        return sem_ce_loss_val, metrics

    def forward_decoder(self, latent):
        latent = latent.permute(0, 4, 1, 2, 3)
        out = self.semantic_decoder(latent)
        B = latent.shape[0]
        out = out.view(B, 1216, 40, 40, 40)
        out = out.permute(0, 2, 3, 4, 1)
        return out

    def forward(self, x):
        # original_x = x.clone().detach()
        # Forward pass through the SwinTransformer base

        # print("x", x.shape)

        # pad x to have same res for all voxels in the batch (masking is taken care of my output alpha>0.01 i think?)
        x, _ = self.transform(x, resolution=self.input_resolution)
        x = torch.cat((x), dim=0)
        x = self.patch_partition(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)

        pred = self.forward_decoder(x)
        # loss_rgb = self.forward_loss(original_x, pred)
        return pred

    def loss_fn(self, x, pred):
        loss_rgb = self.forward_loss(x, pred)
        return loss_rgb


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

    # backbone = SwinTransformer_VoxelSemantics_Skip(
    #     patch_size=[4, 4, 4],
    #     embed_dim=swin[backbone_type]["embed_dim"],
    #     depths=swin[backbone_type]["depths"],
    #     num_heads=swin[backbone_type]["num_heads"],
    #     window_size=[4, 4, 4],
    #     stochastic_depth_prob=0,
    #     expand_dim=True,
    #     input_dim=4,
    # )

    # backbone = SwinTransformer_VoxelSemantics_Pretrained_Skip(
    #     resolution=160,
    #     checkpoint_path="/home/mirshad7/Downloads/epoch_1200.pt",
    #     is_eval=False,
    # )

    backbone = SwinTransformer_FPN_Pretrained_Skip(checkpoint_path="NeRF-MAE/checkpoints/nerf_mae_pretrained.pt", 
                                                   is_eval=False)

    # backbone = SwinTransformer_VoxelSR_Skip(
    #     patch_size=[4, 4, 4],
    #     embed_dim=swin[backbone_type]["embed_dim"],
    #     depths=swin[backbone_type]["depths"],
    #     num_heads=swin[backbone_type]["num_heads"],
    #     window_size=[4, 4, 4],
    #     stochastic_depth_prob=0,
    #     expand_dim=True,
    #     input_dim=4,
    # )

    # backbone = SwinTransformer_VoxelSR_Pretrained_Skip(is_eval=True, out_resolution=384)

    num_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print("num_params", num_params)

    grid = torch.randn((1, 4, 160, 160, 160))

    # # sem_feat = [torch.randn((1, 120, 106, 160)), torch.randn((1, 105, 110, 160))]

    # # sem_feat = [torch.randn((1, 120, 106, 160))]

    # vox_feat_out = torch.randn((1, 4, 384, 384, 384))
    out = backbone(grid)

    print("out", out[0].shape)

    # loss = backbone.loss_fn(vox_feat_out, out)

    # print("loss", loss)

    # metrics = backbone.output_metrics(out_feat, out)

    # print("metrics", metrics["MSE"], metrics["PSNR"])

    # backbone = SwinTransformer_FPN(
    #     patch_size=[4, 4, 4],
    #     embed_dim=swin[backbone_type]["embed_dim"],
    #     depths=swin[backbone_type]["depths"],
    #     num_heads=swin[backbone_type]["num_heads"],
    #     window_size=[4, 4, 4],
    #     stochastic_depth_prob=0.1,
    #     expand_dim=True,
    # )

    # total_params = 0
    # for i, stage in enumerate(backbone.stages):
    #     num_params = sum(p.numel() for p in stage.parameters() if p.requires_grad)
    #     total_params += num_params
    #     print(f"Stage {i+1} - num_params: {num_params}")

    # print("Number of Swin parameters:", total_params)

    # num_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    # print("num_params", num_params)

    # grid = torch.randn((2, 4, 160, 160, 160)).cuda()
    # backbone = backbone.cuda()

    # out = backbone(grid)
