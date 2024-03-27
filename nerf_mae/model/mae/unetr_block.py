# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import numpy as np

# from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer


class UnetResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        norm_name="instancenorm",
        act_name="leakyrelu",
        dropout=None,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding=(kernel_size // 2)
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size // 2),
        )
        self.norm1 = self.get_norm_layer(norm_name, out_channels)
        self.norm2 = self.get_norm_layer(norm_name, out_channels)
        self.activation = self.get_activation_layer(act_name)
        self.dropout = self.get_dropout_layer(dropout) if dropout is not None else None

        self.downsample = in_channels != out_channels
        if self.downsample:
            self.conv3 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
            self.norm3 = self.get_norm_layer(norm_name, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.dropout:
            out = self.dropout(out)
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)
        out += residual
        out = self.activation(out)
        return out

    def get_norm_layer(self, norm_name, num_channels):
        if norm_name == "batchnorm":
            return nn.BatchNorm3d(num_channels)
        elif norm_name == "instancenorm":
            return nn.InstanceNorm3d(num_channels)
        else:
            raise ValueError(f"Unsupported normalization: {norm_name}")

    def get_activation_layer(self, act_name):
        if act_name == "leakyrelu":
            return nn.LeakyReLU(0.01, inplace=True)
        elif act_name == "relu":
            return nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {act_name}")

    def get_dropout_layer(self, dropout):
        if isinstance(dropout, float):
            return nn.Dropout3d(dropout)
        else:
            raise ValueError(f"Unsupported dropout type: {type(dropout)}")


class UnetOutBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = None):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        if dropout is not None:
            self.dropout = nn.Dropout3d(p=dropout)
        else:
            self.dropout = None

    def forward(self, inp):
        x = self.conv(inp)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: str = "instancenorm",
        res_block: bool = False,
        use_skip: bool = True,
        input_padding=0,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        self.use_skip = use_skip
        upsample_stride = upsample_kernel_size
        self.transp_conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            upsample_kernel_size,
            stride=upsample_stride,
            padding=input_padding,
            output_padding=0,
        )

        if res_block:
            if use_skip:
                self.conv_block = UnetResBlock(
                    out_channels + out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                )
            else:
                self.conv_block = UnetResBlock(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                )
            # self.conv_block = UnetResBlock(
            #     out_channels + out_channels,
            #     out_channels,
            #     kernel_size=kernel_size,
            #     stride=1,
            #     norm_name=norm_name,
            # )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip=None):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        if self.use_skip:
            out = torch.cat((out, skip), dim=1)
        # out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetResBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: str = "instancenorm",
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        return self.layer(inp)
