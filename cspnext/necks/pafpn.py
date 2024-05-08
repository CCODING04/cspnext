import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from cspnext.core import ConvModule, DepthwiseSeparableConvModule
from cspnext.layers import CSPLayer

from .base_neck import BaseYOLONeck


class CSPNeXtPAFPN(BaseYOLONeck):
    """Path Aggregation Network with CSPNeXt blocks."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        freeze_all: bool = False,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        upsample_cfg: Optional[Dict] = dict(scale_factor=2, mode="nearest"),
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = dict(type="BN"),
        act_cfg: Optional[Dict] = dict(type="SiLU", inplace=True),
        init_cfg: Optional[Dict] = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ) -> None:
        self.num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.upsample_cfg = upsample_cfg
        self.expand_ratio = expand_ratio
        self.conv_cfg = conv_cfg

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg
        )

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(**self.upsample_cfg)
    
    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build top down layer."""
        if idx == len(self.in_channels) - 1:
            layer = self.conv(
                self.in_channels[idx],
                self.in_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        else:
            layer = nn.Identity()
        return layer
    
    def build_top_down_layer(self, idx: int) -> nn.Module:
        if idx == 1:
            return CSPLayer(
                self.in_channels[idx - 1] * 2,
                self.in_channels[idx - 1],
                num_blocks=self.num_csp_blocks,
                add_identity=False,
                use_cspnext_block=True,
                expand_ratio=self.expand_ratio,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        else:
            return nn.Sequential(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1],
                    num_blocks = self.num_csp_blocks,
                    add_identity=False,
                    use_cspnext_block=True,
                    expand_ratio=self.expand_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                self.conv(
                    self.in_channels[idx - 1],
                    self.in_channels[idx - 2],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
        )

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer"""
        return self.conv(
            self.in_channels[idx],
            self.in_channels[idx],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
    
    def build_bottom_up_layer(self, idx: int):
        """build bottom up layer"""
        return CSPLayer(
            self.in_channels[idx] * 2,
            self.in_channels[idx + 1],
            num_blocks=self.num_csp_blocks,
            add_identity=False,
            use_cspnext_block=True,
            expand_ratio=self.expand_ratio,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
    
    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer."""
        return self.conv(
            self.in_channels[idx],
            self.out_channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
