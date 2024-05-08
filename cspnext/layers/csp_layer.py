from typing import Dict, List, Optional

import torch
import torch.nn as nn

from cspnext.core import ConvModule, DepthwiseSeparableConvModule

from .channel_attention import ChannelAttention


class CSPNeXtBlock(nn.Module):
    """The basic bottleneck block used in CSPNeXt"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 kernel_size: int = 5,
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] =  dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[Dict] = dict(type='SiLU'),
                 init_cfg: Optional[Dict] = None
                 ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # TODO: use_depthwise = True
        assert use_depthwise == False
        conv = ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size//2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity:
            return out + identity
        else:
            return out



class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 channel_attention: bool = False,
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = dict(
                     type='BN', momentum=0.03, eps=0.001
                 ),
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__()

        block = CSPNeXtBlock # TODO: if use_cspnext_block else DarknetBOttlenet
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.blocks = nn.Sequential(*[
            block(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)