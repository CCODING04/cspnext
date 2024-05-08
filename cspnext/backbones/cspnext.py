import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from cspnext.core import ConvModule
from cspnext.layers import CSPLayer, SPPFBottleneck


class CSPNext(nn.Module):

    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 1024, 3, False, True],
        ],
        "P6": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 768, 3, True, False],
            [768, 1024, 3, False, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_overwrite: dict = None,
        channel_attention: bool = True,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = dict(type="BN"),
        act_cfg: Optional[Dict] = dict(type="SiLU", inplace=True),
        norm_eval: bool = False,
        init_cfg: Optional[Dict] = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ) -> None:
        
        arch_setting = self.arch_settings[arch]
        if arch_overwrite:
            arch_setting = arch_overwrite

        self.channel_attention = channel_attention
        self.use_depthwise = use_depthwise
        # TODO: use_depthwise = True
        assert use_depthwise == False
        self.conv = ConvModule
        self.expand_ratio = expand_ratio
        self.conv_cfg = conv_cfg

        super().__init__()

        self.num_stages = len(arch_setting)
        self.arch_setting = arch_setting

        assert set(out_indices).issubset(
            i for i in range(self.num_stages + 1)
        )

        if frozen_stages not in range(-1, self.num_stages + 1):
            raise ValueError('"fronzen_stages" must be in '
                             'range(-1, len(arch_setting) + 1),'  
                             f'but received {frozen_stages}')

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        self.stem = self.build_stem_layer()
        self.layers = ['stem']

        for idx, setting in enumerate(arch_setting):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            self.add_module(f'stage{idx+1}', nn.Sequential(*stage))
            self.layers.append(f'stage{idx + 1}')

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer"""
        stem = nn.Sequential(
            ConvModule(
                3,
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                int(self.arch_setting[0][0] * self.widen_factor //  2),
                int(self.arch_setting[0][0] * self.widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        )
        return stem
    

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer"""
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)
        num_blocks = max(round(num_blocks * self.deepen_factor), 1)

        stage = []
        conv_layer = self.conv(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        stage.append(conv_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            stage.append(spp)
        
        # csp_layer
        csp_layer = CSPLayer(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            use_depthwise=self.use_depthwise,
            use_cspnext_block=True,
            expand_ratio=self.expand_ratio,
            channel_attention=self.channel_attention,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        stage.append(csp_layer)
        return stage
    
    def forward(self, x: torch.Tensor) -> tuple:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
        
