from typing import Dict, Optional

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention Module"""
    def __init__(self, 
                 channels: int, 
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out