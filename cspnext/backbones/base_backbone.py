import copy
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 arch_setting: list,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = None,
                 norm_eval: bool = False,
                 init_cfg: Optional[Dict] = None):
        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

        