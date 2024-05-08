from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair


class MlvlPointGenerator:
    """Standard points generator for multi-level(Mlvl) 
    features maps in 2D points-based detectors"""

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 offset: float = 0.5) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int],
                                 level_idx: int)