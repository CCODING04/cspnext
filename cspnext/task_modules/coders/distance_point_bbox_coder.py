from typing import Optional, Sequence, Union

import torch
from torch import Tensor


class DistancePointBBoxCoder:
    """Distance Point Bbox coder."""
    def __init__(self,
                 points: Tensor,
                 pred_bboxes: Tensor,
                 stride: Tensor,
                 max_shape: Optional[])