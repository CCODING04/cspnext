from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class BatchDynamicSoftLabelAssigner(nn.Module):
    def __init__(self,
                 num_classes,
                 soft_center_radius: float = 3.0,
                 topk: int = 13,
                 iou_weight: float = 3.0,
                 iou_calculator: Dict = dict(type='BboxOverlaps2D'),
                 batch_iou: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        # BboxOverlap2D
        self.batch_iou = batch_iou