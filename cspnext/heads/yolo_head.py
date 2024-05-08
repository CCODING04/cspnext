from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class YOLOv5Head(nn.Module):
    def __init__(self,
                 head_module: Dict,
                 prior_generator: Dict = dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[
                         [(10, 13), (16, 30), (33, 23)],
                         [(30, 61), (62, 45), (59, 119)],
                         [(116, 90), (156, 198), (373, 326)]
                     ],
                     strides=[8, 16, 32]
                 ),
                 bbox_coder: Dict = dict(type='YOLOv5BBoxCoder'),
                 loss_cls: Dict = dict(
                     type='CrossEntroyLoss'
                 ),
                 loss_bbox: Dict = dict(
                     type='IoULoss'
                 ),
                 loss_obj: Dict = dict(
                     type='CrossEntropyLoss'
                 ),
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5,
                 ignore_iof_thr: float = -1.0,
                 obj_level_weights: List[float] = [4.0, 1.0, 0.4],
                 init_cfg: Optional[Dict] = None
                 ):
        super().__init__()