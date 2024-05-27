from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

# def batched_nms(
#     boxes: Tensor,
#     scores: Tensor,
#     idxs: Tensor,
#     nms_cfg: Optional[Dict],
#     class_agnostic: bool = False,
# ) -> Tuple[Tensor]:
#     if nms_cfg is None:
#         scores, inds = scores.sort(descending=True)
#         boxes = boxes[inds]
#         return torch.cat([boxes, scores[:, None]], -1), inds
    
#     nms_cfg_ = deepcopy(nms_cfg)
#     class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
#     if class_agnostic:
#         boxes_for_nms = boxes
#     else:
