from copy import deepcopy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .iou2d_calculator import BboxOverlaps2D


def find_inside_points(boxes: Tensor,
                       points: Tensor,
                       box_dim: int = 4,
                       eps: float = 0.01) -> Tensor:
    if box_dim == 4:
        # horizontal boxes
        lt_ = points[:, None, None] - boxes[..., :2]
        rb_ = boxes[..., 2:] - points[:, None, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
    elif box_dim == 5:
        # rotated boxes
        # TODO
        raise NotImplementedError('box_dim = 5 has not been implemented')
    return is_in_gts

    


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
        _iou_calculator = deepcopy(iou_calculator)
        assert _iou_calculator.pop('type', None) == 'BboxOverlaps2D'
        self.iou_calculator = BboxOverlaps2D(**_iou_calculator)
        self.batch_iou = batch_iou

    @torch.no_grad
    def forward(self,
                pred_bboxes: Tensor,
                pred_scores: Tensor,
                priors: Tensor,
                gt_labels: Tensor,
                gt_bboxes: Tensor,
                pad_bbox_flag: Tensor) -> dict:
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        batch_size, num_bboxes, box_dim = decoded_bboxes.size()

        if num_gt == 0 or num_bboxes == 0:
            return {
                'assigned_labels':
                gt_labels.new_full(
                    pred_scores[..., 0].shape,
                    self.num_classes,
                    dtype=torch.long
                ),
                'assigned_labels_weights':
                gt_bboxes.new_full(pred_scores[..., 0].shape, 1),
                'assigned_bboxes':
                gt_bboxes.new_full(pred_bboxes.shape, 0),
                'assign_metrics':
                gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
            }
        
        prior_center = priors[:, :2]
        is_in_gts = find_inside_points(gt_bboxes, prior_center, box_dim)

        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]

        is_in_gts = is_in_gts.permute(1, 0, 2)

        valid_mask = is_in_gts.sum(dim=-1) > 0
        