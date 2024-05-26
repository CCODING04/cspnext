from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .iou2d_calculator import BboxOverlaps2D


def find_inside_points(
    boxes: Tensor, points: Tensor, box_dim: int = 4, eps: float = 0.01
) -> Tensor:
    if box_dim == 4:
        # horizontal boxes
        lt_ = points[:, None, None] - boxes[..., :2]
        rb_ = boxes[..., 2:] - points[:, None, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
    elif box_dim == 5:
        # rotated boxes
        # TODO
        raise NotImplementedError("box_dim = 5 has not been implemented")
    return is_in_gts


def get_box_center(boxes: Tensor, box_dim: int = 4) -> Tensor:
    """Return a tensor representing the centers of boxes"""
    if box_dim == 4:
        return (boxes[..., :2] + boxes[..., 2:]) / 2.0
    elif box_dim == 5:
        return boxes[..., :2]
    else:
        raise NotImplementedError(f"Unsupported box_dim: {box_dim}")


class BatchDynamicSoftLabelAssigner(nn.Module):
    def __init__(
        self,
        num_classes,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
        iou_calculator: Dict = dict(type="BboxOverlaps2D"),
        batch_iou: bool = True,
        eps: float = 1e-7,
        inf: float = 1e8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        # BboxOverlap2D
        _iou_calculator = deepcopy(iou_calculator)
        assert _iou_calculator.pop("type", None) == "BboxOverlaps2D"
        self.iou_calculator = BboxOverlaps2D(**_iou_calculator)
        self.batch_iou = batch_iou
        self.eps = eps
        self.inf = inf

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        priors: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_bbox_flag: Tensor,
    ) -> dict:
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        batch_size, num_bboxes, box_dim = decoded_bboxes.size()

        if num_gt == 0 or num_bboxes == 0:
            return {
                "assigned_labels": gt_labels.new_full(
                    pred_scores[..., 0].shape, self.num_classes, dtype=torch.long
                ),
                "assigned_labels_weights": gt_bboxes.new_full(
                    pred_scores[..., 0].shape, 1
                ),
                "assigned_bboxes": gt_bboxes.new_full(pred_bboxes.shape, 0),
                "assign_metrics": gt_bboxes.new_full(pred_scores[..., 0].shape, 0),
            }

        prior_center = priors[:, :2]
        is_in_gts = find_inside_points(gt_bboxes, prior_center, box_dim)

        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]

        is_in_gts = is_in_gts.permute(1, 0, 2)

        valid_mask = is_in_gts.sum(dim=-1) > 0

        gt_center = get_box_center(gt_bboxes, box_dim)

        strides = priors[..., 2]
        distance = (priors[None].unsqueeze(2)[..., :2] - gt_center[:, None, :, :]).pow(
            2
        ).sum(-1).sqrt() / strides[None, :, None]

        distance = distance * valid_mask.unsqueeze(-1)

        # ???
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        # pred bbox cost
        if self.batch_iou:
            pairwise_ious = self.iou_calculator(decoded_bboxes, gt_bboxes)
        else:
            ious = []
            for box, gt in zip(decoded_bboxes, gt_bboxes):
                iou = self.iou_calculator(box, gt)
                ious.append(iou)
            pairwise_ious = torch.stack(ious, dim=0)

        iou_cost = -torch.log(pairwise_ious + self.eps) * self.iou_weight

        pairwise_pred_scores = pred_scores.permute(0, 2, 1)
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.long().squeeze(-1)
        pairwise_pred_scores = pairwise_pred_scores[idx[0], idx[1]].permute(0, 2, 1)

        # class cost
        scale_factor = pairwise_ious - pairwise_pred_scores.sigmoid()

        pairwise_cls_cost = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pairwise_ious, reduce="none"
        ) * scale_factor.abs().pow(2.0)

        cost_matrix = pairwise_cls_cost + iou_cost + soft_center_prior

        max_pad_value = torch.ones_like(cost_matrix) * self.inf
        cost_matrix = torch.where(
            valid_mask[..., None].repeat(1, 1, num_gt), cost_matrix, max_pad_value
        )

        # dynamic_k_matching
        (matched_pred_ious, matched_gt_inds, fg_mask_inboxes) = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, pad_bbox_flag
        )

        del pairwise_ious, cost_matrix

        batch_index = (fg_mask_inboxes > 0).nonzero(as_tuple=True)[0]

        assigned_labels = gt_labels.new_full(
            pred_scores[..., 0].shape, self.num_classes
        )
        assigned_labels = assigned_labels.long()
        assigned_labels_weights = gt_bboxes.new_full(pred_scores[..., 0].shape, 1)

        assigned_bboxes = gt_bboxes.new_full(pred_bboxes.shape, 0)
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[batch_index, matched_gt_inds]

        assign_metrics = gt_bboxes.new_full(pred_bboxes[..., 0].shape, 0)
        assign_metrics[fg_mask_inboxes] = matched_pred_ious
        return dict(
            assigned_labels=assigned_labels,
            assigned_labels_weights=assigned_labels_weights,
            assigned_bboxes=assigned_bboxes,
            assign_metrics=assign_metrics,
        )

    def dynamic_k_matching(
        self, cost_matrix: Tensor, pairwise_ious: Tensor, pad_bbox_flag: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive targets."""
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)
        candidate_topk = min(self.topk, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        num_gts = pad_bbox_flag.sum((1, 2)).int()

        _, sorted_indices = torch.sort(cost_matrix, dim=1)
        for b in range(pad_bbox_flag.shape[0]):
            for gt_idx in range(num_gts[b]):
                topk_ids = sorted_indices[b, : dynamic_ks[b, gt_idx], gt_idx]
                matching_matrix[b, :, gt_idx][topk_ids] = 1

        del topk_ious, dynamic_ks

        prior_match_gt_mask = matching_matrix.sum(2) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost_matrix[prior_match_gt_mask, :], dim=1
            )
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        fg_mask_inboxes = matching_matrix.sum(2) > 0
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(2)[fg_mask_inboxes]
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes
