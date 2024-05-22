import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .accuracy import accuracy
from .utils import weight_reduce_loss


def cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
    avg_non_ignore=False,
):
    """Calculate the CrossEntropy loss"""
    ignore_index = (
        -100 if ignore_index is None else ignore_index
    )
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction="none",
        ignore_index=ignore_index,
    )
    if (
        (avg_factor is None)
        and avg_non_ignore
        and reduction == "mean"
    ):
        avg_factor = (
            label.numel()
            - (label == ignore_index).sum().item()
        )
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss,
        weight=weight,
        reduction=reduction,
        avg_factor=avg_factor,
    )
    return loss


def _epxand_onehot_labels(
    labels, label_weights, label_channels, ignore_index
):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full(
        (labels.size(0), label_channels), 0
    )
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels),
        as_tuple=False,
    )
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = (
        valid_mask.view(-1, 1)
        .expand(labels.size(0), label_channels)
        .float()
    )

    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(
            -1, 1
        ).repeat(1, label_channels)
        bin_label_weights *= valid_mask
    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
    avg_non_ignore=False,
):
    """Calculate the binary CrossEntropy loss."""
    ignore_index = (
        -100 if ignore_index is None else ignore_index
    )

    if pred.dim() != label.dim():
        label, weight, valid_mask = _epxand_onehot_labels(
            label, weight, pred.size(-1), ignore_index
        )
    else:
        valid_mask = (
            (label >= 0) & (label != ignore_index)
        ).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask

    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred,
        label.float(),
        pos_weight=class_weight,
        reduction="none",
    )
    loss = weight_reduce_loss(
        loss,
        weight,
        reduction=reduction,
        avg_factor=avg_factor,
    )
    return loss


def mask_cross_entropy(
    pred,
    target,
    label,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=None,
    **kwargs
):
    """Calculate the CrossEntropy loss for masks"""
    assert (
        ignore_index is None
    ), "BCE loss does not suppoert ignore_index"
    assert reduction == "mean" and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(
        0, num_rois, dtype=torch.long, device=pred.device
    )
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice,
        target,
        weight=class_weight,
        reduction="mean",
    )[None]


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction="mean",
        num_classes=-1,
        class_weight=None,
        ignore_index=None,
        loss_weight=1.0,
        avg_non_ignore=False,
    ):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if (
            (ignore_index is not None)
            and not self.avg_non_ignore
            and self.reduction == "mean"
        ):
            warnings.warn(
                'XXXXXXXXXXXXXXX'
            )
        
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        
        self.num_classes = num_classes

        assert self.num_classes != -1

        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True

    def get_cls_channesl(self, num_classes):
        assert num_classes == self.num_classes
        if not self.use_sigmoid:
            return num_classes + 1
        else:
            return num_classes

    
    def get_activation(self, cls_score):
        fine_cls_score = cls_score[:, :self.num_classes]

        if not self.use_sigmoid:
            bg_score = cls_score[:, [-1]]
            new_score = torch.cat([fine_cls_score, bg_score], dim=-1)
            scores = F.softmax(new_score, dim=-1)
        else:
            score_classes = fine_cls_score.sigmoid()
            score_neg = 1 - score_classes.sum(dim=1, keepdim=True)
            score_neg = score_neg.clamp(min=0, max=1)
            scores = torch.cat([score_classes, score_neg], dim=1)
        return scores


    def get_accuracy(self, cls_score, labels):
        fine_cls_score = cls_score[:, :self.num_classes]
        
        pos_inds = labels < self.num_classes
        acc_classes = accuracy(fine_cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        return acc
