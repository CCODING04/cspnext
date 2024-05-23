from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def quality_focal_loss(
    pred,
    target,
    weight=None,
    beta=2.0,
    reduction=None,
    avg_factor=None,
    activated=False,
):
    """Quaility Focal Loss"""
    assert (
        len(target) == 2
    ), """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    label, score = target

    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = (
        ((label >= 0) & (label < bg_class_ind))
        .nonzero()
        .squeeze(1)
    )
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = (
        F.binary_cross_entropy_with_logits(
            pred[pos, pos_label],
            score[pos],
            reduction="none",
        )
        * scale_factor.abs().pow(beta)
    )
    loss = loss.sum(dim=1, keepdim=False)
    return loss


def quality_focal_loss_with_prob(
    pred,
    target,
    weight=None,
    beta=2.0,
    reduction=None,
    avg_factor=None,
    activated=False,
):
    assert (
        len(target) == 2
    ), """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    label, score = target
    pred_sigmoid = pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)
    bg_class_ind = pred.size(1)
    pos = (
        ((label >= 0) & (label < bg_class_ind))
        .nonzero()
        .squeeze(1)
    )
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)
    loss = loss.sum(dim=1, keepdim=False)
    return loss


def quality_focal_loss_tensor_target(
    pred,
    target,
    weight=None,
    beta=2.0,
    reduction=None,
    avg_factor=None,
    activated=False,
):
    assert pred.size() == target.size()
    if activated:
        pred_sigmoid = pred
        loss_function = F.binary_cross_entropy
    else:
        pred_sigmoid = pred.sigmoid()
        loss_function = F.binary_cross_entropy_with_logits

    scale_factor = pred_sigmoid
    target = target.type_as(pred)

    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = loss_function(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)

    pos = target != 0
    scale_factor = target[pos] - pred_sigmoid[pos]
    loss[pos] = loss_function(
        pred[pos], target[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=True)
    return loss


class QualityFocalLoss(nn.Module):

    def __init__(
        self,
        use_sigmoid=True,
        beta=2.0,
        reduction="mean",
        loss_weight=1.0,
        activated=False,
    ):
        super().__init__()
        assert (
            use_sigmoid is True
        ), "Only sigmoid in QFL supported now."
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ) -> torch.Tensor:
        assert reduction_override in (
            None,
            "none",
            "mean",
            "sum",
        )
        reduction = (
            reduction_override
            if reduction_override
            else self.reduction
        )
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = (
                    quality_focal_loss_with_prob
                )
            else:
                calculate_loss_func = quality_focal_loss
            if isinstance(target, torch.Tensor):
                calculate_loss_func = partial(
                    quality_focal_loss_tensor_target,
                    activated=self.activated,
                )

            loss_cls = (
                self.loss_weight
                * calculate_loss_func(
                    pred,
                    target,
                    weight=weight,
                    beta=self.beta,
                    reduction=reduction,
                    avg_factor=avg_factor,
                )
            )
        else:
            raise NotImplementedError
        return loss_cls
