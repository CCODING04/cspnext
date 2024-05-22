import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from cspnext.structures.bbox import bbox_overlaps


def iou_loss(
    pred: Tensor,
    target: Tensor,
    linear: bool = False,
    mode: str = "log",
    eps: float = 1e-6,
) -> Tensor:
    assert mode in ["linear", "square", "log"]
    if linear:
        mode = "linear"
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'iou_loss is deprecated, please use "mode=`linear`" '
            "instead."
        )
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    ious = bbox_overlaps(
        pred, target, is_aligned=True
    ).clamp(min=eps)

    if fp16:
        ious = ious.to(torch.float16)

    if mode == "linear":
        loss = 1 - ious
    elif mode == "square":
        loss = 1 - ious**2
    elif mode == "log":
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


class IoULoss(nn.Module):
    def __init__(
        self,
        linear: bool = False,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        mode: str = "log",
    ) -> None:
        super().__init__()
        assert mode in ["linear", "square", "log"]
        if linear:
            mode = "linear"
            warnings.warn(
                'DeprecationWarning: Setting "linear=True" in '
                'IOULoss is deprecated, please use "mode=`linear`" '
                "instead."
            )
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        **kwargs
    ) -> Tensor:
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
        if (
            (weight is not None)
            and (not torch.any(weight > 0))
            and (reduction != "none")
        ):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()

        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss
