from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def reduce_loss(loss: Tensor, )

def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)


def cross_entropy(
        pred,
        label,
        weight=None,
        reduction='mean',
        avg_factor=None,
        class_weight=None,
        ignore_index=-100,
        avg_non_ignore=False
):
    """Calculate the CrossEntropy loss"""
    ignore_index = -100 if ignore_index is None else ignore_index
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index
    )
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    if weight is not None:
        weight = weight.float()
    