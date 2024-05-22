import torch
import torch.nn as nn
import torch.nn.functional as F


def quality_focal_loss(pred, target, beta=2.0):
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
        .sequeeze(1)
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
