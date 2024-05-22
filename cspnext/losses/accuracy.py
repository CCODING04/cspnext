import torch.nn as nn


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target."""
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.0) for i in range(len(topk))]
        return accu[0] if return_single else accu

    assert pred.ndim == 2 and target.ndim == 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), f"maxk {maxk} exceeds pred dimension {pred.size(1)}"
    pred_value, pred_label = pred.topk(max, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    if thresh is not None:
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res

class Accuracy(nn.Module):
    def __init__(self, topk=(1, ),  thresh=None):
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk, self.thresh)