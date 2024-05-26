from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from cspnext.structures.bbox import bbox2distance, distance2bbox


class DistancePointBBoxCoder:
    """Distance Point Bbox coder."""

    def __init__(
        self,
        clip_boder: Optional[bool] = True,
        use_box_type: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.use_box_type = use_box_type
        self.clip_boder = clip_boder

    def encode(
        self,
        points: Tensor,
        gt_bboxes: Tensor,
        max_dis: float = 16,
        eps: float = 0.01,
    ) -> Tensor:
        """Encode bounding box to distances."""
        assert points.size(-2) == gt_bboxes.size(-2)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)

    def decode(
        self,
        points: Tensor,
        pred_bboxes: Tensor,
        stride: Tensor,
        max_shape: Optional[
            Union[
                Sequence[int],
                torch.Tensor,
                Sequence[Sequence[int]],
            ]
        ] = None,
    ) -> Tensor:
        """Decode distance prediction to bounding box."""
        assert points.size(-2) == pred_bboxes.size(-2)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_boder is False:
            max_shape = None
        pred_bboxes = pred_bboxes * stride[None, :, None]
        return distance2bbox(points, pred_bboxes, max_shape)
