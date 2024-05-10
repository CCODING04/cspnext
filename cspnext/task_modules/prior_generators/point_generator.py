from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

from cspnext.utils import DeviceType


class MlvlPointGenerator:
    """Standard points generator for multi-level(Mlvl)
    features maps in 2D points-based detectors"""

    def __init__(
        self, strides: Union[List[int], List[Tuple[int, int]]], offset: float = 0.5
    ) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset


    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)
    
    @property
    def num_base_priors(self) -> List[int]:
        """list[int]: The number of priors(points) at 
        a point on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(
        self, x: Tensor, y: Tensor, row_major: bool = True
    ) -> Tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)
        else:
            return yy.reshape(-1), xx.reshape(-1)

    def single_level_grid_priors(
        self,
        featmap_size: Tuple[int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: DeviceType = "cuda",
        with_stride: bool = False,
    ) -> Tensor:
        """Generate grid Points of a single level."""
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        # easy convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
        shift_y = shift_y.to(dtype)

        # self._meshgrid
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
            stride_h = shift_yy.new_full((shift_yy.shape[0],), stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        
        all_points = shifts.to(device)
        return all_points
    

    def single_level_valid_flags(self,
                                 featmap_size: Tuple[int, int],
                                 valid_size: Tuple[int, int],
                                 device: DeviceType = 'cuda') -> Tensor:
        """Generate the valid flags of points of a single feature map"""
        feat_h ,feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid
    
    def single_level_valid_flags_V2(self,
                                    featmap_size: Tuple[int, int],
                                    valid_size: Tuple[int, int],
                                    device: DeviceType = 'cuda') -> Tensor:
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid = torch.zeros((feat_h, feat_w), dtype=torch.bool, device=device)
        valid[:valid_h, :valid_w] = 1
        return valid
    
    def grid_priors(self,
                    featmap_sizes: List[Tuple],
                    dtype:torch.dtype = torch.float32,
                    device: DeviceType = 'cuda',
                    with_stride: bool = False) -> List[Tensor]:
        """Generate grid points of multiple feature levels"""
        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride
            )
            multi_level_priors.append(priors)
        return multi_level_priors
    
    def valid_flags(self,
                    featmap_sizes: List[Tuple[int, int]],
                    pad_shape: Tuple[int],
                    device: DeviceType = 'cuda') -> List[Tensor]:
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags(
                (feat_h, feat_w),
                (valid_feat_h, valid_feat_w),
                device=device
            )
            multi_level_flags.append(flags)
        return multi_level_flags
    
    def sparse_priors(self,
                      prior_idxs: Tensor,
                      featmap_size: Tuple[int],
                      level_idx: int,
                      dtype: torch.dtype = torch.float32,
                      device: DeviceType = 'cuda') -> Tensor:
        """Generate sparse points according to the ''prior_idxs''"""
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height + self.offset) * self.strides[level_idx][1]
        priors = torch.stack([x, y], 1).to(dtype)
        return priors

    

