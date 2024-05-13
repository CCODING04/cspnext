import torch

from cspnext.structures.bbox import bbox_overlaps


def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        x = (x / scale).half()
    return x

class BboxOverlaps2D:
    """2D Overlaps Calculator"""
    def __init__(self, scale=1.,dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]

        if self.dtype == 'fp16':
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, )
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                overlaps = overlaps.float()
            return overlaps
        
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
    
    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(' \
                   f'scale={self.scale}, dtype={self.dtype}'
        return repr_str