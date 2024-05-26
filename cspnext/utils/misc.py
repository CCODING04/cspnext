from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

from cspnext.structures import InstanceData


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates."""
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(
                f"Only supports dict or list or Tensor, " f"but get {type(results)}."
            )
    return scores, labels, keep_idxs, filtered_results


def gt_instances_preprocess(
    batch_gt_instances: Union[Tensor, Sequence], batch_size: int
) -> Tensor:
    """Split batch_gt_instances with batch size"""
    batch_instance = None
    if isinstance(batch_gt_instances, Sequence):
        max_gt_bbox_len = max(
            [len(gt_instances) for gt_instances in batch_gt_instances]
        )
        batch_instance_list = []
        for index, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            labels = gt_instance.labels
            box_dim = bboxes.size(-1)
            batch_instance_list.append(torch.cat((labels[:, None], bboxes), dim=-1))
            if bboxes.shape[0] >= max_gt_bbox_len:
                continue

            fill_tensor = bboxes.new_full(
                [max_gt_bbox_len - bboxes.shape[0], box_dim + 1], 0
            )
            batch_instance_list[index] = torch.cat(
                (batch_instance_list[index], fill_tensor), dim=0
            )
        return torch.stack(batch_instance_list)
    else:
        # format of batch_gt_instances
        # [img_ind, cls_ind, (box)]
        assert isinstance(batch_gt_instances, Tensor)
        box_dim = batch_gt_instances.size(-1) - 2
        if len(batch_gt_instances) > 0:
            gt_images_indexes = batch_gt_instances[:, 0]
            max_gt_bbox_len = gt_images_indexes.unique(return_counts=True)[1].max()
            batch_instance = torch.zeros(
                (batch_size, max_gt_bbox_len, box_dim + 1),
                dtype=batch_gt_instances.dtype,
                device=batch_gt_instances.device,
            )
            for i in range(batch_size):
                match_indexes = gt_images_indexes == i
                gt_num = match_indexes.sum()
                if gt_num:
                    batch_instance[i, :gt_num] = batch_gt_instances[match_indexes, 1:]
        else:
            batch_instance = torch.zeros(
                (batch_size, 0, box_dim + 1),
                dtype=batch_gt_instances.dtype,
                device=batch_gt_instances.device,
            )
    return batch_instance
