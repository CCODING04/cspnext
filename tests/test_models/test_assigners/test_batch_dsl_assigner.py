from unittest import TestCase

import pytest
import torch

from cspnext.task_modules.assigners import BatchDynamicSoftLabelAssigner


class TestBatchDynamicSoftLabelAssigner(TestCase):
    def test_assign(self):
        num_classes = 2
        batch_size = 2

        assigner = BatchDynamicSoftLabelAssigner(
            num_classes=num_classes, soft_center_radius=3.0, topk=1, iou_weight=3.0
        )

        pred_bboxes = (
            torch.FloatTensor(
                [
                    [23, 23, 43, 43],
                    [4, 5, 6, 7],
                ]
            )
            .unsqueeze(0)
            .repeat(batch_size, 10, 1)
        )

        pred_scores = (
            torch.FloatTensor(
                [
                    [0.2],
                    [0.8],
                ]
            )
            .unsqueeze(0)
            .repeat(batch_size, 10, 1)
        )

        priors = torch.FloatTensor([[30, 30, 8, 8], [4, 5, 6, 7]]).repeat(10, 1)

        gt_bboxes = (
            torch.FloatTensor([[23, 23, 43, 43]]).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        gt_labels = torch.LongTensor([[0]]).unsqueeze(0).repeat(batch_size, 1, 1)
        pad_bbox_flag = torch.FloatTensor([[1]]).unsqueeze(0).repeat(batch_size, 1, 1)

        assign_result = assigner.forward(
            pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes, pad_bbox_flag
        )

        assigned_labels = assign_result["assigned_labels"]
        assigned_labels_weights = assign_result["assigned_labels_weights"]
        assigned_bboxes = assign_result["assigned_bboxes"]
        assign_metrics = assign_result["assign_metrics"]

        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 20]))
        self.assertEqual(assigned_bboxes.shape, torch.Size([batch_size, 20, 4]))
        self.assertEqual(assigned_labels_weights.shape, torch.Size([batch_size, 20]))
        self.assertEqual(assign_metrics.shape, torch.Size([batch_size, 20]))
