from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torchvision.ops import batched_nms

from cspnext.core import ConvModule
from cspnext.losses import CrossEntropyLoss, GIoULoss, QualityFocalLoss
from cspnext.structures.bbox import bbox2distance, distance2bbox
from cspnext.structures.bbox.transforms import get_box_wh, scale_boxes
from cspnext.structures.data import InstanceData
from cspnext.task_modules.assigners import BatchDynamicSoftLabelAssigner
from cspnext.task_modules.coders import DistancePointBBoxCoder
from cspnext.task_modules.prior_generators import MlvlPointGenerator
from cspnext.utils import (InstanceList, OptInstanceList,
                           filter_scores_and_topk, gt_instances_preprocess)


class RTMDetSepBNHeadModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widen_factor: float = 1.0,
        num_base_priors: int = 1,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = [8, 16, 32],
        share_conv: bool = True,
        pred_kernel_size: int = 1,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = dict(type="BN"),
        act_cfg: Optional[Dict] = dict(
            type="SiLU", inplace=True
        ),
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.share_conv = share_conv
        self.num_classes = num_classes
        self.pred_kernel_size = pred_kernel_size
        self.feat_channels = int(
            feat_channels * widen_factor
        )
        self.stacked_convs = stacked_convs
        self.num_base_priors = num_base_priors

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        self.in_channels = int(in_channels * widen_factor)

        self._init_layers()

    def _init_layers(self):
        """Initalize layers"""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()

        for n in range(len(self.featmap_strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = (
                    self.in_channels
                    if i == 0
                    else self.feat_channels
                )
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.num_classes,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                )
            )
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                )
            )
        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = (
                        self.cls_convs[0][i].conv
                    )
                    self.reg_convs[n][i].conv = (
                        self.reg_convs[0][i].conv
                    )

    def forward(
        self, feats: Tuple[torch.Tensor, ...]
    ) -> tuple:
        """Forward features from the upstream network"""
        cls_scores = []
        bbox_preds = []
        for idx, x in enumerate(feats):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            reg_dist = self.rtm_reg[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return tuple(cls_scores), tuple(bbox_preds)


class RTMDetHead(nn.Module):
    def __init__(
        self,
        head_module: Dict,
        prior_generator: Optional[Dict] = dict(
            type="MlvlPointGenerator",
            offset=0,
            strides=[8, 16, 32],
        ),
        bbox_coder: Optional[Dict] = dict(
            type="DistancePointBBoxCoder"
        ),
        loss_cls: Optional[Dict] = dict(
            type="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_bbox: Optional[Dict] = dict(
            type="GIoULoss", loss_weight=2.0
        ),
        assinger: Optional[Dict] = dict(
            type="BatchDynamicSoftLabelAssigner",
            num_classes=80,
            topk=13,
            iou_calculator=dict(type="BboxOverlaps2D"),
        ),
        test_cfg: Optional[dict] = dict(
            multi_label=True,
            # The number of boxes before NMS
            nms_pre=30000,
            score_thr=0.3,  # Threshold to filter out boxes.
            nms=dict(
                type="nms", iou_threshold=0.45
            ),  # NMS type and threshold
            max_per_img=300,
        ),
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        # super().__init__(
        #     head_module=head_module,
        #     prior_generator=prior_generator,
        #     bbox_coder=bbox_coder,
        #     loss_cls=loss_cls,
        #     init_cfg=init_cfg
        # )

        # head module
        _head_module = deepcopy(head_module)
        assert (
            _head_module.pop("type", None)
            == "RTMDetSepBNHeadModule"
        )
        self.head_module = RTMDetSepBNHeadModule(
            **_head_module
        )

        self.num_classes = self.head_module.num_classes
        self.featmap_strides = (
            self.head_module.featmap_strides
        )
        self.num_levels = len(self.featmap_strides)

        # coder
        _bbox_coder = deepcopy(bbox_coder)
        assert (
            _bbox_coder.pop("type", None)
            == "DistancePointBBoxCoder"
        )
        self.bbox_coder = DistancePointBBoxCoder(
            **_bbox_coder
        )

        # loss func
        # rtmdet doesn't need loss_obj
        self.loss_obj = None
        _loss_cls = deepcopy(loss_cls)
        assert (
            _loss_cls.pop("type", None)
            == "QualityFocalLoss"
        )
        self.loss_cls = QualityFocalLoss(**_loss_cls)
        _loss_bbox = deepcopy(loss_bbox)
        assert _loss_bbox.pop("type", None) == "GIoULoss"
        self.loss_bbox = GIoULoss(**_loss_bbox)

        # prior generator
        _prior_generator = deepcopy(prior_generator)
        assert (
            _prior_generator.pop("type", None)
            == "MlvlPointGenerator"
        )
        self.prior_generaotor = MlvlPointGenerator(
            **_prior_generator
        )
        self.num_base_priors = (
            self.prior_generaotor.num_base_priors[0]
        )
        # box coder

        self.featmap_sizes = [
            torch.empty(1)
        ] * self.num_levels

        self.use_sigmoid_cls = loss_cls.get(
            "use_sigmoid", False
        )
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        # special init
        _assigner = deepcopy(assinger)
        assert (
            _assigner.pop("type", None)
            == "BatchDynamicSoftLabelAssigner"
        )
        self.assigner = BatchDynamicSoftLabelAssigner(
            **_assigner
        )

        self.featmap_sizes_train = None
        self.flatten_priors_train = None

        # config
        self.test_cfg = test_cfg

    def forward(
        self, x: Tuple[torch.Tensor]
    ) -> Tuple[List]:
        """forward features from upstream network"""
        return self.head_module(x)

    def loss(
        self,
        x: Tuple[torch.Tensor],
        batch_data_samples: Union[list, dict],
    ) -> dict:
        if isinstance(batch_data_samples, list):
            # TODO
            pass
        else:
            outs = self(x)
            loss_inputs = outs + (
                batch_data_samples["bbox_labels"],
                batch_data_samples["img_metas"],
            )
            losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        num_imgs = len(batch_img_metas)
        featmap_sizes = [
            featmap.size()[-2:] for featmap in cls_scores
        ]
        assert (
            len(featmap_sizes)
            == self.prior_generaotor.num_levels
        )
        gt_info = gt_instances_preprocess(
            batch_gt_instances, num_imgs
        )
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (
            gt_bboxes.sum(-1, keepdim=True) > 0
        ).float()

        device = cls_scores[0].device

        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = (
                self.prior_generaotor.grid_priors(
                    featmap_sizes,
                    device=device,
                    with_stride=True,
                )
            )
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0
            )

        flatten_cls_scores = torch.cat(
            [
                cls_score.permute(0, 2, 3, 1).reshape(
                    num_imgs, -1, self.cls_out_channels
                )
                for cls_score in cls_scores
            ],
            1,
        ).contiguous()

        flatten_bboxes = torch.cat(
            [
                bbox_pred.permute(0, 2, 3, 1).reshape(
                    num_imgs, -1, 4
                )
                for bbox_pred in bbox_preds
            ],
            1,
        )
        # flatten_bboxes_1 = flatten_bboxes * self.flatten_priors_train[..., -1, None]
        flatten_bboxes = (
            flatten_bboxes
            * self.flatten_priors_train[
                None,
                ...,
                -2:,
            ].repeat(1, 1, 2)
        )
        flatten_bboxes = distance2bbox(
            self.flatten_priors_train[..., :2],
            flatten_bboxes,
        )

        # assigner
        assigned_result = self.assigner(
            flatten_bboxes.detach(),
            flatten_cls_scores.detach(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag,
        )

        labels = assigned_result["assigned_labels"].reshape(
            -1
        )
        label_weights = assigned_result[
            "assigned_labels_weights"
        ].reshape(-1)
        bbox_targets = assigned_result[
            "assigned_bboxes"
        ].reshape(-1, 4)
        assign_metric = assigned_result[
            "assign_metrics"
        ].reshape(-1)
        cls_preds = flatten_cls_scores.reshape(
            -1, self.num_classes
        )
        bbox_preds = flatten_bboxes.reshape(-1, 4)

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = (
            ((labels >= 0) & (labels < bg_class_ind))
            .nonzero()
            .squeeze(1)
        )
        avg_factor = (
            (assign_metric.sum()).clamp_(min=1).item()
        )

        # loss_cls
        loss_cls = self.loss_cls(
            cls_preds,
            (labels, assign_metric),
            label_weights,
            avg_factor=avg_factor,
        )

        # los_bbox
        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                bbox_preds[pos_inds],
                bbox_targets[pos_inds],
                weight=assign_metric[pos_inds],
                avg_factor=avg_factor,
            )
        else:
            loss_bbox = bbox_preds.sum() * 0

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def predict(
        self,
        x: Tuple[torch.Tensor],
        batch_data_samples: InstanceList,
        rescale: bool = False,
    ) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo
            for data_samples in batch_data_samples
        ]
        outs = self(x)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            rescale=rescale
        )
        return predictions

    def _bbox_post_process(
        self,
        results: InstanceData,
        cfg: Dict,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: Optional[dict] = None,
    ) -> InstanceData:
        """bbox post-processing method"""
        if rescale:
            assert img_meta.get("scale_factor") is not None
            scale_factor = [
                1 / s for s in img_meta["scale_factor"]
            ]
            results.bboxes = scale_boxes(
                results.bboxes, scale_factor
            )

        if hasattr(results, "score_factors"):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop("score_factors")
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get("min_bbox_size", -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (
                h > cfg.min_bbox_size
            )
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            iou_threshold = cfg['nms'].get("iou_threshold", 0)
            keep_idxs = batched_nms(
                results.bboxes,
                results.scores,
                results.labels,
                iou_threshold,
            )
            # det_bboxes = bboxes[keep_idxs]
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            # results.scores = results.scores[keep_idxs, -1]
            results = results[: cfg['max_per_img']]

        return results

    def predict_by_feat(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        score_factors: Optional[List[torch.Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[Dict] = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceList:

        assert len(cls_scores) == len(bbox_preds)
        cfg = self.test_cfg if cfg is None else cfg
        cfg = deepcopy(cfg)

        # with_objectness = False
        multi_label = cfg.get('multi_label', False)
        multi_label &= self.num_classes > 1
        cfg['multi_label'] = multi_label
        num_imgs = len(batch_img_metas)

        featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]

        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = (
                self.prior_generaotor.grid_priors(
                    featmap_sizes,
                    dtype=cls_scores[0].dtype,
                    device=cls_scores[0].device,
                )
            )
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (
                    featmap_size.numel()
                    * self.num_base_priors,
                ),
                stride,
            )
            for featmap_size, stride in zip(
                featmap_sizes, self.featmap_strides
            )
        ]
        flatten_stride = torch.cat(mlvl_strides)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_classes
            )
            for cls_score in cls_scores
        ]

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, 4
            )
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(
            flatten_cls_scores, dim=1
        ).sigmoid()

        flatten_bbox_preds = torch.cat(
            flatten_bbox_preds, dim=1
        )

        # bbox_coder
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None],
            flatten_bbox_preds,
            flatten_stride,
        )

        results_list = []
        for bboxes, scores, img_meta in zip(
            flatten_decoded_bboxes,
            flatten_cls_scores,
            batch_img_metas,
        ):
            ori_shape = img_meta["ori_shape"]
            scale_factor = img_meta["scale_factor"]
            if "pad_param" in img_meta:
                pad_param = img_meta["pad_param"]
            else:
                pad_param = None

            score_thr = cfg.get("score_thr", -1)

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get("nms_pre", 100000)
            if cfg['multi_label'] is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = (
                    filter_scores_and_topk(
                        scores,
                        score_thr,
                        nms_pre,
                        results=dict(labels=labels[:, 0]),
                    )
                )
                labels = results["labels"]

            else:
                scores, labels, keep_idxs, _ = (
                    filter_scores_and_topk(
                        scores, score_thr, nms_pre
                    )
                )

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
            )

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor(
                        [
                            pad_param[2],
                            pad_param[0],
                            pad_param[2],
                            pad_param[0],
                        ]
                    )
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor
                ).repeat((1, 2))


            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta
            )
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
