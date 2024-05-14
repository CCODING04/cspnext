from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cspnext.core import ConvModule
from cspnext.task_modules.prior_generators import MlvlPointGenerator
from cspnext.utils import InstanceList, OptInstanceList


class RTMDetSepBNHeadModule(nn.Module):
    def __init__(self,
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
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[Dict] = None):
        super().__init__()
        self.share_conv = share_conv
        self.num_classes = num_classes
        self.pred_kernel_size = pred_kernel_size
        self.feat_channels = int(feat_channels * widen_factor)
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
                chn = self.in_channels if i == 0  else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
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
                        act_cfg=self.act_cfg
                    )
                )
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.num_classes,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2
                )
            )
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2
                )
            )
        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
    

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> tuple:
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
    def __init__(self,
                 head_module: Dict,
                 prior_generator: Optional[Dict] = dict(
                     type='MlvlPointGenerator',
                     offset=0,
                     strides=[8, 16, 32]
                 ),
                 bbox_coder: Optional[Dict] = dict(
                     type='DistancePointBBoxCoder'
                 ),
                 loss_cls: Optional[Dict]=dict(
                     type='QualityFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0
                 ),
                 loss_bbox: Optional[Dict] = dict(
                     type='GIoULoss',
                     loss_weight=2.0
                 ),
                 init_cfg: Optional[Dict] = None
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
        assert _head_module.pop('type', None) == 'RTMDetSepBNHeadModule'
        self.head_module = RTMDetSepBNHeadModule(**_head_module)

        self.num_classes = self.head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)


        # prior generator
        _prior_generator = deepcopy(prior_generator)
        assert _prior_generator.pop('type', None) == 'MlvlPointGenerator'
        self.prior_generaotor = MlvlPointGenerator(**_prior_generator)
        self.num_base_priors = self.prior_generaotor.num_base_priors[0]
        # box coder


        self.featmap_sizes = [torch.empty(1)] * self.num_levels

        

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        # rtmdet doesn't need loss_obj
        self.loss_obj = None

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[List]:
        """forward features from upstream network"""
        return self.head_module(x)
    
    def loss_by_feat(
            self,
            cls_scores: List[torch.Tensor],
            bbox_preds: List[torch.Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> dict:
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:]  for featmap in cls_scores]
        