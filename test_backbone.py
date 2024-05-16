from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from cspnext.backbones import CSPNext
from cspnext.heads import RTMDetHead
from cspnext.necks import CSPNeXtPAFPN
from cspnext.structures import InstanceData


class M(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 80
        self.backbone = CSPNext(
            arch="P5",
            expand_ratio=0.5,
            deepen_factor=1.0,  # 0.33,
            widen_factor=1.0,  # 0.5,
            channel_attention=True,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="SiLU", inplace=True),
        )
        self.neck = CSPNeXtPAFPN(
            in_channels=[256, 512, 1024],
            out_channels=256,
            deepen_factor=1.0,
            widen_factor=1.0,
            num_csp_blocks=3,
            expand_ratio=0.5,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="SiLU", inplace=True),
        )
        self.bbox_head = RTMDetHead(
            head_module=dict(
                type="RTMDetSepBNHeadModule",
                num_classes=num_classes,
                in_channels=256,
                stacked_convs=2,
                feat_channels=256,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="SiLU", inplace=True),
                share_conv=True,
                pred_kernel_size=1,
                featmap_strides=[8, 16, 32],
            ),
            prior_generator=dict(
                type="MlvlPointGenerator", offset=0, strides=[8, 16, 32]
            ),
            bbox_coder=dict(type="DistancePointBBoxCoder"),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.bbox_head(x)
        return x

    def extract_feat(self, batch_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.backbone(batch_inputs)
        x = self.neck(x)
        return x

    def loss(
        self, batch_inputs: torch.Tensor, batch_data_smaples: Union[dict, list]
    ) -> Union[dict, list]:
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_smaples)
        return losses


if __name__ == "__main__":

    ckpt = torch.load(
        "ckpt/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth",
        map_location="cpu",
    )["state_dict"]
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if (
            k.startswith("backbone.")
            or k.startswith("neck.")
            or k.startswith("bbox_head.")
        ):
            new_ckpt[k] = v

    m = M()
    m.eval()

    # for k, v in m.state_dict().items():
    #     if k.startswith('box_heads.'):
    #         print(k)
    # exit(0)

    m.load_state_dict(new_ckpt)

    # for k1, k2 in zip(ckpt.keys(), m.state_dict().keys()):
    #     print(k1, k2)

    x = torch.randn((1, 3, 640, 640), dtype=torch.float32)
    with torch.no_grad():
        cls_scores, bbox_preds = m(x)
    for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
        print(cls_score.shape, bbox_pred.shape)

    s = 640
    img_metas = [
        {
            "img_shape": (s, s, 3),
            "batch_input_shape": (s, s),
            "scale_factor": 1,
        }
    ]
    # gt_instances = InstanceData(bboxes=torch.empty((0, 4)), labels=torch.LongTensor([]))
    gt_instances = InstanceData(
        bboxes=torch.Tensor([
            [23.6667, 23.8757, 238.6326, 151.8874],
            [52.123, 120.224, 99.345, 230.256]
        ]),
        labels=torch.LongTensor([1, 1]),
    )
    empty_gt_losses = m.loss(x, dict(bbox_labels=[gt_instances], img_metas=img_metas))
    empty_cls_loss = empty_gt_losses["loss_cls"].sum()
    empty_box_loss = empty_gt_losses["loss_bbox"].sum()
    print("loss_cls =", empty_cls_loss.item())
    print("loss_bbox =", empty_box_loss.item())
