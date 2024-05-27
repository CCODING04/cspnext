from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor

from cspnext.backbones import CSPNext
from cspnext.heads import RTMDetHead
from cspnext.necks import CSPNeXtPAFPN
from cspnext.structures import InstanceData
from cspnext.utils import InstanceList


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
                type="MlvlPointGenerator",
                offset=0,
                strides=[8, 16, 32],
            ),
            bbox_coder=dict(type="DistancePointBBoxCoder"),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.bbox_head(x)
        return x

    def extract_feat(
        self, batch_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        x = self.backbone(batch_inputs)
        x = self.neck(x)
        return x

    def loss(
        self,
        batch_inputs: torch.Tensor,
        batch_data_smaples: Union[dict, list],
    ) -> Union[dict, list]:
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_smaples)
        return losses

    def predict(
        self,
        batch_inputs: torch.Tensor,
        batch_data_samples: InstanceList,
        rescale: bool = True,
    ) -> InstanceList:
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale
        )
        return results_list


if __name__ == "__main__":
    import random

    import numpy as np

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

    m.load_state_dict(new_ckpt)

    # inference test
    image_ori = cv2.imread("assets\demo.jpg")
    image = image_ori.copy()
    # scale = (640, 640), pad_val = 114
    scale = [640, 640]  # hw

    image_shape = image.shape[:2]
    ratio = min(
        scale[0] / image_shape[0], scale[1] / image_shape[1]
    )
    ratio = [ratio, ratio]
    no_pad_shape = (
        int(round(image_shape[0] * ratio[0])),
        int(round(image_shape[1] * ratio[1])),
    )
    padding_h, padding_w = [
        scale[0] - no_pad_shape[0],
        scale[1] - no_pad_shape[1],
    ]
    if image_shape != no_pad_shape:
        image = cv2.resize(
            image,
            (no_pad_shape[1], no_pad_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    scale_factor = (
        no_pad_shape[1] / image_shape[1],
        no_pad_shape[0] / image_shape[0],
    )

    top_padding, left_padding = int(
        round(padding_h // 2 - 0.1)
    ), int(round(padding_w // 2 - 0.1))
    bottom_padding = padding_h - top_padding
    right_padding = padding_w - left_padding

    padding_list = [
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
    ]

    if (
        top_padding != 0
        or bottom_padding != 0
        or left_padding != 0
        or right_padding != 0
    ):
        pad_val = 114
        image = cv2.copyMakeBorder(
            image,
            top_padding,
            bottom_padding,
            left_padding,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=(pad_val, pad_val, pad_val),
        )

    image_info = InstanceData(
        metainfo=dict(
            scale_factor=scale_factor,
            img=image,
            img_shape=image.shape,
            ori_shape=image_ori.shape[:2],
            pad_param=np.array(
                padding_list, dtype=np.float32
            ),
        )
    )
    # x = ToTensor()(Image.fromarray(image)).unsqueeze(0).to(torch.float32)
    x = torch.randn((1, 3, 640, 640), dtype=torch.float32)

    with torch.no_grad():
        outs = m.predict([x], [image_info])
    print(outs)
