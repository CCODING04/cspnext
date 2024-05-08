from collections import OrderedDict

import torch
import torch.nn as nn

from cspnext.backbones import CSPNext
from cspnext.necks import CSPNeXtPAFPN


class M(nn.Module):
    def __init__(self):
        super().__init__()
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
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU', inplace=True)
        )
        

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x


if __name__ == "__main__":

    ckpt = torch.load(
        "ckpt/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth",
        map_location="cpu",
    )["state_dict"]
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('backbone.') or k.startswith('neck.'):
            new_ckpt[k] = v
    
    
    # print(ckpt.keys())

    m = M()
    m.eval()

    m.load_state_dict(new_ckpt)

    # for k1, k2 in zip(ckpt.keys(), m.state_dict().keys()):
    #     print(k1, k2)

    x = torch.randn((1, 3, 480, 480), dtype=torch.float32)
    with torch.no_grad():
        outs = m(x)
    for out in outs:
        print(out.shape)
