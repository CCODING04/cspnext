from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .utils import constant_init, kaiming_init


class ConvModule(nn.Module):

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = None,
        inplace: bool = True,
        order: tuple = ("conv", "norm", "act"),
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace

        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        # conv part
        if conv_cfg is not None:
            _conv_cfg = conv_cfg.copy()
            _conv_type = _conv_cfg.pop("type")
        else:
            _conv_cfg = dict()
        # TODO: specify conv method according to conv_type

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **_conv_cfg
        )

        # norm part
        if norm_cfg is not None:
            _norm_cfg = norm_cfg.copy()
            _norm_type = _norm_cfg.pop("type")
            assert _norm_type == "BN"

        if self.with_norm:
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = _norm_type.lower(), nn.BatchNorm2d(
                norm_channels, **_norm_cfg
            )
            self.add_module(self.norm_name, norm)

        else:
            self.norm_name = None

        # activation part

        if self.with_activation:
            _act_cfg = act_cfg.copy()
            if 'inplace' not in _act_cfg:
                _act_cfg.setdefault('inplace', inplace)
            _act_type = _act_cfg.pop("type")
            self.act = eval("nn." + _act_type)(**_act_cfg)

        self.init_weights()

    def init_weights(self):
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(
        self, x: torch.Tensor, activate: bool = True, norm: bool = True
    ) -> torch.Tensor:
        for layer_name in self.order:
            m = getattr(self, layer_name)
            x = m(x)
        return x

