import torch.nn as nn


def build_activation_layer(cfg):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "GELU":
        return nn.GELU()
    if kind == "ReLU":
        return nn.ReLU(inplace=cfg.get("inplace", True))
    if kind == "LeakyReLU":
        return nn.LeakyReLU(cfg.get("negative_slope", 0.01), inplace=cfg.get("inplace", True))
    if kind == "SiLU":
        return nn.SiLU()
    if kind == "Sigmoid":
        return nn.Sigmoid()
    raise KeyError(f"activation {kind} not in shim")


def build_norm_layer(cfg, num_features, postfix=""):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    eps = cfg.get("eps", 1e-5)
    if kind in ("BN", "BN2d", "SyncBN"):
        layer = nn.BatchNorm2d(num_features, eps=eps, momentum=cfg.get("momentum", 0.1))
        name = "bn"
    elif kind == "GN":
        layer = nn.GroupNorm(cfg["num_groups"], num_features, eps=eps)
        name = "gn"
    elif kind == "LN":
        layer = nn.LayerNorm(num_features, eps=eps)
        name = "ln"
    else:
        raise KeyError(f"norm {kind} not in shim")
    for p in layer.parameters():
        p.requires_grad = cfg.get("requires_grad", True)
    return name + str(postfix), layer


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias="auto", conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type="ReLU"), inplace=True, order=("conv", "norm", "act")):
        super().__init__()
        if bias == "auto":
            bias = norm_cfg is None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1] if norm_cfg is not None else None
        self.activate = build_activation_layer(act_cfg) if act_cfg is not None else None
        # mmcv ConvModule.init_weights: kaiming (fan_out, relu) for conv, constant 1/0 for norm
        nn.init.kaiming_normal_(self.conv.weight, a=0, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        if self.norm is not None and getattr(self.norm, "weight", None) is not None:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x
