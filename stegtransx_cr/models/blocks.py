"""Building blocks shared by the hiding and reveal networks.

Specification section 3 asks for depthwise-separable convolution blocks and
lightweight Transformer blocks with 4 heads and an 8 x 8 window. Attention is
window-based everywhere: at 256 x 256 the reveal network operates at native
resolution (65,536 tokens), so global self-attention is not tractable, while
8 x 8 windows keep the model inside the lightweight budget of the paper.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an (B, C, H, W) tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """(B, C, H, W) -> (B * num_windows, window * window, C)."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // window, window, w // window, window)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window * window, c)


def window_reverse(windows: torch.Tensor, window: int, b: int, c: int, h: int, w: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`."""
    x = windows.view(b, h // window, w // window, window, window, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, h, w)


def _pad_to_window(x: torch.Tensor, window: int) -> Tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    if pad_h or pad_w:
        # reflect requires pad < spatial size; 4x4 features with window 8 cannot
        # reflect-pad by 4. Replicate is correct for those leftover pixels.
        mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, pad_h, pad_w


class WindowAttention(nn.Module):
    """Multi-head attention inside non-overlapping windows.

    Setting ``cross=True`` lets the queries come from one feature map and the
    keys/values from another, which is what the specification calls a
    "Transformer cross-attention layer" inside the hybrid blocks.
    """

    def __init__(self, dim: int, num_heads: int = 4, window: int = 8, cross: bool = False) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.window = window
        self.cross = cross
        self.norm_q = LayerNorm2d(dim)
        self.norm_kv = LayerNorm2d(dim) if cross else None
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        # Learned positional bias per head over the flattened window.
        self.position_bias = nn.Parameter(torch.zeros(num_heads, window * window, window * window))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        q_src = self.norm_q(x)
        kv_src = q_src if context is None else self.norm_kv(context)

        q_src, pad_h, pad_w = _pad_to_window(q_src, self.window)
        kv_src, _, _ = _pad_to_window(kv_src, self.window)
        _, _, hp, wp = q_src.shape

        q_tokens = window_partition(q_src, self.window)
        kv_tokens = window_partition(kv_src, self.window)

        n_win, tokens, _ = q_tokens.shape
        q = self.to_q(q_tokens)
        k, v = self.to_kv(kv_tokens).chunk(2, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(n_win, tokens, self.num_heads, c // self.num_heads).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # Explicit attention rather than scaled_dot_product_attention: A100
        # FlashAttention kernels reject a dense additive bias and can abort
        # with "No available kernel" instead of falling back.
        bias = self.position_bias.unsqueeze(0).to(dtype=q.dtype)
        scale = (c // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(n_win, tokens, c)
        out = self.proj(out)

        out = window_reverse(out, self.window, b, c, hp, wp)
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network implemented with 1 x 1 convolutions."""

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = LayerNorm2d(dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class TransformerBlock(nn.Module):
    """Pre-norm residual Transformer block with windowed self-attention."""

    def __init__(self, dim: int, num_heads: int = 4, window: int = 8, expansion: int = 2) -> None:
        super().__init__()
        self.attn = WindowAttention(dim, num_heads=num_heads, window=window, cross=False)
        self.ffn = FeedForward(dim, expansion=expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class ConvBlock(nn.Module):
    """Depthwise 3 x 3 followed by pointwise 1 x 1 with a residual addition."""

    def __init__(self, channels: int, slope: float = 0.2) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.LeakyReLU(slope, inplace=True)
        self.norm = LayerNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.act(self.depthwise(out))
        out = self.pointwise(out)
        return residual + out


class HybridBlock(nn.Module):
    """CNN layer paired with a lightweight Transformer cross-attention layer.

    The convolutional branch produces the queries and the block input supplies
    the keys and values, so the attention refines the convolutional features
    using the wider context of the incoming feature map.
    """

    def __init__(self, dim: int, num_heads: int = 4, window: int = 8, expansion: int = 2, slope: float = 0.2) -> None:
        super().__init__()
        self.conv = ConvBlock(dim, slope=slope)
        self.cross_attn = WindowAttention(dim, num_heads=num_heads, window=window, cross=True)
        self.ffn = FeedForward(dim, expansion=expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        conv_out = conv_out + self.cross_attn(conv_out, context=x)
        return conv_out + self.ffn(conv_out)


class Downsample(nn.Module):
    """Strided 3 x 3 convolution (specification: encoder stage transition)."""

    def __init__(self, in_channels: int, out_channels: int, slope: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.act = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Upsample(nn.Module):
    """Bilinear upsampling paired with a 1 x 1 convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x)
