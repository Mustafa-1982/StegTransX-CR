"""Gradient Approximation Function (GAF) differentiable rounding.

Specification section 3.4, equations (7) and (8), following PRIS:

    sign(x) = +1 if floor(x) is odd else -1
    GAF(x)  = sign(x) * 0.5 * cos(pi * x) + 0.5 + floor(x)
    dGAF/dx = -0.5 * pi * sign(x) * sin(pi * x)

Written out, GAF is a smooth ramp from floor(x) to floor(x) + 1 whose gradient
is non-zero almost everywhere. The specification also requires the forward pass
to equal true rounding. Both properties are obtained with a straight-through
estimator: the forward value is round(x) and the backward path uses the GAF
derivative above.
"""

from __future__ import annotations

import math

import torch


def gaf_smooth(x: torch.Tensor) -> torch.Tensor:
    """The smooth surrogate itself (differentiable, not exactly round(x))."""
    floor = torch.floor(x)
    # sign is +1 when floor(x) is odd, -1 otherwise.
    sign = torch.where(
        torch.remainder(floor, 2.0) == 1.0,
        torch.ones_like(x),
        -torch.ones_like(x),
    )
    return sign * 0.5 * torch.cos(math.pi * x) + 0.5 + floor


def gaf_round(x: torch.Tensor) -> torch.Tensor:
    """Rounding with an exact forward value and a GAF-shaped gradient."""
    surrogate = gaf_smooth(x)
    hard = torch.round(x)
    return surrogate + (hard - surrogate).detach()


def straight_through_gate(soft: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Hard one-hot selection in the forward pass, soft weights in the backward.

    A soft mixture of independently quantised reconstructions averages out
    quantisation noise, which makes a simulator far gentler than the codec it
    models. Selecting one branch in the forward pass avoids that while keeping
    the gate trainable.
    """
    index = soft.argmax(dim=dim, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(dim, index, 1.0)
    return soft + (hard - soft).detach()


def straight_through_binary(soft: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binary version of :func:`straight_through_gate` for a two-way blend."""
    hard = (soft > threshold).to(dtype=soft.dtype)
    return soft + (hard - soft).detach()


def straight_through(real: torch.Tensor, differentiable: torch.Tensor) -> torch.Tensor:
    """Forward the ``real`` value, backpropagate through ``differentiable``.

    Used to push a genuine codec into the forward pass while keeping the
    gradient path of the differentiable simulator. Same idea as GAF, applied at
    the level of a whole codec instead of a single rounding operation.
    """
    return differentiable + (real.detach() - differentiable.detach())
