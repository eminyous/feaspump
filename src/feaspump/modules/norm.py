from enum import StrEnum
from typing import Protocol

import torch


def normalize(x: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


def dot(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.sum(x * y, dim=dim, keepdim=keepdim)


def norm2(
    x: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.linalg.norm(x, ord=2, dim=dim, keepdim=keepdim)


class _NormCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _NormalizeAndProjectGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _NormCtx, x: torch.Tensor) -> torch.Tensor:
        xn = normalize(x)
        norm = norm2(x)
        ctx.save_for_backward(xn, norm)
        return xn

    @staticmethod
    def backward(ctx: _NormCtx, g: torch.Tensor) -> torch.Tensor:
        xn, norm = ctx.saved_tensors
        dotg = dot(xn, g).unsqueeze(-1)
        return (g - dotg * xn) / norm


class _NormalizeOnlyFn(torch.autograd.Function):
    @staticmethod
    def forward(_: _NormCtx, x: torch.Tensor) -> torch.Tensor:
        return normalize(x)

    @staticmethod
    def backward(_: _NormCtx, g: torch.Tensor) -> torch.Tensor:
        return g


class _ProjectGradOnlyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _NormCtx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx: _NormCtx, g: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        xn = normalize(x)
        dotg = dot(xn, g).unsqueeze(-1)
        return g - dotg * xn


class _IdentityFn(torch.autograd.Function):
    @staticmethod
    def forward(_: _NormCtx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(_: _NormCtx, g: torch.Tensor) -> torch.Tensor:
        return g


class NormMode(StrEnum):
    NP = "np"  # Normalize and project gradient
    NI = "ni"  # Normalize only
    IP = "ip"  # Project gradient only
    II = "ii"  # Identity

    @property
    def fn(self) -> type[torch.autograd.Function]:
        match self:
            case self.NP:
                return _NormalizeAndProjectGradFn
            case self.NI:
                return _NormalizeOnlyFn
            case self.IP:
                return _ProjectGradOnlyFn
            case self.II:
                return _IdentityFn


class Normalizer(torch.nn.Module):
    fn: type[torch.autograd.Function]

    def __init__(
        self,
        *,
        mode: NormMode = NormMode.NP,
    ) -> None:
        super().__init__()
        self.fn = NormMode(mode).fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        y = self.fn.apply(x)
        return torch.squeeze(y, 0) if squeeze else y
