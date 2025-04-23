from enum import StrEnum
from typing import Protocol, final

import torch


class X1mXMode(StrEnum):
    MIN = "min"
    PROD = "prod"


class _X1mXCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, torch.Tensor]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _x1mx(x: torch.Tensor, *, mode: X1mXMode) -> torch.Tensor:
    if mode == X1mXMode.MIN:
        return torch.minimum(x, 1 - x)
    return x * (1 - x)


def _dx1mx(x: torch.Tensor, *, mode: X1mXMode) -> torch.Tensor:
    onehalf = 0.5
    if mode == X1mXMode.MIN:
        return torch.where(x > onehalf, -1.0, 1.0)
    return 1 - 2 * x


class X1mX(torch.nn.Module):
    mode: X1mXMode
    fn: type[torch.autograd.Function]

    def __init__(
        self,
        *,
        mode: X1mXMode = X1mXMode.MIN,
    ) -> None:
        super().__init__()
        self.mode = X1mXMode(mode)
        self.fn = self._init_fn()

    def _init_fn(self) -> type[torch.autograd.Function]:
        mode = self.mode

        @final
        class Fn(torch.autograd.Function):
            __slots__ = ()

            @staticmethod
            def forward(ctx: _X1mXCtx, x: torch.Tensor) -> torch.Tensor:
                x = torch.clip(x, min=0, max=1)
                y = _x1mx(x, mode=mode)
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: _X1mXCtx, g: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                dg = _dx1mx(x, mode=mode)
                return dg * g

        return Fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn.apply(x)
