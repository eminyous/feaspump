from typing import Protocol

import torch


class _FlipCtx(Protocol):
    saved_tensors: tuple[torch.Tensor]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class FlipFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _FlipCtx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return 1.0 - x

    @staticmethod
    def backward(ctx: _FlipCtx, g: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        onehalf = torch.tensor(0.5, dtype=x.dtype, device=x.device)
        return torch.where(torch.isclose(x, onehalf), -1.0, 1.0) * g
