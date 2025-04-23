from abc import ABC, abstractmethod
from typing import Protocol, final

import torch

from .distribution import Distribution


class BaseRound(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Round(BaseRound):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_rounding(x)

    @staticmethod
    def _apply_rounding(x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)


class _RoundCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class BaseDiffRound(BaseRound):
    def __init__(self) -> None:
        super().__init__()


class AutoRound(BaseDiffRound):
    @final
    class Fn(torch.autograd.Function):
        __slots__ = ()

        @staticmethod
        def forward(ctx: _RoundCtx, x: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x)
            return torch.round(x)

        @staticmethod
        def backward(ctx: _RoundCtx, g: torch.Tensor) -> torch.Tensor:
            (x,) = ctx.saved_tensors
            xi = 0.5 + torch.floor(x)
            z = x - xi
            grad = torch.zeros_like(z)
            grad[z > 0] = 1
            grad[z < 0] = -1
            return grad * g

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Fn.apply(x)


class PerturbedRound(BaseDiffRound):
    eps: float
    dist: torch.distributions.Distribution

    def __init__(
        self,
        *,
        eps: float = 0.15,
        dist: torch.distributions.Distribution | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        if dist is None:
            dist = torch.distributions.Normal(0, 1)
        self.dist = dist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = self._get_fn(device=x.device)
        return fn.apply(x)

    def _get_fn(
        self,
        *,
        device: torch.device,
    ) -> type[torch.autograd.Function]:
        eps = self.eps
        dist = Distribution(self.dist).to(device=device)

        @final
        class Fn(torch.autograd.Function):
            __slots__ = ()

            @staticmethod
            def forward(ctx: _RoundCtx, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return torch.round(x)

            @staticmethod
            def backward(ctx: _RoundCtx, g: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                xi = 0.5 + torch.floor(x)
                z = (x - xi) / eps
                grad = torch.exp(dist.log_prob(z)) / eps
                return grad * g

        return Fn
