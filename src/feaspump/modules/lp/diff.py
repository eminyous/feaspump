from abc import ABC
from typing import Protocol, final

import gurobipy as gp
import torch

from ._utils import Utils
from .base import BaseLP


class _LPCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class BaseDiffLP(BaseLP, ABC):
    fn: type[torch.autograd.Function]

    def __init__(self, model: gp.Model) -> None:
        super().__init__(model)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.fn.apply(theta)


class AutoLP(BaseDiffLP):
    def __init__(self, model: gp.Model) -> None:
        super().__init__(model)
        self.fn = self._init_fn()

    def _init_fn(self) -> type[torch.autograd.Function]:
        model = self.model

        @final
        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(_: _LPCtx, theta: torch.Tensor) -> torch.Tensor:
                return Utils.solve(model, theta)

            @staticmethod
            def backward(_: _LPCtx, g: torch.Tensor) -> torch.Tensor:
                return -g

        return Fn


class PerturbedLP(BaseDiffLP):
    n_samples: int
    eps: float
    dist: torch.distributions.Distribution

    def __init__(
        self,
        model: gp.Model,
        *,
        n_samples: int = 1,
        eps: float = 0.15,
        dist: torch.distributions.Distribution | None = None,
    ) -> None:
        super().__init__(model)
        self.n_samples = n_samples
        if dist is None:
            dist = torch.distributions.Normal(0, 1)
        self.dist = dist
        self.eps = eps
        self.fn = self._init_fn()

    def _init_fn(self) -> type[torch.autograd.Function]:
        model = self.model
        variables = self.variables
        n_variables = len(variables)
        n_samples = self.n_samples
        eps = self.eps
        dist = self.dist

        @final
        class Fn(torch.autograd.Function):
            __slots__ = ()

            @staticmethod
            def forward(ctx: _LPCtx, theta: torch.Tensor) -> torch.Tensor:
                z = dist.sample((n_samples, n_variables))
                z = z.to(dtype=theta.dtype, device=theta.device)
                thetas = theta.unsqueeze(0) + eps * z
                x = torch.empty_like(thetas)
                for i in range(n_samples):
                    x[i] = Utils.solve(model, thetas[i], variables=variables)
                ctx.save_for_backward(x, z)
                return x.mean(dim=0)

            @staticmethod
            def backward(ctx: _LPCtx, g: torch.Tensor) -> torch.Tensor:
                x, z = ctx.saved_tensors
                dotg = z @ g
                grad = dotg.unsqueeze(-1) * x
                return grad.mean(dim=0) / eps

        return Fn
