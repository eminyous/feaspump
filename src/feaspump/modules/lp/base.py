from abc import ABC, abstractmethod

import gurobipy as gp
import torch

from ._utils import Utils


class BaseLP(torch.nn.Module, ABC):
    model: gp.Model
    variables: list[gp.Var]

    def __init__(self, model: gp.Model) -> None:
        super().__init__()
        self.model = model
        self.variables = model.getVars()
        self.setup()

    @abstractmethod
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def setup(self) -> None:
        self.model.setParam("Method", 4)
        self.model.setParam("OptimalityTol", 1e-9)
        self.model.setParam("NumericFocus", 3)
        self.model.setParam("FeasibilityTol", 1e-9)
        self.model.setParam("BarConvTol", 1e-16)
        self.model.update()

    def solve(
        self,
        *,
        theta: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return Utils.solve(
            self.model,
            theta=theta,
            device=device,
            dtype=dtype,
            variables=self.variables,
        )
