from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import torch

from .base import Pump
from .core import (
    Event,
    Status,
)
from .modules import (
    FlipFn,
    IntegerLossParam,
    IntegLossMode,
    LPMode,
    Normalizer,
    NormMode,
    OptimizerMode,
    OptimizerParam,
    PerturbedRound,
    Reducer,
    Regularization,
)


class InitMode(StrEnum):
    RANDOM = "random"
    ZEROS = "zeros"
    OBJECTIVE = "objective"
    ONES = "ones"
    NEGATIVE_OBJECTIVE = "negative_objective"


@dataclass
class DiffPump(Pump):
    round_eps: float = 0.15
    round_dist: torch.distributions.Distribution | None = field(
        default=None,
    )

    init_mode: InitMode = InitMode.OBJECTIVE

    norm_mode: NormMode = NormMode.II

    opt_mode: OptimizerMode = OptimizerMode.SGD
    opt_settings: Mapping[str, OptimizerParam] = field(default_factory=dict)

    lp_mode: LPMode = LPMode.AUTO

    slacks_reducer: Reducer = Reducer.MEAN

    integ_mode: IntegLossMode = IntegLossMode.X1MX
    integ_reducer: Reducer = Reducer.SUM
    integ_settings: Mapping[str, IntegerLossParam] = field(default_factory=dict)

    reg: Regularization = Regularization.L2

    w_reg: float = 0.5
    w_obj: float = 0.0
    w_feas: float = 0.0
    w_integ: float = 1.0

    _normalizer: Normalizer | None = field(default=None, init=False)
    _integ: torch.nn.Module = field(init=False)

    theta: torch.Tensor = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.opt_mode = OptimizerMode(self.opt_mode)
        self.reg = Regularization(self.reg)
        self.slacks_reducer = Reducer(self.slacks_reducer)
        self.integ_mode = IntegLossMode(self.integ_mode)
        self.integ_reducer = Reducer(self.integ_reducer)
        self.lp_mode = LPMode(self.lp_mode)
        self._integ = self.integ_mode(**self.integ_settings)
        self._round = PerturbedRound(eps=self.round_eps, dist=self.round_dist)
        self._normalizer = Normalizer(mode=self.norm_mode)

    def reset(self) -> None:
        super().reset()
        self._lp = self.lp_mode(self.mip.relax())

    def pre_loop(self) -> None:
        super().pre_loop()
        self.theta = self.init_theta()
        self.optimizer = self.opt_mode([self.theta], **self.opt_settings)

    def do_step(self) -> None:
        theta0 = self.normalize(self.theta)

        self.xlp = self._lp(theta0)

        non_integrality = self.compute_integrality(self.xlp).item()
        if np.isclose(non_integrality, 0.0):
            self._status = Status.INTEGRAL
            return

        self.x = self.round(self.xlp)
        slacks = self.compute_slacks(self.x)
        feas_loss = self.slacks_reducer(slacks)
        non_feasibility = feas_loss.item()
        if np.isclose(non_feasibility, 0.0):
            self._status = Status.FEASIBLE
            return

        # Update the xp
        if self.iteration < self.max_iterations:
            self.update_xp(self.xlp)

        obj_loss = self.mip.obj @ self.x
        reg_loss = self.reg(self.theta)
        integ_loss = self.compute_integrality(self.xp)

        loss = (
            self.w_obj * obj_loss
            + self.w_feas * feas_loss
            + self.w_integ * integ_loss
            + self.w_reg * reg_loss
        )

        objective = obj_loss.item()
        self.emit(
            Event.ITERATION,
            iteration=self.iteration,
            objective=objective,
            non_feasibility=non_feasibility,
            non_integrality=non_integrality,
            loss=loss.item(),
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def init_theta(self) -> torch.Tensor:
        obj = self.mip.obj
        match self.init_mode:
            case InitMode.RANDOM:
                theta = torch.randn_like(obj)
            case InitMode.OBJECTIVE:
                theta = torch.clone(obj).detach()
            case InitMode.ONES:
                theta = torch.ones_like(obj)
            case InitMode.NEGATIVE_OBJECTIVE:
                theta = torch.clone(-obj).detach()
            case _:
                theta = torch.zeros_like(obj)
        return theta.requires_grad_()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalizer(x)

    def compute_integrality(self, x: torch.Tensor) -> torch.Tensor:
        y = self._integ(x[self.integers])
        return self.integ_reducer(y)

    @staticmethod
    def flip(x: torch.Tensor) -> torch.Tensor:
        return FlipFn.apply(x)
