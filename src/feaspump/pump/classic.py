from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
import torch

from ..modules import LP, Round
from .base import CorePump
from .event import Event
from .status import Status


@dataclass
class FeasPump(CorePump):
    _aux_variables: list[gp.Var] = field(init=False, default_factory=list)
    _aux_constrs: list[gp.Constr] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._round = Round()

    def reset(self) -> None:
        super().reset()
        self._lp = LP(self.mip.relax())

    def do_step(self) -> None:
        # Update the LP.
        if self.iteration > 0:
            self.update_lp(self.xp)

        self.xlp = self.solve_lp()

        # Check if xlp is integral
        integ = self.compute_integrality(self.xlp)
        non_integrality = torch.sum(integ).item()
        if np.isclose(non_integrality, 0.0):
            self._status = Status.INTEGRAL
            return

        # Round xlp
        self.x = self.round(self.xlp)

        # Check if x is feasible
        slacks = self.compute_slacks(self.x)
        non_feasibility = torch.sum(slacks).item()
        if np.isclose(non_feasibility, 0.0):
            self._status = Status.FEASIBLE
            return

        # Update the xp
        if self.iteration < self.max_iterations:
            self.update_xp(self.x)

        objective = (self.mip.obj @ self.x).item()
        self.emit(
            Event.ITERATION,
            iteration=self.iteration,
            objective=objective,
            non_feasibility=non_feasibility,
            non_integrality=non_integrality,
        )

    def compute_integrality(self, x: torch.Tensor) -> torch.Tensor:
        xi = x[self.integers]
        y = torch.minimum(xi - torch.floor(xi), torch.ceil(xi) - xi)
        return torch.sum(y)

    @staticmethod
    def flip(x: torch.Tensor) -> torch.Tensor:
        return 1 - x

    def solve_lp(self) -> torch.Tensor:
        return self._lp.solve(device=self.mip.device, dtype=self.mip.dtype)

    def clear_lp(self) -> None:
        model = self._lp.model
        model.remove(self._aux_constrs)
        model.remove(self._aux_variables)
        model.update()
        self._aux_variables.clear()
        self._aux_constrs.clear()

    def update_lp(self, x: torch.Tensor) -> None:
        self.clear_lp()

        obj = gp.LinExpr()
        model = self._lp.model
        for i in self.integers:
            val = x[i].item()
            var = self._lp.variables[i]
            if np.isclose(val, var.LB):
                obj += var
            elif np.isclose(val, var.UB):
                obj += -var
            else:
                aux_var = model.addVar()
                self._aux_variables.append(aux_var)

                obj += aux_var
                aux_constr = model.addConstr(aux_var >= var - val)
                self._aux_constrs.append(aux_constr)
                aux_constr = model.addConstr(aux_var >= val - var)
                self._aux_constrs.append(aux_constr)

        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.update()
