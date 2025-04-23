from enum import StrEnum

import gurobipy as gp

from .diff import AutoLP, BaseDiffLP, PerturbedLP


class LPMode(StrEnum):
    AUTO = "auto"
    PERTURBED = "perturbed"

    def __call__(self, model: gp.Model) -> BaseDiffLP:
        match self:
            case LPMode.PERTURBED:
                return PerturbedLP(model)
            case _:
                return AutoLP(model)
