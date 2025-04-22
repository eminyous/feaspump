from enum import StrEnum

from ..mip import MIP
from ..modules import (
    AutoLP,
    BaseDiffLP,
    PerturbedLP,
)


class LPMode(StrEnum):
    AUTO = "auto"
    PERTURBED = "perturbed"

    def __call__(self, mip: MIP) -> BaseDiffLP:
        match self:
            case LPMode.PERTURBED:
                return PerturbedLP(mip.relax())
            case _:
                return AutoLP(mip.relax())
