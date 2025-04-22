from enum import StrEnum

import torch


class Regularization(StrEnum):
    L1 = "l1"
    L2 = "l2"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        match self:
            case Regularization.L2:
                return torch.square(x).sum()
            case _:
                return torch.abs(x).sum()
