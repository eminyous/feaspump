from enum import StrEnum

import torch


class Reducer(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    SQUARE = "square"
    IDENTITY = "identity"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        match self:
            case Reducer.MEAN:
                return torch.mean(x)
            case Reducer.SQUARE:
                return torch.square(x).sum()
            case Reducer.SUM:
                return torch.sum(x)
            case _:
                return x
