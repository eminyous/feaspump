from enum import StrEnum

import torch

from .x1mx import X1mX

type IntegLossParam = float | int | torch.Tensor


class IntegLossMode(StrEnum):
    X1MX = "x1mx"

    def __call__(self, **params: IntegLossParam) -> torch.nn.Module:
        match self:
            case IntegLossMode.X1MX:
                return X1mX(**params)
            case _:
                msg = f"Unsupported integer loss type: {self}"
                raise ValueError(msg) from None
