from collections.abc import Iterable
from enum import StrEnum

import torch

type OptimizerParam = float | int | bool | tuple[float, float]


class OptimizerMode(StrEnum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"

    @property
    def fn(self) -> type[torch.optim.Optimizer]:
        mapping = {
            OptimizerMode.ADAM: torch.optim.Adam,
            OptimizerMode.SGD: torch.optim.SGD,
            OptimizerMode.ADAMW: torch.optim.AdamW,
            OptimizerMode.ADAGRAD: torch.optim.Adagrad,
            OptimizerMode.RMSPROP: torch.optim.RMSprop,
        }
        try:
            return mapping[self]
        except KeyError:
            msg = f"Optimizer {self} is not supported."
            raise ValueError(msg) from None

    def __call__(
        self,
        tensors: torch.Tensor | Iterable[torch.Tensor],
        **params: OptimizerParam,
    ) -> torch.optim.Optimizer:
        return self.fn(tensors, **params)
