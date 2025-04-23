from typing import Protocol

import torch

from .event import Event

type CallbackParam = str | int | float | torch.Tensor


class Callback(Protocol):
    def __call__(self, event: Event, **params: CallbackParam) -> None: ...
