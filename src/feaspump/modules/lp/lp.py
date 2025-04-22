import torch

from .base import BaseLP


class LP(BaseLP):
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.solve(theta)
