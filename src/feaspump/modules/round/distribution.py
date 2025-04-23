from enum import Enum
from typing import TypeVar

import torch


class DistributionType(Enum):
    Normal = torch.distributions.Normal
    Laplace = torch.distributions.Laplace

    def to(
        self,
        distribution: torch.distributions.Distribution,
        *,
        device: torch.device,
    ) -> torch.distributions.Distribution:
        loc = torch.as_tensor(distribution.mean, device=device)
        scale = torch.as_tensor(distribution.stddev, device=device)

        match self:
            case DistributionType.Normal:
                return torch.distributions.Normal(loc=loc, scale=scale)
            case DistributionType.Laplace:
                return torch.distributions.Laplace(loc=loc, scale=scale)


DT = TypeVar("DT", bound=torch.distributions.Distribution)


class Distribution:
    distribution: torch.distributions.Distribution

    def __init__(self, distribution: torch.distributions.Distribution) -> None:
        self.distribution = distribution

    def to(self, *, device: torch.device) -> torch.distributions.Distribution:
        fn = DistributionType(type(self.distribution))
        return fn.to(self.distribution, device=device)
