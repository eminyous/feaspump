from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import gurobipy as gp
import torch
import typer

from feaspump.mip import MIP

app = typer.Typer(
    help="Feasibility Pump CLI: classic and differentiable variants."
)


class ExportMode(StrEnum):
    LIGHT = "light"
    FULL = "full"


def _init_env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


@dataclass
class Options:
    remote: bool = False
    file: Path | None = None
    instance: str | None = None
    gpu: bool = False
    seeds: list[int] = field(default_factory=list)
    n_trials: int | None = None
    export_mode: ExportMode = ExportMode.LIGHT
    export_path: Path = field(init=True)

    _env: gp.Env = field(default_factory=_init_env, init=False, repr=False)

    def validate(self) -> None:
        if self.remote and self.file is not None:
            msg = "Cannot specify both --remote and --file."
            raise typer.BadParameter(msg)
        if not self.remote and self.file is None:
            msg = "Must specify --file in local mode."
            raise typer.BadParameter(msg)
        if self.remote and self.instance is None:
            msg = "Must specify --instance in remote mode."
            raise typer.BadParameter(msg)
        if self.n_trials is not None:
            self.seeds = list(range(self.n_trials))
        elif not self.seeds:
            self.seeds = [0]
        self.n_trials = len(self.seeds)

    @property
    def path(self) -> Path:
        if self.remote:
            msg = "Remote mode not supported for classic pump."
            raise ValueError(msg)
        return self.file

    @property
    def device(self) -> torch.device:
        if self.gpu:
            if not torch.cuda.is_available():
                msg = "GPU not available, use CPU instead."
                raise RuntimeError(msg)
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def mip(self) -> MIP:
        model = gp.read(str(self.path), env=self._env)
        mip = MIP(model=model, env=self._env, device=self.device, sparse=True)
        model.dispose()
        return mip

    def set_seed(self, seed: int) -> None:
        if self.gpu:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

    def close(self) -> None:
        self._env.close()
