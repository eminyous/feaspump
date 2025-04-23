from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time

import torch

from ..mip import MIP
from ..modules import BaseLP, BaseRound, Slacks
from .callback import Callback
from .event import Event
from .notifier import Notifier
from .status import Status
from .syncable import Syncable


@dataclass
class CorePump(ABC, Notifier, Syncable):
    max_iterations: int = 1000

    perturb_freq: int = 100
    perturb_rho: float = -0.3

    flip_temp: float = 50.0
    flip_temp_low: float = 0.5
    flip_temp_up: float = 1.5

    use_restarts: bool = True

    iteration: int = field(default=0, init=False)

    x: torch.Tensor = field(init=False)
    xp: torch.Tensor = field(init=False)
    xlp: torch.Tensor = field(init=False)

    history_length: int = 2
    x_history: list[torch.Tensor] = field(default_factory=list, init=False)
    xlp_history: list[torch.Tensor] = field(default_factory=list, init=False)

    _status: Status = field(init=False)
    _mip: MIP = field(init=False)

    _lp: BaseLP = field(init=False)
    _round: BaseRound = field(init=False)
    _slacks: Slacks = field(init=False)

    _integers: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.perturb_freq = min(self.perturb_freq, self.max_iterations)

    @property
    def integers(self) -> torch.Tensor:
        return self._integers

    @property
    def status(self) -> Status:
        return self._status

    def reset(self) -> None:
        self.iteration = 0
        self.x_history.clear()
        self.xlp_history.clear()
        A, b = self.mip.Ab
        self._slacks = Slacks(A, b)
        self._integers = torch.tensor(
            self.mip.integers,
            dtype=torch.long,
            device=self.mip.device,
        )
        self.sync_required = self.mip.device.type == "cuda"
        self._status = Status.PENDING
        self.emit(Event.RESET)

    @property
    def mip(self) -> MIP:
        return self._mip

    @mip.setter
    def mip(self, mip: MIP) -> None:
        self._mip = mip
        self.reset()

    def run(self, mip: MIP, *callbacks: Callback) -> None:
        self.on(*callbacks)
        self.mip = mip

        self.check_no_integers()

        if not self.status.is_pending:
            return

        self.setup()

        self.pre_loop()
        self.loop()
        self.post_loop()

    def pre_loop(self) -> None:
        self.emit(Event.START, iterations=self.max_iterations)
        self._status = Status.RUNNING

    def loop(self) -> None:
        while not self.is_complete():
            self.step()
            self.save_step()
            self.check_success()
            self.iteration += 1

    def is_complete(self) -> bool:
        return not (
            self.status.is_running and self.iteration <= self.max_iterations
        )

    def post_loop(self) -> None:
        if self.status.is_running:
            self._status = Status.FAILED
            self.fail()

    def step(self) -> None:
        tick = time()
        self.do_step()
        self.synchronize()
        tock = time()
        self.emit(Event.TIME, iteration=self.iteration, elapsed=tock - tick)

    def save_step(self) -> None:
        if getattr(self, "x", None) is not None:
            self.x_history.append(torch.clone(self.x))
        if getattr(self, "xlp", None) is not None:
            self.xlp_history.append(torch.clone(self.xlp))

    def setup(self) -> None:
        self.emit(Event.SETUP)

    @abstractmethod
    def do_step(self) -> None:
        raise NotImplementedError

    def round(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.clone(x)
        y[self.integers] = self._round(x[self.integers])
        self.emit(Event.ROUND, x=x, y=y)
        return y

    def compute_slacks(self, x: torch.Tensor) -> torch.Tensor:
        s = self._slacks(x)
        self.emit(Event.SLACKS, x=x, s=s)
        return s

    def check_no_integers(self) -> None:
        if self.integers.numel() == 0:
            self._status = Status.FEASIBLE
            self.emit(Event.NO_INTEGERS)

    def check_perturb(self) -> bool:
        perturb = (
            self.iteration % self.perturb_freq == 0 or self.check_lp_cycling()
        )
        self.emit(Event.PERTURBATION_CHECK, perturb=perturb)
        return perturb

    def check_lp_cycling(self) -> bool:
        if len(self.xlp_history) < self.history_length:
            return False
        x = self.xlp
        history = self.xlp_history[-self.history_length :]
        cycling = any(torch.allclose(x, xp) for xp in history)
        self.emit(Event.LP_CYCLING_CHECK, cycling=cycling)
        return cycling

    def check_cycling(self) -> bool:
        if len(self.x_history) < 1:
            return False
        x = self.x
        xp = self.x_history[-1]
        cycling = torch.allclose(x[self.integers], xp[self.integers])
        self.emit(Event.CYCLING_CHECK, cycling=cycling)
        return cycling

    def update_xp(self, x: torch.Tensor) -> None:
        if not self.use_restarts:
            self.xp = torch.clone(x)
            return

        if self.iteration == 0:
            self.xp = torch.clone(x)
        elif self.check_perturb():
            self.xp = self.do_perturb(x)
        elif self.check_cycling():
            self.xp = self.do_flip(x)
        else:
            self.xp = torch.clone(x)

    def do_perturb(self, x: torch.Tensor) -> torch.Tensor:
        rho = torch.clamp(self.perturb_rho + torch.rand_like(self.x), min=0.0)
        aux = torch.abs(self.xlp - self.x) + rho - 0.5
        idx = (aux > 0)[self.integers]
        y = torch.clone(x)
        indices = self.integers[idx]
        y[indices] = self.flip(x[indices])
        self.emit(
            Event.PERTURBED,
            iteration=self.iteration,
            x=x,
            y=y,
            indices=indices,
        )
        return y

    def do_flip(self, x: torch.Tensor) -> torch.Tensor:
        low = int(self.flip_temp * self.flip_temp_low)
        high = int(self.flip_temp * self.flip_temp_up)
        n = len(self.integers)
        rand = torch.randint(low, high + 1, size=(1,))
        flips = torch.clamp(rand, min=1, max=n).item()
        xi = self.xlp[self.integers]
        non_integrality = (xi - torch.floor(xi)) * (torch.ceil(xi) - xi)
        idx = torch.argsort(non_integrality)[-flips:]
        y = torch.clone(x)
        indices = self.integers[idx]
        y[indices] = self.flip(x[indices])
        self.emit(
            Event.FLIPPED,
            iteration=self.iteration,
            x=x,
            y=y,
            indices=indices,
        )
        return y

    @abstractmethod
    def compute_integrality(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def flip(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def check_success(self) -> bool:
        if self.status.is_success:
            self.success()

    def success(self) -> None:
        self.emit(Event.SUCCEEDED, status=self._status)

    def fail(self) -> None:
        self.emit(Event.FAILED, iteration=self.iteration)
