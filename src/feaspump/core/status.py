from enum import StrEnum


class Status(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTEGRAL = "integral"
    FEASIBLE = "feasible"
    FAILED = "failed"

    @property
    def is_success(self) -> bool:
        return self in {self.INTEGRAL, self.FEASIBLE}

    @property
    def is_running(self) -> bool:
        return self == self.RUNNING

    @property
    def is_pending(self) -> bool:
        return self == self.PENDING

    @property
    def has_failed(self) -> bool:
        return self == self.FAILED
