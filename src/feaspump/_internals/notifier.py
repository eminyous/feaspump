import warnings
from dataclasses import dataclass, field

from ..core import Callback, CallbackParam, Event


@dataclass
class Notifier:
    callbacks: set[Callback] = field(default_factory=set, init=False)

    def on(self, *callbacks: Callback) -> None:
        self.callbacks.update(callbacks)

    def off(self, callback: Callback) -> None:
        self.callbacks.remove(callback)

    def emit(self, event: Event, **params: CallbackParam) -> None:
        for callback in self.callbacks:
            self.emit_to(callback, event, **params)

    @staticmethod
    def emit_to(
        callback: Callback,
        event: Event,
        **params: CallbackParam,
    ) -> None:
        try:
            callback(event, **params)
        except Exception as e:  # noqa: BLE001
            msg = f"Callback {callback} raised an error on event {event}: {e}"
            warnings.warn(msg, UserWarning, stacklevel=2)
