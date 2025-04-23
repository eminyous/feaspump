from dataclasses import dataclass

import torch


@dataclass
class Syncable:
    sync_required: bool = False

    def synchronize(self) -> None:
        if (
            torch.cuda.is_available()
            and torch.cuda.is_initialized()
            and self.sync_required
        ):
            torch.cuda.synchronize()
