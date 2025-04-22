import torch


class Slacks(torch.nn.Module):
    A: torch.Tensor
    b: torch.Tensor

    def __init__(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, d = x.shape
        m, n = self.A.shape
        if n != d:
            msg = f"Shape mismatch: expected {n}, got {d}"
            raise ValueError(msg)

        y = x.reshape(-1, n)
        Ax = torch.matmul(y, self.A.T)
        s = Ax - self.b
        v = torch.nn.functional.relu(s)
        return v.reshape(*batch, m)
