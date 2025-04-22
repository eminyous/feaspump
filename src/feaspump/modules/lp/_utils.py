import gurobipy as gp
import torch


class Utils:
    @staticmethod
    def solve(
        model: gp.Model,
        theta: torch.Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        variables: list[gp.Var] | None = None,
    ) -> torch.Tensor:
        if theta is None:
            device = device or torch.device("cpu")
            dtype = dtype or torch.double
        else:
            device = device or theta.device
            dtype = dtype or theta.dtype
        variables = variables or model.getVars()
        if theta is not None:
            Utils.set_objective(theta, model, variables=variables)

        Utils.optimize(model)
        x = [var.X for var in variables]
        return torch.tensor(x, device=device, dtype=dtype)

    @staticmethod
    def set_objective(
        theta: torch.Tensor,
        model: gp.Model,
        variables: list[gp.Var],
    ) -> None:
        obj = gp.LinExpr()
        for i, v in enumerate(variables):
            obj += theta[i].item() * v
        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.update()

    @staticmethod
    def optimize(model: gp.Model) -> None:
        model.optimize()
        if model.Status not in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL}:
            msg = f"LP optimization failed with status {model.Status}."
            raise RuntimeError(msg)
        if model.SolCount == 0:
            msg = "LP optimization failed: no solution found."
            raise RuntimeError(msg)
