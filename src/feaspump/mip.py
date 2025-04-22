from collections.abc import Mapping
from typing import overload

import gurobipy as gp
import torch


def copy_params(src: gp.Model, dst: gp.Model) -> None:
    for param in dir(src):
        if param.startswith("setParam"):
            setattr(dst, param, getattr(src, param))
    dst.update()


def copy_variables(src: gp.Model, dst: gp.Model) -> Mapping[int, gp.Var]:
    mapping = {}
    for var in src.getVars():
        new_var = dst.addVar(
            lb=var.LB,
            ub=var.UB,
            obj=var.Obj,
            vtype=var.VType,
            name=var.VarName,
        )
        mapping[var.index] = new_var
    dst.update()
    return mapping


def copy_constrs(
    src: gp.Model,
    dst: gp.Model,
    mapping: Mapping[int, gp.Var],
) -> None:
    for constr in src.getConstrs():
        expr = src.getRow(constr)
        lhs = gp.quicksum(
            expr.getCoeff(i) * mapping[expr.getVar(i).index]
            for i in range(expr.size())
        )
        if constr.Sense == gp.GRB.LESS_EQUAL:
            new_constr = lhs <= constr.RHS
        elif constr.Sense == gp.GRB.GREATER_EQUAL:
            new_constr = lhs >= constr.RHS
        elif constr.Sense == gp.GRB.EQUAL:
            new_constr = lhs == constr.RHS
        else:
            msg = f"Unknown constraint sense: {constr.Sense}"
            raise ValueError(msg)
        dst.addConstr(new_constr, name=constr.ConstrName)
    dst.update()


def copy_objective(
    src: gp.Model,
    dst: gp.Model,
    mapping: Mapping[int, gp.Var],
) -> None:
    dst.setObjective(
        gp.quicksum(v.Obj * mapping[v.index] for v in src.getVars()),
        sense=src.ModelSense,
    )
    dst.update()


def copy_model(src: gp.Model, dst: gp.Model) -> None:
    copy_params(src, dst)
    mapping = copy_variables(src, dst)
    copy_constrs(src, dst, mapping)
    copy_objective(src, dst, mapping)
    dst.update()


class MIP(gp.Model):
    device: torch.device
    dtype: torch.dtype
    sparse: bool = False

    @overload
    def __init__(
        self,
        *,
        name: str = "",
        env: gp.Env | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        sparse: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        model: gp.Model | None = None,
        env: gp.Env | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        sparse: bool = False,
    ) -> None: ...
    def __init__(
        self,
        *,
        name: str = "",
        model: gp.Model | None = None,
        env: gp.Env | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        sparse: bool = False,
    ) -> None:
        if model is None:
            super().__init__(name, env)
        else:
            super().__init__(model.ModelName, env)
            copy_model(model, self)
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.double
        self.sparse = sparse

    def __setattr__(self, name: str, value) -> None:  # noqa: ANN001
        return object.__setattr__(self, name, value)

    @property
    def variables(self) -> list[gp.Var]:
        return self.getVars()

    @property
    def binaries(self) -> list[int]:
        return [v.index for v in self.variables if v.VType == gp.GRB.BINARY]

    @property
    def integers(self) -> list[int]:
        return [v.index for v in self.variables if v.VType == gp.GRB.INTEGER]

    @property
    def n(self) -> int:
        return self.NumVars

    @property
    def m(self) -> int:
        m = 0
        for constr in self.constrs:
            expr = self.getRow(constr)
            if expr.size() == 0:
                continue
            if constr.Sense in {gp.GRB.LESS_EQUAL, gp.GRB.GREATER_EQUAL}:
                m += 1
            elif constr.Sense == gp.GRB.EQUAL:
                m += 2

        for v in self.variables:
            if v.LB > -gp.GRB.INFINITY:
                m += 1
            if v.UB < gp.GRB.INFINITY:
                m += 1
        return m

    @property
    def constrs(self) -> list[gp.Constr]:
        return self.getConstrs()

    @property
    def obj(self) -> torch.Tensor:
        obj = self._zeros(self.n)
        for i, v in enumerate(self.variables):
            obj[i] = v.Obj
        return obj

    @property
    def Ab(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.sparse:
            A, b = self._get_Ab()
        else:
            A, b = self._get_Ab_sparse()
        return A, b

    def to(self, device: torch.device) -> None:
        self.device = device

    def _zeros(self, *shape: int) -> torch.Tensor:
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)

    def _get_Ab_sparse(self) -> tuple[torch.Tensor, torch.Tensor]:
        A_rows, A_cols, A_values, b = [], [], [], self._zeros(self.m)
        m = self._process_constrs_sparse(A_rows, A_cols, A_values, b, 0)
        m = self._process_var_bnds_sparse(A_rows, A_cols, A_values, b, m)

        A = torch.sparse_coo_tensor(
            (A_rows, A_cols),
            A_values,
            size=(m, self.n),
            dtype=self.dtype,
            device=self.device,
        )
        return A, b

    def _process_constrs_sparse(
        self,
        A_rows: list,
        A_cols: list,
        A_values: list,
        b: torch.Tensor,
        m: int,
    ) -> int:
        for constr in self.constrs:
            expr = self.getRow(constr)
            if expr.size() == 0:
                continue

            rhs = constr.RHS
            for i in range(expr.size()):
                v = expr.getVar(i)
                a = expr.getCoeff(i)
                match constr.Sense:
                    case gp.GRB.LESS_EQUAL:
                        A_rows.append(m)
                        A_cols.append(v.index)
                        A_values.append(-a)
                    case gp.GRB.GREATER_EQUAL:
                        A_rows.append(m)
                        A_cols.append(v.index)
                        A_values.append(a)
                    case gp.GRB.EQUAL:
                        A_rows.extend((m, m + 1))
                        A_cols.extend((v.index, v.index))
                        A_values.extend((a, -a))

            match constr.Sense:
                case gp.GRB.LESS_EQUAL:
                    b[m] = -rhs
                case gp.GRB.GREATER_EQUAL:
                    b[m] = rhs
                case gp.GRB.EQUAL:
                    b[m], b[m + 1] = rhs, -rhs
                    m += 1
            m += 1
        return m

    def _process_var_bnds_sparse(
        self,
        A_rows: list,
        A_cols: list,
        A_values: list,
        b: torch.Tensor,
        m: int,
    ) -> int:
        for v in self.variables:
            if v.LB > -gp.GRB.INFINITY:
                A_rows.append(m)
                A_cols.append(v.index)
                A_values.append(-1)
                b[m] = -v.LB
                m += 1

            if v.UB < gp.GRB.INFINITY:
                A_rows.append(m)
                A_cols.append(v.index)
                A_values.append(1)
                b[m] = v.UB
                m += 1
        return m

    def _get_Ab(self) -> tuple[torch.Tensor, torch.Tensor]:
        A = self._zeros(self.m, self.n)
        b = self._zeros(self.m)

        m = self._process_constrs(A, b, 0)
        self._process_var_bnds(A, b, m)
        return A, b

    def _process_constrs(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        m: int = 0,
    ) -> int:
        for constr in self.constrs:
            expr = self.getRow(constr)

            if expr.size() == 0:
                continue

            lhs = self._zeros(self.n)
            rhs = constr.RHS
            for i in range(expr.size()):
                v = expr.getVar(i)
                a = expr.getCoeff(i)
                lhs[v.index] = a

            match constr.Sense:
                case gp.GRB.LESS_EQUAL:
                    A[m], b[m] = -lhs, -rhs
                    m += 1
                case gp.GRB.GREATER_EQUAL:
                    A[m], b[m] = lhs, rhs
                    m += 1
                case gp.GRB.EQUAL:
                    A[m], A[m + 1] = lhs, -lhs
                    b[m], b[m + 1] = rhs, -rhs
                    m += 2
        return m

    def _process_var_bnds(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        m: int = 0,
    ) -> int:
        for v in self.variables:
            if v.LB > -gp.GRB.INFINITY:
                A[m][v.index] = -1
                b[m] = -v.LB
                m += 1

            if v.UB < gp.GRB.INFINITY:
                A[m][v.index] = 1
                b[m] = v.UB
                m += 1
        return m
