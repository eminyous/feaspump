import gurobipy as gp

from feaspump import mip


def test_variables() -> None:
    m = mip.MIP()

    x = m.addVar(vtype=gp.GRB.BINARY, name="x")
    y = m.addVar(vtype=gp.GRB.CONTINUOUS, name="y")
    z = m.addVar(vtype=gp.GRB.INTEGER, name="z")
    m.update()

    assert m.variables == [x, y, z]
    assert m.binaries == [0]
    assert m.integers == [2]
