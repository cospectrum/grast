from grast.delta.delta import OneHot

from grast import var, const, Dual
from grast.real import Add, Mul, Const, Var


X = var("x")
Y = var("y")
one = const(1)
two = const(2)


def test_init() -> None:
    assert X == Dual(Var("x"), OneHot(Var("x")))
    assert Y == Dual(Var("y"), OneHot(Var("y")))

    assert (X * Y).real == Mul(Var("x"), Var("y"))
    f = two * (X * Y) + one
    assert f.real == Add(
        Mul(
            Const(2),
            Mul(Var("x"), Var("y")),
        ),
        Const(1),
    )
