from typing import Any, TypeVar

from grast import var
from grast.delta import Add, OneHot
from grast.delta.delta import Delta, Neg, Scale, Sub
from grast.real import Var


T = TypeVar("T", bound=Any)


def one_hot(key: str) -> Delta[T]:
    return var(key).delta


R1: Var[float] = Var("r1")
R2: Var[float] = Var("r2")
R3: Var[float] = Var("r3")
X: Delta[float] = one_hot("x")
Y: Delta[float] = one_hot("y")
Z: Delta[float] = one_hot("z")


def test_add() -> None:
    add = X.add(Y)
    assert add == Add(X, OneHot(Var("y")))  # type: ignore
    assert add.add(Z) == Add(add, Z)
    assert X.add(Y.neg()) == Sub(X, Y)


def test_scale() -> None:
    s = X.scale(R1)
    assert s == Scale(R1, X)
    p = s.scale(R2)

    tmp = R2.mul(R1)
    assert p == Scale(tmp, X)
    assert p.scale(R3) == Scale(R3.mul(tmp), X)


def test_neg() -> None:
    assert X.neg() == Neg(X)
    assert X.neg().neg() == X
    assert Neg(Neg(X)).neg() == Neg(X)


def test_sub() -> None:
    assert X.sub(Y) == Sub(X, Y)
    assert X.sub(Neg(Z)) == Add(X, Z)
