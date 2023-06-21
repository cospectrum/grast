from grast.delta import Algebra as A
from grast.delta import Add, OneHot

from grast import one_hot
from grast.delta.delta import Neg, Scale, Sub
from grast.real.expression import Var


R1 = Var("r1")
R2 = Var("r2")
R3 = Var("r3")
X = one_hot("x")
Y = one_hot("y")
Z = one_hot("z")


def test_add() -> None:
    add = A.add(X, Y)
    assert add == Add(X, OneHot(Var("y")))
    assert A.add(add, Z) == Add(add, Z)


def test_scale() -> None:
    s = A.scale(R1, X)
    assert s == Scale(R1, X)
    p = A.scale(R2, s)
    assert p == Scale(R2, s)


def test_neg() -> None:
    assert A.neg(X) == Neg(X)
    assert A.neg(A.neg(X)) == X


def test_sub() -> None:
    sub = A.sub(X, Y)
    assert sub == Sub(X, Y)
    z = A.neg(Z)
    assert A.sub(X, z) == Add(X, Z)
