from grast.delta import Add, OneHot

from grast import one_hot
from grast.delta.delta import Neg, Scale, Sub
from grast.real import Var


R1 = Var("r1")
R2 = Var("r2")
R3 = Var("r3")
X = one_hot("x")
Y = one_hot("y")
Z = one_hot("z")


def test_add() -> None:
    add = X.add(Y)
    assert add == Add(X, OneHot(Var("y")))
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
