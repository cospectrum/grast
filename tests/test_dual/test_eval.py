import random
import math

from grast import var, const


X = var("x")
Y = var("y")
one = const(1)
two = const(2)
three = const(3)


def eq(a: float, b: float) -> bool:
    return math.isclose(a, b)


def test_add() -> None:
    x = random.random()
    y = random.random()

    f = X + X
    assert f(x=x) == x + x
    df = f.grad()
    assert df["x"](x=x) == 2

    f = X + Y
    assert f(x=x, y=y) == x + y
    df = f.grad()
    assert df["x"]() == 1
    assert df["y"]() == 1

    f = X + one
    assert f(x=x) == x + 1
    df = f.grad()
    assert df["x"]() == 1

    f = two + X
    assert f(x=x) == 2 + x
    df = f.grad()
    assert df["x"]() == 1

    f = two + one + X
    assert f(x=x) == 2 + 1 + x
    df = f.grad()
    assert df["x"]() == 1


def test_mul() -> None:
    x = random.random()
    y = random.random()

    f = X * two
    assert f(x=x) == x * 2
    df = f.grad()
    assert df["x"]() == 2

    f = two * X
    assert f(x=x) == 2 * x
    df = f.grad()
    assert df["x"]() == 2

    f = X * two * three
    assert eq(f(x=x), x * 2 * 3)
    df = f.grad()
    assert df["x"]() == 2 * 3

    f = X * Y
    assert f(x=x, y=y) == x * y
    df = f.grad()
    assert df["x"](y=y) == y
    assert df["y"](x=x) == x
