import random
import math
from typing import Any

from grast import var


X = var("x")
Y = var("y")


def eq(a: Any, b: Any) -> bool:
    assert isinstance(a, float)
    assert isinstance(b, float)
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

    f = X + 1
    assert f(x=x) == x + 1
    df = f.grad()
    assert df["x"]() == 1

    f = 2 + X
    assert f(x=x) == 2 + x
    df = f.grad()
    assert df["x"]() == 1

    f = 2 + 1 + X
    assert f(x=x) == 2 + 1 + x
    df = f.grad()
    assert df["x"]() == 1


def test_mul() -> None:
    x = random.random()
    y = random.random()

    f = X * X
    assert f(x=x) == x * x
    df = f.grad()
    assert df["x"](x=x) == 2 * x

    f = X * 2
    assert f(x=x) == x * 2
    df = f.grad()
    assert df["x"]() == 2

    f = 2 * X
    assert f(x=x) == 2 * x
    df = f.grad()
    assert df["x"]() == 2

    f = X * 2 * 3
    assert eq(f(x=x), x * 2 * 3)
    df = f.grad()
    assert df["x"]() == 2 * 3

    f = X * Y
    assert f(x=x, y=y) == x * y
    df = f.grad()
    assert df["x"](y=y) == y
    assert df["y"](x=x) == x

    f = X * Y * X
    assert eq(f(x=x, y=y), x * y * x)
    df = f.grad()
    assert eq(df["x"](x=x, y=y), 2 * x * y)
    assert eq(df["y"](x=x, y=y), x * x)


def test_sub() -> None:
    x = random.random()
    y = random.random()

    f = X - 3
    assert f(x=x) == x - 3
    df = f.grad()
    assert df["x"]() == 1

    f = 4 - X
    assert f(x=x) == 4 - x
    df = f.grad()
    assert df["x"]() == -1

    f = X - Y
    assert f(x=x, y=y) == x - y
    df = f.grad()
    assert df["x"]() == 1
    assert df["y"]() == -1


def test_div() -> None:
    x = random.random()
    y = random.random()

    f = X / 3
    assert f(x=x) == x / 3
    df = f.eval_grad()
    assert eq(df["x"], 1 / 3)

    f = 3 / X
    assert f(x=x) == 3 / x
    df = f.eval_grad(x=x)
    assert eq(df["x"], -3 / x**2)

    f = X / Y
    assert f(x=x, y=y) == x / y
    df = f.eval_grad(x=x, y=y)
    assert eq(df["x"], 1 / y)
    assert eq(df["y"], -x / y**2)


def test_pow() -> None:
    x = random.random()
    y = random.random()
    a = random.random()

    f = X**a
    assert eq(f(x=x), x**a)
    df = f.eval_grad(x=x)
    assert eq(df["x"], a * x ** (a - 1))

    f = a**X
    assert eq(f(x=x), a**x)
    df = f.eval_grad(x=x)
    assert eq(df["x"], a**x * math.log(a))

    f = X**Y
    assert eq(f(x=x, y=y), x**y)
    df = f.eval_grad(x=x, y=y)
    assert eq(df["x"], y * x ** (y - 1))
    assert eq(df["y"], x**y * math.log(x))
