import random
import math
from typing import Any

from grast import var
from grast.dual import Dual


X: Dual[float] = var("x")
Y: Dual[float] = var("y")


def eq(a: Any, b: Any) -> bool:
    assert isinstance(a, float)
    assert isinstance(b, float)
    return math.isclose(a, b)


def test_add() -> None:
    x = random.random()
    y = random.random()

    f = X + X
    assert f(dict(x=x)) == x + x
    df = f.grad()
    assert df["x"](dict(x=x)) == 2

    f = X + Y
    assert f(dict(x=x, y=y)) == x + y
    df = f.grad()
    assert df["x"]() == 1
    assert df["y"]() == 1

    f = X + 1
    assert f(dict(x=x)) == x + 1
    df = f.grad()
    assert df["x"]() == 1

    f = 2 + X
    assert f(dict(x=x)) == 2 + x
    df = f.grad()
    assert df["x"]() == 1

    f = 2 + 1 + X
    assert f(dict(x=x)) == 2 + 1 + x
    df = f.grad()
    assert df["x"]() == 1


def test_mul() -> None:
    x = random.random()
    y = random.random()

    f = X * X
    assert f(dict(x=x)) == x * x
    df = f.grad()
    assert df["x"](dict(x=x)) == 2 * x

    f = X * 2
    assert f(dict(x=x)) == x * 2
    df = f.grad()
    assert df["x"]() == 2

    f = 2 * X
    assert f(dict(x=x)) == 2 * x
    df = f.grad()
    assert df["x"]() == 2

    f = X * 2 * 3
    assert eq(f(dict(x=x)), x * 2 * 3)
    df = f.grad()
    assert df["x"]() == 2 * 3

    f = X * Y
    assert f(dict(x=x, y=y)) == x * y
    df = f.grad()
    assert df["x"](dict(y=y)) == y
    assert df["y"](dict(x=x)) == x

    f = X * Y * X
    assert eq(f(dict(x=x, y=y)), x * y * x)
    df = f.grad()
    assert eq(df["x"](dict(x=x, y=y)), 2 * x * y)
    assert eq(df["y"](dict(x=x, y=y)), x * x)


def test_sub() -> None:
    x = random.random()
    y = random.random()

    f = X - 3
    assert f(dict(x=x)) == x - 3
    df = f.grad()
    assert df["x"]() == 1

    f = 4 - X
    assert f(dict(x=x)) == 4 - x
    df = f.grad()
    assert df["x"]() == -1

    f = X - Y
    assert f(dict(x=x, y=y)) == x - y
    df = f.grad()
    assert df["x"]() == 1
    assert df["y"]() == -1


def test_div() -> None:
    x = random.random()
    y = random.random()

    f = X / 3
    assert f(dict(x=x)) == x / 3
    df = f.eval_grad()
    assert eq(df["x"], 1 / 3)

    f = 3 / X
    assert f(dict(x=x)) == 3 / x
    df = f.eval_grad(dict(x=x))
    assert eq(df["x"], -3 / x**2)

    f = X / Y
    assert f(dict(x=x, y=y)) == x / y
    df = f.eval_grad(dict(x=x, y=y))
    assert eq(df["x"], 1 / y)
    assert eq(df["y"], -x / y**2)


def test_pow() -> None:
    x = random.random()
    y = random.random()
    a = random.random()

    f = X**a
    assert eq(f(dict(x=x)), x**a)
    df = f.eval_grad(dict(x=x))
    assert eq(df["x"], a * x ** (a - 1))

    f = a**X
    assert eq(f(dict(x=x)), a**x)
    df = f.eval_grad(dict(x=x))
    assert eq(df["x"], a**x * math.log(a))

    f = X**Y
    args = dict(x=x, y=y)
    assert eq(f(args), x**y)
    df = f.eval_grad(args)
    assert eq(df["x"], y * x ** (y - 1))
    assert eq(df["y"], x**y * math.log(x))


def test_poly() -> None:
    f = lambda t: 1 - 3 * 2 * (t + 1) ** 2 - 4 * t
    df = lambda t: -6 * 2 * (t + 1) - 4

    x = random.random()
    args = dict(x=x)
    assert eq(f(X)(args), f(x))

    grad = f(X).eval_grad(args)
    assert eq(grad["x"], df(x))


def test_ln() -> None:
    x = random.random()
    f = X.ln()
    args = dict(x=x)
    assert f(args) == math.log(x)
    df = f.eval_grad(args)
    assert eq(df["x"], 1 / x)
