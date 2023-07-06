import random
import math
from typing import Any

from grast import var
from grast.dual import Dual


X: Dual[float] = var("x")
Y: Dual[float] = var("y")


def rand() -> float:
    return random.random()


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


def test_sin() -> None:
    a = rand()
    sin_a = math.sin(a)
    cos_a = math.cos(a)
    args = dict(x=a)

    f = X.sin()
    val = f(args)
    d = f.eval_grad(args)
    assert val == sin_a
    assert d["x"] == cos_a

    sin_sin_a = math.sin(sin_a)
    cos_sin_a = math.cos(sin_a)

    f = X.sin().cos()
    val = f(args)
    d = f.eval_grad(args)
    assert eq(val, cos_sin_a)
    assert eq(d["x"], sin_sin_a * (-cos_a))

    f = X.sin().sin()
    val = f(args)
    d = f.eval_grad(args)
    assert eq(val, sin_sin_a)
    assert eq(d["x"], cos_a * cos_sin_a)


def test_cos() -> None:
    a = rand()
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    args = dict(x=a)

    val = X.cos()(args)
    d = X.cos().eval_grad(args)

    assert val == cos_a
    assert d["x"] == -sin_a

    val = X.cos().cos()(args)
    d = X.cos().cos().eval_grad(args)
    assert val == math.cos(cos_a)
    assert eq(d["x"], sin_a * math.sin(cos_a))


def test_exp() -> None:
    a = rand()
    exp_a = math.exp(a)
    args = dict(x=a)

    f = X.exp()
    d = f.eval_grad(args)
    assert f(args) == exp_a
    assert d["x"] == exp_a

    f = f.exp()
    d = f.eval_grad(args)
    assert f(args) == math.exp(exp_a)
    assert eq(d["x"], math.exp(a + exp_a))


def test_abs() -> None:
    a = rand()
    args = dict(x=a)
    f = abs(X)
    assert f(args) == abs(a)
    df = f.grad()
    assert eq(df["x"](args), a / abs(a))


def test_tan() -> None:
    a = rand()
    tg_a = math.sin(a) / math.cos(a)
    sec_a = 1 / math.cos(a)
    args = dict(x=a)

    f = X.tan()
    val = f(args)
    d = f.eval_grad(args)
    assert eq(val, tg_a)
    assert eq(d["x"], sec_a**2)

    tan = X.sin() / X.cos()
    dtan = tan.eval_grad(args)
    assert eq(tan(args), val)
    assert eq(dtan["x"], d["x"])
