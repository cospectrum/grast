import math


def test_readme() -> None:
    from grast import var

    X = var("x")
    Y = var("y")
    Z = var("z").freeze()  # arg without derivative

    h = X / Y + Y**X
    f = Z * h + 3

    df = f.grad()
    df_dx = df["x"]
    df_dy = df["y"]

    x: float = -3.0
    y: float = 5.0
    z: float = 2.0
    args = dict(x=x, y=y, z=z)
    assert f(args) == z * h(args) + 3
    assert h(args) == x / y + y**x

    assert eq(
        df_dx(args),
        z * (y**x * math.log(y) + 1 / y),
    )
    assert eq(
        df_dy(args),
        x * z * (y ** (x + 1) - 1) / y**2,
    )

    print(str(f))
    print(str(df_dx))
    print(str(df_dy))


def eq(a: float, b: float) -> bool:
    return math.isclose(a, b)
