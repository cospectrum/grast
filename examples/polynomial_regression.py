import random
import math

import matplotlib.pyplot as plt

from grast import var, Dual


def rand() -> float:
    return random.uniform(0, math.pi)


def unknown_fn(x: float) -> float:
    return math.sin(x)


def main() -> None:
    n = 5
    param_keys = [f"w_{i}" for i in range(n)]
    X: Dual = var("x")
    Y: Dual = var("y")  # right answer

    f: Dual = sum(var(k) * X**i for i, k in enumerate(param_keys))  # type: ignore
    loss = (f - Y) ** 2
    dl = loss.grad()

    args = {k: 0.0 for k in param_keys}
    lr = 3e-4

    for _ in range(2000):
        x = rand()
        y = unknown_fn(x)
        args["x"] = x
        args["y"] = y

        for k in param_keys:
            args[k] -= lr * dl[k](args)

    del args["x"]
    del args["y"]
    print(f"{args=}")

    xs = [rand() for _ in range(150)]
    ys = [unknown_fn(x) for x in xs]
    predictions = []

    for x in xs:
        args["x"] = x
        pred = f(args)
        predictions.append(pred)

    plt.scatter(xs, ys, c="green")
    plt.scatter(xs, predictions, c="red")
    plt.show()


if __name__ == "__main__":
    main()
