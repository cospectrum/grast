import grast.delta as de
import grast.real as re

from typing import Any, Callable, Generic, TypeVar


T = TypeVar("T", bound=Any)

R = re.Real
D = de.Delta
Grad = dict[str, R[T]]


def add_at(grad: Grad[T], var: re.Var[T], real: R[T]) -> None:
    key = var.key
    if key in grad:
        grad[key] = grad[key].add(real)
    else:
        grad[key] = real


One = re.Const(1)


class Eval(Generic[T]):
    real: R[T]

    def __init__(self, real: R[T] | None = None) -> None:
        self.real = One if real is None else real  # type: ignore

    def grad(self, expression: D[T]) -> Grad[T]:
        g: Grad[T] = dict()
        return self(expression)(g)

    def __call__(self, expression: D[T]) -> Callable[[Grad[T]], Grad[T]]:
        def callback(grad: Grad[T]) -> Grad[T]:
            match expression:
                case de.OneHot(var):
                    add_at(grad, var, self.real)

                case de.Scale(scalar, expr):
                    if self.real == One:
                        real = scalar
                    elif scalar == One:
                        real = self.real
                    elif self.real == One.neg():
                        real = scalar.neg()
                    elif scalar == One.neg():
                        real = self.real.neg()
                    else:
                        real = self.real.mul(scalar)
                    Eval(real)(expr)(grad)

                case de.Neg(expr):
                    real = self.real.neg()
                    Eval(real)(expr)(grad)

                case de.Add(left, right):
                    self(left)(self(right)(grad))

                case de.Sub(left, right):
                    self(left)(self(right.neg())(grad))

                case de.Zero():
                    pass

                case expr:
                    raise TypeError(f"unrecognized expression: {expr}")
            return grad

        return callback
