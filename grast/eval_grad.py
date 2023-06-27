import grast.delta as de
import grast.real as re

from .real import Algebra as ra

from typing import Callable, Hashable


R = re.Real
D = de.Delta
Grad = dict[Hashable, R]


def add_at(grad: Grad, var: re.Var, real: R) -> None:
    key = var.val
    if key in grad:
        grad[key] = ra.add(grad[key], real)
    else:
        grad[key] = real


One = re.Const(1)


class Eval:
    real: R

    def __init__(self, real: R | None = None) -> None:
        if real is None:
            real = One
        self.real = real

    def grad(self, expression: D) -> Grad:
        g: Grad = dict()
        return self(expression)(g)

    def __call__(self, expression: D) -> Callable[[Grad], Grad]:
        def callback(grad: Grad) -> Grad:
            match expression:
                case de.OneHot(var):
                    add_at(grad, var, self.real)

                case de.Scale(scalar, expr):
                    if self.real == One:
                        real = scalar
                    else:
                        real = ra.mul(self.real, scalar)
                    Eval(real)(expr)(grad)

                case de.Neg(expr):
                    real = ra.neg(self.real)
                    Eval(real)(expr)(grad)

                case de.Add(left, right):
                    self(right)(self(left)(grad))

                case de.Sub(left, right):
                    self(left)(self(de.Neg(right))(grad))

                case de.Zero():
                    pass

                case expr:
                    raise TypeError(f"unrecognized expression: {expr}")
            return grad

        return callback
