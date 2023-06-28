from __future__ import annotations

import grast.real as re

from typing import Any
from dataclasses import dataclass

from .real import Real as R
from .delta import (
    Delta as D,
    OneHot,
    Zero,
)

from .eval_grad import Grad, Eval


@dataclass
class Dual:
    real: R
    delta: D

    def grad(self, one: R | None = None) -> Grad:
        delta = self.delta
        return Eval(one).grad(delta)

    @property
    def tup(self) -> tuple[R, D]:
        return self.real, self.delta

    def __add__(self, other: Dual) -> Dual:
        a, b = self.tup
        c, d = other.tup
        return Dual(a.add(c), b.add(d))

    def __sub__(self, other: Dual) -> Dual:
        a, b = self.tup
        c, d = other.tup
        return Dual(a.sub(c), b.sub(d))

    def __neg__(self) -> Dual:
        a, b = self.tup
        return Dual(a.neg(), b.neg())

    def __mul__(self, other: Dual) -> Dual:
        a, b = self.tup
        c, d = other.tup
        delta = d.scale(a).add(b.scale(c))
        return Dual(a.mul(c), delta)

    def __truediv__(self, other: Dual) -> Dual:
        a, b = self.tup
        c, d = other.tup
        first = b.scale(c.inv())
        second = d.scale(a).scale(c.mul(c).inv())
        return Dual(a.div(c), first.sub(second))

    def __pow__(self, other: Dual) -> Dual:
        a, b = self.tup
        c, d = other.tup
        val = a.pow(c)
        left = b.scale(c.mul(val.div(a)))
        right = d.scale(val.mul(a.ln()))
        return Dual(val, left.add(right))

    def ln(self) -> Dual:
        a, b = self.tup
        return Dual(a.ln(), b.scale(a.inv()))


def var(key: str) -> Dual:
    real = re.Var(val=key)
    delta = OneHot(var=real)
    return Dual(real, delta)


def const(val: Any) -> Dual:
    real = re.Const(val=val)
    delta = Zero()
    return Dual(real, delta)
