from __future__ import annotations

import grast.real as re

from typing import Hashable
from dataclasses import dataclass

from .real import Algebra as ra
from .real import Real as R
from .delta import (
    Delta as D,
    OneHot,
    Zero,
    Algebra as da,
)

from .eval_grad import Grad, Eval


T = Hashable


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
        return Dual(ra.add(a, c), da.add(b, d))

    def __sub__(self, other: Dual) -> Dual:
        a, b = self.tup
        c, d = other.tup
        return Dual(ra.sub(a, c), da.sub(b, d))

    def __neg__(self) -> Dual:
        a, b = self.tup
        return Dual(ra.neg(a), da.neg(b))

    def __mul__(self, other: Dual) -> Dual:
        a, b = self.tup
        c, d = other.tup
        delta = da.add(da.scale(a, d), da.scale(c, b))
        return Dual(ra.mul(a, c), delta)

    def __truediv__(self, other: Dual) -> Dual:
        a, b = self.tup
        c, d = other.tup
        r = ra.div(a, c)
        first = da.scale(ra.inv(c), b)
        num = da.scale(a, d)
        second = da.scale(ra.inv(ra.mul(c, c)), num)
        delta = da.sub(first, second)
        return Dual(r, delta)


def var(val: T) -> Dual:
    real = re.Var(val=val)
    delta = OneHot(var=real)
    return Dual(real, delta)


def const(val: T) -> Dual:
    real = re.Const(val=val)
    delta = Zero()
    return Dual(real, delta)
