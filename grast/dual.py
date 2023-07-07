from __future__ import annotations
import grast.real as re

from typing import Generic, TypeVar
from dataclasses import dataclass


from .cfg import Cfg
from .real import Real
from .delta import (
    Delta,
    OneHot,
    Zero,
)


T = TypeVar("T")
Args = dict[str, T]


@dataclass
class Dual(Generic[T]):
    real: Real[T]
    delta: Delta[T]

    def __call__(
        self,
        args: Args[T] | None = None,
        cfg: Cfg[T] | None = None,
    ) -> T:
        return self.real(args, cfg)

    def grad(self, one: Real[T] | None = None) -> dict[str, Real[T]]:
        return self.delta(one)

    def eval_grad(
        self,
        args: Args[T] | None = None,
        cfg: Cfg[T] | None = None,
    ) -> Args[T]:
        grad = self.grad()
        return {k: v(args, cfg=cfg) for k, v in grad.items()}

    def freeze(self) -> Dual[T]:
        return Dual(self.real, Zero())

    def __str__(self) -> str:
        return str(self.real)

    def __add__(self, other: Dual[T] | T) -> Dual[T]:
        other = wrap(other)
        a, b = self.tup
        c, d = other.tup
        return Dual(a.add(c), b.add(d))

    def __radd__(self, other: Dual[T] | T) -> Dual[T]:
        return wrap(other) + self

    def __sub__(self, other: Dual[T] | T) -> Dual[T]:
        other = wrap(other)
        a, b = self.tup
        c, d = other.tup
        return Dual(a.sub(c), b.sub(d))

    def __rsub__(self, other: Dual[T] | T) -> Dual[T]:
        return wrap(other) - self

    def __neg__(self) -> Dual[T]:
        a, b = self.tup
        return Dual(a.neg(), b.neg())

    def __mul__(self, other: Dual[T] | T) -> Dual[T]:
        other = wrap(other)
        a, b = self.tup
        c, d = other.tup
        delta = d.scale(a).add(b.scale(c))
        return Dual(a.mul(c), delta)

    def __rmul__(self, other: Dual[T] | T) -> Dual[T]:
        return wrap(other) * self

    def __truediv__(self, other: Dual[T] | T) -> Dual[T]:
        other = wrap(other)
        a, b = self.tup
        c, d = other.tup
        first = b.scale(c.inv())
        second = d.scale(a).scale(c.mul(c).inv())
        return Dual(a.div(c), first.sub(second))

    def __rtruediv__(self, other: Dual[T] | T) -> Dual[T]:
        return wrap(other) / self

    def __pow__(self, other: Dual[T] | T) -> Dual[T]:
        other = wrap(other)
        a, b = self.tup
        c, d = other.tup
        val = a.pow(c)
        left = b.scale(c.mul(val.div(a)))
        right = d.scale(val.mul(a.ln()))
        return Dual(val, left.add(right))

    def __rpow__(self, other: Dual[T] | T) -> Dual[T]:
        return wrap(other) ** self

    def __abs__(self) -> Dual[T]:
        a, b = self.tup
        val = a.abs()
        delta = b.scale(a.div(val))
        return Dual(val, delta)

    @property
    def tup(self) -> tuple[Real[T], Delta[T]]:
        return self.real, self.delta

    def ln(self) -> Dual[T]:
        a, b = self.tup
        return Dual(a.ln(), b.scale(a.inv()))

    def exp(self) -> Dual[T]:
        a, b = self.tup
        val = a.exp()
        return Dual(val, b.scale(val))

    def cos(self) -> Dual[T]:
        a, b = self.tup
        val = a.cos()
        delta = b.scale(a.sin().neg())
        return Dual(val, delta)

    def sin(self) -> Dual[T]:
        a, b = self.tup
        val = a.sin()
        delta = b.scale(a.cos())
        return Dual(val, delta)

    def tan(self) -> Dual[T]:
        a, b = self.tup
        cos = a.cos()
        delta = b.scale(cos.mul(cos).inv())
        return Dual(a.tan(), delta)


def wrap(val: Dual[T] | T) -> Dual[T]:
    if isinstance(val, Dual):
        return val
    return const(val)


def var(key: str, requires_grad: bool = True) -> Dual:
    real: re.Var = re.Var(key=key)
    delta = OneHot(var=real) if requires_grad else Zero()
    return Dual(real, delta)


def const(val: T) -> Dual[T]:
    real = re.Const(val=val)
    delta = Zero()
    return Dual(real, delta)
