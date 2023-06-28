from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar


__all__ = [
    "Real",
    "Value",
    "Var",
    "Const",
    "BinaryFn",
    "UnaryFn",
]


T = TypeVar("T")


class Real:
    def to_str(self) -> str:
        import grast.utils as utils

        return utils.to_str(self)

    def __call__(self, **kwargs: float) -> float:
        import grast.forward as forward

        return forward.Forward(kwargs)(self)

    def add(self, other: Real) -> Real:
        return A().add(self, other)

    def mul(self, other: Real) -> Real:
        return A().mul(self, other)

    def sub(self, other: Real) -> Real:
        return A().sub(self, other)

    def neg(self) -> Real:
        return A().neg(self)

    def inv(self) -> Real:
        return A().inv(self)

    def div(self, other: Real) -> Real:
        return A().div(self, other)

    def pow(self, other: Real) -> Real:
        return A().pow(self, other)

    def ln(self) -> Real:
        return A().ln(self)


def A():
    import grast.real.algebra as algebra

    return algebra.Algebra


R = Real


@dataclass
class Value(R, Generic[T]):
    val: T


class Var(Value[T]):
    pass


class Const(Value[T]):
    pass


@dataclass
class UnaryFn(R):
    arg: R


@dataclass
class BinaryFn(R):
    left: R
    right: R
