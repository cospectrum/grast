from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar

from grast.cfg import Cfg


__all__ = [
    "Real",
    "Value",
    "Var",
    "Const",
    "BinaryFn",
    "UnaryFn",
]


T = TypeVar("T")


class Real(Generic[T]):
    def __str__(self) -> str:
        import grast.utils as utils

        return utils.to_str(self)

    def __call__(
        self, args: dict[str, T] | None = None, cfg: Cfg[T] | None = None
    ) -> T:
        import grast.forward as forward

        return forward.Forward(args, cfg=cfg)(self)

    def add(self, other: Real[T]) -> Real[T]:
        return A().add(self, other)

    def mul(self, other: Real[T]) -> Real[T]:
        return A().mul(self, other)

    def sub(self, other: Real[T]) -> Real[T]:
        return A().sub(self, other)

    def neg(self) -> Real[T]:
        return A().neg(self)

    def inv(self) -> Real[T]:
        return A().inv(self)

    def abs(self) -> Real[T]:
        return A().abs(self)

    def div(self, other: Real[T]) -> Real[T]:
        return A().div(self, other)

    def pow(self, other: Real[T]) -> Real[T]:
        return A().pow(self, other)

    def ln(self) -> Real[T]:
        return A().ln(self)

    def exp(self) -> Real[T]:
        return A().exp(self)

    def cos(self) -> Real[T]:
        return A().cos(self)

    def sin(self) -> Real[T]:
        return A().sin(self)

    def tan(self) -> Real[T]:
        return A().tan(self)


def A():
    import grast.real.algebra as algebra

    return algebra.Algebra


R = Real


class Value(R[T]):
    pass


@dataclass
class Var(Value[T]):
    key: str


@dataclass
class Const(Value[T]):
    val: T


@dataclass
class UnaryFn(R[T]):
    arg: R[T]


@dataclass
class BinaryFn(R[T]):
    left: R[T]
    right: R[T]
