from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from grast.real import Real, Var


__all__ = ["Delta", "Zero", "OneHot", "Add", "Neg", "Scale", "Sub"]

T = TypeVar("T")


class Delta(Generic[T]):
    def __call__(self, one: Real[T] | None = None) -> dict[str, Real[T]]:
        import grast.grad as grad

        return grad.Eval(one).grad(self)

    def __str__(self) -> str:
        import grast.utils as utils

        return utils.to_str(self)

    def scale(self, scalar: Real[T]) -> Delta[T]:
        return A().scale(scalar, self)

    def neg(self) -> Delta[T]:
        return A().neg(self)

    def add(self, other: Delta[T]) -> Delta[T]:
        return A().add(self, other)

    def sub(self, other: Delta[T]) -> Delta[T]:
        return A().sub(self, other)


def A():
    import grast.delta.algebra as algebra

    return algebra.Algebra


D = Delta


class Zero(D[T]):
    _instance: Zero | None = None

    def __new__(cls) -> Zero:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Zero()"


@dataclass
class OneHot(D[T]):
    var: Var[T]


@dataclass
class Add(D[T]):
    left: D[T]
    right: D[T]


@dataclass
class Scale(D[T]):
    real: Real[T]
    delta: D[T]


@dataclass
class Neg(D[T]):
    delta: D[T]


@dataclass
class Sub(D[T]):
    left: D[T]
    right: D[T]
