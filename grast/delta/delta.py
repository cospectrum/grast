from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from grast.real import Real, Var


__all__ = ["Delta", "Zero", "OneHot", "Add", "Neg", "Scale", "Sub"]

T = TypeVar("T")


class Delta:
    def to_str(self) -> str:
        import grast.utils as utils

        return utils.to_str(self)

    def scale(self, scalar: Real) -> Delta:
        return A().scale(scalar, self)

    def neg(self) -> Delta:
        return A().neg(self)

    def add(self, other: Delta) -> Delta:
        return A().add(self, other)

    def sub(self, other: Delta) -> Delta:
        return A().sub(self, other)


def A():
    import grast.delta.algebra as algebra

    return algebra.Algebra


D = Delta


class Zero(D):
    _instance: Zero | None = None

    def __new__(cls) -> Zero:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Zero()"


@dataclass
class OneHot(D, Generic[T]):
    var: Var[T]


@dataclass
class Add(D):
    left: D
    right: D


@dataclass
class Scale(D):
    real: Real
    delta: D


@dataclass
class Neg(D):
    delta: D


@dataclass
class Sub(D):
    left: D
    right: D
