from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from grast.real import Real, Var


__all__ = ["Delta", "Zero", "OneHot", "Add", "Neg", "Scale", "Sub"]

T = TypeVar("T")


class Delta:
    pass


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
