from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar


__all__ = [
    "Expression",
    "Value",
    "Var",
    "Const",
    "BinaryFn",
    "UnaryFn",
]


T = TypeVar("T")


class Expression:
    pass


E = Expression


@dataclass
class Value(E, Generic[T]):
    val: T

    def __repr__(self) -> str:
        return f"{self.val}"


class Var(Value[T]):
    pass


class Const(Value[T]):
    pass


@dataclass
class UnaryFn(E):
    arg: E

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}({self.arg})"


@dataclass
class BinaryFn(E):
    left: E
    right: E

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}({self.left}, {self.right})"
