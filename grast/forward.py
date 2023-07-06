from __future__ import annotations

import grast.real as re

from typing import Any, Generic, TypeVar
from grast.real import Real

from .cfg import Cfg


__all__ = [
    "Forward",
]

T = TypeVar("T", bound=Any)


class Forward(Generic[T]):
    args: dict[str, T]
    cfg: Cfg[T]

    def __init__(
        self,
        args: dict[str, T] | None = None,
        cfg: Cfg[T] | None = None,
    ) -> None:
        self.args = dict() if args is None else args
        self.cfg = Cfg.float() if cfg is None else cfg  # type: ignore

    def __call__(self, real: Real[T]) -> T:
        cfg = self.cfg
        match real:
            case re.Const(val):
                return val
            case re.Var(key):
                return self.args[key]
            case re.Add(left, right):
                return self(left) + self(right)
            case re.Sub(left, right):
                return self(left) - self(right)
            case re.Mul(left, right):
                return self(left) * self(right)
            case re.Div(left, right):
                return self(left) / self(right)
            case re.Pow(left, right):
                return self(left) ** self(right)
            case re.Neg(arg):
                return -self(arg)
            case re.Abs(arg):
                return abs(self(arg))
            case re.Inv(arg):
                return cfg.inv(self(arg))
            case re.Ln(arg):
                return cfg.ln(self(arg))
            case re.Exp(arg):
                return cfg.exp(self(arg))
            case re.Cos(arg):
                return cfg.cos(self(arg))
            case re.Sin(arg):
                return cfg.sin(self(arg))
            case re.Tan(arg):
                return cfg.tan(self(arg))
        raise TypeError
