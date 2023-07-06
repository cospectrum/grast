from __future__ import annotations
import math

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


T = TypeVar("T", bound=Any)
UnaryFn = Callable[[T], T]


@dataclass
class Cfg(Generic[T]):
    ln: UnaryFn[T]
    inv: UnaryFn[T] = lambda x: x**-1  # type: ignore

    @staticmethod
    def float() -> Cfg[float]:
        ln = lambda x: math.log(x)
        return Cfg(ln=ln)
