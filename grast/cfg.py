from __future__ import annotations
import math

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


T = TypeVar("T", bound=Any)
UnaryFn = Callable[[T], T]


@dataclass
class Cfg(Generic[T]):
    ln: UnaryFn[T]
    exp: UnaryFn[T]
    cos: UnaryFn[T]
    sin: UnaryFn[T]
    tan: UnaryFn[T]
    inv: UnaryFn[T] = lambda x: x**-1  # type: ignore

    @staticmethod
    def float() -> Cfg[float]:
        ln = lambda x: math.log(x)
        exp = lambda x: math.exp(x)
        cos = lambda x: math.cos(x)
        sin = lambda x: math.sin(x)
        tan = lambda x: math.tan(x)
        return Cfg(ln=ln, exp=exp, cos=cos, sin=sin, tan=tan)
