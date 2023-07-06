from typing import Any, TypeVar
from grast.delta import Delta

from .dual import Dual, var as var
from .dual import const as const

from .cfg import Cfg as Cfg


__all__ = [
    "Cfg",
    "Dual",
    "var",
    "const",
    "one_hot",
]


T = TypeVar("T", bound=Any)


def one_hot(key: str) -> Delta[T]:
    return var(key).delta
