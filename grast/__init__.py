from typing import TypeVar

from grast.delta import OneHot

from .dual import var as var
from .dual import const as const


T = TypeVar("T")


def one_hot(val: T) -> OneHot[T]:
    d = var(val).delta
    assert isinstance(d, OneHot)
    return d
