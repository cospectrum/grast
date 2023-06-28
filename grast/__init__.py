from grast.delta import OneHot

from .dual import var as var
from .dual import const as const


def one_hot(key: str) -> OneHot[str]:
    d = var(key).delta
    assert isinstance(d, OneHot)
    return d
