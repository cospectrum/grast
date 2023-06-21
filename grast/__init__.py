from typing import TypeVar

from grast.delta import OneHot
from grast.eval_grad import Grad, Eval

from .dual import var as var
from .dual import const as const
from .dual import Dual


T = TypeVar("T")


def one_hot(val: T) -> OneHot[T]:
    d = var(val).delta
    assert isinstance(d, OneHot)
    return d


def grad(dual: Dual, one=None) -> Grad:
    delta = dual.delta
    return Eval(one).grad(delta)
