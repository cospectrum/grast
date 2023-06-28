import math
import grast.real as re

from grast.real import Real
from dataclasses import dataclass


T = float


@dataclass
class Forward:
    kwargs: dict[str, T]

    def __call__(self, real: Real) -> T:
        match real:
            case re.Const(val):
                return val
            case re.Var(key):
                return self.kwargs[key]
            case re.Add(left, right):
                return self(left) + self(right)
            case re.Sub(left, right):
                return self(left) - self(right)
            case re.Mul(left, right):
                return self(left) * self(right)
            case re.Div(left, right):
                return self(left) / self(right)
            case re.Neg(arg):
                return -self(arg)
            case re.Inv(arg):
                return 1 / self(arg)
            case re.Pow(left, right):
                return self(left) ** self(right)
            case re.Ln(arg):
                return math.log(self(arg))
        raise TypeError
