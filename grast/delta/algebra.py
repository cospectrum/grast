from .delta import (
    Add,
    Scale,
    Zero,
    Sub,
    Neg,
    Delta as D,
)

from ..real import Expression as R


class Algebra:
    @staticmethod
    def add(left: D, right: D) -> D:
        match left, right:
            case Zero(), Zero():
                return Zero()
            case _, Zero():
                return left
            case Zero(), _:
                return right
            case _:
                return Add(left, right)

    @classmethod
    def sub(cls, left: D, right: D) -> D:
        match left, right:
            case Zero(), Zero():
                return Zero()
            case Zero(), _:
                return cls.neg(right)
            case _, Zero():
                return left
            case _, Neg(d):
                return cls.add(left, d)
            case _:
                return Sub(left, right)

    @staticmethod
    def scale(real: R, delta: D) -> D:
        match delta:
            case Zero():
                return Zero()
            case _:
                return Scale(real, delta)

    @staticmethod
    def neg(delta: D) -> D:
        match delta:
            case Zero():
                return delta
            case Neg(d):
                return d
            case _:
                return Neg(delta)
