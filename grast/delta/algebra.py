from .delta import (
    Add,
    Scale,
    Zero,
    Sub,
    Neg,
    Delta as D,
)

from ..real import Real as R


class Algebra:
    @classmethod
    def add(cls, left: D, right: D) -> D:
        match left, right:
            case Zero(), Zero():
                return Zero()
            case _, Zero():
                return left
            case Zero(), _:
                return right
            case _, Neg(delta):
                return cls.sub(left, delta)
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

    @classmethod
    def scale(cls, real: R, delta: D) -> D:
        match delta:
            case Zero():
                return Zero()
            case Scale(scalar, inner):
                return cls.scale(real.mul(scalar), inner)
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
