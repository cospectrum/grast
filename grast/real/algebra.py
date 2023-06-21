import grast.real as re
from .expression import Expression as R


class Algebra:
    @staticmethod
    def add(left: R, right: R) -> R:
        return re.Add(left, right)

    @staticmethod
    def mul(left: R, right: R) -> R:
        return re.Mul(left, right)

    @classmethod
    def div(cls, left: R, right: R) -> R:
        match left, right:
            case _, re.Inv(r):
                return cls.mul(left, r)
        return re.Div(left, right)

    @classmethod
    def sub(cls, left: R, right: R) -> R:
        match left, right:
            case _, re.Neg(r):
                return cls.add(left, r)
            case re.Neg(l), re.Neg(r):
                return cls.sub(r, l)
            case re.Neg(l), _:
                return cls.neg(cls.add(l, right))
        return re.Sub(left, right)

    @staticmethod
    def neg(real: R) -> R:
        match real:
            case re.Neg(r):
                return r
        return re.Neg(real)

    @staticmethod
    def inv(real: R) -> R:
        match real:
            case re.Inv(r):
                return r
        return re.Inv(real)
