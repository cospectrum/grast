import grast.real as re

from typing import Generic

from .real import T, Real as R


class Algebra(Generic[T]):
    @classmethod
    def add(cls, left: R[T], right: R[T]) -> R[T]:
        match left, right:
            case re.Neg(l), re.Neg(r):
                return cls.neg(cls.add(l, r))
            case _, re.Neg(r):
                return cls.sub(left, r)
            case re.Neg(l), _:
                return cls.sub(right, l)
        return re.Add(left, right)

    @classmethod
    def mul(cls, left: R[T], right: R[T]) -> R[T]:
        match left, right:
            case re.Inv(l), re.Inv(r):
                return cls.inv(cls.mul(l, r))
            case re.Inv(l), _:
                return cls.div(right, l)
            case _, re.Inv(r):
                return cls.div(left, r)
        return re.Mul(left, right)

    @classmethod
    def div(cls, left: R[T], right: R[T]) -> R[T]:
        match left, right:
            case re.Inv(l), re.Inv(r):
                return cls.div(r, l)
            case re.Inv(l), _:
                return cls.inv(cls.mul(l, right))
            case _, re.Inv(r):
                return cls.mul(left, r)
        return re.Div(left, right)

    @classmethod
    def sub(cls, left: R[T], right: R[T]) -> R[T]:
        match left, right:
            case _, re.Neg(r):
                return cls.add(left, r)
            case re.Neg(l), re.Neg(r):
                return cls.sub(r, l)
            case re.Neg(l), _:
                return cls.neg(cls.add(l, right))
        return re.Sub(left, right)

    @staticmethod
    def neg(real: R[T]) -> R[T]:
        match real:
            case re.Neg(r):
                return r
        return re.Neg(real)

    @staticmethod
    def inv(real: R[T]) -> R[T]:
        match real:
            case re.Inv(r):
                return r
        return re.Inv(real)

    @staticmethod
    def pow(left: R[T], right: R[T]) -> R[T]:
        return re.Pow(left, right)

    @staticmethod
    def abs(real: R[T]) -> R[T]:
        match real:
            case re.Abs(_):
                return real
        return re.Abs(real)

    @staticmethod
    def ln(real: R[T]) -> R[T]:
        return re.Ln(real)

    @staticmethod
    def exp(real: R[T]) -> R[T]:
        return re.Exp(real)

    @staticmethod
    def cos(real: R[T]) -> R[T]:
        return re.Cos(real)

    @staticmethod
    def sin(real: R[T]) -> R[T]:
        return re.Sin(real)

    @staticmethod
    def tan(real: R[T]) -> R[T]:
        return re.Tan(real)
