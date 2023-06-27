import grast.real as re
import grast.real.unary as uf
import grast.real.binary as bf

from grast.real import Real as R

from typing import Callable
from dataclasses import dataclass

from .brackets import brackets, is_atom


def unary_prefix(expr: R) -> Callable[[str], str]:
    assert isinstance(expr, re.UnaryFn)
    match expr:
        case uf.Pos(_):
            return lambda s: f"{s}"
        case uf.Neg(e):
            return lambda s: f"-{s if is_atom(e) else brackets(s)}"
    return unary_postfix(expr, throw=True)


def unary_postfix(expr: R, throw: bool = False) -> Callable[[str], str]:
    assert isinstance(expr, re.UnaryFn)
    match expr:
        case uf.Inv(e):
            return lambda s: f"{s if is_atom(e) else brackets(s)}^-1"
        case _:
            if throw:
                raise TypeError
            return unary_prefix(expr)


@dataclass
class RealStr:
    real: R

    @classmethod
    def unary_fn(cls, expr: R) -> str:
        assert isinstance(expr, re.UnaryFn)
        f = unary_prefix(expr)
        return f(real_str(expr.arg))

    def __repr__(self) -> str:
        return self.to_str()

    @classmethod
    def binary_fn(cls, expr: R) -> str:
        assert isinstance(expr, re.BinaryFn)
        left = expr.left
        right = expr.right
        match type(expr):
            case bf.Add:
                return f"{cls(left)} + {cls(right)}"
            case bf.Mul:
                return f"{real_str(left)} * {real_str(right)}"
            case bf.Sub:
                return f"{cls(left)} - {real_str(right)}"

        raise TypeError

    def to_str(self) -> str:
        cls = RealStr
        expr = self.real
        match expr:
            case re.UnaryFn(_):
                return cls.unary_fn(expr)
            case re.BinaryFn(_, _):
                return cls.binary_fn(expr)
            case re.Value(_):
                return str(expr)
            case _:
                return str(expr)


def real_str(real: R, with_brackets: bool = True) -> str:
    s = str(RealStr(real))
    if with_brackets:
        s = s if is_atom(real) else brackets(s)
    return s
