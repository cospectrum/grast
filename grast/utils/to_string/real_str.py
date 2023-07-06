import grast.real as re
import grast.real.unary as uf
import grast.real.binary as bf

from grast.real import Real as R
from dataclasses import dataclass

from .utils import brackets, is_primitive_fn


@dataclass
class RealStr:
    real: R

    @classmethod
    def unary_fn(cls, expr: R) -> str:
        assert isinstance(expr, re.UnaryFn)
        arg = expr.arg

        match type(expr):
            case uf.Neg:
                return f"-{real_str(arg)}"
            case uf.Inv:
                return f"{real_str(arg)}^-1"

        name = expr.__class__.__name__.lower()
        return f"{name}({cls(arg)})"

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
            case bf.Div:
                return f"{real_str(left)} / {real_str(right)}"
            case bf.Pow:
                return f"{real_str(left)} ^ {real_str(right)}"

        name = expr.__class__.__name__.lower()
        return f"{name}({cls(left)}, {cls(right)})"

    def to_str(self) -> str:
        cls = RealStr
        expr = self.real
        match expr:
            case re.UnaryFn(_):
                return cls.unary_fn(expr)
            case re.BinaryFn(_, _):
                return cls.binary_fn(expr)
            case re.Const(val):
                return f"{val}"
            case re.Var(key):
                return f"{key}"
        raise TypeError


def real_str(real: R) -> str:
    s = str(RealStr(real))
    if is_primitive_fn(real):
        return brackets(s)
    return s
