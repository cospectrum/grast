import grast.real as re
import grast.delta as de

from grast.real import Real as R
from grast.delta import Delta as D

from typing import Any, Iterable, Type


def brackets(s: str, left_bracket: str = "(", right_bracket: str = ")") -> str:
    if s[0] == left_bracket and s[-1] == right_bracket:
        return s
    return f"{left_bracket}{s}{right_bracket}"


def type_of_any(val: Any, types: Iterable[Type]) -> bool:
    return any(isinstance(val, t) for t in types)


def is_atom(expr: R | D | Any) -> bool:
    types = [re.Value, de.OneHot, de.Zero]
    return type_of_any(expr, types)


def is_primitive_binary_fn(fn: re.BinaryFn) -> bool:
    types = [re.Sub, re.Mul, re.Add, re.Div, re.Pow]
    return type_of_any(fn, types)


def is_primitive_unary_fn(fn: re.UnaryFn) -> bool:
    types = [re.Neg, re.Inv]
    return type_of_any(fn, types)


def is_primitive_fn(fn: re.Real) -> bool:
    if isinstance(fn, re.UnaryFn):
        return is_primitive_unary_fn(fn)
    elif isinstance(fn, re.BinaryFn):
        return is_primitive_binary_fn(fn)
    return False


def not_primitive_fn(fn: re.Real) -> bool:
    return not is_primitive_fn(fn)
