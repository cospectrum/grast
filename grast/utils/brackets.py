from typing import Any
import grast.real as re
import grast.delta as de

from grast.real import Expression as R
from grast.delta import Delta as D


def brackets(s: str, left_bracket: str = "(", right_bracket: str = ")") -> str:
    if s[0] == left_bracket and s[-1] == right_bracket:
        return s
    return f"{left_bracket}{s}{right_bracket}"


def is_atom(expr: R | D | Any) -> bool:
    if (
        isinstance(expr, re.Value)
        or isinstance(expr, de.OneHot)
        or isinstance(expr, de.Zero)
    ):
        return True
    return False
