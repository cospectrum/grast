import grast.delta as d
from dataclasses import dataclass

from ..delta import Delta as D

from .real_str import real_str
from .brackets import brackets, is_atom


@dataclass
class DeltaStr:
    delta: D
    zero: str = "0"

    def __repr__(self) -> str:
        return self.to_str(self.delta)

    @classmethod
    def to_str(cls, expr: D) -> str:
        match expr:
            case d.Zero():
                return cls.zero

            case d.OneHot(var):
                return f"d{var.val}"

            case d.Add(l, r):
                return f"{cls(l)} + {cls(r)}"

            case d.Sub(l, r):
                return f"{cls(l)} - {delta_str(r)}"

            case d.Neg(e):
                return f"-{delta_str(e)}"

            case d.Scale(scalar, delta):
                return f"{real_str(scalar)} * {delta_str(delta)}"

            case _:
                raise TypeError


def delta_str(delta: D, with_brackets: bool = True) -> str:
    s = DeltaStr.to_str(delta)
    if with_brackets:
        s = s if is_atom(delta) else brackets(s)
    return s
