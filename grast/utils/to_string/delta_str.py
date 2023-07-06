import grast.delta as d

from grast.delta import Delta as D
from dataclasses import dataclass

from .real_str import real_str
from .utils import brackets, is_atom


@dataclass
class DeltaStr:
    delta: D
    zero: str = "0"

    def __repr__(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        cls = DeltaStr
        match self.delta:
            case d.Zero():
                return self.zero

            case d.OneHot(var):
                return f"d{var.key}"

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


def delta_str(delta: D) -> str:
    s = str(DeltaStr(delta))
    if is_atom(delta):
        return s
    return brackets(s)
