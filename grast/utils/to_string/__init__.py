from grast.delta import Delta
from grast.real import Real

from .real_str import RealStr
from .delta_str import DeltaStr


def to_str(number: Real | Delta) -> str:
    if isinstance(number, Real):
        return str(RealStr(number))
    return str(DeltaStr((number)))
