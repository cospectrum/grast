from grast.delta import Delta
from grast.real import Real

from .real_str import real_str
from .delta_str import delta_str


def to_str(number: Real | Delta) -> str:
    if isinstance(number, Real):
        return real_str(number, False)
    return delta_str(number, False)
