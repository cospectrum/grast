import grast.real as re
import grast.delta as de

from .real_str import real_str
from .delta_str import delta_str


def to_str(number: re.Expression | de.Delta) -> str:
    if isinstance(number, re.Expression):
        return real_str(number, False)
    return delta_str(number, False)
