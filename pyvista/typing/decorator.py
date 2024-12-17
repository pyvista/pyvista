"""Decorator used by the mypy plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable


def promote_type(*types: type) -> Callable[[Any], Any]:
    """Duck-type type-promotion decorator used by the mypy plugin.

    Apply this decorator to a class to promote its type statically.
    This tells `mypy` to treat the decorated class as though it's
    equivalent to another class.

    .. note::
        This decorator does nothing at runtime and merely passes the object through.

    Parameters
    ----------
    types : type
        Type(s) to promote the class to. The types are only used statically by mypy.

    Returns
    -------
    type
        Decorated class.

    """
    return lambda obj: obj
