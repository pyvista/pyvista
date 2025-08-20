"""PyVista plugin for static type-checking with mypy."""

from __future__ import annotations

__all__: list[str] = ['promote_type']

import importlib.util
from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from typing import Any

    from mypy.plugin import ClassDefContext
    from mypy.types import Instance
    from typing_extensions import Self


T = TypeVar('T', bound=type)


def promote_type(*types: type[Any]) -> Callable[[T], T]:  # noqa: ARG001
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
    Callable
        Decorated class.

    """
    return lambda obj: obj


if importlib.util.find_spec('mypy'):  # pragma: no cover
    from mypy.nodes import CallExpr
    from mypy.nodes import NameExpr
    from mypy.nodes import RefExpr
    from mypy.plugin import Plugin

    def _promote_type_callback(ctx: ClassDefContext) -> None:
        """Apply the `promote_type` decorator.

        The decorated class is captured and promoted to the type(s) provided
        by the decorator's argument(s).
        """
        for decorator in ctx.cls.decorators:
            if isinstance(decorator, CallExpr):
                callee = decorator.callee
                if isinstance(callee, NameExpr):
                    name = callee.name
                    if name == promote_type.__name__:
                        decorated_type: Instance = ctx.api.named_type(ctx.cls.fullname)
                        args = decorator.args
                        for arg in args:
                            if isinstance(arg, RefExpr):
                                named_type: Instance = ctx.api.named_type(arg.fullname)
                                decorated_type.type._promote.append(named_type)

    class _PyVistaPlugin(Plugin):
        """Mypy plugin to enable static type promotions."""

        def get_class_decorator_hook(
            self: Self, fullname: str
        ) -> Callable[[ClassDefContext], None] | None:
            def _get_type_fullname(typ: Any) -> str:
                return f'{typ.__module__}.{typ.__qualname__}'

            if fullname == _get_type_fullname(promote_type):
                return _promote_type_callback
            return None

    def plugin(version: str) -> type[_PyVistaPlugin]:  # numpydoc ignore: RT01  # noqa: ARG001
        """Entry-point for mypy."""
        return _PyVistaPlugin
