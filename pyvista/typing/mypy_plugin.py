"""PyVista plugin for static type-checking with mypy."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from mypy.nodes import CallExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.plugin import ClassDefContext
from mypy.plugin import Plugin

from .decorator import promote_type

if TYPE_CHECKING:
    from typing import Any

    from mypy.types import Instance
    from typing_extensions import Self


__all__: list[str] = []


def _promote_type_callback(ctx: ClassDefContext) -> None:
    """Apply the `promote_type` decorator.

    The decorated class is captured and promoted it to the type(s) provided
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


def plugin(version: str) -> type[_PyVistaPlugin]:  # numpydoc ignore: RT01
    """Entry-point for mypy."""
    return _PyVistaPlugin
