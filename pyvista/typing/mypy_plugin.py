"""PyVista plugin for static type-checking with mypy."""

from __future__ import annotations

__all__: list[str] = ['promote_type']

import importlib.util
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Final

    from mypy.plugin import ClassDefContext
    from mypy.types import Instance
    from typing_extensions import Self


NUMPY_SIGNED_INTEGER_TYPE_FULLNAME: Final = 'numpy.signedinteger'
NUMPY_FLOATING_TYPE_FULLNAME: Final = 'numpy.floating'
NUMPY_BOOL_TYPE_FULLNAME: Final = 'numpy.bool'


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

    def _promote_bool_callback(ctx: ClassDefContext) -> None:
        """Add two-way type promotion between `bool` and `numpy.bool_`.

        This promotion allows for use of NumPy typing annotations with `bool`,
        e.g. npt.NDArray[bool].

        See mypy.semanal_classprop.add_type_promotion for a similar promotion
        between `int` and `i64` types.
        """
        assert ctx.cls.fullname == NUMPY_BOOL_TYPE_FULLNAME
        numpy_bool: Instance = ctx.api.named_type(NUMPY_BOOL_TYPE_FULLNAME)
        builtin_bool: Instance = ctx.api.named_type('builtins.bool')

        builtin_bool.type._promote.append(numpy_bool)
        numpy_bool.type.alt_promote = builtin_bool

    def _promote_int_callback(ctx: ClassDefContext) -> None:
        """Add two-way type promotion between `int` and `numpy.signedinteger`.

        This promotion allows for use of NumPy typing annotations with `int`,
        e.g. npt.NDArray[int].

        See mypy.semanal_classprop.add_type_promotion for a similar promotion
        between `int` and `i64` types.
        """
        assert ctx.cls.fullname == NUMPY_SIGNED_INTEGER_TYPE_FULLNAME
        numpy_signed_integer: Instance = ctx.api.named_type(NUMPY_SIGNED_INTEGER_TYPE_FULLNAME)
        builtin_int: Instance = ctx.api.named_type('builtins.int')

        builtin_int.type._promote.append(numpy_signed_integer)
        numpy_signed_integer.type.alt_promote = builtin_int

    def _promote_float_callback(ctx: ClassDefContext) -> None:
        """Add two-way type promotion between `float` and `numpy.floating`.

        This promotion allows for use of NumPy typing annotations with `float`,
        e.g. npt.NDArray[float].

        See mypy.semanal_classprop.add_type_promotion for a similar promotion
        between `int` and `i64` types.
        """
        assert ctx.cls.fullname == NUMPY_FLOATING_TYPE_FULLNAME
        numpy_floating: Instance = ctx.api.named_type(NUMPY_FLOATING_TYPE_FULLNAME)
        builtin_float: Instance = ctx.api.named_type('builtins.float')

        builtin_float.type._promote.append(numpy_floating)
        numpy_floating.type.alt_promote = builtin_float

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

        def get_customize_class_mro_hook(
            self, fullname: str
        ) -> Callable[[ClassDefContext], None] | None:
            """Customize class definitions before semantic analysis."""
            if fullname == NUMPY_FLOATING_TYPE_FULLNAME:
                return _promote_float_callback
            elif fullname == NUMPY_SIGNED_INTEGER_TYPE_FULLNAME:
                return _promote_int_callback
            elif fullname == NUMPY_BOOL_TYPE_FULLNAME:
                return _promote_bool_callback
            return None

    def plugin(version: str) -> type[_PyVistaPlugin]:  # numpydoc ignore: RT01
        """Entry-point for mypy."""
        return _PyVistaPlugin
