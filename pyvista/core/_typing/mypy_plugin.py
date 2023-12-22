"""Mypy plugin to enable type annotations of NumPy arrays with builtin types.

This plugin adds type promotions for `int` and `float` types so that
`NDArray[float]` and `NDArray[int]` can be used as a valid type annotations.


.. versionadded:: 0.44

Examples
--------
To enable the plugin, it must be added to the mypy configuration file along
with numpy's mypy plugin:

.. code-block::

    [mypy]
    plugins = [
        'numpy.typing.mypy_plugin',
        'pyvista.core._typing.mypy_plugin',
    ]

"""
from typing import TYPE_CHECKING, Any, Callable, Final, Optional, Type

import numpy as np

try:
    from mypy.plugin import ClassDefContext, Plugin
    from mypy.types import Instance

    MYPY_EXCEPTION: Optional[ModuleNotFoundError] = None
except ModuleNotFoundError as ex:
    MYPY_EXCEPTION = ex


__all__: list[str] = []


def _get_type_fullname(typ: Any) -> str:
    return f"{typ.__module__}.{typ.__qualname__}"


NUMPY_SIGNED_INTEGER_TYPE_FULLNAME: Final = _get_type_fullname(np.signedinteger)
NUMPY_FLOATING_TYPE_FULLNAME: Final = _get_type_fullname(np.floating)
# TODO: Add type promotion for `bool` and `numpy.bool_`(only if/when mypy complains?)
# NUMPY_BOOL_TYPE_FULLNAME: Final = _get_type_fullname(np.bool_)

if TYPE_CHECKING or MYPY_EXCEPTION is None:

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

    class _PyvistaPlugin(Plugin):
        """Mypy plugin to enable type annotations of NumPy arrays with builtin types."""

        def get_customize_class_mro_hook(
            self, fullname: str
        ) -> Optional[Callable[[ClassDefContext], None]]:
            """Customize class definitions before semantic analysis."""
            if fullname == NUMPY_FLOATING_TYPE_FULLNAME:
                return _promote_float_callback
            elif fullname == NUMPY_SIGNED_INTEGER_TYPE_FULLNAME:
                return _promote_int_callback
            return None

    def plugin(version: str) -> Type[_PyvistaPlugin]:  # numpydoc ignore=PR01,RT01
        """Entry-point for mypy."""
        return _PyvistaPlugin

else:

    def plugin(version: str) -> Any:  # numpydoc ignore=PR01,RT01
        """Entry-point for mypy."""
        raise MYPY_EXCEPTION
