"""Mypy plugin to enable generic use of builtin types with numpy's NDArray.

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
from time import time
from typing import Any, Callable, Dict, Final, Optional, Tuple, Type

from mypy.build import PRI_MED
from mypy.nodes import MypyFile
from mypy.plugin import ClassDefContext, Plugin, ReportConfigContext
from mypy.types import Instance
import numpy as np

__all__: list[str] = []


def _get_type_fullname(typ: Any) -> str:
    return f"{typ.__module__}.{typ.__qualname__}"


NUMPY_NUMBER_TYPE_FULLNAME: Final = _get_type_fullname(np.number)
NUMPY_BOOL_TYPE_FULLNAME: Final = _get_type_fullname(np.bool_)
NUMPY_INTEGER_TYPE_FULLNAME: Final = _get_type_fullname(np.integer)
NUMPY_FLOATING_TYPE_FULLNAME: Final = _get_type_fullname(np.floating)
NUMPY_DTYPE_TYPE_FULLNAME: Final = _get_type_fullname(np.dtype)


def _promote_int(ctx: ClassDefContext) -> None:
    """Enable generic use of numpy.dtype[int]."""
    assert ctx.cls.fullname == NUMPY_INTEGER_TYPE_FULLNAME

    # Promote `numpy.integer` as a subtype of `int`
    int_inst: Instance = ctx.api.named_type('builtins.int')
    ctx.cls.info._promote.append(int_inst)

    # Promote `int` as a subtype of `numpy.number`
    number_inst: Instance = ctx.api.named_type(NUMPY_NUMBER_TYPE_FULLNAME)
    int_inst.type._promote.append(number_inst)


def _promote_float(ctx: ClassDefContext) -> None:
    """Enable generic use of numpy.dtype[float]."""
    assert ctx.cls.fullname == NUMPY_FLOATING_TYPE_FULLNAME

    # Promote `numpy.floating` as a subtype of `float`
    float_inst: Instance = ctx.api.named_type('builtins.float')
    ctx.cls.info._promote.append(float_inst)

    # Promote `float` as a subtype of `numpy.number`
    number_inst: Instance = ctx.api.named_type(NUMPY_NUMBER_TYPE_FULLNAME)
    float_inst.type._promote.append(number_inst)


def _add_dependency(module: str) -> Tuple[int, str, int]:
    """Return a (priority, module name, line number) tuple.

    The line number can be -1 when there is not a known real line number.

    Priorities are defined in mypy.build. 10 (PRI_MED) is a good choice
    for priority.
    """
    priority = PRI_MED
    line_number = -1
    return priority, module, line_number


class _PyvistaPlugin(Plugin):
    """Mypy plugin to enable generic use of builtin types with numpy's NDArray."""

    def get_customize_class_mro_hook(
        self, fullname: str
    ) -> Optional[Callable[[ClassDefContext], None]]:
        """Customize MRO for given classes.

        The plugin can modify the class MRO (or other properties) _in place_.
        This method is called with the class full name before its body is
        semantically analyzed.
        """
        # Customize numpy data type definitions
        if fullname == NUMPY_FLOATING_TYPE_FULLNAME:
            return _promote_float
        elif fullname == NUMPY_INTEGER_TYPE_FULLNAME:
            return _promote_int
        return None

    def get_additional_deps(self, file: MypyFile) -> list[tuple[int, str, int]]:
        """Customize dependencies for a module.

        This hook allows adding in new dependencies for a module. It
        is called after parsing a file but before analysis. This can
        be useful if a library has dependencies that are dynamic based
        on configuration information, for example.
        """
        # Add numpy numeric types for import so that their class definitions
        # can be customized (e.g. add type promotions).
        return [
            _add_dependency(NUMPY_FLOATING_TYPE_FULLNAME),
            _add_dependency(NUMPY_INTEGER_TYPE_FULLNAME),
            _add_dependency(NUMPY_NUMBER_TYPE_FULLNAME),
        ]

    def report_config_data(self, ctx: ReportConfigContext) -> Optional[Dict[str, Any]]:
        """Get representation of configuration data for a module.

        The data must be encodable as JSON and will be stored in the
        cache metadata for the module. A mismatch between the cached
        values and the returned will result in that module's cache
        being invalidated and the module being rechecked.

        This can be called twice for each module, once after loading
        the cache to check if it is valid and once while writing new
        cache information.

        If is_check in the context is true, then the return of this
        call will be checked against the cached version. Otherwise the
        call is being made to determine what to put in the cache. This
        can be used to allow consulting extra cache files in certain
        complex situations.

        This can be used to incorporate external configuration information
        that might require changes to typechecking.
        """
        # Always invalidate cached dtype configurations so that it is always
        # re-checked. This ensures type promotions of 'float' or 'int' as a
        # subtype of 'numpy.number' are reflected in the config and prevents
        # the type-var error: "dtype" must be a subtype of "generic" from occurring
        if NUMPY_DTYPE_TYPE_FULLNAME in ctx.id:
            # Return dict with a unique value to invalidate mypy cache.
            return {'': time()}
        else:
            return None


def plugin(version: str) -> Type[_PyvistaPlugin]:  # numpydoc ignore=PR01,RT01
    """Entry-point for mypy."""
    return _PyvistaPlugin
