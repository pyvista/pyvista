"""Miscellaneous core utilities."""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Sequence
import enum
from functools import cache
import importlib
import sys
import threading
import traceback
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    from typing import Any

    from pyvista._typing_core import ArrayLike
    from pyvista._typing_core import NumpyArray
    from pyvista._typing_core import VectorLike

    _T = TypeVar('_T')

T = TypeVar('T', bound='AnnotatedIntEnum')


def assert_empty_kwargs(**kwargs) -> bool:
    """Assert that all keyword arguments have been used (internal helper).

    If any keyword arguments are passed, a ``TypeError`` is raised.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments passed to the function.

    Returns
    -------
    bool
        ``True`` when successful.

    Raises
    ------
    TypeError
        If any keyword arguments are passed, a ``TypeError`` is raised.

    """
    n = len(kwargs)
    if n == 0:
        return True
    caller = sys._getframe(1).f_code.co_name
    keys = list(kwargs.keys())
    bad_arguments = ', '.join([f'"{key}"' for key in keys])
    grammar = 'is an invalid keyword argument' if n == 1 else 'are invalid keyword arguments'
    message = f'{bad_arguments} {grammar} for `{caller}`'
    raise TypeError(message)


def check_valid_vector(point: VectorLike[float], name: str = '') -> None:
    """Check if a vector contains three components.

    Parameters
    ----------
    point : VectorLike[float]
        Input vector to check. Must be an iterable with exactly three components.
    name : str, optional
        Name to use in the error messages. If not provided, "Vector" will be used.

    Raises
    ------
    TypeError
        If the input is not an iterable.
    ValueError
        If the input does not have exactly three components.

    """
    if not isinstance(point, (Sequence, np.ndarray)):
        msg = f'{name} must be a length three iterable of floats.'
        raise TypeError(msg)
    if len(point) != 3:
        if name == '':
            name = 'Vector'
        msg = f'{name} must be a length three iterable of floats.'
        raise ValueError(msg)


def abstract_class(cls_):  # noqa: ANN001, ANN201 # numpydoc ignore=RT01
    """Decorate a class, overriding __new__.

    Preventing a class from being instantiated similar to abc.ABCMeta
    but does not require an abstract method.

    Parameters
    ----------
    cls_ : type
        The class to be decorated as abstract.

    """

    def __new__(cls, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001, N807
        if cls is cls_:
            msg = f'{cls.__name__} is an abstract class and may not be instantiated.'
            raise TypeError(msg)
        return super(cls_, cls).__new__(cls)

    cls_.__new__ = __new__
    return cls_


class AnnotatedIntEnum(int, enum.Enum):
    """Annotated enum type."""

    annotation: str

    def __new__(cls, value: int, annotation: str) -> Self:
        """Initialize."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.annotation = annotation
        return obj

    @classmethod
    def from_str(cls, input_str: str) -> Self:
        """Create an enum member from a string.

        Parameters
        ----------
        input_str : str
            The string representation of the annotation for the enum member.

        Returns
        -------
        AnnotatedIntEnum
            The enum member with the specified annotation.

        Raises
        ------
        ValueError
            If there is no enum member with the specified annotation.

        """
        for value in cls:
            if value.annotation.lower() == input_str.lower():
                return value
        msg = f'{cls.__name__} has no value matching {input_str}'
        raise ValueError(msg)

    @classmethod
    def from_any(cls, value: AnnotatedIntEnum | int | str) -> Self:
        """Create an enum member from a string, int, etc.

        Parameters
        ----------
        value : int | str | AnnotatedIntEnum
            The value used to determine the corresponding enum member.

        Returns
        -------
        AnnotatedIntEnum
            The enum member matching the specified value.

        Raises
        ------
        ValueError
            If there is no enum member matching the specified value.

        """
        if isinstance(value, cls):
            return value
        elif isinstance(value, int):
            return cls(value)  # type: ignore[call-arg]
        elif isinstance(value, str):
            return cls.from_str(value)
        else:
            msg = f'Invalid type {type(value)} for class {cls.__name__}.'  # type: ignore[unreachable]
            raise TypeError(msg)


@cache
def has_module(module_name: str) -> bool:
    """Return if a module can be imported.

    Parameters
    ----------
    module_name : str
        Name of the module to check.

    Returns
    -------
    bool
        ``True`` if the module can be imported, otherwise ``False``.

    """
    module_spec = importlib.util.find_spec(module_name)
    return module_spec is not None


def try_callback(func, *args) -> None:  # noqa: ANN001
    """Wrap a given callback in a try statement.

    Parameters
    ----------
    func : callable
        Callable object.

    *args
        Any arguments.

    """
    try:
        func(*args)
    except Exception:  # noqa: BLE001  # pragma: no cover
        etype, exc, tb = sys.exc_info()
        stack = traceback.extract_tb(tb)[1:]
        formatted_exception = 'Encountered issue in callback (most recent call last):\n' + ''.join(
            traceback.format_list(stack) + traceback.format_exception_only(etype, exc),
        ).rstrip('\n')
        warnings.warn(formatted_exception)


def threaded(fn):  # noqa: ANN001, ANN201
    """Call a function using a thread.

    Parameters
    ----------
    fn : callable
        Callable object.

    Returns
    -------
    function
        Wrapped function.

    """

    def wrapper(*args, **kwargs):  # noqa: ANN202
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class conditional_decorator:  # noqa: N801
    """Conditional decorator for methods.

    Parameters
    ----------
    dec : callable
        The decorator to be applied conditionally.
    condition : bool
        Condition to match. If ``True``, the decorator is applied. If
        ``False``, the function is returned unchanged.

    """

    def __init__(self, dec, condition) -> None:  # noqa: ANN001
        """Initialize."""
        self.decorator = dec
        self.condition = condition

    def __call__(self, func):  # noqa: ANN001, ANN204
        """Call the decorated function if condition is matched."""
        if not self.condition:
            # Return the function unchanged, not decorated.
            return func
        return self.decorator(func)


def _check_range(value: float, rng: Sequence[float], parm_name: str) -> None:
    """Check if a parameter is within a range."""
    if value < rng[0] or value > rng[1]:
        msg = (
            f'The value {float(value)} for `{parm_name}` is outside the '
            f'acceptable range {tuple(rng)}.'
        )
        raise ValueError(msg)


class _AutoFreezeMeta(type):
    """Metaclass to automatically freeze a class when called."""

    def __call__(cls: type[_T], *args, **kwargs) -> _T:
        obj = super().__call__(*args, **kwargs)  # type: ignore[misc]
        obj._no_new_attributes(cls)
        return obj


class _AutoFreezeABCMeta(_AutoFreezeMeta, ABCMeta):
    """Metaclass to combine automatic attribute freezing with ABC support."""


class _NoNewAttrMixin(metaclass=_AutoFreezeABCMeta):
    """Mixin to prevent adding new attributes.

    This class is mainly used to prevent users from setting the wrong attributes on an
    object. It freezes the attributes when called and prevents setting new ones via
    "normal" methods like ``obj.foo = 42``.
    """

    def _no_new_attributes(self, this_class: type) -> None:
        """Prevent setting additional attributes."""
        object.__setattr__(self, '__frozen', True)
        object.__setattr__(self, '__frozen_by_class', this_class)

    def __setattr__(self, key: str, value: Any) -> None:
        """Prevent adding new attributes to classes using "normal" methods."""
        if not key.startswith('_'):
            # Check if this class froze itself. Any frozen state already set by parent classes,
            # e.g. by calling super().__init__(), will be ignored. This allows subclasses to set
            # attributes during init without being affect by a parent class init.
            frozen = self.__dict__.get('__frozen', False)
            frozen_by = self.__dict__.get('__frozen_by_class', None)
            if (
                frozen
                and frozen_by is type(self)
                and not (key in type(self).__dict__ or hasattr(self, key))
            ):
                from pyvista import PyVistaAttributeError  # noqa: PLC0415

                msg = (
                    f'Attribute {key!r} does not exist and cannot be added to class '
                    f'{self.__class__.__name__!r}\nUse `pv.set_new_attribute` to set new '
                    f'attributes or consider setting a private variable (with `_` prefix) instead.'
                )
                raise PyVistaAttributeError(msg)
        object.__setattr__(self, key, value)


def set_new_attribute(obj: object, name: str, value: Any) -> None:
    """Set a new attribute for this object.

    Python allows arbitrarily setting new attributes on objects at any time,
    but PyVista's classes do not allow this. If an attribute is not part of
    PyVista's API, an ``AttributeError`` is normally raised when attempting
    to set it.

    Use :func:`set_new_attribute` to override this and set a new attribute anyway.

    Examples
    --------
    Set a new custom attribute on a mesh.

    >>> import pyvista as pv
    >>> mesh = pv.PolyData()
    >>> pv.set_new_attribute(mesh, 'foo', 42)
    >>> mesh.foo
    42

    .. versionadded:: 0.46

    """
    if hasattr(obj, name):
        from pyvista import PyVistaAttributeError  # noqa: PLC0415

        msg = (
            f'Attribute {name!r} already exists. '
            '`set_new_attribute` can only be used for setting NEW attributes.'
        )
        raise PyVistaAttributeError(msg)
    object.__setattr__(obj, name, value)


def _reciprocal(
    x: ArrayLike[float], tol: float = 1e-8, value_if_division_by_zero: float = 0.0
) -> NumpyArray[float]:
    """Compute the element-wise reciprocal and avoid division by zero.

    The reciprocal of elements with an absolute value less than a
    specified tolerance has the value specified by ``default_if_div_by_zero``.

    Parameters
    ----------
    x : array_like
        Input array.
    tol : float
        Tolerance value. Values smaller than ``tol`` have a reciprocal of zero.
    value_if_division_by_zero : float
        Default value given to values less than ``tol``, i.e. the value given if division
        by zero is detected.

    Returns
    -------
    numpy.ndarray
        Element-wise reciprocal of the input.

    """
    x = np.array(x)
    x = x if np.issubdtype(x.dtype, np.floating) else x.astype(float)
    zero = np.abs(x) < tol
    x[~zero] = np.reciprocal(x[~zero])
    x[zero] = value_if_division_by_zero
    return x


class _classproperty(property):  # noqa: N801
    """Read-only class property decorator.

    Use this decaorator as an alternative to chaining `@classmethod`
    and `@property` which is deprecated.

    See:
    - https://docs.python.org/library/functions.html#classmethod
    - https://stackoverflow.com/a/13624858

    Examples
    --------
    >>> from pyvista.core.utilities.misc import _classproperty
    >>> class Foo:
    ...     @_classproperty
    ...     def bar(cls): ...

    """

    def __get__(self: property, owner_self: Any, owner_cls: type | None = None) -> Any:
        return self.fget(owner_cls)  # type: ignore[misc]


class _NameMixin:
    """Add a 'name' property to a class.

    .. versionadded:: 0.45

    """

    @property
    def name(self) -> str:  # numpydoc ignore=RT01
        """Get or set the unique name identifier used by PyVista."""
        if not hasattr(self, '_name') or self._name is None:
            address = (
                self.GetAddressAsString('')
                if hasattr(self, 'GetAddressAsString')
                else hex(id(self))
            )
            return f'{type(self).__name__}({address})'
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not value:
            msg = 'Name must be truthy.'
            raise ValueError(msg)
        object.__setattr__(self, '_name', str(value))


class _BoundsSizeMixin:
    @property
    def bounds_size(self) -> tuple[float, float, float]:
        """Return the size of each axis of the object's bounding box.

        .. versionadded:: 0.46

        Returns
        -------
        tuple[float, float, float]
            Size of each x-y-z axis.

        Examples
        --------
        Get the size of a cube. The cube has edge lengths af ``(1.0, 1.0, 1.0)``
        by default.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.bounds_size
        (1.0, 1.0, 1.0)

        """
        bounds = self.bounds  # type: ignore[attr-defined]
        return (
            bounds.x_max - bounds.x_min,
            bounds.y_max - bounds.y_min,
            bounds.z_max - bounds.z_min,
        )
