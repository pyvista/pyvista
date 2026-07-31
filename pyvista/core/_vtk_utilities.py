"""Utilities related to VTK."""

from __future__ import annotations

from functools import cache
import sys
from typing import Literal
from typing import NamedTuple
from typing import cast

import pyvista as pv
from pyvista import _vtk
from pyvista._warn_external import warn_external
from pyvista.core.config import global_config
from pyvista.core.errors import ObsoleteVTKVersionWarning
from pyvista.core.errors import VTKVersionError


class VersionInfo(NamedTuple):
    """Version information as a named tuple."""

    major: int
    minor: int
    micro: int

    def __str__(self):
        return str((self.major, self.minor, self.micro))

    @staticmethod
    def _format(version: tuple[int, int, int]):
        return '.'.join(map(str, version))


def _get_vtk_version():
    """Return the vtk version as a namedtuple.

    Returns
    -------
    VersionInfo
        Version information as a named tuple.

    """
    try:
        ver = _vtk.vtkVersion()
        major = ver.GetVTKMajorVersion()
        minor = ver.GetVTKMinorVersion()
        micro = ver.GetVTKBuildVersion()
    except AttributeError:  # pragma: no cover
        msg = (
            f'Unable to detect VTK version. '
            f'Defaulting to {VersionInfo._format(_MIN_SUPPORTED_VTK_VERSION)}'
        )
        warn_external(msg)
        major, minor, micro = _MIN_SUPPORTED_VTK_VERSION
    return VersionInfo(major, minor, micro)


class VTKVersionInfo(VersionInfo):
    def _check_min_supported(self, other: tuple[int, int, int]) -> None:
        if isinstance(other, tuple) and other < _MIN_SUPPORTED_VTK_VERSION:  # type: ignore[redundant-expr]
            msg = (
                f'Comparing against unsupported VTK version {VersionInfo._format(other):}. '
                f'Minimum supported is {VersionInfo._format(_MIN_SUPPORTED_VTK_VERSION):}.'
            )
            warn_external(msg, ObsoleteVTKVersionWarning)

    def __lt__(self, other):
        self._check_min_supported(other)
        return super().__lt__(other)

    def __le__(self, other):
        self._check_min_supported(other)
        return super().__le__(other)

    def __gt__(self, other):
        self._check_min_supported(other)
        return super().__gt__(other)

    def __ge__(self, other):
        self._check_min_supported(other)
        return super().__ge__(other)


vtk_version_info = VTKVersionInfo(*_get_vtk_version())
_MIN_SUPPORTED_VTK_VERSION = (9, 3, 1)

_VERSION_LENGTH = 3


def _pad_version(version: tuple[int, ...] | None) -> tuple[int, int, int] | None:
    """Pad a version tuple with zeros so that it has major, minor, and micro values."""
    if version is None:
        return None

    if not all(isinstance(item, int) for item in version):
        msg = f'Version must be a tuple of integers, got {version!r}.'
        raise TypeError(msg)

    if (length := len(version)) > _VERSION_LENGTH:
        msg = f'Version tuple incorrect length (needs <= {_VERSION_LENGTH}), got {version!r}.'
        raise ValueError(msg)

    return (*version, *(0,) * (_VERSION_LENGTH - length))  # type: ignore[return-value]


def _parse_vtk_version_constraint(
    versions: tuple[int | tuple[int, ...], ...],
    at_least: tuple[int, ...] | None,
    less_than: tuple[int, ...] | None,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """Normalize a version constraint as a pair of minimum and maximum versions."""
    if len(versions) > 0 and at_least is not None:
        msg = 'Cannot specify both positional versions and the `at_least` keyword argument.'
        raise ValueError(msg)

    minimum_: tuple[int, ...] | None
    if len(versions) > 0:
        # Positional versions are either a single version tuple or variadic integers
        first = versions[0]
        if len(versions) == 1 and isinstance(first, tuple):
            minimum_ = first
        else:
            minimum_ = cast('tuple[int, ...]', versions)
    else:
        minimum_ = at_least

        if minimum_ is None and less_than is None:
            msg = 'Need to specify either `at_least` or `less_than`.'
            raise ValueError(msg)

    minimum = _pad_version(minimum_)
    maximum = _pad_version(less_than)

    if minimum is not None and maximum is not None and minimum > maximum:
        msg = (
            f'Cannot specify a minimum version greater than the maximum one, got '
            f'at_least={minimum} and less_than={maximum}.'
        )
        raise ValueError(msg)

    return minimum, maximum


def _warn_if_obsolete_constraint(
    minimum: tuple[int, int, int] | None, maximum: tuple[int, int, int] | None
) -> None:
    """Warn if a version constraint is always or never satisfied by supported VTK versions."""
    min_supported = pv._MIN_SUPPORTED_VTK_VERSION
    for keyword, version in (('at_least', minimum), ('less_than', maximum)):
        if version is not None and version <= min_supported:
            msg = (
                f'The VTK version constraint `{keyword}={version}` is obsolete and can be '
                f'removed. The minimum supported VTK version is '
                f'{VersionInfo._format(min_supported)}.'
            )
            warn_external(msg, ObsoleteVTKVersionWarning)


def _default_reason(
    minimum: tuple[int, int, int] | None,
    maximum: tuple[int, int, int] | None,
    current: tuple[int, ...],
) -> str:
    """Generate a message describing an unsatisfied version constraint."""
    if maximum is None:
        requirement = f'VTK version {VersionInfo._format(minimum)} or greater'  # type: ignore[arg-type]
    elif minimum is None:
        requirement = f'a VTK version less than {VersionInfo._format(maximum)}'
    else:
        requirement = (
            f'a VTK version of at least {VersionInfo._format(minimum)} and less than '
            f'{VersionInfo._format(maximum)}'
        )
    return (
        f'This feature requires {requirement}. '
        f'The installed version is {VersionInfo._format(current)}.'  # type: ignore[arg-type]
    )


def require_vtk_version(
    *versions: int | tuple[int, ...],
    at_least: tuple[int, ...] | None = None,
    less_than: tuple[int, ...] | None = None,
    reason: str | None = None,
) -> None:
    """Raise an error if the installed VTK version does not satisfy a constraint.

    Use this function to guard code which requires a specific range of VTK versions,
    e.g. a keyword argument which is only supported by newer versions of VTK. To guard
    an entire function or method, call it as the first statement of that callable.

    The minimum version may be specified positionally, either as separate integers or
    as a single tuple, or with the ``at_least`` keyword. All three forms below are
    equivalent:

    - ``require_vtk_version(9, 6)``
    - ``require_vtk_version((9, 6))``
    - ``require_vtk_version(at_least=(9, 6))``

    Versions are padded with zeros, e.g. ``(9, 6)`` is interpreted as ``(9, 6, 0)``.
    The minimum is inclusive and the maximum is exclusive.

    .. versionadded:: 0.49

    Parameters
    ----------
    *versions : int | tuple[int, ...]
        Minimum (inclusive) VTK version required, specified positionally. May not be
        used together with ``at_least``.

    at_least : tuple[int, ...], optional
        Minimum (inclusive) VTK version required.

    less_than : tuple[int, ...], optional
        Maximum (exclusive) VTK version required.

    reason : str, optional
        Message of the raised error. If unspecified, a default message describing the
        required and installed versions is used.

    Raises
    ------
    pyvista.core.errors.VTKVersionError
        If the installed VTK version does not satisfy the constraint.

    Warns
    -----
    pyvista.core.errors.ObsoleteVTKVersionWarning
        If the constraint is at or below the minimum VTK version supported by PyVista,
        since such a constraint is always (or never) satisfied and can be removed.

    See Also
    --------
    pyvista.core.errors.VTKVersionError
        Error raised when a version constraint is not satisfied.

    Examples
    --------
    Guard a keyword argument which requires a newer version of VTK.

    >>> import pyvista as pv
    >>> def scale_mesh(mesh, factor, fast_mode=False):
    ...     if fast_mode:
    ...         pv.require_vtk_version(
    ...             99, 0, reason='`fast_mode` requires VTK 99.0.'
    ...         )
    ...     return mesh.scale(factor)

    The guard is only triggered when the keyword is used.

    >>> mesh = pv.Sphere()
    >>> scaled = scale_mesh(mesh, 2)

    >>> try:
    ...     scaled = scale_mesh(mesh, 2, fast_mode=True)
    ... except pv.VTKVersionError as error:
    ...     print(error)
    `fast_mode` requires VTK 99.0.

    Omit ``reason`` to use a default message instead.

    >>> try:
    ...     pv.require_vtk_version(99, 0)
    ... except pv.VTKVersionError as error:
    ...     print(error)
    This feature requires VTK version 99.0.0 or greater. The installed version is ...

    A maximum version may be required instead of, or in addition to, a minimum.

    >>> try:
    ...     pv.require_vtk_version(at_least=(99, 0), less_than=(99, 1))
    ... except pv.VTKVersionError as error:
    ...     print(error)
    This feature requires a VTK version of at least 99.0.0 and less than 99.1.0. ...

    """
    minimum, maximum = _parse_vtk_version_constraint(versions, at_least, less_than)
    _warn_if_obsolete_constraint(minimum, maximum)

    # Compare plain tuples since comparing `vtk_version_info` against an obsolete
    # version emits its own warning, which is already handled above
    current = tuple(pv.vtk_version_info)

    if (minimum is not None and current < minimum) or (maximum is not None and current >= maximum):
        raise VTKVersionError(
            reason if reason is not None else _default_reason(minimum, maximum, current)
        )


class vtkPyVistaOverride:  # noqa: N801
    """Base class to automatically override VTK classes with PyVista classes."""

    def __init_subclass__(cls, **kwargs):
        if vtk_version_info >= (9, 4):
            # Check for VTK base classes and call the override method
            for base in cls.__bases__:
                if (
                    hasattr(base, '__module__')
                    and base.__module__.startswith('vtkmodules.')
                    and hasattr(base, 'override')
                ):
                    # For now, just remove any overrides for these classes
                    # There are clear issues with the current implementation
                    # of overriding these classes upstream and until they are
                    # resolved, we will entirely remove the overrides.
                    # See https://gitlab.kitware.com/vtk/vtk/-/merge_requests/11698
                    # See https://gitlab.kitware.com/vtk/vtk/-/issues/19550#note_1598883
                    base.override(None)
                    break

        return cls


_VTK_SNAKE_CASE_STATE: Literal['allow', 'warning', 'error'] = 'error'


class DisableVtkSnakeCase:
    """Base class to raise error if using VTK's `snake_case` API."""

    @staticmethod
    def check_attribute(target, attr):
        # Skip check and exit early if possible
        if (
            _VTK_SNAKE_CASE_STATE == 'allow'
            or not attr
            or not attr[0].islower()
            or attr in ('__class__', '__init__')
            or vtk_version_info < (9, 4)
        ):
            return

        # Check if we have a vtk-defined attribute using cached lookup
        cls = target if isinstance(target, type) else target.__class__
        if not _is_vtk_attribute_cached(cls, attr):
            return

        # We have a VTK attribute, so raise or warn
        if sys.meta_path is not None:  # Avoid dynamic imports when Python is shutting down
            msg = f'The attribute {attr!r} is defined by VTK and is not part of the PyVista API'
            if _VTK_SNAKE_CASE_STATE == 'error':
                from pyvista import PyVistaAttributeError  # noqa: PLC0415

                raise PyVistaAttributeError(msg)
            else:
                warn_external(msg, RuntimeWarning)
        return

    def __getattribute__(self, item):
        DisableVtkSnakeCase.check_attribute(self, item)
        return object.__getattribute__(self, item)

    def __dir__(self) -> list[str]:
        """Return a filtered attribute listing for :func:`dir` and tab-completion.

        VTK-inherited names are hidden by default so PyVista objects present
        a curated public surface in data-science IDEs (Positron Variables
        pane, VS Code Jupyter extension, ...) and in IPython / Jupyter
        tab-completion. VTK methods remain fully callable; only their
        enumeration is suppressed.

        - CamelCase VTK attributes (``GetNumberOfPoints``, ``DeepCopy``, ...) are
          hidden unless :attr:`pyvista.global_config.show_vtk_api` is
          ``True``.
        - snake_case VTK aliases (``number_of_points``, ``deep_copy``, ...) are
          hidden unless VTK snake_case is allowed via
          :func:`pyvista.vtk_snake_case`, since they would otherwise raise
          ``PyVistaAttributeError`` on access.
        """
        cls: type = type(self)
        listing = super().__dir__()
        show_camel = global_config.show_vtk_api
        show_snake = _VTK_SNAKE_CASE_STATE == 'allow'
        if show_camel and show_snake:
            return sorted(listing)

        def keep(attr: str) -> bool:
            if not _is_vtk_attribute_cached(cls, attr):
                return True
            if attr and attr[0].islower():
                return show_snake
            return show_camel

        return sorted(attr for attr in listing if keep(attr))


def is_vtk_attribute(obj: object, attr: str):  # numpydoc ignore=RT01
    """Return True if the attribute is defined by a vtk class.

    Parameters
    ----------
    obj : object
        Class or instance to check.

    attr : str
        Name of the attribute to check.

    """

    def _find_defining_class(cls, attr):
        """Find the class that defines a given attribute."""
        for base in cls.__mro__:
            if attr in base.__dict__:
                return base
        return None

    cls = _find_defining_class(obj if isinstance(obj, type) else obj.__class__, attr)
    return cls is not None and cls.__module__.startswith('vtkmodules')


# Wrap the check in an LRU cache
@cache
def _is_vtk_attribute_cached(target_type, attr):
    return is_vtk_attribute(target_type, attr)


class VTKObjectWrapperCheckSnakeCase(_vtk.VTKObjectWrapper):
    """Superclass for classes that wrap VTK objects with Python objects.

    This class overrides __getattr__ to disable the VTK snake case API.
    """

    def __getattr__(self, name: str):
        """Forward unknown attribute requests to VTKArray's __getattr__."""
        if self.VTKObject is not None:
            # Check if forwarding snake_case attributes
            DisableVtkSnakeCase.check_attribute(self.VTKObject, name)
            return getattr(self.VTKObject, name)
        raise AttributeError
