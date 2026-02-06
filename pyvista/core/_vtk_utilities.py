"""Utilities related to VTK."""

from __future__ import annotations

from functools import cache
import sys
from typing import Literal
from typing import NamedTuple

from pyvista._warn_external import warn_external
from pyvista.core import _vtk_core as _vtk


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
            from pyvista.core.errors import VTKVersionError  # noqa: PLC0415

            msg = (
                f'Comparing against unsupported VTK version {VersionInfo._format(other):}. '
                f'Minimum supported is {VersionInfo._format(_MIN_SUPPORTED_VTK_VERSION):}.'
            )
            raise VTKVersionError(msg)

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
_MIN_SUPPORTED_VTK_VERSION = (9, 2, 2)


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
