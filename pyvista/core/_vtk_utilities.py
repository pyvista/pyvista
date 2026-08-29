"""Utilities related to VTK."""

from __future__ import annotations

import functools
import sys
from typing import Literal
from typing import NamedTuple

from pyvista import _vtk
from pyvista._warn_external import warn_external
from pyvista.core.config import global_config
from pyvista.core.errors import VTKVersionError

# A wrapped VTK class' ``__module__`` is rooted at the active backend; accept both.
_VTK_MODULE_PREFIXES = tuple({'vtkmodules.', f'{_vtk._VTK_ROOT}.'})


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
_MIN_SUPPORTED_VTK_VERSION = (9, 3, 1)


def vtk_backend() -> str:
    """Return the name of the VTK build PyVista is running against.

    ``'vtk'`` is stock VTK (the default); ``'cvista'`` is the community fork,
    selected by installing ``pyvista[cvista]`` or setting ``PYVISTA_VTK_BACKEND``
    before importing PyVista. Use it to branch on features a given build ships.
    The returned name round-trips through ``PYVISTA_VTK_BACKEND``.

    .. versionadded:: 0.49

    Returns
    -------
    str
        Name of the active backend: ``'vtk'`` for stock VTK, otherwise the
        backend's package name (for example, ``'cvista'``).

    Examples
    --------
    The value depends on which build is installed, so this example is not run.

    >>> import pyvista as pv
    >>> pv.vtk_backend()  # doctest: +SKIP
    'vtk'

    Raise a clear error for a build that cannot support a feature:

    >>> if pv.vtk_backend() != 'vtk':  # doctest: +SKIP
    ...     msg = (
    ...         f'This feature is not supported on the {pv.vtk_backend()} backend.'
    ...     )
    ...     raise RuntimeError(msg)

    """
    return 'vtk' if _vtk._VTK_ROOT == 'vtkmodules' else _vtk._VTK_ROOT


class vtkPyVistaOverride:  # noqa: N801
    """Base class to automatically override VTK classes with PyVista classes."""

    def __init_subclass__(cls, **kwargs):
        if vtk_version_info >= (9, 4):
            # Check for VTK base classes and call the override method
            for base in cls.__bases__:
                if (
                    hasattr(base, '__module__')
                    and base.__module__.startswith(_VTK_MODULE_PREFIXES)
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

# VTK only exposes the snake_case API from 9.4 on, so below that there is nothing
# to check for. `check_attribute` runs on every attribute access, so bind it once here.
_VTK_SNAKE_CASE_MIN_VERSION_MET = vtk_version_info >= (9, 4)

# VTK 9.6.2 adds fixed-size cell array storage (regular cell arrays store their
# connectivity without an explicit offsets array). Bind the capability check once
# here and reuse it everywhere instead of re-inlining the version comparison.
_SUPPORTS_FIXED_SIZE_STORAGE = vtk_version_info >= (9, 6, 2)

# Since VTK 9.6, `vtkCellArray.SetData` references the arrays handed to it (their
# reference count goes 1 -> 2). Before that it does not, so the caller must keep them
# alive itself -- see `CellArray._set_data`.
_SETDATA_TAKES_OWNERSHIP = vtk_version_info >= (9, 6)

# VTK 9.4 keeps the polyhedron faces and face locations in two cell arrays, reachable
# from `GetPolyhedronFaces` and `GetPolyhedronFaceLocations`. Before that a polyhedron
# is a single padded face stream with no offsets or connectivity of its own, so the
# properties built on those cell arrays have nothing to read.
_SUPPORTS_POLYHEDRON_FACE_CELL_ARRAYS = vtk_version_info >= (9, 4)


# Per-class attribute names verified safe to access in every ``vtk_snake_case`` state.
_SAFE_ATTRS_BY_CLASS: dict[type, set[str]] = {}


class DisableVtkSnakeCase:
    """Base class to raise error if using VTK's ``snake_case`` API."""

    @staticmethod
    def check_attribute(target, attr):
        """Raise or warn if ``attr`` is a VTK-defined ``snake_case`` name on ``target``."""
        safe_by_class = _SAFE_ATTRS_BY_CLASS
        if safe_by_class is None:  # pragma: no cover  # Python is shutting down
            return  # type: ignore[unreachable]
        # `target.__class__` and `isinstance` would recurse into `__getattribute__`
        cls: type = target if issubclass(type(target), type) else type(target)
        safe = safe_by_class.get(cls)
        if safe is None:
            safe = safe_by_class[cls] = set()
        elif attr in safe:
            return

        # Names that no state can flag: no snake_case API, or not a VTK lowercase name
        if (
            not _VTK_SNAKE_CASE_MIN_VERSION_MET
            or not attr
            or not attr[0].islower()
            or not _is_vtk_attribute_cached(cls, attr)
        ):
            safe.add(attr)
            return

        # We have a VTK attribute, so raise or warn
        if _VTK_SNAKE_CASE_STATE == 'allow':
            return
        if sys.meta_path is not None:  # Avoid dynamic imports when Python is shutting down
            msg = f'The attribute {attr!r} is defined by VTK and is not part of the PyVista API'
            if _VTK_SNAKE_CASE_STATE == 'error':
                from pyvista import PyVistaAttributeError  # noqa: PLC0415

                raise PyVistaAttributeError(msg)
            else:
                warn_external(msg, RuntimeWarning)
        return

    def __getattribute__(self, item):
        """Get an attribute after checking it is part of the PyVista API."""
        # Hot path: inline the cache lookup so verified-safe names skip the check call
        try:
            if item in _SAFE_ATTRS_BY_CLASS[type(self)]:
                return object.__getattribute__(self, item)
        except (KeyError, TypeError):  # unseen class, or None during Python shutdown
            pass
        DisableVtkSnakeCase.check_attribute(self, item)
        return object.__getattribute__(self, item)

    def __dir__(self) -> list[str]:
        """Return a filtered attribute listing for :func:`dir` and tab-completion.

        VTK-inherited names are hidden by default so PyVista objects present
        a curated public surface in data-science IDEs (Positron Variables
        pane, VS Code Jupyter extension, and so on) and in IPython / Jupyter
        tab-completion. VTK methods remain fully callable; only their
        enumeration is suppressed.

        - CamelCase VTK attributes (``GetNumberOfPoints``, ``DeepCopy``, and so on) are
          hidden unless :attr:`pyvista.global_config.show_vtk_api` is
          ``True``.
        - ``snake_case`` VTK aliases (``number_of_points``, ``deep_copy``, and so on) are
          hidden unless VTK ``snake_case`` is allowed via
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
    """Return True if the attribute is defined by a VTK class.

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
    return cls is not None and cls.__module__.startswith(_VTK_MODULE_PREFIXES)


# Wrap the check in an LRU cache
@functools.cache
def _is_vtk_attribute_cached(target_type, attr):
    return is_vtk_attribute(target_type, attr)


class VTKObjectWrapperCheckSnakeCase(_vtk.VTKObjectWrapper):
    """Superclass for classes that wrap VTK objects with Python objects.

    This class overrides ``__getattr__`` to disable the VTK snake case API.
    """

    def __getattr__(self, name: str):
        """Forward unknown attribute requests to ``VTKArray``'s ``__getattr__``."""
        if self.VTKObject is not None:
            # Check if forwarding snake_case attributes
            DisableVtkSnakeCase.check_attribute(self.VTKObject, name)
            return getattr(self.VTKObject, name)
        raise AttributeError
