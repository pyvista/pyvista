"""Miscellaneous pyvista functions."""
from collections import namedtuple
from functools import lru_cache
import importlib
import os
import warnings

import numpy as np

import pyvista
from pyvista import _vtk


def _lazy_vtk_instantiation(module_name, class_name):
    """Lazy import and instantiation of a class from vtkmodules."""
    module = importlib.import_module(f"vtkmodules.{module_name}")
    return getattr(module, class_name)()


def _set_plot_theme_from_env():
    """Set plot theme from an environment variable."""
    from pyvista.themes import _NATIVE_THEMES, set_plot_theme

    if 'PYVISTA_PLOT_THEME' in os.environ:
        try:
            theme = os.environ['PYVISTA_PLOT_THEME']
            set_plot_theme(theme.lower())
        except ValueError:
            allowed = ', '.join([item.name for item in _NATIVE_THEMES])
            warnings.warn(
                f'\n\nInvalid PYVISTA_PLOT_THEME environment variable "{theme}". '
                f'Should be one of the following: {allowed}'
            )


def _check_range(value, rng, parm_name):
    """Check if a parameter is within a range."""
    if value < rng[0] or value > rng[1]:
        raise ValueError(
            f'The value {float(value)} for `{parm_name}` is outside the acceptable range {tuple(rng)}.'
        )


def _try_imageio_imread(filename):
    """Attempt to read a file using ``imageio.imread``.

    Parameters
    ----------
    filename : str
        Name of the file to read using ``imageio``.

    Returns
    -------
    imageio.core.util.Array
        Image read from ``imageio``.

    Raises
    ------
    ModuleNotFoundError
        Raised when ``imageio`` is not installed when attempting to read
        ``filename``.

    """
    try:
        from imageio import imread
    except ModuleNotFoundError:  # pragma: no cover
        raise ModuleNotFoundError(
            'Problem reading the image with VTK. Install imageio to try to read the '
            'file using imageio with:\n\n'
            '   pip install imageio'
        ) from None

    return imread(filename)


@lru_cache(maxsize=None)
def has_module(module_name):
    """Return if a module can be imported."""
    module_spec = importlib.util.find_spec(module_name)
    return module_spec is not None


def raise_has_duplicates(arr):
    """Raise a ValueError if an array is not unique."""
    if has_duplicates(arr):
        raise ValueError("Array contains duplicate values.")


def has_duplicates(arr):
    """Return if an array has any duplicates."""
    s = np.sort(arr, axis=None)
    return (s[1:] == s[:-1]).any()


def _get_vtk_id_type():
    """Return the numpy datatype responding to ``vtk.vtkIdTypeArray``."""
    VTK_ID_TYPE_SIZE = _vtk.vtkIdTypeArray().GetDataTypeSize()
    if VTK_ID_TYPE_SIZE == 4:
        return np.int32
    elif VTK_ID_TYPE_SIZE == 8:
        return np.int64
    return np.int32


class PyVistaDeprecationWarning(Warning):
    """Non-supressed Depreciation Warning."""

    pass


class PyVistaFutureWarning(Warning):
    """Non-supressed Future Warning."""

    pass


class PyVistaEfficiencyWarning(Warning):
    """Efficiency warning."""

    pass


def VTKVersionInfo():
    """Return the vtk version as a namedtuple."""
    version_info = namedtuple('VTKVersionInfo', ['major', 'minor', 'micro'])

    try:
        ver = _vtk.vtkVersion()
        major = ver.GetVTKMajorVersion()
        minor = ver.GetVTKMinorVersion()
        micro = ver.GetVTKBuildVersion()
    except AttributeError:  # pragma: no cover
        warnings.warn("Unable to detect VTK version. Defaulting to v4.0.0")
        major, minor, micro = (4, 0, 0)

    return version_info(major, minor, micro)


def uses_egl() -> bool:
    """Check if VTK has been compiled with EGL support via OSMesa."""
    ren_win_str = str(type(_vtk.vtkRenderWindow()))
    return 'EGL' in ren_win_str or 'OSOpenGL' in ren_win_str


def copy_vtk_array(array, deep=True):
    """Create a deep or shallow copy of a VTK array.

    Parameters
    ----------
    array : vtk.vtkDataArray | vtk.vtkAbstractArray
        VTK array.

    deep : bool, optional
        When ``True``, create a deep copy of the array. When ``False``, returns
        a shallow copy.

    Returns
    -------
    vtk.vtkDataArray or vtk.vtkAbstractArray
        Copy of the original VTK array.

    Examples
    --------
    Perform a deep copy of a vtk array.

    >>> import vtk
    >>> import pyvista
    >>> arr = vtk.vtkFloatArray()
    >>> _ = arr.SetNumberOfValues(10)
    >>> arr.SetValue(0, 1)
    >>> arr_copy = pyvista.utilities.misc.copy_vtk_array(arr)
    >>> arr_copy.GetValue(0)
    1.0

    """
    if not isinstance(array, (_vtk.vtkDataArray, _vtk.vtkAbstractArray)):
        raise TypeError(f"Invalid type {type(array)}.")

    new_array = type(array)()
    if deep:
        new_array.DeepCopy(array)
    else:
        new_array.ShallowCopy(array)

    return new_array


vtk_version_info = VTKVersionInfo()


def set_pickle_format(format: str):
    """Set the format used to serialize :class:`pyvista.DataObject` when pickled."""
    supported = {'xml', 'legacy'}
    format = format.lower()
    if format not in supported:
        raise ValueError(
            f'Unsupported pickle format `{format}`. Valid options are `{"`, `".join(supported)}`.'
        )
    pyvista.PICKLE_FORMAT = format


def no_new_attr(cls):
    """Override __setattr__ to not permit new attributes."""
    if not hasattr(cls, '_new_attr_exceptions'):
        cls._new_attr_exceptions = []

    def __setattr__(self, name, value):
        """Do not allow setting attributes."""
        if hasattr(self, name) or name in cls._new_attr_exceptions:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f'Attribute "{name}" does not exist and cannot be added to type '
                f'{self.__class__.__name__}'
            )

    setattr(cls, '__setattr__', __setattr__)
    return cls
