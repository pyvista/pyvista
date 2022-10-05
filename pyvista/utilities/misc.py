"""Miscellaneous pyvista functions."""
from collections import namedtuple
from functools import lru_cache
import importlib
import os
import warnings

import numpy as np

import pyvista
from pyvista import _vtk


def _set_plot_theme_from_env():
    """Set plot theme from an environment variable."""
    from pyvista.themes import _ALLOWED_THEMES, set_plot_theme

    if 'PYVISTA_PLOT_THEME' in os.environ:
        try:
            theme = os.environ['PYVISTA_PLOT_THEME']
            set_plot_theme(theme.lower())
        except KeyError:
            allowed = ', '.join([item.name for item in _ALLOWED_THEMES])
            warnings.warn(
                f'\n\nInvalid PYVISTA_PLOT_THEME environment variable "{theme}". '
                f'Should be one of the following: {allowed}'
            )


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
    array : vtk.vtkDataArray or vtk.vtkAbstractArray
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


def can_create_mpl_figure():  # pragma: no cover
    """Return if a figure can be created with matplotlib."""
    try:
        import matplotlib.pyplot as plt

        figure = plt.figure()
        plt.close(figure)
        return True
    except:
        return False


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

    def __setattr__(self, name, value):
        """Do not allow setting attributes."""
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f'Attribute "{name}" does not exist and cannot be added to type '
                f'{self.__class__.__name__}'
            )

    setattr(cls, '__setattr__', __setattr__)
    return cls
