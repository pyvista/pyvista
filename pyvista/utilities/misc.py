"""Miscellaneous pyvista functions."""
from collections import namedtuple
import warnings

import numpy as np

from pyvista import _vtk


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


class PyvistaDeprecationWarning(Warning):
    """Non-supressed Depreciation Warning."""

    pass


class PyvistaFutureWarning(Warning):
    """Non-supressed Future Warning."""

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


vtk_version_info = VTKVersionInfo()
