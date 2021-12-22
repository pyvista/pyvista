"""Miscellaneous pyvista functions."""
from collections import namedtuple
import warnings

import numpy as np

from pyvista import _vtk


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
