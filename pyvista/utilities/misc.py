"""Miscellaneous pyvista functions."""
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


def _detect_os_mesa() -> bool:
    """Determine VTK compiled with OSMesa.

    Returns
    -------
    bool
        ``True`` when VTK compiled with OSMesa.

    """
    from pyvista import Plotter
    try:
        # expect vtkOSOpenGLRenderWindow
        return 'vtkOSOpenGL' in Plotter().ren_win.__class__.__name__
    except:
        return False
