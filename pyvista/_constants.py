"""Constants used throughout PyVista."""

from __future__ import annotations

import vtkmodules.vtkCommonCore as _vtk

# Maximum number of color bars allowed in plotting
MAX_N_COLOR_BARS = 10

# Name used for unnamed scalars
DEFAULT_SCALARS_NAME = 'Data'

# VTK version information
vtk_version_info = tuple(
    int(x) for x in _vtk.vtkVersion.GetVTKVersion().split('.')
)