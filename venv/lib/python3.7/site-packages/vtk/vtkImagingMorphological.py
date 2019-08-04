"""
This module is an adapter for scripts that explicitly import from named
submodules as opposed to from the top-level vtk module. This is necessary
because the specific submodules might not exist when VTK_ENABLE_KITS is enabled.
"""

from __future__ import absolute_import

try:
    # use relative import for installed modules
    from .vtkImagingKitPython import *
except ImportError:
    # during build and testing, the modules will be elsewhere,
    # e.g. in lib directory or Release/Debug config directories
    from vtkImagingKitPython import *
