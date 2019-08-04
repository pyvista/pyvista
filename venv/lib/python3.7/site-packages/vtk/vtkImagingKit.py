from __future__ import absolute_import

try:
    # use relative import for installed modules
    from .vtkImagingKitPython import *
except ImportError:
    # during build and testing, the modules will be elsewhere,
    # e.g. in lib directory or Release/Debug config directories
    from vtkImagingKitPython import *
