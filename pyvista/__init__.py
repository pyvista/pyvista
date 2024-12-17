"""PyVista package for 3D plotting and mesh analysis."""

from __future__ import annotations

import os
import sys

from typing import Literal
import warnings


from pyvista.core import *
from pyvista.core import _validation as _validation


# catch annoying numpy/vtk future warning:
warnings.simplefilter(action='ignore', category=FutureWarning)

# A simple flag to set when generating the documentation
OFF_SCREEN = os.environ.get('PYVISTA_OFF_SCREEN', 'false').lower() == 'true'

# flag for when building the sphinx_gallery
BUILDING_GALLERY = os.environ.get('PYVISTA_BUILDING_GALLERY', 'false').lower() == 'true'

# A threshold for the max cells to compute a volume for when repr-ing
REPR_VOLUME_MAX_CELLS = 1e6

# Set where figures are saved
FIGURE_PATH = os.environ.get('PYVISTA_FIGURE_PATH', None)

ON_SCREENSHOT = os.environ.get('PYVISTA_ON_SCREENSHOT', 'false').lower() == 'true'

# theme to use by default for the plot directive
PLOT_DIRECTIVE_THEME = None

# Set a parameter to control default print format for floats outside
# of the plotter
FLOAT_FORMAT = '{:.3e}'

# Serialization format to be used when pickling `DataObject`
PICKLE_FORMAT: Literal['vtk', 'xml', 'legacy'] = 'vtk' if vtk_version_info >= (9, 3) else 'xml'

# Name used for unnamed scalars
DEFAULT_SCALARS_NAME = 'Data'

MAX_N_COLOR_BARS = 10


# Lazily import/access the plotting module
def __getattr__(name):
    """Fetch an attribute ``name`` from ``globals()`` or the ``pyvista.plotting`` module.

    This override is implemented to prevent importing all of the plotting module
    and GL-dependent VTK modules when importing PyVista.

    Raises
    ------
    AttributeError
        If the attribute is not found.

    """
    import importlib
    import inspect

    allow = {
        'demos',
        'examples',
        'ext',
        'trame',
        'utilities',
    }
    if name in allow:
        return importlib.import_module(f'pyvista.{name}')

    try:
        feature = inspect.getattr_static(sys.modules['pyvista.plotting'], name)
    except AttributeError:
        raise AttributeError(f"module 'pyvista' has no attribute '{name}'") from None

    return feature
