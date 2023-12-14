"""PyVista package for 3D plotting and mesh analysis."""
# flake8: noqa: F401
import os
import sys
import warnings

from pyvista._plot import plot
from pyvista._version import __version__
from pyvista.core import *
from pyvista.core.cell import _get_vtk_id_type
from pyvista.core.utilities.observers import send_errors_to_logging
from pyvista.core.wrappers import _wrappers
from pyvista.jupyter import set_jupyter_backend
from pyvista.report import GPUInfo, Report, get_gpu_info, vtk_version_info

# get the int type from vtk
ID_TYPE = _get_vtk_id_type()

# determine if using at least vtk 9.0.0
if vtk_version_info.major < 9:  # pragma: no cover
    from pyvista.core.errors import VTKVersionError

    raise VTKVersionError('VTK version must be 9.0.0 or greater.')

# catch annoying numpy/vtk future warning:
warnings.simplefilter(action='ignore', category=FutureWarning)

# A simple flag to set when generating the documentation
OFF_SCREEN = os.environ.get("PYVISTA_OFF_SCREEN", "false").lower() == "true"

# flag for when building the sphinx_gallery
BUILDING_GALLERY = os.environ.get("PYVISTA_BUILDING_GALLERY", "false").lower() == "true"

# A threshold for the max cells to compute a volume for when repr-ing
REPR_VOLUME_MAX_CELLS = 1e6

# Set where figures are saved
FIGURE_PATH = os.environ.get("PYVISTA_FIGURE_PATH", None)

ON_SCREENSHOT = os.environ.get("PYVISTA_ON_SCREENSHOT", "false").lower() == "true"

# Send VTK messages to the logging module:
send_errors_to_logging()

# theme to use by default for the plot directive
PLOT_DIRECTIVE_THEME = None

# Set a parameter to control default print format for floats outside
# of the plotter
FLOAT_FORMAT = "{:.3e}"

# Serialization format to be used when pickling `DataObject`
PICKLE_FORMAT = 'xml'

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

    # avoid recursive import
    if 'pyvista.plotting' not in sys.modules:
        import pyvista.plotting

    try:
        feature = inspect.getattr_static(sys.modules['pyvista.plotting'], name)
    except AttributeError:
        raise AttributeError(f"module 'pyvista' has no attribute '{name}'") from None

    return feature
