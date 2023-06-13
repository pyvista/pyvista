"""PyVista package for 3D plotting and mesh analysis."""
# flake8: noqa: F401

MAX_N_COLOR_BARS = 10

import os
import warnings

from pyvista._plot import plot
from pyvista._version import __version__
from pyvista.core import *
from pyvista.core.cell import _get_vtk_id_type
from pyvista.core.utilities.observers import send_errors_to_logging
from pyvista.core.wrappers import _wrappers
from pyvista.errors import InvalidCameraError, RenderWindowUnavailable
from pyvista.jupyter import set_jupyter_backend
from pyvista.plotting import *
from pyvista.plotting import _typing, _vtk
from pyvista.plotting.utilities.sphinx_gallery import _get_sg_image_scraper
from pyvista.report import GPUInfo, Report, get_gpu_info, vtk_version_info
from pyvista.themes import (
    DocumentTheme as _GlobalTheme,
    _rcParams,
    _set_plot_theme_from_env,
    load_theme,
    set_plot_theme,
)

global_theme = _GlobalTheme()
rcParams = _rcParams()  # raises DeprecationError when used

# Set preferred plot theme
_set_plot_theme_from_env()

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
BUILDING_GALLERY = False
if 'PYVISTA_BUILDING_GALLERY' in os.environ:
    if os.environ['PYVISTA_BUILDING_GALLERY'].lower() == 'true':
        BUILDING_GALLERY = True

# A threshold for the max cells to compute a volume for when repr-ing
REPR_VOLUME_MAX_CELLS = 1e6

# Set where figures are saved
FIGURE_PATH = None

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
