"""PyVista package for 3D plotting and mesh analysis."""

MAX_N_COLOR_BARS = 10

from typing import Optional
import warnings
import os

# Load default theme.  Must be loaded first.
from pyvista._version import __version__
from pyvista.plotting import *
from pyvista.utilities import *
from pyvista.core import *
from pyvista.utilities.misc import (
    _get_vtk_id_type,
    vtk_version_info,
    _set_plot_theme_from_env,
    set_pickle_format,
)
from pyvista import _vtk
from pyvista.jupyter import set_jupyter_backend, PlotterITK
from pyvista.themes import set_plot_theme, load_theme, _rcParams
from pyvista.themes import DefaultTheme as _GlobalTheme  # hide this

# Per contract with Sphinx-Gallery, this method must be available at top level
from pyvista.utilities.sphinx_gallery import _get_sg_image_scraper

from pyvista.utilities.wrappers import _wrappers

global_theme = _GlobalTheme()
rcParams = _rcParams()  # raises DeprecationError when used

# Set preferred plot theme
_set_plot_theme_from_env()

# get the int type from vtk
ID_TYPE = _get_vtk_id_type()

# determine if using at least vtk 5.0.0
if vtk_version_info.major < 5:  # pragma: no cover
    from pyvista.core.errors import VTKVersionError

    raise VTKVersionError('VTK version must be 5.0 or greater.')

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
