"""PyVista package for 3D plotting and mesh analysis."""

MAX_N_COLOR_BARS = 10

from typing import Optional
import warnings
import os
import appdirs

# Load default theme.  Must be loaded first.
from pyvista._version import __version__
from pyvista.plotting import *
from pyvista.utilities import *
from pyvista.core import *
from pyvista.utilities.misc import _get_vtk_id_type, vtk_version_info
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
if 'PYVISTA_PLOT_THEME' in os.environ:
    set_plot_theme(os.environ['PYVISTA_PLOT_THEME'].lower())

# get the int type from vtk
ID_TYPE = _get_vtk_id_type()

# determine if using at least vtk 5.0.0
if vtk_version_info.major < 5:
    raise RuntimeError('VTK version must be 5.0 or greater.')

# catch annoying numpy/vtk future warning:
warnings.simplefilter(action='ignore', category=FutureWarning)

# A simple flag to set when generating the documentation
OFF_SCREEN = False
try:
    if os.environ['PYVISTA_OFF_SCREEN'].lower() == 'true':
        OFF_SCREEN = True
except KeyError:
    pass

# If available, a local vtk-data instance will be used for examples
VTK_DATA_PATH: Optional[str] = None
if 'PYVISTA_VTK_DATA' in os.environ:
    VTK_DATA_PATH = os.environ['PYVISTA_VTK_DATA']
    if not os.path.isdir(VTK_DATA_PATH):
        warnings.warn(f"VTK_DATA_PATH: {VTK_DATA_PATH} is an invalid path")
    if not os.path.isdir(os.path.join(VTK_DATA_PATH, 'Data')):
        warnings.warn(f"VTK_DATA_PATH: {os.path.join(VTK_DATA_PATH, 'Data')} does not exist")

# flag for when building the sphinx_gallery
BUILDING_GALLERY = False
if 'PYVISTA_BUILDING_GALLERY' in os.environ:
    if os.environ['PYVISTA_BUILDING_GALLERY'].lower() == 'true':
        BUILDING_GALLERY = True

# A threshold for the max cells to compute a volume for when repr-ing
REPR_VOLUME_MAX_CELLS = 1e6

# Set where figures are saved
FIGURE_PATH = None


# allow user to override the examples path
if 'PYVISTA_USERDATA_PATH' in os.environ:
    USER_DATA_PATH = os.environ['PYVISTA_USERDATA_PATH']
    if not os.path.isdir(USER_DATA_PATH):
        raise FileNotFoundError(f'Invalid PYVISTA_USERDATA_PATH at {USER_DATA_PATH}')

else:
    USER_DATA_PATH = appdirs.user_data_dir('pyvista')
    try:
        # Set up data directory
        os.makedirs(USER_DATA_PATH, exist_ok=True)
    except Exception as e:
        warnings.warn(f'Unable to create `PYVISTA_USERDATA_PATH` at "{USER_DATA_PATH}"\n'
                      f'Error: {e}\n\n'
                      'Override the default path by setting the environmental variable '
                      '`PYVISTA_USERDATA_PATH` to a writable path.')
        USER_DATA_PATH = ''

EXAMPLES_PATH = os.path.join(USER_DATA_PATH, 'examples')
try:
    os.makedirs(EXAMPLES_PATH, exist_ok=True)
except Exception as e:
    warnings.warn(f'Unable to create `EXAMPLES_PATH` at "{EXAMPLES_PATH}"\n'
                  f'Error: {e}\n\n'
                  'Override the default path by setting the environmental variable '
                  '`PYVISTA_USERDATA_PATH` to a writable path.')
    EXAMPLES_PATH = ''

# Send VTK messages to the logging module:
send_errors_to_logging()


# Set a parameter to control default print format for floats outside
# of the plotter
FLOAT_FORMAT = "{:.3e}"
