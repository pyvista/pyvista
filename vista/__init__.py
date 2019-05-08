import warnings
from vista._version import __version__
from vista.plotting import *
from vista.utilities import *
from vista.errors import *
from vista.readers import *
from vista.colors import *
from vista.features import *
from vista.filters import DataSetFilters
from vista.common import Common
from vista.pointset import PointGrid
from vista.pointset import PolyData
from vista.pointset import UnstructuredGrid
from vista.pointset import StructuredGrid
from vista.grid import Grid
from vista.grid import RectilinearGrid
from vista.grid import UniformGrid
from vista.geometric_objects import *
from vista.container import MultiBlock
from vista.qt_plotting import QtInteractor
from vista.qt_plotting import BackgroundPlotter
from vista.export import export_plotter_vtkjs, get_vtkjs_url
from vista.renderer import Renderer

# IPython interactive tools
from vista.ipy_tools import OrthogonalSlicer
from vista.ipy_tools import ManySlicesAlongAxis
from vista.ipy_tools import Threshold
from vista.ipy_tools import Clip
from vista.ipy_tools import ScaledPlotter
from vista.ipy_tools import Isocontour

# Sphinx-gallery tools
from vista.sphinx_gallery import Scraper

import numpy as np
import vtk

# get the int type from vtk
ID_TYPE = np.int32
if vtk.VTK_ID_TYPE == 12:
    ID_TYPE = np.int64


# determine if using vtk > 5
if vtk.vtkVersion().GetVTKMajorVersion() <= 5:
    raise AssertionError('VTK version must be 5.0 or greater.')

# catch annoying numpy/vtk future warning:
warnings.simplefilter(action='ignore', category=FutureWarning)

# A simple flag to set when generating the documentation
OFF_SCREEN = False
try:
    if os.environ['VISTA_OFF_SCREEN'].lower() == 'true':
        OFF_SCREEN = True
except KeyError:
    pass

# A threshold for the max cells to compute a volume for when repr-ing
REPR_VOLUME_MAX_CELLS = 1e6

# Set where figures are saved
FIGURE_PATH = None

# Set up data directory
import appdirs
import os

USER_DATA_PATH = appdirs.user_data_dir('vista')
if not os.path.exists(USER_DATA_PATH):
    os.makedirs(USER_DATA_PATH)

EXAMPLES_PATH = os.path.join(USER_DATA_PATH, 'examples')
if not os.path.exists(EXAMPLES_PATH):
    os.makedirs(EXAMPLES_PATH)

# Send VTK messages to the logging module:
send_errors_to_logging()


# Set up panel for interactive notebook rendering
try:
    if os.environ['VISTA_USE_PANEL'].lower() == 'false':
        rcParams['use_panel'] = False
    elif os.environ['VISTA_USE_PANEL'].lower() == 'true':
        rcParams['use_panel'] = True
except KeyError:
    pass
try:
    import panel
    panel.extension('vtk')
except ImportError:
    pass

# Set preferred plot theme
try:
    theme = os.environ['VISTA_PLOT_THEME'].lower()
    set_plot_theme(theme)
except KeyError:
    pass
