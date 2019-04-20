import warnings
from vtki._version import __version__
from vtki.plotting import *
from vtki.utilities import *
from vtki.errors import *
from vtki.readers import *
from vtki.colors import *
from vtki.filters import DataSetFilters
from vtki.common import Common
from vtki.pointset import PointGrid
from vtki.pointset import PolyData
from vtki.pointset import UnstructuredGrid
from vtki.pointset import StructuredGrid
from vtki.grid import Grid
from vtki.grid import RectilinearGrid
from vtki.grid import UniformGrid
from vtki.geometric_objects import *
from vtki.container import MultiBlock
from vtki.qt_plotting import QtInteractor
from vtki.qt_plotting import BackgroundPlotter
from vtki.export import export_plotter_vtkjs, get_vtkjs_url
from vtki.renderer import Renderer

# IPython interactive tools
from vtki.ipy_tools import OrthogonalSlicer
from vtki.ipy_tools import ManySlicesAlongAxis
from vtki.ipy_tools import Threshold
from vtki.ipy_tools import Clip
from vtki.ipy_tools import ScaledPlotter
from vtki.ipy_tools import Isocontour

# Sphinx-gallery tools
from vtki.sphinx_gallery import Scraper

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
    if os.environ['VTKI_OFF_SCREEN'] == 'True':
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

USER_DATA_PATH = appdirs.user_data_dir('vtki')
if not os.path.exists(USER_DATA_PATH):
    os.makedirs(USER_DATA_PATH)

EXAMPLES_PATH = os.path.join(USER_DATA_PATH, 'examples')
if not os.path.exists(EXAMPLES_PATH):
    os.makedirs(EXAMPLES_PATH)

# Send VTK messages to the logging module:
send_errors_to_logging()


# Set up panel for interactive notebook rendering
try:
    import panel
    panel.extension('vtk')
except ImportError:
    pass
