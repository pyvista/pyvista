import warnings
from vtki._version import __version__
from vtki.plotting import *
from vtki.utilities import *
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
TESTING_OFFSCREEN = False

# A threshold for the max cells to compute a volume for when repr-ing
REPR_VOLUME_MAX_CELLS = 1e6
