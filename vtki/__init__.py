# import numpy
# with numpy.warnings.catch_warnings():
#     # numpy.warnings.filterwarnings('ignore', r"== np.dtype(complex).type`.")
#     numpy.warnings.filterwarnings('ignore')
#     from vtk.util import numpy_support as VN
#     VN.numpy_to_vtk(numpy.zeros(10))

import warnings
from vtki._version import __version__
from vtki.plotting import *
from vtki.utilities import *
from vtki.colors import *
from vtki.common import Common
from vtki.polydata import PolyData
from vtki.grid import UnstructuredGrid
from vtki.grid import StructuredGrid
from vtki.grid import Grid
from vtki.grid import RectilinearGrid
from vtki.grid import UniformGrid
from vtki.geometric_objects import *

import numpy as np

import vtk

# get the int type from vtk
if vtk.VTK_ID_TYPE == 12:
    ID_TYPE = np.int64
else:
    ID_TYPE = np.int32

# determine if using vtk > 5
if vtk.vtkVersion().GetVTKMajorVersion() < 5:
    raise Exception('VTK version must be 5.0 or greater.')


# catch annoying numpy/vtk future warning:
# vtk/util/numpy_support.py:135: FutureWarning: Conversion of the second argument of issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will be treated as `np.complex128 == np.dtype(complex).type`.
#   assert not numpy.issubdtype(z.dtype, complex), \
warnings.simplefilter(action='ignore', category=FutureWarning)
