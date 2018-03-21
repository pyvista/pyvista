# import numpy
# with numpy.warnings.catch_warnings():
#     # numpy.warnings.filterwarnings('ignore', r"== np.dtype(complex).type`.")
#     numpy.warnings.filterwarnings('ignore')
#     from vtk.util import numpy_support as VN
#     VN.numpy_to_vtk(numpy.zeros(10))

import warnings
from vtkInterface._version import __version__
from vtkInterface.plotting import *
from vtkInterface.utilities import *
from vtkInterface.colors import *
from vtkInterface.common import Common
from vtkInterface.polydata import PolyData
from vtkInterface.grid import UnstructuredGrid
from vtkInterface.grid import StructuredGrid
from vtkInterface.grid import Grid

import numpy as np

try:
    import vtk

    # get the int type from vtk
    if vtk.VTK_ID_TYPE == 12:
        ID_TYPE = np.int64
    else:
        ID_TYPE = np.int32

    # determine if using vtk > 5
    if vtk.vtkVersion().GetVTKMajorVersion() < 5:
        warnings.warn('VTK version must be 5.0 or greater.')

except Exception as e:
    warnings.warn(str(e))
    warnings.warn('Unable to import VTK.  Check if installed')
    ID_TYPE = np.int64


# catch annoying numpy/vtk future warning:
# vtk/util/numpy_support.py:135: FutureWarning: Conversion of the second argument of issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will be treated as `np.complex128 == np.dtype(complex).type`.
#   assert not numpy.issubdtype(z.dtype, complex), \
warnings.simplefilter(action='ignore', category=FutureWarning)
