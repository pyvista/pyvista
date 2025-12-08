"""Core routines."""

from __future__ import annotations

from . import _vtk_core as _vtk_core
from ._typing_core import *
from .cell import Cell as Cell
from .cell import CellArray as CellArray
from .celltype import CellType as CellType
from .composite import MultiBlock as MultiBlock
from .dataobject import DataObject as DataObject
from .dataset import DataSet as DataSet
from .datasetattributes import DataSetAttributes as DataSetAttributes
from .errors import AmbiguousDataError as AmbiguousDataError
from .errors import DeprecationError as DeprecationError
from .errors import MissingDataError as MissingDataError
from .errors import NotAllTrianglesError as NotAllTrianglesError
from .errors import PointSetCellOperationError as PointSetCellOperationError
from .errors import PointSetDimensionReductionError as PointSetDimensionReductionError
from .errors import PointSetNotSupported as PointSetNotSupported
from .errors import PyVistaAttributeError as PyVistaAttributeError
from .errors import PyVistaDeprecationWarning as PyVistaDeprecationWarning
from .errors import PyVistaEfficiencyWarning as PyVistaEfficiencyWarning
from .errors import PyVistaFutureWarning as PyVistaFutureWarning
from .errors import PyVistaInvalidMeshWarning as PyVistaInvalidMeshWarning
from .errors import PyVistaPipelineError as PyVistaPipelineError
from .errors import VTKExecutionError as VTKExecutionError
from .errors import VTKExecutionWarning as VTKExecutionWarning
from .errors import VTKVersionError as VTKVersionError
from .filters import CompositeFilters as CompositeFilters
from .filters import DataObjectFilters as DataObjectFilters
from .filters import DataSetFilters as DataSetFilters
from .filters import ImageDataFilters as ImageDataFilters
from .filters import PolyDataFilters as PolyDataFilters
from .filters import UnstructuredGridFilters as UnstructuredGridFilters
from .grid import Grid as Grid
from .grid import ImageData as ImageData
from .grid import RectilinearGrid as RectilinearGrid
from .objects import Table as Table
from .partitioned import PartitionedDataSet as PartitionedDataSet
from .pointset import ExplicitStructuredGrid as ExplicitStructuredGrid
from .pointset import PointGrid as PointGrid
from .pointset import PointSet as PointSet
from .pointset import PolyData as PolyData
from .pointset import StructuredGrid as StructuredGrid
from .pointset import UnstructuredGrid as UnstructuredGrid
from .pyvista_ndarray import pyvista_ndarray as pyvista_ndarray
from .utilities import *
from .wrappers import _wrappers as _wrappers
