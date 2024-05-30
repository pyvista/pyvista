"""Core routines."""

# flake8: noqa: F401
from __future__ import annotations

from . import _vtk_core
from ._typing_core import *
from .cell import Cell
from .cell import CellArray
from .celltype import CellType
from .composite import MultiBlock
from .dataset import DataObject
from .dataset import DataSet
from .datasetattributes import DataSetAttributes
from .errors import (
    AmbiguousDataError,
    DeprecationError,
    MissingDataError,
    NotAllTrianglesError,
    PointSetCellOperationError,
    PointSetDimensionReductionError,
    PointSetNotSupported,
    PyVistaDeprecationWarning,
    PyVistaEfficiencyWarning,
    PyVistaFutureWarning,
    PyVistaPipelineError,
    VTKVersionError,
)
from .filters import (
    CompositeFilters,
    DataSetFilters,
    ImageDataFilters,
    PolyDataFilters,
    UnstructuredGridFilters,
)
from .grid import Grid, ImageData, RectilinearGrid
from .molecule import Molecule
from .objects import Table
from .partitioned import PartitionedDataSet
from .pointset import ExplicitStructuredGrid
from .pointset import PointGrid
from .pointset import PointSet
from .pointset import PolyData
from .pointset import StructuredGrid
from .pointset import UnstructuredGrid
from .pyvista_ndarray import pyvista_ndarray
from .utilities import *
from .wrappers import _wrappers
