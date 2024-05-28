"""Core routines."""

# flake8: noqa: F401
from __future__ import annotations

from . import _vtk_core
from ._typing_core import *
from .cell import Cell, CellArray
from .celltype import CellType
from .composite import MultiBlock
from .dataset import DataObject, DataSet
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
from .objects import Table
from .partitioned import PartitionedDataSet
from .pointset import (
    ExplicitStructuredGrid,
    PointGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
)
from .pyvista_ndarray import pyvista_ndarray
from .utilities import *
from .wrappers import _wrappers
