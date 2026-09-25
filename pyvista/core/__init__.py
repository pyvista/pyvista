"""Core routines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._typing_core import BoundsTuple as BoundsTuple
from ._typing_core import NumpyArray as NumpyArray
from .cell import Cell as Cell
from .cell import CellArray as CellArray
from .celltype import CellType as CellType
from .composite import MultiBlock as MultiBlock
from .config import global_config as global_config
from .dataobject import DataObject as DataObject
from .dataset import DataSet as DataSet
from .datasetattributes import DataSetAttributes as DataSetAttributes
from .errors import AmbiguousDataError as AmbiguousDataError
from .errors import DeprecationError as DeprecationError
from .errors import InvalidMeshError as InvalidMeshError
from .errors import InvalidMeshWarning as InvalidMeshWarning
from .errors import MissingDataError as MissingDataError
from .errors import NotAllTrianglesError as NotAllTrianglesError
from .errors import PointSetCellOperationError as PointSetCellOperationError
from .errors import PointSetDimensionReductionError as PointSetDimensionReductionError
from .errors import PointSetNotSupported as PointSetNotSupported
from .errors import PrecisionWarning as PrecisionWarning
from .errors import PyVistaAttributeError as PyVistaAttributeError
from .errors import PyVistaDeprecationWarning as PyVistaDeprecationWarning
from .errors import PyVistaEfficiencyWarning as PyVistaEfficiencyWarning
from .errors import PyVistaFutureWarning as PyVistaFutureWarning
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
from .filters.data_object import CellStatus as CellStatus
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

_TYPE_ALIASES = (
    'ArrayLike',
    'CellArrayLike',
    'CellsLike',
    'InteractionEventType',
    'LineStyle',
    'MatrixLike',
    'Number',
    'NumberType',
    'RotationLike',
    'TransformLike',
    'VectorLike',
)


if not TYPE_CHECKING:  # pragma: no branch

    def __getattr__(name: str) -> object:
        """Forward the type aliases that moved to ``pyvista.typing`` with a deprecation warning."""
        if name in _TYPE_ALIASES:
            from pyvista.typing import _get_deprecated_alias  # noqa: PLC0415

            return _get_deprecated_alias(__name__, name)
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)
