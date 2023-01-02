"""Core routines."""

from .dataset import DataSet, DataObject
from .composite import MultiBlock
from .datasetattributes import DataSetAttributes
from .filters import (
    CompositeFilters,
    DataSetFilters,
    PolyDataFilters,
    UnstructuredGridFilters,
    UniformGridFilters,
)
from .grid import Grid, RectilinearGrid, UniformGrid
from .objects import Table, Texture
from .pointset import (
    PointGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
    ExplicitStructuredGrid,
)
from .pyvista_ndarray import pyvista_ndarray
from .cell import CellType


# __all__ only left for mypy --strict to work when pyvista is a dependency
__all__ = [
    'CellType',
    'CompositeFilters',
    'DataObject',
    'DataSet',
    'DataSetAttributes',
    'DataSetFilters',
    'ExplicitStructuredGrid',
    'Grid',
    'MultiBlock',
    'PointGrid',
    'PointSet',
    'PolyData',
    'PolyDataFilters',
    'RectilinearGrid',
    'StructuredGrid',
    'Table',
    'Texture',
    'UniformGrid',
    'UniformGridFilters',
    'UnstructuredGrid',
    'UnstructuredGridFilters',
    'pyvista_ndarray',
]
