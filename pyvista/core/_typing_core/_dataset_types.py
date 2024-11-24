"""PyVista dataset types."""

from __future__ import annotations

from typing import TypeVar

from pyvista.core.composite import MultiBlock
from pyvista.core.grid import ImageData
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import ExplicitStructuredGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import StructuredGrid
from pyvista.core.pointset import UnstructuredGrid

# Use this typevar wherever a `DataSet` type hint may be used
# Unlike `DataSet`, the concrete classes here also inherit from `vtkDataSet`
DataSetType = TypeVar(
    'DataSetType',
    ImageData,
    RectilinearGrid,
    ExplicitStructuredGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
)
DataSetType.__doc__ = """Type variable of all concrete :class:`~pyvista.DataSet` classes."""

# Use this typevar wherever a `DataSet | MultiBlock` type hint may be used
# This should be identical to above, but with `MultiBlock`
DataSetMultiBlockType = TypeVar(
    'DataSetMultiBlockType',
    ImageData,
    RectilinearGrid,
    ExplicitStructuredGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
    MultiBlock,
)
DataSetMultiBlockType.__doc__ = """Type variable of :class:`~pyvista.MultiBlock` and all concrete :class:`~pyvista.DataSet` classes."""
