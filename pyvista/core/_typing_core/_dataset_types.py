"""PyVista dataset types."""

from __future__ import annotations

from typing import TypeVar

from pyvista.core.grid import ImageData
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import ExplicitStructuredGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import StructuredGrid
from pyvista.core.pointset import UnstructuredGrid

# Use this typevar wherever a `DataSet` type hint may be used
# Unlike `DataSet`, the concrete classes here also inherit from `vtkDataSet`
_DataSetType_co = TypeVar(  # noqa: PYI018
    '_DataSetType_co',
    ImageData,
    RectilinearGrid,
    ExplicitStructuredGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
    covariant=True,
)

# Use this typevar wherever a `DataSet | MultiBlock` type hint may be used
# This should be identical to above, but with `MultiBlock`
_DataSetOrMultiBlockType_co = TypeVar(  # noqa: PYI018
    '_DataSetOrMultiBlockType_co',
    ImageData,
    RectilinearGrid,
    ExplicitStructuredGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
    covariant=True,
)
