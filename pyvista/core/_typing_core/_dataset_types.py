"""PyVista dataset types."""

from __future__ import annotations

from typing import TypeVar
from typing import Union

from pyvista.core.composite import MultiBlock
from pyvista.core.grid import ImageData
from pyvista.core.grid import RectilinearGrid
from pyvista.core.objects import Table
from pyvista.core.partitioned import PartitionedDataSet
from pyvista.core.pointset import ExplicitStructuredGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import StructuredGrid
from pyvista.core.pointset import UnstructuredGrid

# Use this typevar wherever a `DataSet` type hint may be used
# Unlike `DataSet`, the concrete classes here also inherit from `vtkDataSet`

ConcretePointGridAlias = Union[
    ExplicitStructuredGrid,
    StructuredGrid,
    UnstructuredGrid,
]
ConcretePointGridType = TypeVar('ConcretePointGridType', bound=ConcretePointGridAlias)

ConcretePointSetAlias = Union[
    ExplicitStructuredGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
]
ConcretePointSetType = TypeVar('ConcretePointSetType', bound=ConcretePointSetAlias)

ConcreteGridAlias = Union[
    ImageData,
    RectilinearGrid,
]
ConcreteGridType = TypeVar('ConcreteGridType', bound=ConcreteGridAlias)

ConcreteDataSetAlias = Union[ConcreteGridAlias, ConcretePointSetAlias]
ConcreteDataSetType = TypeVar('ConcreteDataSetType', bound=ConcreteDataSetAlias)
ConcreteDataSetType.__doc__ = """Type variable of all concrete :class:`~pyvista.DataSet` classes."""

ConcreteDataObjectAlias = Union[
    ConcreteDataSetAlias,
    Table,
    MultiBlock,
    PartitionedDataSet,
]
ConcreteDataObjectType = TypeVar('ConcreteDataObjectType', bound=ConcreteDataObjectAlias)
