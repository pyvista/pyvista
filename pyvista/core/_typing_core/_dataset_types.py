"""PyVista dataset types."""

from __future__ import annotations

from typing import TypeVar
from typing import Union

from pyvista.core.dataset import DataObject
from pyvista.core.dataset import DataSet
from pyvista.core.grid import ImageData
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import ExplicitStructuredGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import StructuredGrid
from pyvista.core.pointset import UnstructuredGrid
from pyvista.core.pointset import _PointSet

# Use these typevars wherever shared `pyvista` and `vtk` object attributes are
# required for type-checking. The abstract classes like `DataSet`, `DataObject`
# do not inherit VTK methods whereas the concrete classes do

ConcreteGridAlias = Union[
    ImageData,
    RectilinearGrid,
]
ConcreteGridType = TypeVar('ConcreteGridType', bound=ConcreteGridAlias)
ConcreteGridType.__doc__ = """Type variable of all concrete PyVista ``Grid`` classes."""

ConcretePointGridAlias = Union[
    ExplicitStructuredGrid,
    StructuredGrid,
    UnstructuredGrid,
]
ConcretePointGridType = TypeVar('ConcretePointGridType', bound=ConcretePointGridAlias)
ConcretePointGridType.__doc__ = """Type variable of all concrete PyVista ``PointGrid``` classes."""

ConcretePointSetAlias = Union[
    ExplicitStructuredGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
]
_PointSetType = TypeVar('_PointSetType', bound=_PointSet)
_PointSetType.__doc__ = """Type variable of all concrete PyVista ``PointSet`` classes."""

DataSetType = TypeVar('DataSetType', bound=DataSet)
DataSetType.__doc__ = """Type variable for :class:`~pyvista.DataSet` classes."""

DataObjectType = TypeVar('DataObjectType', bound=DataObject)
DataObjectType.__doc__ = """Type variable for :class:`~pyvista.DataObject` classes."""
