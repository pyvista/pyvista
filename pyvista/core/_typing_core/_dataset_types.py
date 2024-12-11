"""PyVista dataset types."""

from __future__ import annotations

from typing import TypeVar

from pyvista.core.dataset import DataObject
from pyvista.core.dataset import DataSet
from pyvista.core.pointset import PointGrid
from pyvista.core.pointset import _PointSet

PointGridType = TypeVar('PointGridType', bound=PointGrid)
PointGridType.__doc__ = """Type variable of all concrete PyVista ``PointGrid``` classes."""

_PointSetType = TypeVar('_PointSetType', bound=_PointSet)
_PointSetType.__doc__ = """Type variable of all concrete PyVista ``PointSet`` classes."""

DataSetType = TypeVar('DataSetType', bound=DataSet)
DataSetType.__doc__ = """Type variable for :class:`~pyvista.DataSet` classes."""

DataObjectType = TypeVar('DataObjectType', bound=DataObject)
DataObjectType.__doc__ = """Type variable for :class:`~pyvista.DataObject` classes."""
