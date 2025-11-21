"""PyVista dataset types."""

from __future__ import annotations

from typing import TypeVar

from pyvista.core.composite import MultiBlock
from pyvista.core.dataobject import DataObject
from pyvista.core.dataset import DataSet
from pyvista.core.grid import Grid
from pyvista.core.pointset import PointGrid
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import UnstructuredGrid
from pyvista.core.pointset import _PointSet

_GridType = TypeVar('_GridType', bound=Grid)
"""Type variable for PyVista ``Grid`` classes."""

_PointGridType = TypeVar('_PointGridType', bound=PointGrid)
"""Type variable for PyVista ``PointGrid`` classes."""

_PointSetType = TypeVar('_PointSetType', bound=_PointSet)
"""Type variable for PyVista ``PointSet`` classes."""

_DataSetType = TypeVar('_DataSetType', bound=DataSet)
"""Type variable for :class:`~pyvista.DataSet` classes."""

_DataSetOrMultiBlockType = TypeVar('_DataSetOrMultiBlockType', bound=DataSet | MultiBlock)
"""Type variable for :class:`~pyvista.DataSet` or :class:`~pyvista.MultiBlock` classes."""

_DataObjectType = TypeVar('_DataObjectType', bound=DataObject)
"""Type variable for :class:`~pyvista.DataObject` classes."""


# Undocumented
_PolyDataType = TypeVar('_PolyDataType', bound=PolyData)
_UnstructuredGridType = TypeVar('_UnstructuredGridType', bound=UnstructuredGrid)
