"""PyVista dataset types."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
    from pyvista.core.composite import MultiBlock
    from pyvista.core.dataobject import DataObject
    from pyvista.core.dataset import DataSet
    from pyvista.core.grid import Grid
    from pyvista.core.pointset import PointGrid
    from pyvista.core.pointset import PolyData
    from pyvista.core.pointset import UnstructuredGrid
    from pyvista.core.pointset import _PointSet

_GridType = TypeVar('_GridType', bound='Grid')
_PointGridType = TypeVar('_PointGridType', bound='PointGrid')
_PointSetType = TypeVar('_PointSetType', bound='_PointSet')
_DataSetType = TypeVar('_DataSetType', bound='DataSet')
_MultiBlockType = TypeVar('_MultiBlockType', bound='MultiBlock')
_DataSetOrMultiBlockType = TypeVar('_DataSetOrMultiBlockType', bound='DataSet | MultiBlock')
_DataObjectType = TypeVar('_DataObjectType', bound='DataObject')

# Undocumented
_PolyDataType = TypeVar('_PolyDataType', bound='PolyData')
_UnstructuredGridType = TypeVar('_UnstructuredGridType', bound='UnstructuredGrid')
