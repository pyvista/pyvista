"""PyVista dataset types."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
    from pyvista import DataObject
    from pyvista import DataSet
    from pyvista import Grid
    from pyvista import MultiBlock
    from pyvista import PointGrid
    from pyvista import PolyData
    from pyvista import UnstructuredGrid
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
