"""PyVista dataset types."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import Union

if TYPE_CHECKING:
    from pyvista import DataObject
    from pyvista import DataSet
    from pyvista import Grid
    from pyvista import MultiBlock
    from pyvista import PointGrid
    from pyvista import PointSet
    from pyvista import PolyData
    from pyvista import UnstructuredGrid
    from pyvista.core.pointset import _PointSetBase

_GridType = TypeVar('_GridType', bound='Grid')
_PointGridType = TypeVar('_PointGridType', bound='PointGrid')
_PointSetBaseType = TypeVar('_PointSetBaseType', bound='_PointSetBase')
_DataSetType = TypeVar('_DataSetType', bound='DataSet')
_MultiBlockType = TypeVar('_MultiBlockType', bound='MultiBlock[Any]')
_DataSetOrMultiBlockType = TypeVar('_DataSetOrMultiBlockType', bound='DataSet | MultiBlock[Any]')
_DataObjectType = TypeVar('_DataObjectType', bound='DataObject')

# The dataset classes a filter can build, and the composite that holds them
_OutputDataSet = Union['PolyData', 'PointSet', 'UnstructuredGrid']
_OutputDataObject = Union[_OutputDataSet, 'MultiBlock[Any]']

# Undocumented
_PolyDataType = TypeVar('_PolyDataType', bound='PolyData')
_UnstructuredGridType = TypeVar('_UnstructuredGridType', bound='UnstructuredGrid')
