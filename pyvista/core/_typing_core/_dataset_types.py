"""PyVista dataset types."""

from __future__ import annotations

from typing import TypeVar
from typing import Union

from pyvista.core.composite import MultiBlock
from pyvista.core.dataobject import DataObject
from pyvista.core.dataset import DataSet
from pyvista.core.grid import Grid
from pyvista.core.pointset import PointGrid
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import UnstructuredGrid
from pyvista.core.pointset import _PointSet

_GridType = TypeVar('_GridType', bound=Grid)  # noqa: PYI018

_PointGridType = TypeVar('_PointGridType', bound=PointGrid)  # noqa: PYI018

_PointSetType = TypeVar('_PointSetType', bound=_PointSet)  # noqa: PYI018

_DataSetType = TypeVar('_DataSetType', bound=DataSet)  # noqa: PYI018

_DataSetOrMultiBlockType = TypeVar('_DataSetOrMultiBlockType', bound=Union[DataSet, MultiBlock])  # noqa: PYI018

_DataObjectType = TypeVar('_DataObjectType', bound=DataObject)  # noqa: PYI018


# Undocumented
_PolyDataType = TypeVar('_PolyDataType', bound=PolyData)  # noqa: PYI018
_UnstructuredGridType = TypeVar('_UnstructuredGridType', bound=UnstructuredGrid)  # noqa: PYI018
