"""These classes hold methods to apply general filters to any data type.

By inheriting these classes into the wrapped VTK data structures, a user
can easily apply common filters in an intuitive manner.

Examples
--------
>>> import pyvista as pv
>>> from pyvista import examples
>>> dataset = examples.load_uniform()

>>> # Threshold
>>> thresh = dataset.threshold([100, 500])

>>> # Slice
>>> slc = dataset.slice()

>>> # Clip
>>> clp = dataset.clip(invert=True)

>>> # Contour
>>> iso = dataset.contour()

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pyvista
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.observers import ProgressMonitor

if TYPE_CHECKING:
    from pyvista.core import _vtk_core as _vtk


def _update_alg(alg, *, progress_bar: bool = False, message='') -> None:
    """Update an algorithm with or without a progress bar."""
    if progress_bar:
        with ProgressMonitor(alg, message=message):
            alg.Update()
    else:
        alg.Update()


def _get_output(
    algorithm: _vtk.vtkAlgorithm,
    *,
    iport=0,
    iconnection=0,
    oport=0,
    active_scalars=None,
    active_scalars_field='point',
):
    """Get the algorithm's output and copy input's pyvista meta info."""
    ido = cast('pyvista.DataObject', wrap(algorithm.GetInputDataObject(iport, iconnection)))
    data = cast('pyvista.DataObject', wrap(algorithm.GetOutputDataObject(oport)))
    if not isinstance(data, pyvista.MultiBlock):
        data.copy_meta_from(ido, deep=True)
        if not data.field_data and ido.field_data:
            data.field_data.update(ido.field_data)
        if active_scalars is not None:
            data.set_active_scalars(active_scalars, preference=active_scalars_field)
    # return a PointSet if input is a pointset
    if isinstance(ido, pyvista.PointSet):
        return data.cast_to_pointset()
    return data


from .composite import CompositeFilters
from .data_object import DataObjectFilters

# Re-export submodules to maintain the same import paths
# before filters.py was split into submodules
from .data_set import DataSetFilters
from .image_data import ImageDataFilters
from .poly_data import PolyDataFilters
from .rectilinear_grid import RectilinearGridFilters
from .structured_grid import StructuredGridFilters
from .unstructured_grid import UnstructuredGridFilters

__all__ = [
    'CompositeFilters',
    'DataObjectFilters',
    'DataSetFilters',
    'ImageDataFilters',
    'PolyDataFilters',
    'RectilinearGridFilters',
    'StructuredGridFilters',
    'UnstructuredGridFilters',
    '_get_output',
    '_update_alg',
]
