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
from typing import Any
from typing import cast

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.observers import ProgressMonitor

if TYPE_CHECKING:
    from pyvista.core import _vtk_core as _vtk


def _update_alg(alg: _vtk.vtkAlgorithm, *, progress_bar: bool = False, message='') -> None:
    """Update an algorithm with or without a progress bar."""
    # Get the status of the alg update using GetExecutive
    # https://discourse.vtk.org/t/changing-vtkalgorithm-update-return-type-from-void-to-bool/16164
    if pv.vtk_version_info >= (9, 6, 99):  # >= 9.7.0
        to_be_updated: Any = alg
    else:
        try:
            to_be_updated = alg.GetExecutive()
        except AttributeError:
            # Some PyVista classes aren't true vtkAlgorithm types and don't implement GetExecutive
            to_be_updated = alg

    # Do the update
    if progress_bar:
        with ProgressMonitor(alg, message=message):
            status = to_be_updated.Update()
    else:
        status = to_be_updated.Update()

    if status is not None and status == 0:
        # There was an error with the update. Re-run so we can catch it and
        # raise it as a proper Python error.
        # We avoid using VtkErrorCatcher for the initial update because adding and tracking
        # with VTK observers can be slow.
        with pv.VtkErrorCatcher(raise_errors=True, emit_warnings=True):
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
    ido = cast('pv.DataObject', wrap(algorithm.GetInputDataObject(iport, iconnection)))
    data = cast('pv.DataObject', wrap(algorithm.GetOutputDataObject(oport)))
    if not isinstance(data, pv.MultiBlock):
        data.copy_meta_from(ido, deep=True)
        if not data.field_data and ido.field_data:
            data.field_data.update(ido.field_data)
        if active_scalars is not None:
            data.set_active_scalars(active_scalars, preference=active_scalars_field)
    # return a PointSet if input is a pointset
    if isinstance(ido, pv.PointSet):
        return data.cast_to_pointset()

    # Warn if the alg modified points dtype
    points_in = getattr(ido, 'points', None)
    if points_in is not None:
        points_out = getattr(data, 'points', None)
        if points_out is not None and points_in.dtype != points_out.dtype:
            msg = (
                f'The points dtype of {ido.__class__.__name__} '
                f'was modified by {algorithm.__class__.__name__}.\n'
                f'Input dtype: {points_in.dtype.name!r}, output dtype: {points_out.dtype.name!r}.'
            )
            warn_external(msg, RuntimeWarning)
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
