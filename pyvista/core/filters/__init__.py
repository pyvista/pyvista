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

import contextlib
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista._warn_external import warn_external
from pyvista.core.errors import PrecisionWarning
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.observers import ProgressMonitor

if TYPE_CHECKING:
    from collections.abc import Iterator


def _update_alg(alg: _vtk.vtkAlgorithm, *, progress_bar: bool = False, message='') -> None:
    """Update an algorithm with or without a progress bar."""
    # Get the status of the alg update using GetExecutive
    # https://discourse.vtk.org/t/changing-vtkalgorithm-update-return-type-from-void-to-bool/16164
    if pv.vtk_version_info >= (9, 7):
        to_be_updated: Any = alg
    else:
        try:
            to_be_updated = alg.GetExecutive()
        except AttributeError:
            # Some PyVista classes aren't true vtkAlgorithm types and don't implement GetExecutive
            to_be_updated = alg

    # Do the update
    with _requested_points_precision(alg):
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


DEFAULT_PRECISION = _vtk.vtkAlgorithm.DEFAULT_PRECISION
SINGLE_PRECISION = _vtk.vtkAlgorithm.SINGLE_PRECISION
DOUBLE_PRECISION = _vtk.vtkAlgorithm.DOUBLE_PRECISION


def _points_dtype(mesh: Any = None) -> np.dtype[Any] | None:
    """Return the dtype ``mesh``'s points should have, or ``None`` to leave them alone.

    ``mesh`` is the input of the algorithm being run, and is only consulted under the
    ``'preserve'`` setting. Pass nothing for an algorithm with no input, such as a
    geometry source, where ``'preserve'`` has nothing to preserve.

    See :attr:`pyvista.core.config.Config.points_dtype` for what each setting means.
    ``None``, the default, constrains nothing at all. ``'preserve'`` preserves the dtype
    of points a mesh actually stores, so a mesh that generates or lacks them constrains
    nothing either.
    """
    setting = pv.global_config.points_dtype
    if setting is None:
        return None
    if setting != 'preserve':
        return np.dtype(setting)
    # `DataSet.points` installs an empty float64 array when the mesh has no `vtkPoints`,
    # so ask VTK directly: reading the property would report a dtype nobody chose, and
    # modify the input as a side effect of a read.
    if not isinstance(mesh, _vtk.vtkPointSet) or mesh.GetPoints() is None:
        return None
    # Only the two precisions VTK stores points in are meaningful to preserve. Points
    # can be integers -- `pv.PolyData(..., force_float=False)` keeps them -- and casting
    # a filter's interpolated output back to those would truncate it.
    dtype = mesh.points.dtype
    return dtype if dtype in (np.dtype(np.float32), np.dtype(np.float64)) else None


@contextlib.contextmanager
def _requested_points_precision(alg: _vtk.vtkAlgorithm) -> Iterator[None]:
    """Ask an algorithm for the configured points dtype for the duration of an update.

    Only an explicit ``'float32'`` or ``'float64'`` is requested, and the algorithm's
    own setting is restored afterwards so a temporary global does not outlive itself on
    a source the caller configured. Under ``'preserve'`` nothing is asked for: VTK's
    default already matches the input, and asking anyway is not free -- for
    :vtk:`vtkTransformFilter` the request widens the data arrays it transforms as well
    as the points. Algorithms that ignore what they are asked for are corrected by
    ``_enforce_points_dtype`` instead.
    """
    dtype = _points_dtype()
    if dtype is None:
        yield
        return
    set_precision = getattr(alg, 'SetOutputPointsPrecision', None)
    get_precision = getattr(alg, 'GetOutputPointsPrecision', None)
    if set_precision is None or get_precision is None:
        yield
        return
    previous = get_precision()
    set_precision(DOUBLE_PRECISION if dtype == np.float64 else SINGLE_PRECISION)
    try:
        yield
    finally:
        set_precision(previous)


def _enforce_points_dtype(
    mesh_out: Any, dtype: np.dtype[Any] | None, *, algorithm: _vtk.vtkAlgorithm | None = None
) -> None:
    """Cast ``mesh_out``'s points to ``dtype`` in place if the algorithm ignored the request.

    Only meshes that own their points are cast; the rest apply the setting in their own
    ``points`` property. Casting narrow output up to a wider dtype fixes the dtype but
    cannot recover the digits the algorithm already discarded, so every upward cast
    warns. Casting the other way does not: discarding digits below the input's own
    representation error loses nothing that was there.
    """
    if dtype is None:
        return
    if isinstance(mesh_out, pv.MultiBlock):
        for block in mesh_out.recursive_iterator(skip_none=True):
            _enforce_points_dtype(block, dtype, algorithm=algorithm)
        return
    if not isinstance(mesh_out, _vtk.vtkPointSet):
        return
    points = mesh_out.points
    if points.dtype == dtype:
        return
    if (
        algorithm is not None
        and np.issubdtype(points.dtype, np.floating)
        and points.dtype.itemsize < dtype.itemsize
        and points.size
    ):
        # Widening fabricates precision the algorithm already discarded. Narrowing does
        # not, and neither does packaging a caller's own array, which has no algorithm.
        msg = (
            f'{type(algorithm).__name__} generated {points.dtype.name} points, and '
            f'cannot generate the {dtype.name} that '
            f'`pyvista.global_config.points_dtype = '
            f'{pv.global_config.points_dtype!r}` requires here.\n'
            f'The output points are cast to {dtype.name}, but hold '
            f'{points.dtype.name} values.'
        )
        warn_external(msg, PrecisionWarning)
    mesh_out.points = points.astype(dtype)


def _match_points_dtype(
    mesh_out: Any, mesh_in: Any, *, algorithm: _vtk.vtkAlgorithm | None = None
) -> None:
    """Give ``mesh_out`` the dtype the setting asks for, given the algorithm's input.

    Composites are paired block for block where their structure corresponds, so
    ``'preserve'`` preserves each block's own dtype rather than the whole composite's.
    """
    if pv.global_config.points_dtype != 'preserve':
        _enforce_points_dtype(mesh_out, _points_dtype(), algorithm=algorithm)
        return
    if isinstance(mesh_out, pv.MultiBlock) and isinstance(mesh_in, pv.MultiBlock):
        blocks_out = list(mesh_out.recursive_iterator(skip_none=True))
        blocks_in = list(mesh_in.recursive_iterator(skip_none=True))
        if len(blocks_out) == len(blocks_in):
            for block_out, block_in in zip(blocks_out, blocks_in, strict=True):
                _enforce_points_dtype(block_out, _points_dtype(block_in), algorithm=algorithm)
            return
    _enforce_points_dtype(mesh_out, _points_dtype(mesh_in), algorithm=algorithm)


def _apply_points_dtype(mesh: Any, *, algorithm: _vtk.vtkAlgorithm | None = None) -> Any:
    """Apply the configured dtype to a mesh wrapped without ``_get_output``."""
    _enforce_points_dtype(mesh, _points_dtype(), algorithm=algorithm)
    return mesh


def _get_output(
    algorithm: _vtk.vtkAlgorithm,
    *,
    iport=0,
    iconnection=0,
    oport=0,
    active_scalars=None,
    active_scalars_field='point',
    keep_pointset=True,
):
    """Get the algorithm's output and copy input's pyvista meta info."""
    ido = cast('pv.DataObject', wrap(algorithm.GetInputDataObject(iport, iconnection)))
    data = cast('pv.DataObject', wrap(algorithm.GetOutputDataObject(oport)))
    _match_points_dtype(data, ido, algorithm=algorithm)
    if not isinstance(data, pv.MultiBlock):
        data.copy_meta_from(ido, deep=True)
        if not data.field_data and ido.field_data:
            data.field_data.update(ido.field_data)
        if active_scalars is not None:
            data.set_active_scalars(active_scalars, preference=active_scalars_field)
    # return a PointSet if input is a pointset, unless the algorithm generates
    # cells (e.g. glyph), in which case flattening to a PointSet would drop them
    if keep_pointset and isinstance(ido, pv.PointSet):
        return data.cast_to_pointset()
    return data


from .composite import CompositeFilters as CompositeFilters  # noqa: E402
from .data_object import DataObjectFilters as DataObjectFilters  # noqa: E402
from .data_set import DataSetFilters as DataSetFilters  # noqa: E402
from .image_data import ImageDataFilters as ImageDataFilters  # noqa: E402
from .poly_data import PolyDataFilters as PolyDataFilters  # noqa: E402
from .rectilinear_grid import RectilinearGridFilters as RectilinearGridFilters  # noqa: E402
from .structured_grid import StructuredGridFilters as StructuredGridFilters  # noqa: E402
from .unstructured_grid import UnstructuredGridFilters as UnstructuredGridFilters  # noqa: E402
