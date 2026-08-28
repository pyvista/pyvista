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

from typing import Any
from typing import cast

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaPrecisionWarning
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.observers import ProgressMonitor


def _update_alg(alg: _vtk.vtkAlgorithm, *, progress_bar: bool = False, message='') -> None:
    """Update an algorithm with or without a progress bar."""
    _set_output_points_precision(alg)

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


def _points_dtype(mesh: Any = None) -> np.dtype[Any] | None:
    """Return the dtype ``mesh``'s points should have, or ``None`` to leave them alone.

    ``mesh`` is the input of the algorithm being run, and is only consulted under the
    ``'preserve'`` setting. Pass nothing for an algorithm with no input, such as a
    geometry source, where ``'preserve'`` has nothing to preserve.

    ``'preserve'`` preserves the dtype of points a mesh actually stores.
    :class:`~pyvista.ImageData` and :class:`~pyvista.RectilinearGrid` store none --
    theirs are generated on demand from the origin and spacing, or from the coordinate
    arrays -- so they constrain nothing and VTK picks the output precision. Treating
    their generated double as a request would double every image pipeline's output, and
    a filter that builds one as an intermediate would silently widen an unrelated
    single-precision input.
    """
    setting = pv.global_config.points_dtype
    if setting != 'preserve':
        return np.dtype(setting)
    if not isinstance(mesh, _vtk.vtkPointSet):
        return None
    # Only the two precisions VTK stores points in are meaningful to preserve. Points
    # can be integers -- `pv.PolyData(..., force_float=False)` keeps them -- and casting
    # a filter's interpolated output back to those would truncate it.
    dtype = mesh.points.dtype
    return dtype if dtype in (np.dtype(np.float32), np.dtype(np.float64)) else None


def _set_output_points_precision(alg: _vtk.vtkAlgorithm) -> None:
    """Ask an algorithm to generate points with the configured dtype.

    Algorithms that support ``SetOutputPointsPrecision`` compute in the requested
    precision, which is both cheaper and more accurate than casting afterwards. The
    rest ignore this and are corrected by ``_enforce_points_dtype``.

    ``vtkAlgorithm.DEFAULT_PRECISION`` is *supposed* to match the input precision on
    its own, but in practice some filters do not honor it for some mesh types --
    see https://gitlab.kitware.com/vtk/vtk/-/issues/19965 -- so the input is
    inspected and the precision is requested explicitly.
    """
    set_precision = getattr(alg, 'SetOutputPointsPrecision', None)
    if set_precision is None:
        return
    mesh_in = wrap(alg.GetInputDataObject(0, 0)) if alg.GetNumberOfInputPorts() > 0 else None
    dtype = _points_dtype(mesh_in)
    if dtype == np.float64:
        set_precision(alg.DOUBLE_PRECISION)
    elif dtype == np.float32:
        set_precision(alg.SINGLE_PRECISION)


def _enforce_points_dtype(
    mesh_out: Any, dtype: np.dtype[Any] | None, *, algorithm: _vtk.vtkAlgorithm | None = None
) -> None:
    """Cast ``mesh_out``'s points to ``dtype`` in place if the algorithm ignored the request.

    Only meshes that own their points are cast. :class:`~pyvista.ImageData` and
    :class:`~pyvista.RectilinearGrid` generate their points on demand and apply the
    setting in their own ``points`` property instead.

    Casting single-precision output up to a requested ``'float64'`` fixes the dtype but
    cannot recover the digits the algorithm already discarded, so that case warns.
    Under ``'preserve'`` it does not: that setting promises a stable dtype rather than
    any particular precision, and the cast keeps that promise in full.
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
        points.dtype.itemsize < dtype.itemsize
        and points.size
        and pv.global_config.points_dtype == 'float64'
    ):
        name = 'This algorithm' if algorithm is None else type(algorithm).__name__
        msg = (
            f'{name} generated {points.dtype.name} points, and does not support the '
            f"double precision `pyvista.global_config.points_dtype = 'float64'` asks for.\n"
            f'The output points are cast to float64, but hold single-precision values.'
        )
        warn_external(msg, PyVistaPrecisionWarning)
    mesh_out.points = points.astype(dtype)


def _apply_points_dtype(mesh: Any, algorithm: _vtk.vtkAlgorithm | None = None) -> Any:
    """Apply the configured dtype to a mesh PyVista generated without a VTK algorithm."""
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
    _enforce_points_dtype(data, _points_dtype(ido), algorithm=algorithm)
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
