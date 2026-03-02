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

import numpy as np

import pyvista as pv
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.observers import ProgressMonitor

if TYPE_CHECKING:
    from pyvista.core import _vtk_core as _vtk


def _update_alg(alg: _vtk.vtkAlgorithm, *, progress_bar: bool = False, message='') -> None:
    """Update an algorithm with or without a progress bar."""
    # Try to set output precision to match input if the filter supports it.
    # This should not really be necessary since vtkAlgorithm.DEFAULT_PRECISION
    # is *supposed* to handle this automatically, but in practice some filters
    # do not honor this for some mesh types, e.g. https://gitlab.kitware.com/vtk/vtk/-/issues/19965
    # so we need to explicitly set the output points precision.
    if (precision := pv.POINTS_PRECISION) is not None and (
        set_precision := getattr(alg, 'SetOutputPointsPrecision', None)
    ) is not None:
        if precision == np.single:
            set_precision(alg.SINGLE_PRECISION)
        elif precision == np.double:
            set_precision(alg.DOUBLE_PRECISION)
        elif alg.GetNumberOfInputPorts() > 0:
            # default
            alg_input = cast('pv.DataObject', wrap(alg.GetInputDataObject(0, 0)))
            points = getattr(alg_input, 'points', None)
            if points is not None and points.dtype == np.double:
                set_precision(alg.DOUBLE_PRECISION)

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
    points_dtype=None,
):
    """Get the algorithm's output and copy input's pyvista meta info."""
    ido = cast('pv.DataObject', wrap(algorithm.GetInputDataObject(iport, iconnection)))
    data = cast('pv.DataObject', wrap(algorithm.GetOutputDataObject(oport)))
    _check_output_points_precision(ido, data, points_dtype=points_dtype, algorithm=algorithm)
    if not isinstance(data, pv.MultiBlock):
        data.copy_meta_from(ido, deep=True)
        if not data.field_data and ido.field_data:
            data.field_data.update(ido.field_data)
        if active_scalars is not None:
            data.set_active_scalars(active_scalars, preference=active_scalars_field)
    # return a PointSet if input is a pointset
    if isinstance(ido, pv.PointSet):
        return data.cast_to_pointset()
    return data


def _check_output_points_precision(mesh_in, mesh_out, *, points_dtype, algorithm):
    precision = points_dtype if points_dtype is not None else pv.POINTS_PRECISION
    if precision is not None:
        points_in = getattr(mesh_in, 'points', None)
        if points_in is not None:
            points_out = getattr(mesh_out, 'points', None)
            requires_double = precision == np.double or (
                precision == 'default' and points_in.dtype == np.double
            )
            if requires_double and points_out.dtype != np.double:
                # Handle edge case with no points
                if points_out.size == 0:
                    mesh_out.points_to_double()
                    return

                msg = (
                    f'{algorithm.__class__.__name__} did not generate '
                    f'points with double precision.\n'
                    f'Input {mesh_in.__class__.__name__} points dtype is {points_in.dtype.name}, '
                    f'output {mesh_out.__class__.__name__} points dtype is {points_out.dtype.name}.'
                )
                if points_in.dtype != np.double:
                    msg += (
                        '\nTry converting the input to double precision first '
                        'with `points_to_double`.'
                    )
                elif points_out.dtype != np.double:
                    if precision == np.double:
                        msg += (
                            '\npyvista.POINTS_PRECISION cannot be double for '
                            'this filter and mesh type.'
                        )
                    else:
                        msg += (
                            '\nTry converting the input to single precision first '
                            'with `points_to_single`.'
                        )
                raise ValueError(msg)
            requires_single = precision == np.single or (
                precision == 'default' and points_in.dtype == np.single
            )
            if requires_single:
                mesh_out.points_to_single()


from .composite import CompositeFilters as CompositeFilters
from .data_object import DataObjectFilters as DataObjectFilters

# Re-export submodules to maintain the same import paths
# before filters.py was split into submodules
from .data_set import DataSetFilters as DataSetFilters
from .image_data import ImageDataFilters as ImageDataFilters
from .poly_data import PolyDataFilters as PolyDataFilters
from .rectilinear_grid import RectilinearGridFilters as RectilinearGridFilters
from .structured_grid import StructuredGridFilters as StructuredGridFilters
from .unstructured_grid import UnstructuredGridFilters as UnstructuredGridFilters
