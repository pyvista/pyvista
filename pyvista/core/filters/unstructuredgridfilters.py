"""These classes hold methods to apply general filters to any data type.

By inheriting these classes into the wrapped VTK data structures, a user
can easily apply common filters in an intuitive manner.

Example
-------
>>> import pyvista
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
import collections.abc
import logging
from functools import wraps

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from pyvista.utilities import (FieldAssociation, NORMALS, assert_empty_kwargs,
                               generate_plane, get_array, vtk_id_list_to_array,
                               wrap, ProgressMonitor, abstract_class)
from pyvista.utilities.cells import numpy_to_idarr
from pyvista.core.errors import NotAllTrianglesError
from pyvista.utilities import transformations


def _update_alg(alg, progress_bar=False, message=''):
    """Update an algorithm with or without a progress bar."""
    if progress_bar:
        with ProgressMonitor(alg, message=message):
            alg.Update()
    else:
        alg.Update()


def _get_output(algorithm, iport=0, iconnection=0, oport=0, active_scalars=None,
                active_scalars_field='point'):
    """Get the algorithm's output and copy input's pyvista meta info."""
    ido = algorithm.GetInputDataObject(iport, iconnection)
    data = wrap(algorithm.GetOutputDataObject(oport))
    if not isinstance(data, pyvista.MultiBlock):
        data.copy_meta_from(ido)
        if not data.field_arrays and ido.field_arrays:
            data.field_arrays.update(ido.field_arrays)
        if active_scalars is not None:
            data.set_active_scalars(active_scalars, preference=active_scalars_field)
    return data


@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

    def delaunay_2d(ugrid, tol=1e-05, alpha=0.0, offset=1.0, bound=False,
                    progress_bar=False):
        """Apply a delaunay 2D filter along the best fitting plane.

        This extracts the grid's points and performs the triangulation on those alone.

        Parameters
        ----------
        progress_bar : bool, optional
            Display a progress bar to indicate progress.
        """
        return pyvista.PolyData(ugrid.points).delaunay_2d(tol=tol, alpha=alpha,
                                                          offset=offset,
                                                          bound=bound,
                                                          progress_bar=progress_bar)
