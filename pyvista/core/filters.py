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
class CompositeFilters:
    """An internal class to manage filters/algorithms for composite datasets."""

    def extract_geometry(composite):
        """Combine the geomertry of all blocks into a single ``PolyData`` object.

        Place this filter at the end of a pipeline before a polydata
        consumer such as a polydata mapper to extract geometry from all blocks
        and append them to one polydata object.

        """
        gf = vtk.vtkCompositeDataGeometryFilter()
        gf.SetInputData(composite)
        gf.Update()
        return wrap(gf.GetOutputDataObject(0))

    def combine(composite, merge_points=False):
        """Append all blocks into a single unstructured grid.

        Parameters
        ----------
        merge_points : bool, optional
            Merge coincidental points.

        """
        alg = vtk.vtkAppendFilter()
        for block in composite:
            if isinstance(block, vtk.vtkMultiBlockDataSet):
                block = CompositeFilters.combine(block, merge_points=merge_points)
            alg.AddInputData(block)
        alg.SetMergePoints(merge_points)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))

    clip = DataSetFilters.clip

    clip_box = DataSetFilters.clip_box

    slice = DataSetFilters.slice

    slice_orthogonal = DataSetFilters.slice_orthogonal

    slice_along_axis = DataSetFilters.slice_along_axis

    slice_along_line = DataSetFilters.slice_along_line

    extract_all_edges = DataSetFilters.extract_all_edges

    elevation = DataSetFilters.elevation

    compute_cell_sizes = DataSetFilters.compute_cell_sizes

    cell_centers = DataSetFilters.cell_centers

    cell_data_to_point_data = DataSetFilters.cell_data_to_point_data

    point_data_to_cell_data = DataSetFilters.point_data_to_cell_data

    triangulate = DataSetFilters.triangulate

    def outline(composite, generate_faces=False, nested=False):
        """Produce an outline of the full extent for the all blocks in this composite dataset.

        Parameters
        ----------
        generate_faces : bool, optional
            Generate solid faces for the box. This is off by default

        nested : bool, optional
            If True, these creates individual outlines for each nested dataset

        """
        if nested:
            return DataSetFilters.outline(composite, generate_faces=generate_faces)
        box = pyvista.Box(bounds=composite.bounds)
        return box.outline(generate_faces=generate_faces)

    def outline_corners(composite, factor=0.2, nested=False):
        """Produce an outline of the corners for the all blocks in this composite dataset.

        Parameters
        ----------
        factor : float, optional
            controls the relative size of the corners to the length of the
            corresponding bounds

        nested : bool, optional
            If True, these creates individual outlines for each nested dataset

        """
        if nested:
            return DataSetFilters.outline_corners(composite, factor=factor)
        box = pyvista.Box(bounds=composite.bounds)
        return box.outline_corners(factor=factor)
