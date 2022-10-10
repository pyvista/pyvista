"""Filters module with a class to manage filters/algorithms for composite datasets."""

import numpy as np

import pyvista
from pyvista import _vtk, abstract_class, wrap
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters


@abstract_class
class CompositeFilters:
    """An internal class to manage filters/algorithms for composite datasets."""

    def extract_geometry(self):
        """Extract the surface the geometry of all blocks.

        Place this filter at the end of a pipeline before a polydata
        consumer such as a polydata mapper to extract geometry from
        all blocks and append them to one polydata object.

        Returns
        -------
        pyvista.PolyData
            Surface of the composite dataset.

        """
        gf = _vtk.vtkCompositeDataGeometryFilter()
        gf.SetInputData(self)
        gf.Update()
        return wrap(gf.GetOutputDataObject(0))

    def combine(self, merge_points=False, tolerance=0.0):
        """Combine all blocks into a single unstructured grid.

        Parameters
        ----------
        merge_points : bool, optional
            Merge coincidental points.

        tolerance : float, optional
            The absolute tolerance to use to find coincident points when
            ``merge_points=True``. Note, this was added in VTK v9.

        Returns
        -------
        pyvista.UnstructuredGrid
            Combined blocks.

        Examples
        --------
        Combine blocks within a multiblock without merging points.

        >>> import pyvista
        >>> block = pyvista.MultiBlock([
        ...     pyvista.Cube(clean=False),
        ...     pyvista.Cube(center=(1, 0, 0), clean=False)
        ... ])
        >>> merged = block.combine()
        >>> merged.n_points
        48

        Combine blocks and merge points

        >>> merged = block.combine(merge_points=True)
        >>> merged.n_points
        12

        """
        alg = _vtk.vtkAppendFilter()
        for block in self:
            if isinstance(block, _vtk.vtkMultiBlockDataSet):
                block = CompositeFilters.combine(
                    block, merge_points=merge_points, tolerance=tolerance
                )
            alg.AddInputData(block)
        alg.SetMergePoints(merge_points)
        if pyvista.vtk_version_info >= (9,):
            alg.SetTolerance(tolerance)
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

    def outline(self, generate_faces=False, nested=False, progress_bar=False):
        """Produce an outline of the full extent for the all blocks in this composite dataset.

        Parameters
        ----------
        generate_faces : bool, optional
            Generate solid faces for the box. This is disabled by default.

        nested : bool, optional
            If ``True``, these creates individual outlines for each nested dataset.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing the outline.

        """
        if nested:
            return DataSetFilters.outline(
                self, generate_faces=generate_faces, progress_bar=progress_bar
            )
        box = pyvista.Box(bounds=self.bounds)
        return box.outline(generate_faces=generate_faces, progress_bar=progress_bar)

    def outline_corners(self, factor=0.2, nested=False, progress_bar=False):
        """Produce an outline of the corners for the all blocks in this composite dataset.

        Parameters
        ----------
        factor : float, optional
            Controls the relative size of the corners to the length of
            the corresponding bounds.

        nested : bool, optional
            If ``True``, these creates individual outlines for each nested dataset.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing outlined corners.

        """
        if nested:
            return DataSetFilters.outline_corners(self, factor=factor, progress_bar=progress_bar)
        box = pyvista.Box(bounds=self.bounds)
        return box.outline_corners(factor=factor, progress_bar=progress_bar)

    def _compute_normals(
        self,
        cell_normals=True,
        point_normals=True,
        split_vertices=False,
        flip_normals=False,
        consistent_normals=True,
        auto_orient_normals=False,
        non_manifold_traversal=True,
        feature_angle=30.0,
        track_vertices=False,
        progress_bar=False,
    ):
        """Compute point and/or cell normals for a multi-block dataset."""
        if not self.is_all_polydata:
            raise RuntimeError(
                'This multiblock contains non-PolyData datasets. Convert all the '
                'datasets to PolyData with `as_polydata`'
            )

        # track original point indices
        if split_vertices and track_vertices:
            for block in self:
                ids = np.arange(block.n_points, dtype=pyvista.ID_TYPE)
                block.point_data.set_array(ids, 'pyvistaOriginalPointIds')

        alg = _vtk.vtkPolyDataNormals()
        alg.SetComputeCellNormals(cell_normals)
        alg.SetComputePointNormals(point_normals)
        alg.SetSplitting(split_vertices)
        alg.SetFlipNormals(flip_normals)
        alg.SetConsistency(consistent_normals)
        alg.SetAutoOrientNormals(auto_orient_normals)
        alg.SetNonManifoldTraversal(non_manifold_traversal)
        alg.SetFeatureAngle(feature_angle)
        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Computing Normals')
        return _get_output(alg)
