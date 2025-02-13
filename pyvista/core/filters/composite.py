"""Filters module with a class to manage filters/algorithms for composite datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable

    from pyvista import MultiBlock
    from pyvista.core._typing_core import TransformLike


@abstract_class
class CompositeFilters:
    """An internal class to manage filters/algorithms for composite datasets."""

    def _generic_filter(  # type:ignore[misc]
        self: MultiBlock,
        function: str | Callable[..., Any],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        skip_none: bool = True,
        skip_empty: bool = False,
    ) -> MultiBlock:
        """Apply any filter to all nested blocks recursively.

        This filter applies a user-specified function or method to all blocks in
        this :class:`~pyvista.MultiBlock`.

        .. versionadded:: 0.45

        Parameters
        ----------
        function : Callable | str
            Callable function or method to apply to each block.

        args : tuple, optional
            Arguments to use with the specified ``function``.

        kwargs : dict, optional
            Keyword arguments to use with the specified ``function``.

            .. note::

                If the filter includes an ``inplace`` keyword, then this ``MultiBlock``
                is also modified in-place.

        skip_none : bool, default: True
            Do not apply the filter to ``None`` blocks.

        skip_empty : bool, default: False
            Do not apply the filter to empty meshes.

        Returns
        -------
        MultiBlock
            Filtered dataset.

        Examples
        --------
        Create a :class:`~pyvista.MultiBlock` with various mesh types.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> volume = examples.load_uniform()
        >>> poly = examples.load_ant()
        >>> unstructured = examples.load_tetbeam()
        >>> multi = pv.MultiBlock([volume, poly, unstructured])

        >>> type(multi[0]), type(multi[1]), type(multi[2])
        (<class 'pyvista.core.grid.ImageData'>, <class 'pyvista.core.pointset.PolyData'>, <class 'pyvista.core.pointset.UnstructuredGrid'>)

        Use the generic filter to apply :meth:`~pyvista.DataSetFilters.cast_to_unstructured_grid`
        to all blocks.

        >>> filtered = multi._generic_filter('cast_to_unstructured_grid')

        >>> type(filtered[0]), type(filtered[1]), type(filtered[2])
        (<class 'pyvista.core.pointset.UnstructuredGrid'>, <class 'pyvista.core.pointset.UnstructuredGrid'>, <class 'pyvista.core.pointset.UnstructuredGrid'>)

        Apply the :meth:`~pyvista.DataSetFilters.explode` filter to all blocks.

        >>> filtered = multi._generic_filter('explode', kwargs=dict(factor=0.5))

        """

        def apply_filter(block_):
            func = getattr(block_, function) if isinstance(function, str) else function
            return func(**kwargs_) if args is None else func(*args, **kwargs_)

        kwargs_ = kwargs if kwargs else {}

        # Apply filter in-place
        inplace = kwargs_.get('inplace')
        if inplace:
            iterator = self.recursive_iterator('blocks', skip_none=skip_none, skip_empty=skip_empty)
            for block in iterator:
                apply_filter(block)
            return self

        # Create a copy and replace all the blocks
        output = pyvista.MultiBlock()
        output.shallow_copy(self, recursive=True)
        iterator = output.recursive_iterator(
            'all', skip_none=skip_none, skip_empty=False, nested_ids=True
        )
        for ids, _, block in iterator:  # type: ignore[misc, attr-defined]
            # Only copy the block if skip empty
            copy_block = block is not None and skip_empty and block.n_points == 0  # type: ignore[union-attr]
            new_block = block.copy() if copy_block else apply_filter(block)  # type: ignore[union-attr]
            output.replace(ids, new_block)
        return output

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

    def combine(self, merge_points: bool = False, tolerance=0.0):
        """Combine all blocks into a single unstructured grid.

        Parameters
        ----------
        merge_points : bool, default: False
            Merge coincidental points.

        tolerance : float, default: 0.0
            The absolute tolerance to use to find coincident points when
            ``merge_points=True``.

        Returns
        -------
        pyvista.UnstructuredGrid
            Combined blocks.

        Examples
        --------
        Combine blocks within a multiblock without merging points.

        >>> import pyvista as pv
        >>> block = pv.MultiBlock(
        ...     [
        ...         pv.Cube(clean=False),
        ...         pv.Cube(center=(1, 0, 0), clean=False),
        ...     ]
        ... )
        >>> merged = block.combine()
        >>> merged.n_points
        48

        Combine blocks and merge points

        >>> merged = block.combine(merge_points=True)
        >>> merged.n_points
        12

        """
        alg = _vtk.vtkAppendFilter()
        for block in self:  # type: ignore[attr-defined]
            if isinstance(block, _vtk.vtkMultiBlockDataSet):
                block = CompositeFilters.combine(
                    block,  # type: ignore[arg-type]
                    merge_points=merge_points,
                    tolerance=tolerance,
                )
            alg.AddInputData(block)
        alg.SetMergePoints(merge_points)
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

    sample = DataSetFilters.sample

    triangulate = DataSetFilters.triangulate

    def outline(  # type: ignore[misc]
        self: MultiBlock,
        generate_faces: bool = False,
        nested: bool = False,
        progress_bar: bool = False,
    ):
        """Produce an outline of the full extent for the all blocks in this composite dataset.

        Parameters
        ----------
        generate_faces : bool, default: False
            Generate solid faces for the box.

        nested : bool, default: False
            If ``True``, these creates individual outlines for each nested dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing the outline.

        """
        if nested:
            return DataSetFilters.outline(
                self,
                generate_faces=generate_faces,
                progress_bar=progress_bar,
            )
        box = pyvista.Box(bounds=self.bounds)
        return box.outline(generate_faces=generate_faces, progress_bar=progress_bar)

    def outline_corners(  # type: ignore[misc]
        self: MultiBlock, factor=0.2, nested: bool = False, progress_bar: bool = False
    ):
        """Produce an outline of the corners for the all blocks in this composite dataset.

        Parameters
        ----------
        factor : float, default: 0.2
            Controls the relative size of the corners to the length of
            the corresponding bounds.

        nested : bool, default: False
            If ``True``, these creates individual outlines for each nested dataset.

        progress_bar : bool, default: False
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
        cell_normals: bool = True,
        point_normals: bool = True,
        split_vertices: bool = False,
        flip_normals: bool = False,
        consistent_normals: bool = True,
        auto_orient_normals: bool = False,
        non_manifold_traversal: bool = True,
        feature_angle=30.0,
        track_vertices: bool = False,
        progress_bar: bool = False,
    ):
        """Compute point and/or cell normals for a multi-block dataset."""
        if not self.is_all_polydata:  # type: ignore[attr-defined]
            raise RuntimeError(
                'This multiblock contains non-PolyData datasets. Convert all the '
                'datasets to PolyData with `as_polydata`',
            )

        # track original point indices
        if split_vertices and track_vertices:
            for block in self:  # type: ignore[attr-defined]
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

    def transform(  # type:ignore[misc]
        self: MultiBlock,
        trans: TransformLike,
        transform_all_input_vectors: bool = False,
        inplace: bool | None = None,
        progress_bar: bool = False,
    ):
        """Transform all blocks in this composite dataset.

        .. note::
            See also the notes at :func:`pyvista.DataSetFilters.transform` which is
            used by this filter under the hood.

        .. deprecated:: 0.45.0
            `inplace` was previously defaulted to `True`. In the future this will change to `False`.

        Parameters
        ----------
        trans : TransformLike
            Accepts any transformation input such as a :class:`~pyvista.Transform`
            or a 3x3 or 4x4 array.

        transform_all_input_vectors : bool, default: False
            When ``True``, all arrays with three components are transformed.
            Otherwise, only the normals and vectors are transformed.

        inplace : bool, default: True
            When ``True``, modifies the dataset inplace.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.MultiBlock
            Transformed dataset. Return type of all blocks matches input unless
            input dataset is a :class:`pyvista.ImageData`, in which
            case the output datatype is a :class:`pyvista.StructuredGrid`.

        See Also
        --------
        :class:`pyvista.Transform`
            Describe linear transformations via a 4x4 matrix.

        Examples
        --------
        Translate a mesh by ``(50, 100, 200)``. Here a :class:`~pyvista.Transform` is
        used, but any :class:`~pyvista.TransformLike` is accepted.

        >>> import pyvista as pv
        >>> mesh = pv.MultiBlock([pv.Sphere(), pv.Plane()])
        >>> transform = pv.Transform().translate(50, 100, 200)
        >>> transformed = mesh.transform(transform, inplace=False)
        >>> transformed.plot(show_edges=True)

        """
        from ._deprecate_transform_inplace_default_true import check_inplace

        inplace = check_inplace(cls=type(self), inplace=inplace)

        filter_kwargs = dict(
            trans=pyvista.Transform(trans),
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
            progress_bar=progress_bar,
        )
        return self._generic_filter('transform', kwargs=filter_kwargs)
