"""Filters module with a class to manage filters/algorithms for composite datasets."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_object import DataObjectFilters
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista import MultiBlock
    from pyvista.core.composite import _TypeMultiBlockLeaf


@abstract_class
class CompositeFilters(DataObjectFilters):
    """An internal class to manage filters/algorithms for composite datasets."""

    def generic_filter(  # type:ignore[misc]
        self: MultiBlock,
        function: str | Callable[..., _TypeMultiBlockLeaf],
        /,
        *args,
        **kwargs,
    ) -> MultiBlock:
        """Apply any filter to all nested blocks recursively.

        This filter applies a user-specified function or method to all blocks in
        this :class:`~pyvista.MultiBlock`.

        .. note::

            If an ``inplace`` keyword is used, this ``MultiBlock`` is modified
            in-place along with all blocks.

        .. note::

            By default, the specified ``function`` is not applied to any ``None``
            blocks. These are simply skipped and passed through to the output.

            For advanced use, it is possible to apply the filter to ``None`` blocks
            by using the undocumented keyword ``_skip_none=False``.

        .. versionadded:: 0.45

        Parameters
        ----------
        function : Callable | str
            Callable function or name of the method to apply to each block. The function
            should accept a :class:`~pyvista.DataSet` as input and return either a
            :class:`~pyvista.DataSet` or :class:`~pyvista.MultiBlock` as output.

        *args : Any, optional
            Arguments to use with the specified ``function``.

        **kwargs : Any, optional
            Keyword arguments to use with the specified ``function``.

        Returns
        -------
        MultiBlock
            Filtered dataset.

        Raises
        ------
        RuntimeError
            Raised if the filter cannot be applied to any block for any reason. This
            overrides ``TypeError``, ``ValueError``, ``AttributeError`` errors when
            filtering.

        See Also
        --------
        pyvista.MultiBlock.flatten
        pyvista.MultiBlock.recursive_iterator
        pyvista.MultiBlock.clean

        Examples
        --------
        Create a :class:`~pyvista.MultiBlock` with various mesh types.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> import numpy as np
        >>> volume = examples.load_uniform()
        >>> poly = examples.load_ant()
        >>> unstructured = examples.load_tetbeam()
        >>> multi = pv.MultiBlock([volume, poly, unstructured])

        >>> [type(block) for block in multi]  # doctest: +NORMALIZE_WHITESPACE
        [<class 'pyvista.core.grid.ImageData'>,
         <class 'pyvista.core.pointset.PolyData'>,
         <class 'pyvista.core.pointset.UnstructuredGrid'>]

        Use the generic filter to apply :meth:`~pyvista.DataSet.cast_to_unstructured_grid`
        to all blocks.

        >>> filtered = multi.generic_filter('cast_to_unstructured_grid')

        >>> [type(block) for block in filtered]  # doctest: +NORMALIZE_WHITESPACE
        [<class 'pyvista.core.pointset.UnstructuredGrid'>,
         <class 'pyvista.core.pointset.UnstructuredGrid'>,
         <class 'pyvista.core.pointset.UnstructuredGrid'>]

        Use the :meth:`~pyvista.DataSetFilters.partition` filter on all blocks.
        Any arguments can be specified as though the filter is being used directly.

        >>> filtered = multi.generic_filter('partition', 4, as_composite=True)

        Any function can be used as long as it returns a :class:`~pyvista.DataSet` or
        :class:`~pyvista.MultiBlock`. For example, we can normalize each block
        independently to have bounds between ``-0.5`` and ``0.5``.

        >>> def normalize_bounds(dataset):
        ...     # Center the dataset
        ...     dataset = dataset.translate(-np.array(dataset.center))
        ...     # Scale the dataset
        ...     factor = 1 / np.array(dataset.bounds_size)
        ...     return dataset.scale(factor)

        >>> filtered = multi.generic_filter(normalize_bounds)
        >>> filtered
        MultiBlock (...)
          N Blocks:   3
          X Bounds:   -5.000e-01, 5.000e-01
          Y Bounds:   -5.000e-01, 5.000e-01
          Z Bounds:   -5.000e-01, 5.000e-01

        The generic filter will fail if the filter can only be applied to some blocks
        but not others. For example, it is not possible to use the
        :meth:`~pyvista.ImageDataFilters.resample` filter generically since the
        ``MultiBlock`` above is heterogeneous and contains some blocks which are not
        :class:`~pyvista.ImageData`.

        >>> multi.generic_filter('resample', 0.5)  # doctest:+SKIP
        RuntimeError: The filter 'resample' could not be applied to the block at index 1 with
        name 'Block-01' and type PolyData.

        Use a custom function instead to apply the generic filter conditionally. Here we
        filter the image blocks but simply pass-through a copy of any other blocks.

        >>> def conditional_resample(dataset, *args, **kwargs):
        ...     if isinstance(dataset, pv.ImageData):
        ...         return dataset.resample(*args, **kwargs)
        ...     return dataset.copy()

        >>> filtered = multi.generic_filter(conditional_resample, 0.5)

        """
        # Set default undocumented kwargs. A function is used here to prevent IDEs from
        # suggesting these keywords to users.

        def get_iterator_kwargs(kwargs_) -> tuple[bool, bool]:
            # Skip None blocks by default
            skip_none_: bool = kwargs_.pop('_skip_none', True)
            # Do not skip empty blocks by default
            skip_empty_: bool = kwargs_.pop('_skip_empty', False)
            return skip_none_, skip_empty_

        skip_none, skip_empty = get_iterator_kwargs(kwargs)

        def apply_filter(function_, ids_, name_, block_):  # noqa: PLR0917
            try:
                function_ = (
                    getattr(block_, function_)
                    if isinstance(function_, str)
                    else functools.partial(function_, block_)
                )
                output_ = function_(**kwargs) if len(args) == 0 else function_(*args, **kwargs)
            except (AttributeError, ValueError, TypeError, RuntimeError) as e:
                # Construct a helpful error message
                func_name = (
                    function_.func if isinstance(function_, functools.partial) else function_
                )
                obj_name = type(block).__name__
                if len(ids_) == 1:
                    index = ids_[0]
                    nested = ' '
                else:
                    nested = ' nested '
                    index = _format_nested_index(ids)
                msg = (
                    f"The filter '{func_name}'\n"
                    f'could not be applied to the{nested}block at index {index} with '
                    f"name '{name_}' and type {obj_name}."
                )
                raise RuntimeError(msg) from e
            return output_

        def get_iterator(multi, skip_none_, skip_empty_):
            return multi.recursive_iterator(
                'all', skip_none=skip_none_, skip_empty=skip_empty_, nested_ids=True
            )

        # Apply filter in-place
        inplace = kwargs.get('inplace')
        if inplace:
            for ids, name, block in get_iterator(self, skip_none, skip_empty):
                apply_filter(function, ids, name, block)
            return self

        # Create a copy and replace all the blocks
        output = pv.MultiBlock()
        output.shallow_copy(self, recursive=True)
        for ids, name, block in get_iterator(output, skip_none, skip_empty):
            filtered = apply_filter(function, ids, name, block)
            # Only replace if necessary
            if filtered is not block:
                output.replace(ids, filtered)
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
        msg = '`extract_geometry` is deprecated. Use `extract_surface` instead.'
        warn_external(msg, PyVistaDeprecationWarning)
        if pv.version_info >= (0, 50):  # pragma: no cover
            msg = 'Convert this deprecation warning into an error.'
            raise RuntimeError(msg)
        if pv.version_info >= (0, 51):  # pragma: no cover
            msg = 'Remove this deprecated filter.'
            raise RuntimeError(msg)
        return self._composite_geometry_filter()

    def _composite_geometry_filter(self):
        # NOTE: Internally, this filter uses `vtkDataSetSurfaceFilter` on all leaf nodes,
        # It does not use `vtkGeometryFilter`.
        gf = _vtk.vtkCompositeDataGeometryFilter()
        gf.SetInputData(self)
        gf.Update()
        return wrap(gf.GetOutputDataObject(0))

    @_deprecate_positional_args
    def combine(self, merge_points: bool = False, tolerance=0.0):  # noqa: FBT001, FBT002
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
            single_block = (
                CompositeFilters.combine(
                    block,  # type: ignore[arg-type]
                    merge_points=merge_points,
                    tolerance=tolerance,
                )
                if isinstance(block, _vtk.vtkMultiBlockDataSet)
                else block
            )
            alg.AddInputData(single_block)
        alg.SetMergePoints(merge_points)
        alg.SetTolerance(tolerance)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))

    @_deprecate_positional_args
    def outline(  # type: ignore[misc]
        self: MultiBlock,
        generate_faces: bool = False,  # noqa: FBT001, FBT002
        nested: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
        box = pv.Box(bounds=self.bounds)
        return box.outline(generate_faces=generate_faces, progress_bar=progress_bar)

    @_deprecate_positional_args
    def outline_corners(  # type: ignore[misc]
        self: MultiBlock,
        factor=0.2,
        nested: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
        box = pv.Box(bounds=self.bounds)
        return box.outline_corners(factor=factor, progress_bar=progress_bar)

    @_deprecate_positional_args
    def _compute_normals(  # noqa: PLR0917
        self,
        cell_normals: bool = True,  # noqa: FBT001, FBT002
        point_normals: bool = True,  # noqa: FBT001, FBT002
        split_vertices: bool = False,  # noqa: FBT001, FBT002
        flip_normals: bool = False,  # noqa: FBT001, FBT002
        consistent_normals: bool = True,  # noqa: FBT001, FBT002
        auto_orient_normals: bool = False,  # noqa: FBT001, FBT002
        non_manifold_traversal: bool = True,  # noqa: FBT001, FBT002
        feature_angle=30.0,
        track_vertices: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Compute point and/or cell normals for a multi-block dataset."""
        if not self.is_all_polydata:  # type: ignore[attr-defined]
            msg = (
                'This multiblock contains non-PolyData datasets. Convert all the '
                'datasets to PolyData with `as_polydata`'
            )
            raise RuntimeError(msg)

        # track original point indices
        if split_vertices and track_vertices:
            for block in self:  # type: ignore[attr-defined]
                ids = np.arange(block.n_points, dtype=pv.ID_TYPE)
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
        _update_alg(alg, progress_bar=progress_bar, message='Computing Normals')
        return _get_output(alg)


def _format_nested_index(index: tuple[int, ...]) -> str:
    return ''.join([f'[{ind}]' for ind in index])
