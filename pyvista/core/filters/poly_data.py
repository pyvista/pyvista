"""Filters module with a class to manage filters/algorithms for polydata datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import cast
import warnings

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import MissingDataError
from pyvista.core.errors import NotAllTrianglesError
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import PyVistaFutureWarning
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities.arrays import FieldAssociation
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import get_array_association
from pyvista.core.utilities.arrays import set_default_active_scalars
from pyvista.core.utilities.arrays import vtk_id_list_to_array
from pyvista.core.utilities.geometric_objects import NORMALS
from pyvista.core.utilities.helpers import generate_plane
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.misc import assert_empty_kwargs

if TYPE_CHECKING:
    from pyvista import PolyData
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike
    from pyvista.core._typing_core._dataset_types import _PolyDataType


@abstract_class
class PolyDataFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for polydata datasets."""

    @_deprecate_positional_args(allowed=['angle'])
    def edge_mask(self, angle, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Return a mask of the points of a surface mesh with a surface angle greater than angle.

        Parameters
        ----------
        angle : float
            Angle to consider an edge.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        numpy.ndarray
            Mask of points with an angle greater than ``angle``.

        Examples
        --------
        Plot the mask of points that exceed 45 degrees.

        >>> import pyvista as pv
        >>> mesh = pv.Cube().triangulate().subdivide(4)
        >>> mask = mesh.edge_mask(45)
        >>> mesh.plot(scalars=mask)

        Show the array of masked points.

        >>> mask  # doctest:+SKIP
        array([ True,  True,  True, ..., False, False, False])

        """
        poly_data = self
        if not isinstance(poly_data, pyvista.PolyData):  # pragma: no cover
            poly_data = pyvista.PolyData(poly_data)  # type: ignore[arg-type]
        poly_data.point_data['point_ind'] = np.arange(poly_data.n_points)
        featureEdges = _vtk.vtkFeatureEdges()
        featureEdges.SetInputData(poly_data)
        featureEdges.FeatureEdgesOn()
        featureEdges.BoundaryEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.SetFeatureAngle(angle)
        _update_alg(featureEdges, progress_bar=progress_bar, message='Computing Edges')
        edges = _get_output(featureEdges)
        orig_id = cast('NumpyArray[float]', pyvista.point_array(edges, 'point_ind'))

        return np.isin(poly_data.point_data['point_ind'], orig_id, assume_unique=True)

    def _boolean(self, btype, other_mesh, *, tolerance, progress_bar: bool = False):
        """Perform boolean operation."""
        if self.n_points == other_mesh.n_points and np.allclose(self.points, other_mesh.points):  # type: ignore[attr-defined]
            msg = (
                'The input mesh contains identical points to the surface being operated on. '
                'Unable to perform boolean operations on an identical surface.'
            )
            raise ValueError(msg)
        if not isinstance(other_mesh, pyvista.PolyData):
            msg = 'Input mesh must be PolyData.'
            raise TypeError(msg)
        if not self.is_all_triangles or not other_mesh.is_all_triangles:  # type: ignore[attr-defined]
            msg = 'Make sure both the input and output are triangulated.'
            raise NotAllTrianglesError(msg)

        bfilter = _vtk.vtkBooleanOperationPolyDataFilter()
        if btype == 'union':
            bfilter.SetOperationToUnion()
        elif btype == 'intersection':
            bfilter.SetOperationToIntersection()
        elif btype == 'difference':
            bfilter.SetOperationToDifference()
        else:  # pragma: no cover
            msg = f'Invalid btype {btype}'
            raise ValueError(msg)
        bfilter.SetInputData(0, self)
        bfilter.SetInputData(1, other_mesh)
        bfilter.ReorientDifferenceCellsOn()  # this is already default
        bfilter.SetTolerance(tolerance)
        _update_alg(bfilter, progress_bar=progress_bar, message='Performing Boolean Operation')

        return _get_output(bfilter)

    @_deprecate_positional_args(allowed=['other_mesh'])
    def boolean_union(self, other_mesh, tolerance=1e-5, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Perform a boolean union operation on two meshes.

        Essentially, boolean union, difference, and intersection are
        all the same operation. Just different parts of the objects
        are kept at the end.

        The union of two manifold meshes ``A`` and ``B`` is the mesh
        which is in ``A``, in ``B``, or in both ``A`` and ``B``.

        .. note::
           If your boolean operations don't react the way you think they
           should (i.e. the wrong parts disappear), one of your meshes
           probably has its normals pointing inward. Use
           :func:`PolyDataFilters.plot_normals` to visualize the
           normals.

        .. note::
           The behavior of this filter varies from the
           :func:`PolyDataFilters.merge` filter.  This filter attempts
           to create a manifold mesh and will not include internal
           surfaces when two meshes overlap.

        .. note::
           Both meshes must be composed of all triangles.  Check with
           :attr:`PolyData.is_all_triangles` and convert with
           :func:`PolyDataFilters.triangulate`.

        .. versionchanged:: 0.32.0
           Behavior changed to match default VTK behavior.

        .. versionchanged:: 0.45.0
           Define ``|`` operator to perform boolean union.

        Parameters
        ----------
        other_mesh : pyvista.PolyData
            Mesh operating on the source mesh.

        tolerance : float, tolerance: 1e-5
            Tolerance used to determine when a point's absolute
            distance is considered to be zero.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The result of the boolean operation.

        Examples
        --------
        Demonstrate a boolean union with two spheres.  Note how the
        final mesh includes both spheres.

        >>> import pyvista as pv
        >>> sphere_a = pv.Sphere()
        >>> sphere_b = pv.Sphere(center=(0.5, 0, 0))
        >>> result = sphere_a | sphere_b
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere_a, color='r', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(sphere_b, color='b', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(result, color='lightblue')
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        See :ref:`boolean_operations_example` for more examples using this filter.

        """
        return self._boolean('union', other_mesh, tolerance=tolerance, progress_bar=progress_bar)

    @_deprecate_positional_args(allowed=['other_mesh'])
    def boolean_intersection(self, other_mesh, tolerance=1e-5, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Perform a boolean intersection operation on two meshes.

        Essentially, boolean union, difference, and intersection are
        all the same operation. Just different parts of the objects
        are kept at the end.

        The intersection of two manifold meshes ``A`` and ``B`` is the mesh
        which is the volume of ``A`` that is also in ``B``.

        .. note::
           If your boolean operations don't react the way you think they
           should (i.e. the wrong parts disappear), one of your meshes
           probably has its normals pointing inward. Use
           :func:`PolyDataFilters.plot_normals` to visualize the
           normals.

        .. note::
           This method returns the "volume" intersection between two
           meshes whereas the :func:`PolyDataFilters.intersection`
           filter returns the surface intersection between two meshes
           (which often resolves as a line).


        .. note::
           Both meshes must be composed of all triangles.  Check with
           :attr:`PolyData.is_all_triangles` and convert with
           :func:`PolyDataFilters.triangulate`.

        .. versionadded:: 0.32.0

        .. versionchanged:: 0.45.0
           Define ``&`` operator to perform boolean intersection.

        Parameters
        ----------
        other_mesh : pyvista.PolyData
            Mesh operating on the source mesh.

        tolerance : float, default: 1e-5
            Tolerance used to determine when a point's absolute
            distance is considered to be zero.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The result of the boolean operation.

        Examples
        --------
        Demonstrate a boolean intersection with two spheres.  Note how
        the final mesh only includes the intersection of the two.

        >>> import pyvista as pv
        >>> sphere_a = pv.Sphere()
        >>> sphere_b = pv.Sphere(center=(0.5, 0, 0))
        >>> result = sphere_a & sphere_b
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere_a, color='r', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(sphere_b, color='b', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(result, color='lightblue')
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        See :ref:`boolean_operations_example` for more examples using this filter.

        """
        bool_inter = self._boolean(
            'intersection', other_mesh, tolerance=tolerance, progress_bar=progress_bar
        )

        # check if a polydata is completely contained within another
        if bool_inter.is_empty:
            inter, s1, s2 = self.intersection(other_mesh)
            if inter.is_empty and s1.is_empty and s2.is_empty:
                warnings.warn(
                    'Unable to compute boolean intersection when one PolyData is '
                    'contained within another and no faces intersect.',
                )
        return bool_inter

    @_deprecate_positional_args(allowed=['other_mesh'])
    def boolean_difference(self, other_mesh, tolerance=1e-5, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Perform a boolean difference operation between two meshes.

        Essentially, boolean union, difference, and intersection are
        all the same operation. Just different parts of the objects
        are kept at the end.

        The difference of two manifold meshes ``A`` and ``B`` is the
        volume of the mesh in ``A`` not belonging to ``B``.

        .. note::
           If your boolean operations don't react the way you think they
           should (i.e. the wrong parts disappear), one of your meshes
           probably has its normals pointing inward. Use
           :func:`PolyDataFilters.plot_normals` to visualize the
           normals.

        .. note::
           Both meshes must be composed of all triangles.  Check with
           :attr:`PolyData.is_all_triangles` and convert with
           :func:`PolyDataFilters.triangulate`.

        .. versionchanged:: 0.32.0
           Behavior changed to match default VTK behavior.

        Parameters
        ----------
        other_mesh : pyvista.PolyData
            Mesh operating on the source mesh.

        tolerance : float, default: 1e-5
            Tolerance used to determine when a point's absolute
            distance is considered to be zero.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The result of the boolean operation.

        Examples
        --------
        Demonstrate a boolean difference with two spheres.  Note how
        the final mesh only includes ``sphere_a``.

        >>> import pyvista as pv
        >>> sphere_a = pv.Sphere()
        >>> sphere_b = pv.Sphere(center=(0.5, 0, 0))
        >>> result = sphere_a - sphere_b
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere_a, color='r', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(sphere_b, color='b', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(result, color='lightblue')
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        See :ref:`boolean_operations_example` for more examples using this filter.

        """
        return self._boolean(
            'difference', other_mesh, tolerance=tolerance, progress_bar=progress_bar
        )

    def __add__(self: PolyData, dataset):  # type: ignore[misc]
        """Merge these two meshes."""
        return self.merge(dataset)

    def __iadd__(self: PolyData, dataset):  # type: ignore[misc]
        """Merge another mesh into this one if possible.

        "If possible" means that ``dataset`` is also a :class:`PolyData`.
        Otherwise we have to return a :class:`pyvista.UnstructuredGrid`,
        so the in-place merge attempt will raise.

        """
        return self.merge(dataset, inplace=True)

    def append_polydata(
        self,
        *meshes,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        """Append one or more PolyData into this one.

        Under the hood, the VTK :vtk:`vtkAppendPolyData`
        filter is used to perform the append operation.

        .. versionadded:: 0.40.0

        .. note::
            As stated in the VTK documentation of :vtk:`vtkAppendPolyData`,
            point and cell data are added to the output PolyData **only** if they
            are present across **all** input PolyData.

        .. seealso::
            :func:`pyvista.PolyDataFilters.merge`

        Parameters
        ----------
        *meshes : list[pyvista.PolyData]
            The PolyData(s) to append with the current one.

        inplace : bool, default: False
            Whether to update the mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Appended PolyData(s).

        Examples
        --------
        >>> import pyvista as pv
        >>> sp0 = pv.Sphere()
        >>> sp1 = sp0.translate((1, 0, 0))
        >>> appended = sp0.append_polydata(sp1)
        >>> appended.plot()

        Append more than one PolyData.

        >>> sp2 = sp0.translate((-1, 0, 0))
        >>> appended = sp0.append_polydata(sp1, sp2)
        >>> appended.plot()

        """
        if not all(isinstance(mesh, pyvista.PolyData) for mesh in meshes):
            msg = 'All meshes need to be of PolyData type'
            raise TypeError(msg)

        append_filter = _vtk.vtkAppendPolyData()
        append_filter.AddInputData(self)
        for mesh in meshes:
            append_filter.AddInputData(mesh)

        _update_alg(append_filter, progress_bar=progress_bar, message='Append PolyData')
        merged = _get_output(append_filter)

        if inplace:
            self.deep_copy(merged)  # type: ignore[attr-defined]
            return self

        return merged

    @_deprecate_positional_args(allowed=['dataset'])
    def merge(  # type: ignore[override, misc]  # noqa: PLR0917
        self: PolyData,
        dataset,
        merge_points: bool = True,  # noqa: FBT001, FBT002
        tolerance=0.0,
        inplace: bool = False,  # noqa: FBT001, FBT002
        main_has_priority: bool | None = None,  # noqa: FBT001
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Merge this mesh with one or more datasets.

        .. note::
           The behavior of this filter varies from the
           :func:`PolyDataFilters.boolean_union` filter.  This filter
           does not attempt to create a manifold mesh and will include
           internal surfaces when two meshes overlap.

        .. note::
           The ``+`` operator between two meshes uses this filter with
           the default parameters. When the other mesh is also a
           :class:`pyvista.PolyData`, in-place merging via ``+=`` is
           similarly possible.

        .. versionchanged:: 0.39.0
            Before version ``0.39.0``, if all input datasets were of type
            :class:`pyvista.PolyData`, the VTK :vtk:`vtkAppendPolyData` and
            :vtk:`vtkCleanPolyData` filters were used to perform merging.
            Otherwise, :func:`DataSetFilters.merge`, which uses the VTK
            :vtk:`vtkAppendFilter` filter, was called.
            To enhance performance and coherence with merging operations
            available for other datasets in pyvista, the merging operation
            has been delegated in ``0.39.0`` to :func:`DataSetFilters.merge`
            only, irrespectively of input datasets types.
            This induced that points ordering can be altered compared to
            previous pyvista versions when merging only PolyData together.
            To obtain similar results as before ``0.39.0`` for multiple PolyData, combine
            :func:`PolyDataFilters.append_polydata` and :func:`PolyDataFilters.clean`.


        .. warning::

            The merge order of this filter depends on the installed version
            of VTK. For example, if merging meshes ``a``, ``b``, and ``c``,
            the merged order is ``bca`` for VTK<9.5 and ``abc`` for VTK>=9.5.
            This may be a breaking change for some applications. If only
            merging two meshes, it may be possible to maintain `some` backwards
            compatibility by swapping the input order of the two meshes,
            though this may also affect the merged arrays and is therefore
            not fully backwards-compatible.

        .. seealso::
            :func:`PolyDataFilters.append_polydata`

        Parameters
        ----------
        dataset : pyvista.DataSet
            PyVista dataset to merge this mesh with.

        merge_points : bool, optional
            Merge equivalent points when ``True``.

        tolerance : float, default: 0.0
            The absolute tolerance to use to find coincident points when
            ``merge_points=True``.

        inplace : bool, default: False
            Updates grid inplace when ``True`` if the input type is a
            :class:`pyvista.PolyData`. For other input meshes the
            result is a :class:`pyvista.UnstructuredGrid` which makes
            in-place operation impossible.

        main_has_priority : bool, optional
            When this parameter is ``True`` and ``merge_points=True``,
            the arrays of the merging grids will be overwritten
            by the original main mesh.

            .. deprecated:: 0.46

                This keyword will be removed in a future version. The main mesh
                always has priority with VTK 9.5.0 or later.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            :class:`pyvista.PolyData` if ``dataset`` is a
            :class:`pyvista.PolyData`, otherwise a
            :class:`pyvista.UnstructuredGrid`.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere_a = pv.Sphere()
        >>> sphere_b = pv.Sphere(center=(0.5, 0, 0))
        >>> merged = sphere_a.merge(sphere_b)
        >>> merged.plot(style='wireframe', color='lightblue')

        """
        # check if dataset or datasets are not polydata
        if isinstance(dataset, (list, tuple, pyvista.MultiBlock)):
            is_polydata = all(isinstance(data, pyvista.PolyData) for data in dataset)
        else:
            is_polydata = isinstance(dataset, pyvista.PolyData)

        if inplace and not is_polydata:
            msg = 'In-place merge requires both input datasets to be PolyData.'
            raise TypeError(msg)

        merged = DataSetFilters.merge(
            self,
            dataset,
            merge_points=merge_points,
            tolerance=tolerance,
            main_has_priority=main_has_priority,
            inplace=False,
            progress_bar=progress_bar,
        )

        # convert back to a polydata if both inputs were polydata
        if is_polydata:
            # if either of the input datasets contained lines or strips, we
            # must use extract_geometry to ensure they get converted back
            # correctly. This incurrs a performance penalty, but is needed to
            # maintain data consistency.
            if isinstance(dataset, (list, tuple, pyvista.MultiBlock)):
                dataset_has_lines_strips = any(
                    ds.n_lines or ds.n_strips or ds.n_verts for ds in dataset
                )
            else:
                dataset_has_lines_strips = dataset.n_lines or dataset.n_strips or dataset.n_verts

            if self.n_lines or self.n_strips or self.n_verts or dataset_has_lines_strips:
                merged = merged.extract_geometry()
            else:
                polydata_merged = pyvista.PolyData(
                    merged.points,
                    faces=merged.GetCells(),
                    deep=False,
                )
                # Calling update() will modify the active scalars and normals in this
                # specific case. Store values to restore after updating.
                active_point_scalars_name = merged.point_data.active_scalars_name
                active_cell_scalars_name = merged.cell_data.active_scalars_name
                active_point_normals_name = merged.point_data._active_normals_name
                active_cell_normals_name = merged.cell_data._active_normals_name

                polydata_merged.point_data.update(merged.point_data)
                polydata_merged.cell_data.update(merged.cell_data)
                polydata_merged.field_data.update(merged.field_data)

                # restore active scalars and normals
                polydata_merged.point_data.active_scalars_name = active_point_scalars_name
                polydata_merged.cell_data.active_scalars_name = active_cell_scalars_name
                polydata_merged.point_data._active_normals_name = active_point_normals_name
                polydata_merged.cell_data._active_normals_name = active_cell_normals_name

                merged = polydata_merged

        if inplace:
            self.deep_copy(merged)
            return self

        return merged

    @_deprecate_positional_args(allowed=['mesh'])
    def intersection(  # noqa: PLR0917
        self,
        mesh,
        split_first: bool = True,  # noqa: FBT001, FBT002
        split_second: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Compute the intersection between two meshes.

        .. note::
           This method returns the surface intersection from two meshes
           (which often resolves as a line), whereas the
           :func:`PolyDataFilters.boolean_intersection` filter returns
           the "volume" intersection between two closed (manifold)
           meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to intersect with.

        split_first : bool, default: True
            If ``True``, return the first input mesh split by the
            intersection with the second input mesh.

        split_second : bool, default: True
            If ``True``, return the second input mesh split by the
            intersection with the first input mesh.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The intersection line.

        pyvista.PolyData
            The first mesh split along the intersection. Returns the
            original first mesh if ``split_first=False``.

        pyvista.PolyData
            The second mesh split along the intersection. Returns the
            original second mesh if ``split_second=False``.

        Examples
        --------
        Intersect two spheres, returning the intersection and both spheres
        which have new points/cells along the intersection line.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> s1 = pv.Sphere(phi_resolution=15, theta_resolution=15)
        >>> s2 = s1.copy()
        >>> s2.points += np.array([0.25, 0, 0])
        >>> intersection, s1_split, s2_split = s1.intersection(s2)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(s1, style='wireframe')
        >>> _ = pl.add_mesh(s2, style='wireframe')
        >>> _ = pl.add_mesh(intersection, color='r', line_width=10)
        >>> pl.show()

        The mesh splitting takes additional time and can be turned
        off for either mesh individually.

        >>> intersection, _, s2_split = s1.intersection(
        ...     s2, split_first=False, split_second=True
        ... )

        """
        intfilter = _vtk.vtkIntersectionPolyDataFilter()
        intfilter.SetInputDataObject(0, self)
        intfilter.SetInputDataObject(1, mesh)
        intfilter.SetComputeIntersectionPointArray(True)
        intfilter.SetSplitFirstOutput(split_first)
        intfilter.SetSplitSecondOutput(split_second)
        _update_alg(
            intfilter,
            progress_bar=progress_bar,
            message='Computing the intersection between two meshes',
        )

        intersection = _get_output(intfilter, oport=0)
        first = _get_output(intfilter, oport=1)
        second = _get_output(intfilter, oport=2)

        return intersection, first, second

    @_deprecate_positional_args(allowed=['curv_type'])
    def curvature(self, curv_type='mean', progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Return the pointwise curvature of a mesh.

        See :ref:`connectivity_example` for more examples using this
        filter.

        Parameters
        ----------
        curv_type : str, default: "mean"
            Curvature type.  One of the following:

            * ``"mean"``
            * ``"gaussian"``
            * ``"maximum"``
            * ``"minimum"``

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        numpy.ndarray
            Array of curvature values.

        Examples
        --------
        Calculate the mean curvature of the hills example mesh and plot it.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> curv = hills.curvature()
        >>> hills.plot(scalars=curv)

        Show the curvature array.

        >>> curv  # doctest:+SKIP
        array([0.20587616, 0.06747695, ..., 0.11781171, 0.15988467])

        """
        curv_type = curv_type.lower()

        # Create curve filter and compute curvature
        curvefilter = _vtk.vtkCurvatures()
        curvefilter.SetInputData(self)
        if curv_type == 'mean':
            curvefilter.SetCurvatureTypeToMean()
        elif curv_type == 'gaussian':
            curvefilter.SetCurvatureTypeToGaussian()
        elif curv_type == 'maximum':
            curvefilter.SetCurvatureTypeToMaximum()
        elif curv_type == 'minimum':
            curvefilter.SetCurvatureTypeToMinimum()
        else:
            msg = '``curv_type`` must be either "Mean", "Gaussian", "Maximum", or "Minimum".'
            raise ValueError(msg)
        _update_alg(curvefilter, progress_bar=progress_bar, message='Computing Curvature')

        # Compute and return curvature
        curv = _get_output(curvefilter)
        return _vtk.vtk_to_numpy(curv.GetPointData().GetScalars())

    def plot_curvature(self, curv_type='mean', **kwargs):
        """Plot the curvature.

        Parameters
        ----------
        curv_type : str, default: "mean"
            One of the following strings indicating curvature type:

            * ``'mean'``
            * ``'gaussian'``
            * ``'maximum'``
            * ``'minimum'``

        **kwargs : dict, optional
            See :func:`pyvista.plot`.

        Returns
        -------
        pyvista.CameraPosition
            List of camera position, focal point, and view up.
            Returned when ``return_cpos`` is ``True``.

        Examples
        --------
        Plot the Gaussian curvature of an example mesh.  Override the
        default scalar bar range as the mesh edges report high
        curvature.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> hills.plot_curvature(
        ...     curv_type='gaussian', smooth_shading=True, clim=[0, 1]
        ... )

        """
        kwargs.setdefault('scalar_bar_args', {'title': f'{curv_type.capitalize()} Curvature'})
        return self.plot(scalars=self.curvature(curv_type), **kwargs)  # type: ignore[attr-defined]

    def triangulate(  # type: ignore[override]
        self,
        *,
        pass_verts: bool = False,
        pass_lines: bool = False,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        """Return an all triangle mesh.

        More complex polygons will be broken down into triangles.

        Parameters
        ----------
        pass_verts : bool, default: False
            Whether vertices cells are passed to the output PolyData.

            .. versionadded:: 0.45.0

        pass_lines : bool, default: False
            Whether lines cells are passed to the output PolyData.

            .. versionadded:: 0.45.0

        inplace : bool, default: False
            Whether to update the mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing only triangles.

        Examples
        --------
        Generate a mesh with quadrilateral faces.

        >>> import pyvista as pv
        >>> plane = pv.Plane()
        >>> plane.point_data.clear()
        >>> plane.plot(show_edges=True, line_width=5)

        Convert it to an all triangle mesh.

        >>> mesh = plane.triangulate()
        >>> mesh.plot(show_edges=True, line_width=5)

        """
        trifilter = _vtk.vtkTriangleFilter()
        trifilter.SetInputData(self)
        trifilter.SetPassVerts(pass_verts)
        trifilter.SetPassLines(pass_lines)
        _update_alg(trifilter, progress_bar=progress_bar, message='Computing Triangle Mesh')

        mesh = _get_output(trifilter)
        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self
        return mesh

    @_deprecate_positional_args
    def smooth(  # noqa: PLR0917
        self,
        n_iter=20,
        relaxation_factor=0.01,
        convergence=0.0,
        edge_angle=15,
        feature_angle=45,
        boundary_smoothing: bool = True,  # noqa: FBT001, FBT002
        feature_smoothing: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Adjust point coordinates using Laplacian smoothing.

        The effect is to "relax" the mesh, making the cells better shaped and
        the vertices more evenly distributed.

        Parameters
        ----------
        n_iter : int, default: 20
            Number of iterations for Laplacian smoothing.

        relaxation_factor : float, default: 0.01
            Relaxation factor controls the amount of displacement in a single
            iteration. Generally a lower relaxation factor and higher number of
            iterations is numerically more stable.

        convergence : float, default: 0.0
            Convergence criterion for the iteration process. Smaller numbers
            result in more smoothing iterations. Range from (0 to 1).

        edge_angle : float, default: 15
            Edge angle to control smoothing along edges (either interior or boundary).

        feature_angle : float, default: 45
            Feature angle for sharp edge identification.

        boundary_smoothing : bool, default: True
            Flag to control smoothing of boundary edges. When ``True``,
            boundary edges remain fixed.

        feature_smoothing : bool, default: False
            Flag to control smoothing of feature edges.  When ``True``,
            boundary edges remain fixed as defined by ``feature_angle`` and
            ``edge_angle``.

        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Smoothed mesh.

        Examples
        --------
        Smooth the edges of an all triangular cube

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(5)
        >>> smooth_cube = cube.smooth(n_iter=1000, feature_smoothing=False)
        >>> n_edge_cells = cube.extract_feature_edges().n_cells
        >>> n_smooth_cells = smooth_cube.extract_feature_edges().n_cells
        >>> f'Sharp Edges on Cube:        {n_edge_cells}'
        'Sharp Edges on Cube:        384'
        >>> f'Sharp Edges on Smooth Cube: {n_smooth_cells}'
        'Sharp Edges on Smooth Cube: 12'
        >>> smooth_cube.plot()

        See :ref:`surface_smoothing_example` for more examples using this filter.

        """
        alg = _vtk.vtkSmoothPolyDataFilter()
        alg.SetInputData(self)
        alg.SetNumberOfIterations(n_iter)
        alg.SetConvergence(convergence)
        alg.SetFeatureEdgeSmoothing(feature_smoothing)
        alg.SetFeatureAngle(feature_angle)
        alg.SetEdgeAngle(edge_angle)
        alg.SetBoundarySmoothing(boundary_smoothing)
        alg.SetRelaxationFactor(relaxation_factor)
        _update_alg(alg, progress_bar=progress_bar, message='Smoothing Mesh')

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self
        return mesh

    @_deprecate_positional_args
    def smooth_taubin(  # noqa: PLR0917
        self,
        n_iter=20,
        pass_band=0.1,
        edge_angle=15.0,
        feature_angle=45.0,
        boundary_smoothing: bool = True,  # noqa: FBT001, FBT002
        feature_smoothing: bool = False,  # noqa: FBT001, FBT002
        non_manifold_smoothing: bool = False,  # noqa: FBT001, FBT002
        normalize_coordinates: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Smooth a PolyData DataSet with Taubin smoothing.

        This filter allows you to smooth the mesh as in the Laplacian smoothing
        implementation in :func:`smooth() <PolyDataFilters.smooth>`. However,
        unlike Laplacian smoothing the surface does not "shrink" since this
        filter relies on an alternative approach to smoothing. This filter is
        more akin to a low pass filter where undesirable high frequency features
        are removed.

        This PyVista filter uses the VTK :vtk:`vtkWindowedSincPolyDataFilter`
        filter.

        Parameters
        ----------
        n_iter : int, default: 20
            This is the degree of the polynomial used to approximate the
            windowed sync function. This is generally much less than the number
            needed by :func:`smooth() <PolyDataFilters.smooth>`.

        pass_band : float, default: 0.1
            The passband value for the windowed sinc filter. This should be
            between 0 and 2, where lower values cause more smoothing.

        edge_angle : float, default: 15.0
            Edge angle to control smoothing along edges (either interior or
            boundary).

        feature_angle : float, default: 45.0
            Feature angle for sharp edge identification.

        boundary_smoothing : bool, default: True
            Flag to control smoothing of boundary edges. When ``True``,
            boundary edges remain fixed.

        feature_smoothing : bool, default: False
            Flag to control smoothing of feature edges.  When ``True``,
            boundary edges remain fixed as defined by ``feature_angle`` and
            ``edge_angle``.

        non_manifold_smoothing : bool, default: False
            Smooth non-manifold points.

        normalize_coordinates : bool, default: False
            Flag to control coordinate normalization. To improve the
            numerical stability of the solution and minimize the scaling of the
            translation effects, the algorithm can translate and scale the
            position coordinates to within the unit cube ``[-1, 1]``, perform the
            smoothing, and translate and scale the position coordinates back to
            the original coordinate frame.

        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Smoothed mesh.

        Notes
        -----
        For maximum performance, do not enable ``feature_smoothing`` or
        ``boundary_smoothing``. ``feature_smoothing`` is especially expensive.

        References
        ----------
        See `Optimal Surface Smoothing as Filter Design
        <https://dl.acm.org/doi/pdf/10.1145/218380.218473>`_ for details
        regarding the implementation of Taubin smoothing.

        Examples
        --------
        Smooth the example bone mesh. Here, it's necessary to subdivide the
        mesh to increase the number of faces as the original mesh is so coarse.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_foot_bones().subdivide(2)
        >>> smoothed_mesh = mesh.smooth_taubin()
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_text('Original Mesh')
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(smoothed_mesh)
        >>> _ = pl.add_text('Smoothed Mesh')
        >>> pl.show()

        See :ref:`surface_smoothing_example` for more examples using this filter.

        """
        alg = _vtk.vtkWindowedSincPolyDataFilter()
        alg.SetInputData(self)
        alg.SetNumberOfIterations(n_iter)
        alg.SetFeatureEdgeSmoothing(feature_smoothing)
        alg.SetNonManifoldSmoothing(non_manifold_smoothing)
        alg.SetFeatureAngle(feature_angle)
        alg.SetEdgeAngle(edge_angle)
        alg.SetBoundarySmoothing(boundary_smoothing)
        alg.SetPassBand(pass_band)
        alg.SetNormalizeCoordinates(normalize_coordinates)
        _update_alg(
            alg, progress_bar=progress_bar, message='Smoothing Mesh using Taubin Smoothing'
        )

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self
        return mesh

    @_deprecate_positional_args(allowed=['reduction'])
    def decimate_pro(  # noqa: PLR0917
        self,
        reduction,
        feature_angle=45.0,
        split_angle=75.0,
        splitting: bool = True,  # noqa: FBT001, FBT002
        pre_split_mesh: bool = False,  # noqa: FBT001, FBT002
        preserve_topology: bool = False,  # noqa: FBT001, FBT002
        boundary_vertex_deletion: bool = True,  # noqa: FBT001, FBT002
        max_degree=None,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Reduce the number of triangles in a triangular mesh.

        It forms a good approximation to the original geometry. Based
        on the algorithm originally described in "Decimation of
        Triangle Meshes", Proc Siggraph 92
        (https://doi.org/10.1145/133994.134010).

        Parameters
        ----------
        reduction : float
            Reduction factor. A value of 0.9 will leave 10% of the
            original number of vertices.

        feature_angle : float, default: 45.0
            Angle used to define what an edge is (i.e., if the surface
            normal between two adjacent triangles is >= ``feature_angle``,
            an edge exists).

        split_angle : float, default: 75.0
            Angle used to control the splitting of the mesh. A split
            line exists when the surface normals between two edge
            connected triangles are >= ``split_angle``.

        splitting : bool, default: True
            Controls the splitting of the mesh at corners, along
            edges, at non-manifold points, or anywhere else a split is
            required. Turning splitting off will better preserve the
            original topology of the mesh, but may not necessarily
            give the exact requested decimation.

        pre_split_mesh : bool, default: False
            Separates the mesh into semi-planar patches, which are
            disconnected from each other. This can give superior
            results in some cases. If ``pre_split_mesh`` is set to
            ``True``, the mesh is split with the specified
            ``split_angle``. Otherwise mesh splitting is deferred as
            long as possible.

        preserve_topology : bool, default: False
            Controls topology preservation. If on, mesh splitting and
            hole elimination will not occur. This may limit the
            maximum reduction that may be achieved.

        boundary_vertex_deletion : bool, default: True
            Allow deletion of vertices on the boundary of the mesh.
            Turning this off may limit the maximum reduction that may
            be achieved.

        max_degree : float, optional
            The maximum vertex degree. If the number of triangles
            connected to a vertex exceeds ``max_degree``, then the
            vertex will be split. The complexity of the triangulation
            algorithm is proportional to ``max_degree**2``. Setting ``max_degree``
            small can improve the performance of the algorithm.

        inplace : bool, default: False
            Whether to update the mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Decimated mesh.

        See Also
        --------
        decimate
            Another option for triangular meshes.
        decimate_polyline
            For use with polylines.

        Examples
        --------
        Decimate a sphere.  First plot the sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(phi_resolution=60, theta_resolution=60)
        >>> sphere.plot(show_edges=True, line_width=2)

        Now decimate it and plot it.

        >>> decimated = sphere.decimate_pro(0.75)
        >>> decimated.plot(show_edges=True, line_width=2)

        See :ref:`decimate_example` for more examples using this filter.

        """
        if not self.is_all_triangles:  # type: ignore[attr-defined]
            msg = 'Input mesh for decimation must be all triangles.'
            raise NotAllTrianglesError(msg)

        alg = _vtk.vtkDecimatePro()
        alg.SetInputData(self)
        alg.SetTargetReduction(reduction)
        alg.SetPreserveTopology(preserve_topology)
        alg.SetFeatureAngle(feature_angle)
        alg.SetSplitting(splitting)
        alg.SetSplitAngle(split_angle)
        alg.SetPreSplitMesh(pre_split_mesh)
        alg.SetBoundaryVertexDeletion(boundary_vertex_deletion)

        if max_degree is not None:
            alg.SetDegree(max_degree)

        _update_alg(alg, progress_bar=progress_bar, message='Decimating Mesh')

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self

        return mesh

    def decimate_polyline(
        self,
        reduction: float,
        *,
        maximum_error: float = 10.0,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        """Reduce the number of lines in a polyline mesh.

        This filter uses :vtk:`vtkDecimatePolylineFilter`.

        .. versionadded:: 0.45.0

        Parameters
        ----------
        reduction : float
            Reduction factor. A value of 0.9 will leave 10% of the
            original number of vertices.

        maximum_error : float, default: 10.
            Fraction of the maximum length of the input data bounding box
            to limit reduction.  This might prevent the full reduction from
            being achieved. Default of ``10.`` should not limit reduction.

        inplace : bool, default: False
            Whether to update the mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Decimated mesh.

        Warnings
        --------
        From :vtk:`vtkDecimatePolylineFilter` documentation: this algorithm is a very
        simple implementation that overlooks some potential complexities.
        For example, if a vertex is multiply connected, meaning that it is
        used by multiple distinct polylines, then the extra topological
        constraints are ignored. This can produce less than optimal results.

        See Also
        --------
        decimate
            For use with triangular meshes.
        decimate_pro
            Another option for triangular meshes.

        Examples
        --------
        Generate a circle, builtin function returns a Polygon cell.

        >>> import pyvista as pv
        >>> circle = pv.Circle(resolution=30)

        Convert to a Polyline. A polyline requires duplicating reference
        to initial point to close the circle.

        >>> circle_polyline = pv.PolyData(
        ...     circle.points, lines=[31] + list(range(30)) + [0]
        ... )
        >>> circle_polyline.n_points
        30

        Decimate ~50% of points.

        >>> decimate_some = circle_polyline.decimate_polyline(0.5)
        >>> decimate_some.n_points
        14

        Decimate more points.

        >>> decimate_more = circle_polyline.decimate_polyline(0.75)
        >>> decimate_more.n_points
        6

        Compare decimated polylines.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(circle_polyline, label='Circle', color='red', line_width=5)
        >>> _ = pl.add_mesh(
        ...     decimate_some,
        ...     label='Decimated some',
        ...     color='blue',
        ...     line_width=5,
        ... )
        >>> _ = pl.add_mesh(
        ...     decimate_more,
        ...     label='Decimated more',
        ...     color='black',
        ...     line_width=5,
        ... )
        >>> pl.view_xy()
        >>> _ = pl.add_legend(face='line', size=(0.25, 0.25))
        >>> pl.show()

        See :ref:`decimate_example` for more examples using this filter.

        """
        alg = _vtk.vtkDecimatePolylineFilter()
        alg.SetInputData(self)
        alg.SetTargetReduction(reduction)
        alg.SetMaximumError(maximum_error)

        _update_alg(alg, progress_bar=progress_bar, message='Decimating Mesh')

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self

        return mesh

    @_deprecate_positional_args
    def tube(  # noqa: PLR0917
        self,
        radius=None,
        scalars=None,
        capping: bool = True,  # noqa: FBT001, FBT002
        n_sides=20,
        radius_factor=10.0,
        absolute: bool = False,  # noqa: FBT001, FBT002
        preference='point',
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Generate a tube around each input line.

        The radius of the tube can be set to linearly vary with a
        scalar value.

        Parameters
        ----------
        radius : float, optional
            Minimum tube radius (minimum because the tube radius may
            vary).

        scalars : str, optional
            Scalars array by which the radius varies.

        capping : bool, default: True
            Turn on/off whether to cap the ends with polygons.

        n_sides : int, default: 20
            Set the number of sides for the tube. Minimum of 3.

        radius_factor : float, default: 10.0
            Maximum tube radius in terms of a multiple of the minimum
            radius.

        absolute : bool, default: False
            Vary the radius with values from scalars in absolute units.

        preference : str, default: 'point'
            The field preference when searching for the scalars array by
            name.

        inplace : bool, default: False
            Whether to update the mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Tube-filtered mesh.

        Examples
        --------
        Convert a single line to a tube.

        >>> import pyvista as pv
        >>> line = pv.Line()
        >>> tube = line.tube(radius=0.02)
        >>> f'Line Cells: {line.n_cells}'
        'Line Cells: 1'
        >>> f'Tube Cells: {tube.n_cells}'
        'Tube Cells: 22'
        >>> tube.plot(color='lightblue')

        See :ref:`create_spline_example` for more examples using this filter.

        """
        poly_data = self
        if not isinstance(poly_data, pyvista.PolyData):
            poly_data = pyvista.PolyData(poly_data)  # type: ignore[arg-type]
        n_sides = max(n_sides, 3)
        tube = _vtk.vtkTubeFilter()
        tube.SetInputDataObject(poly_data)
        # User Defined Parameters
        tube.SetCapping(capping)
        if radius is not None:
            tube.SetRadius(radius)
        tube.SetNumberOfSides(n_sides)
        tube.SetRadiusFactor(radius_factor)
        # Check if scalars array given
        if scalars is not None:
            if not isinstance(scalars, str):
                msg = 'scalars array must be given as a string name'
                raise TypeError(msg)
            field = poly_data.get_array_association(scalars, preference=preference)
            # args: (idx, port, connection, field, name)
            tube.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
            if absolute:
                tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            else:
                tube.SetVaryRadiusToVaryRadiusByScalar()
        # Apply the filter
        _update_alg(tube, progress_bar=progress_bar, message='Creating Tube')

        mesh = _get_output(tube)
        if inplace:
            poly_data.copy_from(mesh, deep=False)
            return poly_data
        return mesh

    @_deprecate_positional_args(allowed=['nsub', 'subfilter'])
    def subdivide(  # noqa: PLR0917
        self,
        nsub,
        subfilter='linear',
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Increase the number of triangles in a single, connected triangular mesh.

        Uses one of the following vtk subdivision filters to subdivide a mesh:

        * :vtk:`vtkButterflySubdivisionFilter`
        * :vtk:`vtkLoopSubdivisionFilter`
        * :vtk:`vtkLinearSubdivisionFilter`

        Linear subdivision results in the fastest mesh subdivision,
        but it does not smooth mesh edges, but rather splits each
        triangle into 4 smaller triangles.

        Butterfly and loop subdivision perform smoothing when
        dividing, and may introduce artifacts into the mesh when
        dividing.

        .. note::
           Subdivision filter sometimes fails for multiple part
           meshes.  The input should be one connected mesh.

        Parameters
        ----------
        nsub : int
            Number of subdivisions.  Each subdivision creates 4 new
            triangles, so the number of resulting triangles is
            ``nface*4**nsub`` where ``nface`` is the current number of
            faces.

        subfilter : str, default: "linear"
            Can be one of the following:

            * ``'butterfly'``
            * ``'loop'``
            * ``'linear'``

        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Subdivided mesh.

        Examples
        --------
        First, create an example coarse sphere mesh and plot it.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> mesh = pv.Sphere(phi_resolution=10, theta_resolution=10)
        >>> mesh.plot(show_edges=True, line_width=3)

        Subdivide the sphere mesh using linear subdivision.

        >>> submesh = mesh.subdivide(1, 'linear')
        >>> submesh.plot(show_edges=True, line_width=3)

        Subdivide the sphere mesh using loop subdivision.

        >>> submesh = mesh.subdivide(1, 'loop')
        >>> submesh.plot(show_edges=True, line_width=3)

        Subdivide the sphere mesh using butterfly subdivision.

        >>> submesh = mesh.subdivide(1, 'butterfly')
        >>> submesh.plot(show_edges=True, line_width=3)

        """
        if not self.is_all_triangles:  # type: ignore[attr-defined]
            msg = 'Input mesh for subdivision must be all triangles.'
            raise NotAllTrianglesError(msg)

        subfilter = subfilter.lower()
        if subfilter == 'linear':
            sfilter = _vtk.vtkLinearSubdivisionFilter()
        elif subfilter == 'butterfly':
            sfilter = _vtk.vtkButterflySubdivisionFilter()  # type: ignore[assignment]
        elif subfilter == 'loop':
            sfilter = _vtk.vtkLoopSubdivisionFilter()  # type: ignore[assignment]
        else:
            msg = (
                "Subdivision filter must be one of the following: 'butterfly', 'loop', or 'linear'"
            )
            raise ValueError(msg)

        # Subdivide
        sfilter.SetCheckForTriangles(False)  # we already check for this
        sfilter.SetNumberOfSubdivisions(nsub)
        sfilter.SetInputData(self)
        _update_alg(sfilter, progress_bar=progress_bar, message='Subdividing Mesh')

        submesh = _get_output(sfilter)
        if inplace:
            self.copy_from(submesh, deep=False)  # type: ignore[attr-defined]
            return self

        return submesh

    @_deprecate_positional_args
    def subdivide_adaptive(  # noqa: PLR0917
        self,
        max_edge_len=None,
        max_tri_area=None,
        max_n_tris=None,
        max_n_passes=None,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Increase the number of triangles in a triangular mesh based on edge and/or area metrics.

        This filter uses a simple case-based, multi-pass approach to
        repeatedly subdivide the input triangle mesh to meet the area
        and/or edge length criteria. New points may be inserted only
        on edges; depending on the number of edges to be subdivided a
        different number of triangles are inserted ranging from two
        (i.e., two triangles replace the original one) to four.

        Point and cell data is treated as follows: The cell data from
        a parent triangle is assigned to its subdivided
        children. Point data is interpolated along edges as the edges
        are subdivided.

        This filter retains mesh watertightness if the mesh was
        originally watertight; and the area and max triangles criteria
        are not used.

        Parameters
        ----------
        max_edge_len : float, optional
            The maximum edge length that a triangle may have. Edges
            longer than this value are split in half and the
            associated triangles are modified accordingly.

        max_tri_area : float, optional
            The maximum area that a triangle may have. Triangles
            larger than this value are subdivided to meet this
            threshold. Note that if this criterion is used it may
            produce non-watertight meshes as a result.

        max_n_tris : int, optional
            The maximum number of triangles that can be created. If
            the limit is hit, it may result in premature termination
            of the algorithm and the results may be less than
            satisfactory (for example non-watertight meshes may be
            created). By default, the limit is set to a very large
            number (i.e., no effective limit).

        max_n_passes : int, optional
            The maximum number of passes (i.e., levels of
            subdivision). If the limit is hit, then the subdivision
            process stops and additional passes (needed to meet other
            criteria) are aborted. The default limit is set to a very
            large number (i.e., no effective limit).

        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Subdivided mesh.

        Examples
        --------
        First, load the example airplane mesh and plot it.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = pv.PolyData(examples.planefile)
        >>> mesh.plot(show_edges=True, line_width=3)

        Subdivide the mesh

        >>> submesh = mesh.subdivide_adaptive(max_n_passes=2)
        >>> submesh.plot(show_edges=True)

        """
        if not self.is_all_triangles:  # type: ignore[attr-defined]
            msg = 'Input mesh for subdivision must be all triangles.'
            raise NotAllTrianglesError(msg)

        sfilter = _vtk.vtkAdaptiveSubdivisionFilter()
        if max_edge_len:
            sfilter.SetMaximumEdgeLength(max_edge_len)
        if max_tri_area:
            sfilter.SetMaximumTriangleArea(max_tri_area)
        if max_n_tris:
            sfilter.SetMaximumNumberOfTriangles(max_n_tris)
        if max_n_passes:
            sfilter.SetMaximumNumberOfPasses(max_n_passes)

        sfilter.SetInputData(self)
        _update_alg(sfilter, progress_bar=progress_bar, message='Adaptively Subdividing Mesh')
        submesh = _get_output(sfilter)

        if inplace:
            self.copy_from(submesh, deep=False)  # type: ignore[attr-defined]
            return self

        return submesh

    @_deprecate_positional_args(allowed=['target_reduction'])
    def decimate(  # noqa: PLR0917
        self,
        target_reduction,
        volume_preservation: bool = False,  # noqa: FBT001, FBT002
        attribute_error: bool | None = None,  # noqa: FBT001
        scalars: bool | None = None,  # noqa: FBT001
        vectors: bool | None = None,  # noqa: FBT001
        normals: bool | None = None,  # noqa: FBT001
        tcoords: bool | None = None,  # noqa: FBT001
        tensors: bool | None = None,  # noqa: FBT001
        scalars_weight=0.1,
        vectors_weight=0.1,
        normals_weight=0.1,
        tcoords_weight=0.1,
        tensors_weight=0.1,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        boundary_constraints: bool = False,  # noqa: FBT001, FBT002
        boundary_weight: float = 1.0,
        enable_all_attribute_error: bool = False,  # noqa: FBT001, FBT002
    ):
        """Reduce the number of triangles in a triangular mesh using :vtk:`vtkQuadricDecimation`.

        .. versionchanged:: 0.45
          ``scalars``, ``vectors``, ``normals``, ``tcoords`` and ``tensors`` are now disabled by
          default. They can be enabled all together using ``enable_all_attribute_error``.


        Parameters
        ----------
        target_reduction : float
            Fraction of the original mesh to remove.
            If ``target_reduction`` is set to 0.9, this filter will try
            to reduce the data set to 10% of its original size and will
            remove 90% of the input triangles.

        volume_preservation : bool, default: False
            Decide whether to activate volume preservation which greatly
            reduces errors in triangle normal direction. If ``False``,
            volume preservation is disabled and if ``attribute_error``
            is active, these errors can be large.

        attribute_error : bool, default: False
            Decide whether to include data attributes in the error metric. If
            ``False``, then only geometric error is used to control the
            decimation. If ``True``, the following flags are used to specify
            which attributes are to be included in the error calculation.

            .. deprecated:: 0.45.0

        scalars : bool, default: False
            This flags control specifically if the **scalar** attributes
            are to be included in the error calculation.

        vectors : bool, default: False
            See ``scalars`` parameter.

        normals : bool, default: False
            See ``scalars`` parameter.
            .. versionchanged:: 0.45.0

        tcoords : bool, default: False
            See ``scalars`` parameter.

        tensors : bool, default: False
            See ``scalars`` parameter.

        scalars_weight : float, default: 0.1
            The scaling weight contribution of the scalar attribute.
            These values are used to weight the contribution of the
            attributes towards the error metric.

        vectors_weight : float, default: 0.1
            See ``scalars_weight`` parameter.

        normals_weight : float, default: 0.1
            See ``scalars_weight`` parameter.

        tcoords_weight : float, default: 0.1
            See ``scalars_weight`` parameter.

        tensors_weight : float, default: 0.1
            See ``scalars_weight`` parameter.

        inplace : bool, default: False
            Whether to update the mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        boundary_constraints: bool, default: False
            Use the legacy weighting by boundary_edge_length instead of by
            boundary_edge_length^2 for backwards compatibility.
            It requires vtk>=9.3.0.

            .. versionadded:: 0.45.0

        boundary_weight: float, default: 1.0
            A floating point factor to weigh the boundary quadric constraints
            by: higher factors further constrain the boundary.
            It requires vtk>=9.3.0.

            .. versionadded:: 0.45.0

        enable_all_attribute_error: bool, default: False
            This flag control the default value of all attribute metrics to
            eventually include them in the error calculation

            .. versionadded:: 0.45.0

        Returns
        -------
        pyvista.PolyData
            Decimated mesh.

        See Also
        --------
        decimate_pro
            Another option for triangular meshes.
        decimate_polyline
            For use with polylines.

        Notes
        -----
        If you encounter a segmentation fault or other error, consider using
        :func:`pyvista.PolyDataFilters.clean` to remove any invalid cells
        before using this filter.

        Examples
        --------
        Decimate a sphere.  First plot the sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(phi_resolution=60, theta_resolution=60)
        >>> sphere.plot(show_edges=True, line_width=2)

        Now decimate it by 75% and plot it.

        >>> decimated = sphere.decimate(0.75)
        >>> decimated.plot(show_edges=True, line_width=2)

        Decimate taking scalars attributes into account:

        >>> decimated = sphere.decimate(0.5, scalars=True)

        Decimate taking all attributes **except** normals into account:

        >>> decimated = sphere.decimate(
        ...     0.5, enable_all_attribute_error=True, normals=False
        ... )

        See :ref:`decimate_example` for more examples using this filter.

        """
        if not self.is_all_triangles:  # type: ignore[attr-defined]
            msg = 'Input mesh for decimation must be all triangles.'
            raise NotAllTrianglesError(msg)

        has_attribute_error = False if attribute_error is None else attribute_error
        if has_attribute_error:  # pragma: no cover
            warnings.warn(
                'Since 0.45, use of `attribute_error=True` is deprecated.'
                "Use 'enable_all_attribute_error' instead.",
                PyVistaDeprecationWarning,
            )
            enable_all_attribute_error = True
            if normals is None:
                normals = False

        # Get each attributes if defined, otherwise fallback to the
        # enable_all_attribute_error value
        use_scalars = enable_all_attribute_error if scalars is None else scalars
        use_vectors = enable_all_attribute_error if vectors is None else vectors
        use_normals = enable_all_attribute_error if normals is None else normals
        use_tcoords = enable_all_attribute_error if tcoords is None else tcoords
        use_tensors = enable_all_attribute_error if tensors is None else tensors
        use_attribute = use_scalars or use_vectors or use_normals or use_tcoords or use_tensors

        # create decimation filter
        alg = _vtk.vtkQuadricDecimation()

        alg.SetVolumePreservation(volume_preservation)
        alg.SetAttributeErrorMetric(use_attribute)
        alg.SetScalarsAttribute(use_scalars)
        alg.SetVectorsAttribute(use_vectors)
        alg.SetNormalsAttribute(use_normals)
        alg.SetTCoordsAttribute(use_tcoords)
        alg.SetTensorsAttribute(use_tensors)
        alg.SetScalarsWeight(scalars_weight)
        alg.SetVectorsWeight(vectors_weight)
        alg.SetNormalsWeight(normals_weight)
        alg.SetTCoordsWeight(tcoords_weight)
        alg.SetTensorsWeight(tensors_weight)
        alg.SetTargetReduction(target_reduction)
        if pyvista.vtk_version_info < (9, 3, 0):  # pragma: no cover
            if boundary_constraints:
                warnings.warn('`boundary_constraints` requires vtk >= 9.3.')
        else:
            alg.SetWeighBoundaryConstraintsByLength(boundary_constraints)
            alg.SetBoundaryWeightFactor(boundary_weight)

        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Decimating Mesh')

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self

        return mesh

    @_deprecate_positional_args
    def compute_normals(  # noqa: PLR0917
        self,
        cell_normals: bool = True,  # noqa: FBT001, FBT002
        point_normals: bool = True,  # noqa: FBT001, FBT002
        split_vertices: bool = False,  # noqa: FBT001, FBT002
        flip_normals: bool = False,  # noqa: FBT001, FBT002
        consistent_normals: bool = True,  # noqa: FBT001, FBT002
        auto_orient_normals: bool = False,  # noqa: FBT001, FBT002
        non_manifold_traversal: bool = True,  # noqa: FBT001, FBT002
        feature_angle=30.0,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Compute point and/or cell normals for a mesh.

        The filter can reorder polygons to insure consistent
        orientation across polygon neighbors. Sharp edges can be split
        and points duplicated with separate normals to give crisp
        (rendered) surface definition. It is also possible to globally
        flip the normal orientation.

        The algorithm works by determining normals for each polygon
        and then averaging them at shared points. When sharp edges are
        present, the edges are split and new points generated to
        prevent blurry edges (due to Phong shading).

        An array named ``"Normals"`` is stored with the mesh.

        .. warning::

           - Normals can only be computed for polygons and triangle strips.
             Point clouds are not supported.
           - Triangle strips are broken up into triangle polygons.
             You may want to restrip the triangles.
           - Previous arrays named ``"Normals"`` will be overwritten.

        Parameters
        ----------
        cell_normals : bool, default: True
            Calculation of cell normals.

        point_normals : bool, default: True
            Calculation of point normals.

        split_vertices : bool, default: False
            Splitting of sharp edges. Indices to the original points are
            tracked in the ``"pyvistaOriginalPointIds"`` array.

        flip_normals : bool, default: False
            Set global flipping of normal orientation. Flipping
            modifies both the normal direction and the order of a
            cell's points.

        consistent_normals : bool, default: True
            Enforcement of consistent polygon ordering.

        auto_orient_normals : bool, default: False
            Turn on/off the automatic determination of correct normal
            orientation. NOTE: This assumes a completely closed
            surface (i.e. no boundary edges) and no non-manifold
            edges. If these constraints do not hold, all bets are
            off. This option adds some computational complexity, and
            is useful if you do not want to have to inspect the
            rendered image to determine whether to turn on the
            ``flip_normals`` flag.  However, this flag can work with
            the ``flip_normals`` flag, and if both are set, all the
            normals in the output will point "inward".

        non_manifold_traversal : bool, default: True
            Turn on/off traversal across non-manifold edges. Changing
            this may prevent problems where the consistency of
            polygonal ordering is corrupted due to topological
            loops.

        feature_angle : float, default: 30.0
            The angle that defines a sharp edge. If the difference in
            angle across neighboring polygons is greater than this
            value, the shared edge is considered "sharp".

        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Raises
        ------
        TypeError
            If the mesh contains only ``LINE`` or ``VERTEX`` cell types.
            Normals cannot be computed for these cells.

        Returns
        -------
        pyvista.PolyData
            Updated mesh with cell and point normals.

        See Also
        --------
        pyvista.PolyData.point_normals
            Returns the array of point normals.
        pyvista.PolyData.cell_normals
            Returns the array of cell normals.
        pyvista.PolyDataFilters.flip_normal_vectors
            Flip cell and point normals.
        pyvista.PolyDataFilters.flip_faces
            Flip the orientation of the faces.

        Examples
        --------
        Compute the point normals of the surface of a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere = sphere.compute_normals(cell_normals=False)
        >>> normals = sphere['Normals']
        >>> normals.shape
        (842, 3)

        Alternatively, create a new mesh when computing the normals
        and compute both cell and point normals.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere_with_norm = sphere.compute_normals()
        >>> sphere_with_norm.point_data['Normals'].shape
        (842, 3)
        >>> sphere_with_norm.cell_data['Normals'].shape
        (1680, 3)

        See :ref:`compute_normals_example` for more examples using this filter.

        """
        # track original point indices
        if split_vertices:
            self.point_data.set_array(  # type: ignore[attr-defined]
                np.arange(self.n_points, dtype=pyvista.ID_TYPE),  # type: ignore[attr-defined]
                'pyvistaOriginalPointIds',
            )

        normal = _vtk.vtkPolyDataNormals()
        normal.SetComputeCellNormals(cell_normals)
        normal.SetComputePointNormals(point_normals)
        normal.SetSplitting(split_vertices)
        normal.SetFlipNormals(flip_normals)
        normal.SetConsistency(consistent_normals)
        normal.SetAutoOrientNormals(auto_orient_normals)
        normal.SetNonManifoldTraversal(non_manifold_traversal)
        normal.SetFeatureAngle(feature_angle)
        normal.SetInputData(self)
        _update_alg(normal, progress_bar=progress_bar, message='Computing Normals')

        mesh = _get_output(normal)
        try:
            mesh['Normals']
        except KeyError:
            if (self.n_verts + self.n_lines) == self.n_cells:  # type: ignore[attr-defined]
                msg = (
                    'Normals cannot be computed for PolyData containing only vertex cells '
                    '(e.g. point clouds)\n'
                    'and/or line cells. The PolyData cells must be polygons (e.g. triangle cells).'
                )
                raise TypeError(msg)
            else:  # pragma: no cover
                msg = (
                    'Normals could not be computed for unknown reasons.\n'
                    'Please report the issue at https://github.com/pyvista/pyvista/issues.'
                )
                raise RuntimeError(msg)
        if point_normals:
            mesh.GetPointData().SetActiveNormals('Normals')
        if cell_normals:
            mesh.GetCellData().SetActiveNormals('Normals')

        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self

        return mesh

    @_deprecate_positional_args(allowed=['normal'])
    def clip_closed_surface(  # noqa: PLR0917
        self,
        normal='x',
        origin=None,
        tolerance=1e-06,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Clip a closed polydata surface with a plane.

        This currently only supports one plane but could be
        implemented to handle a plane collection.

        It will produce a new closed surface by creating new polygonal
        faces where the input data was clipped.

        Non-manifold surfaces should not be used as input for this
        filter.  The input surface should have no open edges, and must
        not have any edges that are shared by more than two faces. In
        addition, the input surface should not self-intersect, meaning
        that the faces of the surface should only touch at their
        edges.

        Parameters
        ----------
        normal : str, list, optional
            Plane normal to clip with.  Plane is centered at
            ``origin``.  Normal can be either a 3 member list
            (e.g. ``[0, 0, 1]``) or one of the following strings:
            ``'x'``, ``'y'``, ``'z'``, ``'-x'``, ``'-y'``, or
            ``'-z'``.

        origin : list, optional
            Coordinate of the origin (e.g. ``[1, 0, 0]``).  Defaults
            to the center of the mesh.

        tolerance : float, optional
            The tolerance for creating new points while clipping.  If
            the tolerance is too small, then degenerate triangles
            might be produced.

        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The clipped mesh.

        Examples
        --------
        Clip a sphere in the X direction centered at the origin.  This
        will leave behind half a sphere in the positive X direction.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> clipped_mesh = sphere.clip_closed_surface('-z')
        >>> clipped_mesh.plot(show_edges=True, line_width=3)

        Clip the sphere at the XY plane and leave behind half the
        sphere in the positive Z direction.  Shift the clip upwards to
        leave a smaller mesh behind.

        >>> clipped_mesh = sphere.clip_closed_surface('z', origin=[0, 0, 0.3])
        >>> clipped_mesh.plot(show_edges=True, line_width=3)

        """
        # verify it is manifold
        if self.n_open_edges > 0:  # type: ignore[attr-defined]
            msg = 'This surface appears to be non-manifold.'
            raise ValueError(msg)
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = self.center  # type: ignore[attr-defined]

        # create the plane for clipping
        plane = generate_plane(normal, origin)
        collection = _vtk.vtkPlaneCollection()
        collection.AddItem(plane)

        alg = _vtk.vtkClipClosedSurface()
        alg.SetGenerateFaces(True)
        alg.SetInputDataObject(self)
        alg.SetTolerance(tolerance)
        alg.SetClippingPlanes(collection)
        _update_alg(alg, progress_bar=progress_bar, message='Clipping Closed Surface')
        result = _get_output(alg)

        if inplace:
            self.copy_from(result, deep=False)  # type: ignore[attr-defined]
            return self
        else:
            return result

    @_deprecate_positional_args(allowed=['hole_size'])
    def fill_holes(
        self,
        hole_size,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):  # pragma: no cover
        """Fill holes in a :class:`pyvista.PolyData` or :vtk:`vtkPolyData` object.

        Holes are identified by locating boundary edges, linking them
        together into loops, and then triangulating the resulting
        loops. Note that you can specify an approximate limit to the
        size of the hole that can be filled.

        .. warning::
           This method is known to segfault.  Use at your own risk.

        Parameters
        ----------
        hole_size : float
            Specifies the maximum hole size to fill. This is
            represented as a radius to the bounding circumsphere
            containing the hole. Note that this is an approximate
            area; the actual area cannot be computed without first
            triangulating the hole.

        inplace : bool, default: False
            Return new mesh or overwrite input.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh with holes filled if ``inplace=False``.

        Examples
        --------
        Create a partial sphere with a hole and then fill it.

        >>> import pyvista as pv
        >>> sphere_with_hole = pv.Sphere(end_theta=330)
        >>> sphere = sphere_with_hole.fill_holes(1000)  # doctest:+SKIP
        >>> edges = sphere.extract_feature_edges(
        ...     feature_edges=False, manifold_edges=False
        ... )  # doctest:+SKIP
        >>> assert edges.n_cells == 0  # doctest:+SKIP

        """
        alg = _vtk.vtkFillHolesFilter()
        alg.SetHoleSize(hole_size)
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Filling Holes')

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self
        return mesh

    @_deprecate_positional_args
    def clean(  # noqa: PLR0917
        self,
        point_merging: bool = True,  # noqa: FBT001, FBT002
        tolerance=None,
        lines_to_points: bool = True,  # noqa: FBT001, FBT002
        polys_to_lines: bool = True,  # noqa: FBT001, FBT002
        strips_to_polys: bool = True,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
        absolute: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ):
        """Clean the mesh.

        This merges duplicate points, removes unused points, and/or
        removes degenerate cells.

        Parameters
        ----------
        point_merging : bool, optional
            Enables point merging.  ``True`` by default.

        tolerance : float, optional
            Set merging tolerance.  When enabled merging is set to
            absolute distance. If ``absolute`` is ``False``, then the
            merging tolerance is a fraction of the bounding box
            length. The alias ``merge_tol`` is also excepted.

        lines_to_points : bool, optional
            Enable or disable the conversion of degenerate lines to
            points.  Enabled by default.

        polys_to_lines : bool, optional
            Enable or disable the conversion of degenerate polys to
            lines.  Enabled by default.

        strips_to_polys : bool, optional
            Enable or disable the conversion of degenerate strips to
            polys.

        inplace : bool, default: False
            Updates mesh in-place.

        absolute : bool, optional
            Control if ``tolerance`` is an absolute distance or a
            fraction.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Accepts for ``merge_tol`` to replace the ``tolerance``
            keyword argument.  This may be deprecated in future.

        Returns
        -------
        pyvista.PolyData
            Cleaned mesh.

        See Also
        --------
        remove_unused_points
            Strictly remove unused points `without` merging points.

        Examples
        --------
        Create a mesh with a degenerate face and then clean it,
        removing the degenerate face

        >>> import pyvista as pv
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
        >>> faces = np.array([3, 0, 1, 2, 3, 0, 2, 2])
        >>> mesh = pv.PolyData(points, faces)
        >>> mout = mesh.clean()
        >>> mout.faces  # doctest:+SKIP
        array([3, 0, 1, 2])

        """
        if tolerance is None:
            tolerance = kwargs.pop('merge_tol', None)
        assert_empty_kwargs(**kwargs)
        alg = _vtk.vtkCleanPolyData()
        alg.SetPointMerging(point_merging)
        alg.SetConvertLinesToPoints(lines_to_points)
        alg.SetConvertPolysToLines(polys_to_lines)
        alg.SetConvertStripsToPolys(strips_to_polys)
        if isinstance(tolerance, (int, float)):
            if absolute:
                alg.ToleranceIsAbsoluteOn()
                alg.SetAbsoluteTolerance(tolerance)
            else:
                alg.SetTolerance(tolerance)
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Cleaning')
        output = _get_output(alg)

        # Check output so no segfaults occur
        if output.n_points < 1 and self.n_cells > 0:  # type: ignore[attr-defined]
            msg = 'Clean tolerance is too high. Empty mesh returned.'
            raise ValueError(msg)

        if inplace:
            self.copy_from(output, deep=False)  # type: ignore[attr-defined]
            return self
        return output

    @_deprecate_positional_args(allowed=['start_vertex', 'end_vertex'])
    def geodesic(  # noqa: PLR0917
        self,
        start_vertex,
        end_vertex,
        inplace: bool = False,  # noqa: FBT001, FBT002
        keep_order: bool = True,  # noqa: FBT001, FBT002
        use_scalar_weights: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Calculate the geodesic path between two vertices using Dijkstra's algorithm.

        This will add an array titled ``'vtkOriginalPointIds'`` of the input
        mesh's point ids to the output mesh. The default behavior of the
        underlying :vtk:`vtkDijkstraGraphGeodesicPath` filter is that the
        geodesic path is reversed in the resulting mesh. This is overridden
        in PyVista by default.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        inplace : bool, default: False
            Whether the input mesh should be replaced with the path. The
            geodesic path is always returned.

        keep_order : bool, default: True
            If ``True``, the points of the returned path are guaranteed
            to start with the start vertex (as opposed to the end vertex).

            .. versionadded:: 0.32.0

        use_scalar_weights : bool, default: False
            If ``True``, use scalar values in the edge weight.
            This only works for point data.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            ``PolyData`` object consisting of the line segment between the
            two given vertices. If ``inplace`` is ``True`` this is the
            same object as the input mesh.

        Examples
        --------
        Plot the path between two points on the random hills mesh.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> path = hills.geodesic(560, 5820)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(hills)
        >>> _ = pl.add_mesh(path, line_width=5, color='k')
        >>> pl.show()

        See :ref:`geodesic_example` for more examples using this filter.

        """
        if not (0 <= start_vertex < self.n_points and 0 <= end_vertex < self.n_points):  # type: ignore[attr-defined]
            msg = 'Invalid point indices.'
            raise IndexError(msg)
        if not self.is_all_triangles:  # type: ignore[attr-defined]
            msg = 'Input mesh for geodesic path must be all triangles.'
            raise NotAllTrianglesError(msg)

        dijkstra = _vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(self)
        dijkstra.SetStartVertex(start_vertex)
        dijkstra.SetEndVertex(end_vertex)
        dijkstra.SetUseScalarWeights(use_scalar_weights)
        _update_alg(dijkstra, progress_bar=progress_bar, message='Calculating the Geodesic Path')
        original_ids = vtk_id_list_to_array(dijkstra.GetIdList())

        output = _get_output(dijkstra)
        if output.is_empty:
            msg = (
                f'There is no path between vertices {start_vertex} and {end_vertex}. '
                'It is likely the vertices belong to disconnected regions.'
            )
            raise ValueError(msg)

        output['vtkOriginalPointIds'] = original_ids

        # ensure proper order if requested
        if keep_order and original_ids[0] == end_vertex:
            output.points[...] = output.points[::-1, :]
            output['vtkOriginalPointIds'] = output['vtkOriginalPointIds'][::-1]

        if inplace:
            self.copy_from(output, deep=False)  # type: ignore[attr-defined]
            return self

        return output

    @_deprecate_positional_args(allowed=['start_vertex', 'end_vertex'])
    def geodesic_distance(  # noqa: PLR0917
        self,
        start_vertex,
        end_vertex,
        use_scalar_weights: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Calculate the geodesic distance between two vertices using Dijkstra's algorithm.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        use_scalar_weights : bool, default: False
            If ``True``, use scalar values in the edge weight.
            This only works for point data.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        float
            Length of the geodesic segment.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> length = sphere.geodesic_distance(0, 100)
        >>> f'Length is {length:.3f}'
        'Length is 0.812'

        See :ref:`geodesic_example` for more examples using this filter.

        """
        path = self.geodesic(start_vertex, end_vertex, use_scalar_weights=use_scalar_weights)
        sizes = path.compute_cell_sizes(
            length=True,
            area=False,
            volume=False,
            progress_bar=progress_bar,
        )
        distance = np.sum(sizes['Length'])
        del path
        del sizes
        return distance

    @_deprecate_positional_args(allowed=['origin', 'end_point'])
    def ray_trace(  # type: ignore[misc]  # noqa: PLR0917
        self: PolyData,
        origin,
        end_point,
        first_point: bool = False,  # noqa: FBT001, FBT002
        plot: bool = False,  # noqa: FBT001, FBT002
        off_screen=None,
    ):
        """Perform a single ray trace calculation.

        This requires a mesh and a line segment defined by an origin
        and end_point.

        .. warning::

            This filter uses the mesh's :attr:`~pyvista.PolyData.obbTree` property internally to
            compute the intersection. Since the obb tree is cached, the intersection may not be
            valid if the mesh's geometry is modified in between filter calls.

        Parameters
        ----------
        origin : sequence[float]
            Start of the line segment.

        end_point : sequence[float]
            End of the line segment.

        first_point : bool, default: False
            Returns intersection of first point only.

        plot : bool, default: False
            Whether to plot the ray trace results.

        off_screen : bool, optional
            Plots off screen when ``plot=True``.  Used for unit testing.

        Returns
        -------
        intersection_points : numpy.ndarray
            Location of the intersection points.  Empty array if no
            intersections.

        intersection_cells : numpy.ndarray
            Indices of the intersection cells.  Empty array if no
            intersections.

        See Also
        --------
        :ref:`ray_trace_moeller_example`
            Example of ray-tracing using the Moeller-Trumbore intersection algorithm.

        Examples
        --------
        Compute the intersection between a ray from the origin to
        ``[1, 0, 0]`` and a sphere with radius 0.5 centered at the
        origin.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> point, cell = sphere.ray_trace([0, 0, 0], [1, 0, 0], first_point=True)
        >>> f'Intersected at {point[0]:.3f} {point[1]:.3f} {point[2]:.3f}'
        'Intersected at 0.499 0.000 0.000'

        Show a plot of the ray trace.

        >>> point, cell = sphere.ray_trace([0, 0, 0], [1, 0, 0], plot=True)

        See :ref:`ray_trace_example` for more examples using this filter.

        """
        points = _vtk.vtkPoints()
        cell_ids = _vtk.vtkIdList()
        self.obbTree.IntersectWithLine(list(origin), list(end_point), points, cell_ids)

        intersection_points = _vtk.vtk_to_numpy(points.GetData())
        has_intersection = intersection_points.shape[0] >= 1
        if first_point and has_intersection:
            intersection_points = intersection_points[0]

        intersection_cells = []
        if has_intersection:
            ncells = 1 if first_point else cell_ids.GetNumberOfIds()
            intersection_cells = [cell_ids.GetId(i) for i in range(ncells)]
        intersection_cells = np.array(intersection_cells)  # type: ignore[assignment]

        if plot:
            plotter = pyvista.Plotter(off_screen=off_screen)
            plotter.add_mesh(self, label='Test Mesh')
            segment = np.array([origin, end_point])
            plotter.add_lines(segment, color='b', label='Ray Segment')
            plotter.add_mesh(
                intersection_points, color='r', point_size=10, label='Intersection Points'
            )
            plotter.add_legend()
            plotter.add_axes()
            plotter.show()

        return intersection_points, intersection_cells

    @_deprecate_positional_args(allowed=['origins', 'directions'])
    def multi_ray_trace(  # type:ignore[misc]  # noqa: PLR0917
        self: PolyData,
        origins,
        directions,
        first_point: bool = False,  # noqa: FBT001, FBT002
        retry: bool = False,  # noqa: FBT001, FBT002
    ):  # pragma: no cover
        """Perform multiple ray trace calculations.

        This requires a mesh with only triangular faces, an array of
        origin points and an equal sized array of direction vectors to
        trace along.

        The embree library used for vectorization of the ray traces is
        known to occasionally return no intersections where the VTK
        implementation would return an intersection.  If the result
        appears to be missing some intersection points, set
        ``retry=True`` to run a second pass over rays that returned no
        intersections, using :func:`PolyDataFilters.ray_trace`.

        Parameters
        ----------
        origins : array_like[float]
            Starting point for each trace.

        directions : array_like[float]
            Direction vector for each trace.

        first_point : bool, default: False
            Returns intersection of first point only.

        retry : bool, default: False
            Will retry rays that return no intersections using
            :func:`PolyDataFilters.ray_trace`.

        Returns
        -------
        intersection_points : numpy.ndarray
            Location of the intersection points.  Empty array if no
            intersections.

        intersection_rays : numpy.ndarray
            Indices of the ray for each intersection point. Empty array if no
            intersections.

        intersection_cells : numpy.ndarray
            Indices of the intersection cells.  Empty array if no
            intersections.

        Examples
        --------
        Compute the intersection between rays from the origin in
        directions ``[1, 0, 0]``, ``[0, 1, 0]`` and ``[0, 0, 1]``, and
        a sphere with radius 0.5 centered at the origin

        >>> import pyvista as pv  # doctest:+SKIP
        >>> sphere = pv.Sphere()  # doctest:+SKIP
        >>> points, rays, cells = sphere.multi_ray_trace(
        ...     [[0, 0, 0]] * 3,
        ...     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ...     first_point=True,
        ... )  # doctest:+SKIP
        >>> string = ', '.join(
        ...     [
        ...         f'({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})'
        ...         for point in points
        ...     ]
        ... )  # doctest:+SKIP
        >>> f'Rays intersected at {string}'  # doctest:+SKIP
        'Rays intersected at (0.499, 0.000, 0.000), (0.000, 0.497, 0.000), (0.000, 0.000, 0.500)'

        """
        if not self.is_all_triangles:
            msg = 'Input mesh for multi_ray_trace must be all triangles.'
            raise NotAllTrianglesError(msg)

        try:
            import trimesh  # noqa: PLC0415

            if not trimesh.ray.has_embree:
                raise ImportError
        except ImportError:
            msg = (
                'To use multi_ray_trace please install trimesh, embree (v2.17.7) '
                'and pyembree/embreex with:\n'
                '\tconda install embree=2 trimesh pyembree\nOR\n'
                '\tpip install trimesh embreex'
            )
            raise ImportError(msg)

        origins = np.asarray(origins)
        directions = np.asarray(directions)
        tmesh = trimesh.Trimesh(self.points, self.regular_faces)
        locations, index_ray, index_tri = tmesh.ray.intersects_location(
            origins,
            directions,
            multiple_hits=not first_point,
        )
        if retry:
            # gather intersecting rays in lists
            loc_lst = locations.tolist()
            ray_lst = index_ray.tolist()
            tri_lst = index_tri.tolist()

            # find indices that trimesh failed on
            all_ray_indices = np.arange(len(origins))
            retry_ray_indices = np.setdiff1d(all_ray_indices, index_ray, assume_unique=True)

            # compute ray points for all failed rays at once
            origins_retry = origins[retry_ray_indices, :]  # shape (n_retry, 3)
            directions_retry = directions[retry_ray_indices, :]
            unit_directions = directions_retry / np.linalg.norm(
                directions_retry,
                axis=1,
                keepdims=True,
            )

            origin_to_centre_vectors = self.center - origins_retry  # shape (n_retry, 3)
            origin_to_centre_lengths = np.linalg.norm(
                origin_to_centre_vectors,
                axis=-1,
                keepdims=True,
            )
            second_points = origins_retry + unit_directions * (
                origin_to_centre_lengths + self.length
            )

            for id_r, origin, second_point in zip(retry_ray_indices, origins_retry, second_points):
                locs, indices = self.ray_trace(origin, second_point, first_point=first_point)
                if locs.any():
                    if first_point:
                        locs = locs.reshape([1, 3])
                    ray_lst.extend([id_r] * indices.size)
                    tri_lst.extend(indices)
                    loc_lst.extend(locs)

            # sort result arrays by ray index
            index_ray = np.array(ray_lst)
            sorting_inds = index_ray.argsort()
            index_ray = index_ray[sorting_inds]
            index_tri = np.array(tri_lst)[sorting_inds]
            locations = np.array(loc_lst)[sorting_inds]

        return locations, index_ray, index_tri

    @_deprecate_positional_args
    def plot_boundaries(  # type: ignore[misc]
        self: PolyData,
        edge_color='red',
        line_width=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ):
        """Plot boundaries of a mesh.

        Parameters
        ----------
        edge_color : ColorLike, default: "red"
            The color of the edges when they are added to the plotter.

        line_width : int, optional
            Width of the boundary lines.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            All additional keyword arguments will be passed to
            :func:`pyvista.Plotter.add_mesh`.

        Returns
        -------
        pyvista.CameraPosition
            List of camera position, focal point, and view up.
            Returned when ``return_cpos`` is ``True``.

        Examples
        --------
        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> hills.plot_boundaries(line_width=10)

        """
        edges = DataSetFilters.extract_feature_edges(self, progress_bar=progress_bar)

        plotter = pyvista.Plotter(
            off_screen=kwargs.pop('off_screen', None),
            notebook=kwargs.pop('notebook', None),
        )
        plotter.add_mesh(
            edges,
            color=edge_color,
            style='wireframe',
            label='Edges',
            line_width=line_width,
        )
        plotter.add_mesh(self, label='Mesh', **kwargs)
        plotter.add_legend()  # type: ignore[call-arg]
        return plotter.show()

    @_deprecate_positional_args
    def plot_normals(  # type: ignore[misc]  # noqa: PLR0917
        self: PolyData,
        show_mesh: bool = True,  # noqa: FBT001, FBT002
        mag=1.0,
        flip: bool = False,  # noqa: FBT001, FBT002
        use_every=1,
        faces: bool = False,  # noqa: FBT001, FBT002
        color=None,
        **kwargs,
    ):
        """Plot the point normals of a mesh.

        Parameters
        ----------
        show_mesh : bool, default: True
            Plot the mesh itself.

        mag : float, default: 1.0
            Size magnitude of the normal arrows.

        flip : bool, default: False
            Flip the normal direction when ``True``.

        use_every : int, default: 1
            Display every nth normal.  By default every normal is
            displayed.  Display every 10th normal by setting this
            parameter to 10.

        faces : bool, default: False
            Plot face normals instead of the default point normals.

        color : ColorLike, optional
            Color of the arrows.  Defaults to
            :attr:`pyvista.plotting.themes.Theme.edge_color`.

        **kwargs : dict, optional
            All additional keyword arguments will be passed to
            :func:`pyvista.Plotter.add_mesh`.

        Returns
        -------
        pyvista.CameraPosition
            List of camera position, focal point, and view up.
            Returned when ``return_cpos`` is ``True``.

        Examples
        --------
        Plot the point normals of a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(phi_resolution=10, theta_resolution=10)
        >>> sphere.plot_normals(mag=0.1, show_edges=True)

        Plot the face normals of a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(phi_resolution=10, theta_resolution=10)
        >>> sphere.plot_normals(mag=0.1, faces=True, show_edges=True)

        """
        plotter = pyvista.Plotter(
            off_screen=kwargs.pop('off_screen', None),
            notebook=kwargs.pop('notebook', None),
        )
        if show_mesh:
            plotter.add_mesh(self, **kwargs)

        color = pyvista.Color(color, default_color=pyvista.global_theme.edge_color)

        if faces:
            centers = self.cell_centers().points[::use_every]
            normals = self.cell_normals
        else:
            centers = self.points[::use_every]
            normals = self.point_normals

        if flip:
            normals *= -1  # type: ignore[misc]

        plotter.add_arrows(
            centers,
            normals[::use_every],
            mag=mag,
            color=color,
            show_scalar_bar=False,
        )

        return plotter.show()

    @_deprecate_positional_args(allowed=['remove', 'mode'])
    def remove_points(  # noqa: PLR0917
        self,
        remove,
        mode='any',
        keep_scalars: bool = True,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ):
        """Rebuild a mesh by removing points.

        Only valid for all-triangle meshes.

        Parameters
        ----------
        remove : sequence[bool | int]
            If remove is a bool array, points that are ``True`` will
            be removed.  Otherwise, it is treated as a list of
            indices.

        mode : str, default: "any"
            When ``'all'``, only faces containing all points flagged
            for removal will be removed.

        keep_scalars : bool, default: True
            When ``True``, point and cell scalars will be passed on to
            the new mesh.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.PolyData
            Mesh without the points flagged for removal.

        numpy.ndarray
            Indices of new points relative to the original mesh.

        Examples
        --------
        Remove the first 100 points from a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> reduced_sphere, ridx = sphere.remove_points(range(100, 250))
        >>> reduced_sphere.plot(show_edges=True, line_width=3)

        """
        remove = np.asarray(remove)

        # np.asarray will eat anything, so we have to weed out bogus inputs
        if not issubclass(remove.dtype.type, (np.bool_, np.integer)):
            msg = 'Remove must be either a mask or an integer array-like'
            raise TypeError(msg)

        if remove.dtype == np.bool_:
            if remove.size != self.n_points:  # type: ignore[attr-defined]
                msg = 'Mask different size than n_points'
                raise ValueError(msg)
            remove_mask = remove
        else:
            remove_mask = np.zeros(self.n_points, np.bool_)  # type: ignore[attr-defined]
            remove_mask[remove] = True

        if not self.is_all_triangles:  # type: ignore[attr-defined]
            raise NotAllTrianglesError

        f = self.faces.reshape(-1, 4)[:, 1:]  # type: ignore[attr-defined]
        vmask = remove_mask.take(f)
        fmask = ~vmask.all(1) if mode == 'all' else ~vmask.any(1)

        # Regenerate face and point arrays
        uni = np.unique(f.compress(fmask, 0), return_inverse=True)
        new_points = self.points.take(uni[0], 0)

        nfaces = fmask.sum()
        faces = np.empty((nfaces, 4), dtype=pyvista.ID_TYPE)
        faces[:, 0] = 3
        faces[:, 1:] = np.reshape(uni[1], (nfaces, 3))

        newmesh = pyvista.PolyData(new_points, faces, deep=True)
        ridx = uni[0]

        # Add scalars back to mesh if requested
        if keep_scalars:
            for key in self.point_data:  # type: ignore[attr-defined]
                newmesh.point_data[key] = self.point_data[key][ridx]  # type: ignore[attr-defined]

            for key in self.cell_data:  # type: ignore[attr-defined]
                try:
                    newmesh.cell_data[key] = self.cell_data[key][fmask]  # type: ignore[attr-defined]
                except (ValueError, TypeError, KeyError):  # pragma: no cover
                    warnings.warn(f'Unable to pass cell key {key} onto reduced mesh')

        # Return vtk surface and reverse indexing array
        if inplace:
            self.copy_from(newmesh, deep=False)  # type: ignore[attr-defined]
            return self, ridx
        return newmesh, ridx

    def flip_normals(self):
        """Flip normals of a triangular mesh by reversing the point ordering.

        .. deprecated:: 0.45

            This filter is deprecated. Use ``flip_faces(inplace=True)`` instead.

        Examples
        --------
        Flip the normals of a sphere and plot the normals before and
        after the flip.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.plot_normals(mag=0.1)
        >>> sphere.flip_normals()  # doctest:+SKIP
        >>> sphere.plot_normals(mag=0.1, opacity=0.5)

        """
        # Deprecated on v0.45.0, estimated removal on v0.48.0
        warnings.warn(
            '`flip_normals` is deprecated. Use `flip_faces` instead. '
            'Note that `inplace` is now `False` by default for the new filter.',
            PyVistaDeprecationWarning,
        )
        if not self.is_all_triangles:  # type: ignore[attr-defined]
            msg = 'Can only flip normals on an all triangle mesh.'
            raise NotAllTrianglesError(msg)

        f = self._connectivity_array  # type: ignore[attr-defined]

        # swap first and last point index in-place
        # See: https://stackoverflow.com/a/33362288/
        f[::3], f[2::3] = f[2::3], f[::3].copy()

    def _reverse_sense(  # type: ignore[misc]
        self: PolyData,
        *,
        reverse_cells: bool,
        reverse_normals: bool,
        inplace: bool,
        progress_bar: bool,
    ):
        """Flip faces and/or normal vectors."""
        alg = _vtk.vtkReverseSense()
        alg.SetInputData(self)
        alg.SetReverseNormals(reverse_normals)
        alg.SetReverseCells(reverse_cells)
        _update_alg(alg, progress_bar=progress_bar)
        output = _get_output(alg)
        if inplace:
            self.copy_from(output, deep=False)
            return self
        return output

    def flip_faces(  # type: ignore[misc]
        self: PolyData,
        *,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        """Flip the orientation of the faces.

        The flip is performed by reversing the order of indices in the cell
        connectivity list. In other libraries, this operation is sometimes
        referred to as "flip orientation", "reverse cells", "reverse face
        orientations", or similar.

        .. note::

            Polygon cells have an implicitly-defined orientation, and reversing
            the ordering affects how normals are computed by filters like
            :meth:`~pyvista.PolyDataFilters.compute_normals`.

        .. note::
            This filter does not modify any existing normals which may be present
            in the dataset. Use :meth:`~pyvista.PolyDataFilters.flip_normal_vectors`
            to flip the normal vectors.

        .. warning::

            This filter does not produce the correct output for
            :attr:`~pyvista.CellType.TRIANGLE_STRIP` cells,
            see https://gitlab.kitware.com/vtk/vtk/-/issues/18634.
            Use :meth:`~pyvista.PolyDataFilters.triangulate` to triangulate the mesh
            first.

        .. versionadded:: 0.45

        Parameters
        ----------
        inplace : bool, default: False
            Overwrites the original mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh with flipped faces.

        See Also
        --------
        pyvista.PolyDataFilters.flip_normal_vectors
            Flip the direction of the point and cell normals.
        pyvista.PolyDataFilters.compute_normals
            Compute new normals. Includes the option to flip the normals.
        pyvista.PolyData.point_normals
            Returns the array of point normals.
        pyvista.PolyData.cell_normals
            Returns the array of cell normals.

        Examples
        --------
        Flip the faces of a sphere. Show the point ids of the first cell
        before and after the flip.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.regular_faces[0]
        array([ 2, 30,  0])

        >>> sphere_flipped = sphere.flip_faces()
        >>> sphere_flipped.regular_faces[0]
        array([ 0, 30,  2])

        Note that the sphere also has pre-computed normals which have not been
        affected by this filter.

        >>> sphere.point_data['Normals'][0]
        pyvista_ndarray([0., 0., 1.], dtype=float32)

        >>> sphere_flipped.point_data['Normals'][0]
        pyvista_ndarray([0., 0., 1.], dtype=float32)

        See :ref:`boolean_operations_example` for more examples using this filter.

        """
        return self._reverse_sense(
            reverse_cells=True,
            reverse_normals=False,
            inplace=inplace,
            progress_bar=progress_bar,
        )

    def flip_normal_vectors(  # type: ignore[misc]
        self: PolyData,
        *,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        """Flip the direction of the mesh's point and cell normal vectors.

        This filter effectively multiplies the point and cell normals by ``-1``.
        It has no effect if no active normals are currently set.

        .. note::

            Polygon cells have an implicitly-defined orientation which may differ
            from the orientation of the normal vectors. To ensure that the normals
            are consistent with this implicit definition, consider also using
            :meth:`~pyvista.PolyDataFilters.flip_faces` or re-computing normals with
            :meth:`~pyvista.PolyDataFilters.compute_normals` and enabling the
            `flip_normals` option.

        .. versionadded:: 0.45

        Parameters
        ----------
        inplace : bool, default: False
            Overwrites the original mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh with point and cell normal directions flipped.

        See Also
        --------
        pyvista.PolyDataFilters.flip_faces
            Flip the orientation of the faces.
        pyvista.PolyDataFilters.compute_normals
            Compute new normals. Includes the option to flip the normals.
        pyvista.PolyData.point_normals
            Returns the array of point normals.
        pyvista.PolyData.cell_normals
            Returns the array of cell normals.

        Examples
        --------
        Flip the normal vectors of a sphere. Show one of the normal vectors
        before and after the flip.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.point_data['Normals'][0]
        pyvista_ndarray([0., 0., 1.], dtype=float32)

        >>> sphere_flipped = sphere.flip_normal_vectors()
        >>> sphere_flipped.point_data['Normals'][0]
        pyvista_ndarray([-0., -0., -1.], dtype=float32)

        Note that the sphere's cell ordering has not been affected by this filter.

        >>> sphere.regular_faces[0]
        array([ 2, 30,  0])

        >>> sphere_flipped.regular_faces[0]
        array([ 2, 30,  0])

        """
        return self._reverse_sense(
            reverse_cells=False,
            reverse_normals=True,
            inplace=inplace,
            progress_bar=progress_bar,
        )

    @_deprecate_positional_args
    def delaunay_2d(  # noqa: PLR0917
        self,
        tol=1e-05,
        alpha=0.0,
        offset=1.0,
        bound: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
        edge_source=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Apply a 2D Delaunay filter along the best fitting plane.

        This filter can be used to generate a 2d surface from a set of
        points on a plane.  If you want to create a surface from a
        point cloud, see :func:`pyvista.PolyDataFilters.reconstruct_surface`.

        Parameters
        ----------
        tol : float, default: 1e-05
            Specify a tolerance to control discarding of closely
            spaced points. This tolerance is specified as a fraction
            of the diagonal length of the bounding box of the points.

        alpha : float, default: 0.0
            Specify alpha (or distance) value to control output of
            this filter. For a non-zero alpha value, only edges or
            triangles contained within a sphere centered at mesh
            vertices will be output. Otherwise, only triangles will be
            output.

        offset : float, default: 1.0
            Specify a multiplier to control the size of the initial,
            bounding Delaunay triangulation.

        bound : bool, default: False
            Boolean controls whether bounding triangulation points
            and associated triangles are included in the
            output. These are introduced as an initial triangulation
            to begin the triangulation process. This feature is nice
            for debugging output.

        inplace : bool, default: False
            If ``True``, overwrite this mesh with the triangulated
            mesh.

        edge_source : pyvista.PolyData, optional
            Specify the source object used to specify constrained
            edges and loops. If set, and lines/polygons are defined, a
            constrained triangulation is created. The lines/polygons
            are assumed to reference points in the input point set
            (i.e. point ids are identical in the input and
            source).

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh from the 2D delaunay filter.

        Examples
        --------
        First, generate 30 points on circle and plot them.

        >>> import pyvista as pv
        >>> points = pv.Polygon(n_sides=30).points
        >>> circle = pv.PolyData(points)
        >>> circle.plot(show_edges=True, point_size=15)

        Use :func:`delaunay_2d` to fill the interior of the circle.

        >>> filled_circle = circle.delaunay_2d()
        >>> filled_circle.plot(show_edges=True, line_width=5)

        Use the ``edge_source`` parameter to create a constrained delaunay
        triangulation and plot it.

        >>> squar = pv.Polygon(n_sides=4, radius=8, fill=False)
        >>> squar = squar.rotate_z(45, inplace=False)
        >>> circ0 = pv.Polygon(center=(2, 3, 0), n_sides=30, radius=1)
        >>> circ1 = pv.Polygon(center=(-2, -3, 0), n_sides=30, radius=1)
        >>> comb = circ0.append_polydata(circ1, squar)
        >>> tess = comb.delaunay_2d(edge_source=comb)
        >>> tess.plot(cpos='xy', show_edges=True)

        See :ref:`create_tri_surface_example` for more examples using this filter.

        """
        alg = _vtk.vtkDelaunay2D()
        alg.SetProjectionPlaneMode(_vtk.VTK_BEST_FITTING_PLANE)
        alg.SetInputDataObject(self)
        alg.SetTolerance(tol)
        alg.SetAlpha(alpha)
        alg.SetOffset(offset)
        alg.SetBoundingTriangulation(bound)
        if edge_source is not None:
            alg.SetSourceData(edge_source)
        _update_alg(alg, progress_bar=progress_bar, message='Computing 2D Triangulation')

        # Sometimes lines are given in the output. The
        # `.triangulate()` filter cleans those
        mesh = _get_output(alg).triangulate()
        if inplace:
            self.copy_from(mesh, deep=False)  # type: ignore[attr-defined]
            return self
        return mesh

    @_deprecate_positional_args
    def compute_arc_length(self, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Compute the arc length over the length of the probed line.

        It adds a new point-data array named ``"arc_length"`` with the
        computed arc length for each of the polylines in the
        input. For all other cell types, the arc length is set to 0.

        Parameters
        ----------
        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        float
            Arc length of the length of the probed line.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> path = sphere.geodesic(0, 100)
        >>> length = path.compute_arc_length()['arc_length'][-1]
        >>> f'Length is {length:.3f}'
        'Length is 0.812'

        This is identical to the geodesic_distance.

        >>> length = sphere.geodesic_distance(0, 100)
        >>> f'Length is {length:.3f}'
        'Length is 0.812'

        You can also plot the arc_length.

        >>> arc = path.compute_arc_length()
        >>> arc.plot(scalars='arc_length')

        """
        alg = _vtk.vtkAppendArcLength()
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Computing the Arc Length')
        return _get_output(alg)

    @_deprecate_positional_args
    def project_points_to_plane(self, origin=None, normal=(0.0, 0.0, 1.0), inplace: bool = False):  # noqa: FBT001, FBT002
        """Project points of this mesh to a plane.

        Parameters
        ----------
        origin : sequence[float], optional
            Plane origin.  Defaults to the approximate center of the
            input mesh minus half the length of the input mesh in the
            direction of the normal.

        normal : sequence[float], default: (0.0, 0.0, 1.0)
            Plane normal.  Defaults to +Z.

        inplace : bool, default: False
            Whether to overwrite the original mesh with the projected
            points.

        Returns
        -------
        pyvista.PolyData
            The points of this mesh projected onto a plane.

        See Also
        --------
        :ref:`project_plane_example`
        :ref:`project_points_tessellate_example`

        Examples
        --------
        Flatten a sphere to the XY plane.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> projected = sphere.project_points_to_plane()
        >>> projected.plot(show_edges=True, line_width=3)

        """
        if not isinstance(normal, (np.ndarray, Sequence)) or len(normal) != 3:
            msg = 'Normal must be a length three vector'
            raise TypeError(msg)
        if origin is None:
            origin = np.array(self.center) - np.array(normal) * self.length / 2.0  # type: ignore[attr-defined]
        # choose what mesh to use
        mesh = self.copy() if not inplace else self  # type: ignore[attr-defined]
        # Make plane
        plane = generate_plane(normal, origin)
        # Perform projection in place on the copied mesh
        f = lambda p: plane.ProjectPoint(p, p)
        np.apply_along_axis(f, 1, mesh.points)
        return mesh

    @_deprecate_positional_args
    def ribbon(  # noqa: PLR0917
        self,
        width=None,
        scalars=None,
        angle=0.0,
        factor=2.0,
        normal=None,
        tcoords: bool | str = False,  # noqa: FBT001, FBT002
        preference='points',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Create a ribbon of the lines in this dataset.

        .. note::
           If there are no lines in the input dataset, then the output
           will be an empty :class:`pyvista.PolyData` mesh.

        Parameters
        ----------
        width : float, optional
            Set the "half" width of the ribbon. If the width is
            allowed to vary, this is the minimum width. The default is
            10% the length.

        scalars : str, optional
            String name of the scalars array to use to vary the ribbon
            width.  This is only used if a scalars array is specified.

        angle : float, optional
            Angle in degrees of the offset angle of the ribbon from
            the line normal. The default is 0.0.

        factor : float, optional
            Set the maximum ribbon width in terms of a multiple of the
            minimum width. The default is 2.0.

        normal : sequence[float], optional
            Normal to use as default.

        tcoords : bool, str, optional
            If ``True``, generate texture coordinates along the
            ribbon. This can also be specified to generate the texture
            coordinates with either ``'length'`` or ``'normalized'``.

        preference : str, optional
            The field preference when searching for the scalars array by
            name.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Ribbon mesh.  Empty if there are no lines in the input dataset.

        Examples
        --------
        Convert a line to a ribbon and plot it.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> n = 1000
        >>> theta = np.linspace(-10 * np.pi, 10 * np.pi, n)
        >>> z = np.linspace(-2, 2, n)
        >>> r = z**2 + 1
        >>> x = r * np.sin(theta)
        >>> y = r * np.cos(theta)
        >>> points = np.column_stack((x, y, z))
        >>> pdata = pv.PolyData(points)
        >>> pdata.lines = np.hstack((n, range(n)))
        >>> pdata['distance'] = range(n)
        >>> ribbon = pdata.ribbon(width=0.2)
        >>> ribbon.plot(show_scalar_bar=False)

        """
        if scalars is not None:
            field = get_array_association(self, scalars, preference=preference)  # type: ignore[arg-type]
        if width is None:
            width = self.length * 0.1  # type: ignore[attr-defined]
        alg = _vtk.vtkRibbonFilter()
        alg.SetInputDataObject(self)
        alg.SetWidth(width)
        if normal is not None:
            alg.SetUseDefaultNormal(True)
            alg.SetDefaultNormal(normal)
        alg.SetAngle(angle)
        if scalars is not None:
            alg.SetVaryWidth(True)
            alg.SetInputArrayToProcess(
                0,
                0,
                0,
                field.value,
                scalars,
            )  # args: (idx, port, connection, field, name)
            alg.SetWidthFactor(factor)
        else:
            alg.SetVaryWidth(False)
        if tcoords:
            alg.SetGenerateTCoords(True)
            if isinstance(tcoords, str):
                if tcoords.lower() == 'length':
                    alg.SetGenerateTCoordsToUseLength()
                elif tcoords.lower() == 'normalized':
                    alg.SetGenerateTCoordsToNormalizedLength()
            else:
                alg.SetGenerateTCoordsToUseLength()
        else:
            alg.SetGenerateTCoordsToOff()
        _update_alg(alg, progress_bar=progress_bar, message='Creating a Ribbon')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['vector'])
    def extrude(  # noqa: PLR0917
        self,
        vector,
        capping=None,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Sweep polygonal data creating a "skirt" from free edges.

        This will create a line from vertices.

        This takes polygonal data as input and generates polygonal
        data on output. The input dataset is swept according to some
        extrusion function and creates new polygonal primitives. These
        primitives form a "skirt" or swept surface. For example,
        sweeping a line results in a quadrilateral, and sweeping a
        triangle creates a "wedge".

        The skirt is generated by locating certain topological
        features. Free edges (edges of polygons or triangle strips
        only used by one polygon or triangle strips) generate
        surfaces. This is true also of lines or polylines. Vertices
        generate lines.

        .. versionchanged:: 0.32.0
           The ``capping`` keyword was added with a default of ``False``.
           The previously used VTK default corresponds to ``capping=True``.
           In a future version the default will be changed to ``True`` to
           match the behavior of the underlying VTK filter.

        Parameters
        ----------
        vector : numpy.ndarray or sequence
            Direction and length to extrude the mesh in.

        capping : bool, optional
            Control if the sweep of a 2D object is capped. The default is
            ``False``, which differs from VTK's default.

            .. warning::
               The ``capping`` keyword was added in version 0.32.0 with a
               default value of ``False``. In a future version this default
               will be changed to ``True`` to match the behavior of the
               underlying VTK filter. It is recommended to explicitly pass
               a value for this keyword argument to prevent future changes
               in behavior and warnings.

        inplace : bool, default: False
            Overwrites the original mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Extruded mesh.

        Examples
        --------
        Extrude a half circle arc.

        >>> import pyvista as pv
        >>> arc = pv.CircularArc(pointa=[-1, 0, 0], pointb=[1, 0, 0], center=[0, 0, 0])
        >>> mesh = arc.extrude([0, 0, 1], capping=False)
        >>> mesh.plot(color='lightblue')

        Extrude and cap an 8 sided polygon.

        >>> poly = pv.Polygon(n_sides=8)
        >>> mesh = poly.extrude((0, 0, 1.5), capping=True)
        >>> mesh.plot(line_width=5, show_edges=True)

        """
        if capping is None:
            capping = False
            warnings.warn(
                'The default value of the ``capping`` keyword argument will change in '
                'a future version to ``True`` to match the behavior of VTK. We recommend '
                'passing the keyword explicitly to prevent future surprises.',
                PyVistaFutureWarning,
            )

        alg = _vtk.vtkLinearExtrusionFilter()
        alg.SetExtrusionTypeToVectorExtrusion()
        alg.SetVector(*vector)
        alg.SetInputData(self)
        alg.SetCapping(capping)
        _update_alg(alg, progress_bar=progress_bar, message='Extruding')
        output = _get_output(alg)
        if inplace:
            self.copy_from(output, deep=False)  # type: ignore[attr-defined]
            return self
        return output

    @_deprecate_positional_args
    def extrude_rotate(  # noqa: PLR0917
        self,
        resolution=30,
        inplace: bool = False,  # noqa: FBT001, FBT002
        translation=0.0,
        dradius=0.0,
        angle=360.0,
        capping=None,
        rotation_axis=(0, 0, 1),
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Sweep polygonal data creating "skirt" from free edges/lines, and lines from vertices.

        This takes polygonal data as input and generates polygonal
        data on output. The input dataset is swept around the axis
        to create new polygonal primitives. These primitives form a
        "skirt" or swept surface. For example, sweeping a line results
        in a cylindrical shell, and sweeping a circle creates a torus.

        There are a number of control parameters for this filter.  You
        can control whether the sweep of a 2D object (i.e., polygon or
        triangle strip) is capped with the generating geometry via the
        ``capping`` parameter. Also, you can control the angle of
        rotation, and whether translation along the axis is
        performed along with the rotation.  (Translation is useful for
        creating "springs".) You also can adjust the radius of the
        generating geometry with the ``dradius`` parameter.

        The skirt is generated by locating certain topological
        features. Free edges (edges of polygons or triangle strips
        only used by one polygon or triangle strips) generate
        surfaces. This is true also of lines or polylines. Vertices
        generate lines.

        This filter can be used to model axisymmetric objects like
        cylinders, bottles, and wine glasses; or translational
        rotational symmetric objects like springs or corkscrews.

        .. versionchanged:: 0.32.0
           The ``capping`` keyword was added with a default of ``False``.
           The previously used VTK default corresponds to ``capping=True``.
           In a future version the default will be changed to ``True`` to
           match the behavior of the underlying VTK filter.

        Parameters
        ----------
        resolution : int, optional
            Number of pieces to divide line into.

        inplace : bool, default: False
            Overwrites the original mesh inplace.

        translation : float, optional
            Total amount of translation along the axis.

        dradius : float, optional
            Change in radius during sweep process.

        angle : float, optional
            The angle of rotation in degrees.

        capping : bool, optional
            Control if the sweep of a 2D object is capped. The default is
            ``False``, which differs from VTK's default.

            .. warning::
               The ``capping`` keyword was added in version 0.32.0 with a
               default value of ``False``. In a future version this default
               will be changed to ``True`` to match the behavior of the
               underlying VTK filter. It is recommended to explicitly pass
               a value for this keyword argument to prevent future changes
               in behavior and warnings.

        rotation_axis : numpy.ndarray or sequence, optional
            The direction vector of the axis around which the rotation is done.
            It requires vtk>=9.1.0.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Rotationally extruded mesh.

        See Also
        --------
        :ref:`extrude_rotate_example`

        Examples
        --------
        Create a "spring" using the rotational extrusion filter.

        >>> import pyvista as pv
        >>> profile = pv.Polygon(
        ...     center=[1.25, 0.0, 0.0],
        ...     radius=0.2,
        ...     normal=(0, 1, 0),
        ...     n_sides=30,
        ... )
        >>> extruded = profile.extrude_rotate(
        ...     resolution=360,
        ...     translation=4.0,
        ...     dradius=0.5,
        ...     angle=1500.0,
        ...     capping=True,
        ... )
        >>> extruded.plot(smooth_shading=True)

        Create a "wine glass" using the rotational extrusion filter.

        >>> import numpy as np
        >>> points = np.array(
        ...     [
        ...         [-0.18, 0, 0],
        ...         [-0.18, 0, 0.01],
        ...         [-0.18, 0, 0.02],
        ...         [-0.01, 0, 0.03],
        ...         [-0.01, 0, 0.04],
        ...         [-0.02, 0, 0.5],
        ...         [-0.05, 0, 0.75],
        ...         [-0.1, 0, 0.8],
        ...         [-0.2, 0, 1.0],
        ...     ]
        ... )
        >>> spline = pv.Spline(points, 30)
        >>> extruded = spline.extrude_rotate(resolution=20, capping=False)
        >>> extruded.plot(color='lightblue')

        """
        if capping is None:
            capping = False
            warnings.warn(
                'The default value of the ``capping`` keyword argument will change in '
                'a future version to ``True`` to match the behavior of VTK. We recommend '
                'passing the keyword explicitly to prevent future surprises.',
                PyVistaFutureWarning,
            )

        if not isinstance(rotation_axis, (np.ndarray, Sequence)) or len(rotation_axis) != 3:
            msg = 'Vector must be a length three vector'
            raise ValueError(msg)

        if resolution <= 0:
            msg = '`resolution` should be positive'
            raise ValueError(msg)
        alg = _vtk.vtkRotationalExtrusionFilter()
        alg.SetInputData(self)
        alg.SetResolution(resolution)
        alg.SetTranslation(translation)
        alg.SetDeltaRadius(dradius)
        alg.SetCapping(capping)
        alg.SetAngle(angle)
        if pyvista.vtk_version_info >= (9, 1, 0):
            alg.SetRotationAxis(rotation_axis)  # type: ignore[arg-type]
        elif rotation_axis != (0, 0, 1):
            msg = (
                'The installed version of VTK does not support '
                'setting the direction vector of the axis around which the rotation is done.'
            )
            raise VTKVersionError(msg)

        _update_alg(alg, progress_bar=progress_bar, message='Extruding')
        output = wrap(alg.GetOutput())
        if inplace:
            self.copy_from(output, deep=False)  # type: ignore[attr-defined]
            return self
        return output

    @_deprecate_positional_args(allowed=['direction', 'trim_surface'])
    def extrude_trim(  # noqa: PLR0917
        self,
        direction,
        trim_surface,
        extrusion='boundary_edges',
        capping='intersection',
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Extrude polygonal data trimmed by a surface.

        The input dataset is swept along a specified direction forming a
        "skirt" from the boundary edges 2D primitives (i.e., edges used
        by only one polygon); and/or from vertices and lines. The extent
        of the sweeping is defined where the sweep intersects a
        user-specified surface.

        Parameters
        ----------
        direction : numpy.ndarray or sequence
            Direction vector to extrude.

        trim_surface : pyvista.PolyData
            Surface which trims the surface.

        extrusion : str, default: "boundary_edges"
            Control the strategy of extrusion. One of the following:

            * ``"boundary_edges"``
            * ``"all_edges"``

            The default only generates faces on the boundary of the original
            input surface. When using ``"all_edges"``, faces are created along
            interior points as well.

        capping : str, default: "intersection"
            Control the strategy of capping. One of the following:

            * ``"intersection"``
            * ``"minimum_distance"``
            * ``"maximum_distance"``
            * ``"average_distance"``

        inplace : bool, default: False
            Overwrites the original mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Extruded mesh trimmed by a surface.

        See Also
        --------
        :ref:`extrude_trim_example`

        Examples
        --------
        Extrude a disc.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> plane = pv.Plane(i_size=2, j_size=2, direction=[0, 0.8, 1])
        >>> disc = pv.Disc(center=(0, 0, -1), c_res=50)
        >>> direction = [0, 0, 1]
        >>> extruded_disc = disc.extrude_trim(direction, plane)
        >>> extruded_disc.plot(smooth_shading=True, split_sharp_edges=True)

        """
        if not isinstance(direction, (np.ndarray, Sequence)) or len(direction) != 3:
            msg = 'Vector must be a length three vector'
            raise TypeError(msg)

        extrusions = {'boundary_edges': 0, 'all_edges': 1}
        if isinstance(extrusion, str):
            if extrusion not in extrusions:
                msg = f'Invalid strategy of extrusion "{extrusion}".'
                raise ValueError(msg)
            extrusion = extrusions[extrusion]
        else:
            msg = 'Invalid type given to `extrusion`. Must be a string.'
            raise TypeError(msg)

        cappings = {
            'intersection': 0,
            'minimum_distance': 1,
            'maximum_distance': 2,
            'average_distance': 3,
        }
        if isinstance(capping, str):
            if capping not in cappings:
                msg = f'Invalid strategy of capping "{capping}".'
                raise ValueError(msg)
            capping = cappings[capping]
        else:
            msg = 'Invalid type given to `capping`. Must be a string.'
            raise TypeError(msg)

        alg = _vtk.vtkTrimmedExtrusionFilter()
        alg.SetInputData(self)
        alg.SetExtrusionDirection(*direction)
        alg.SetTrimSurfaceData(trim_surface)
        alg.SetExtrusionStrategy(extrusion)
        alg.SetCappingStrategy(capping)
        _update_alg(alg, progress_bar=progress_bar, message='Extruding with trimming')
        output = wrap(alg.GetOutput())
        if inplace:
            self.copy_from(output, deep=False)  # type: ignore[attr-defined]
            return self
        return output

    @_deprecate_positional_args
    def strip(  # noqa: PLR0917
        self,
        join: bool = False,  # noqa: FBT001, FBT002
        max_length=1000,
        pass_cell_data: bool = False,  # noqa: FBT001, FBT002
        pass_cell_ids: bool = False,  # noqa: FBT001, FBT002
        pass_point_ids: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Strip poly data cells.

        Generates triangle strips and/or poly-lines from input
        polygons, triangle strips, and lines.

        Polygons are assembled into triangle strips only if they are
        triangles; other types of polygons are passed through to the
        output and not stripped. (Use ``triangulate`` filter to
        triangulate non-triangular polygons prior to running this
        filter if you need to strip all the data.) The filter will
        pass through (to the output) vertices if they are present in
        the input polydata.

        Also note that if triangle strips or polylines are defined in
        the input they are passed through and not joined nor
        extended. (If you wish to strip these use ``triangulate``
        filter to fragment the input into triangles and lines prior to
        running this filter.)

        This filter implements :vtk:`vtkStripper`.

        Parameters
        ----------
        join : bool, default: False
            If ``True``, the output polygonal segments will be joined
            if they are contiguous. This is useful after slicing a
            surface.

        max_length : int, default: 1000
            Specify the maximum number of triangles in a triangle
            strip, and/or the maximum number of lines in a poly-line.

        pass_cell_data : bool, default: False
            Enable/Disable passing of the CellData in the input to the
            output as FieldData. Note the field data is transformed.

        pass_cell_ids : bool, default: False
            If ``True``, the output polygonal dataset will have a
            celldata array that holds the cell index of the original
            3D cell that produced each output cell. This is useful for
            picking. The default is ``False`` to conserve memory.

        pass_point_ids : bool, default: False
            If ``True``, the output polygonal dataset will have a
            pointdata array that holds the point index of the original
            vertex that produced each output vertex. This is useful
            for picking. The default is ``False`` to conserve memory.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Stripped mesh.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> slc = mesh.slice(normal='z', origin=(0, 0, -10))
        >>> stripped = slc.strip()
        >>> stripped.n_cells
        1
        >>> stripped.plot(show_edges=True, line_width=3)

        """
        alg = _vtk.vtkStripper()
        alg.SetInputDataObject(self)
        alg.SetJoinContiguousSegments(join)
        alg.SetMaximumLength(max_length)
        alg.SetPassCellDataAsFieldData(pass_cell_data)
        alg.SetPassThroughCellIds(pass_cell_ids)
        alg.SetPassThroughPointIds(pass_point_ids)
        _update_alg(alg, progress_bar=progress_bar, message='Stripping Mesh')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['other_mesh'])
    def collision(  # noqa: PLR0917
        self,
        other_mesh,
        contact_mode=0,
        box_tolerance=0.001,
        cell_tolerance=0.0,
        n_cells_per_node=2,
        generate_scalars: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Perform collision determination between two polyhedral surfaces.

        If ``collision_mode`` is set to all contacts, the output will
        be lines of contact. If ``collision_mode`` is first contact or half
        contacts then the Contacts output will be vertices.

        .. warning::
            Currently only triangles are processed. Use
            :func:`PolyDataFilters.triangulate` to convert any strips
            or polygons to triangles.  Otherwise, the mesh will be
            converted for you within this method.

        Parameters
        ----------
        other_mesh : pyvista.DataSet
            Other mesh to test collision with.  If the other mesh is
            not a surface, its external surface will be extracted and
            triangulated.

        contact_mode : int, default: 0
            Contact mode.  One of the following:

            * 0 - All contacts. Find all the contacting cell pairs
              with two points per collision
            * 1 - First contact. Quickly find the first contact point.
            * 2 - Half contacts. Find all the contacting cell pairs
              with one point per collision.

        box_tolerance : float, default: 0.001
             Oriented bounding box (OBB) tree tolerance in world coordinates.

        cell_tolerance : float, default: 0.0
            Cell tolerance (squared value).

        n_cells_per_node : int, default: 2
            Number of cells in each OBB.

        generate_scalars : bool, default: False
            Flag to visualize the contact cells.  If ``True``, the
            contacting cells will be colored from red through blue,
            with collisions first determined colored red.  This array
            is stored as ``"collision_rgba"``.

            .. note::
               This will remove any other cell arrays in the mesh.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing collisions in the ``field_data``
            attribute named ``"ContactCells"``.  Array only exists
            when there are collisions.

        int
            Number of collisions.

        Notes
        -----
        Due to the nature of the :vtk:`vtkCollisionDetectionFilter`,
        repeated uses of this method will be slower that using the
        :vtk:`vtkCollisionDetectionFilter` directly. The first
        update of the filter creates two instances of :vtk:`vtkOBBTree`,
        which can be subsequently updated by modifying the transform or
        matrix of the input meshes.

        This method assumes no transform and is easier to use for
        single collision tests, but it is recommended to use a
        combination of ``pyvista`` and ``vtk`` for rapidly computing
        repeated collisions.  See the `Collision Detection Example
        <https://kitware.github.io/vtk-examples/site/Python/Visualization/CollisionDetection/>`_

        Examples
        --------
        Compute the collision between a sphere and the back faces of a
        cube and output the cell indices of the first 10 collisions.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> mesh_a = pv.Sphere(radius=0.5)
        >>> mesh_b = pv.Cube(center=(0.5, 0.5, 0.5)).extract_cells([0, 2, 4])
        >>> collision, ncol = mesh_a.collision(mesh_b, cell_tolerance=1)
        >>> collision['ContactCells'][:10]
        pyvista_ndarray([464,   0,   0,  29,  29,  27,  27,  28,  28,  23])

        Plot the collisions by creating a collision mask with the
        ``"ContactCells"`` field data.  Cells with a collision are
        colored red.

        >>> scalars = np.zeros(collision.n_cells, dtype=bool)
        >>> scalars[collision.field_data['ContactCells']] = True
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(
        ...     collision,
        ...     scalars=scalars,
        ...     show_scalar_bar=False,
        ...     cmap='bwr',
        ... )
        >>> _ = pl.add_mesh(
        ...     mesh_b,
        ...     color='lightblue',
        ...     line_width=5,
        ...     opacity=0.7,
        ...     show_edges=True,
        ... )
        >>> pl.show()

        Alternatively, simply plot the collisions using the default
        ``'collision_rgba'`` array after enabling ``generate_scalars``.

        >>> collision, ncol = mesh_a.collision(
        ...     mesh_b, cell_tolerance=1, generate_scalars=True
        ... )
        >>> collision.plot()

        See :ref:`collision_example` for more examples using this filter.

        """
        # other mesh must be a polydata
        if not isinstance(other_mesh, pyvista.PolyData):
            other_mesh = other_mesh.extract_surface()

        # according to VTK limitations
        poly_data = self
        if not poly_data.is_all_triangles:  # type: ignore[attr-defined]
            poly_data = poly_data.triangulate()
        if not other_mesh.is_all_triangles:
            other_mesh = other_mesh.triangulate()

        alg = _vtk.vtkCollisionDetectionFilter()
        alg.SetInputData(0, poly_data)
        alg.SetTransform(0, _vtk.vtkTransform())
        alg.SetInputData(1, other_mesh)
        alg.SetMatrix(1, _vtk.vtkMatrix4x4())
        alg.SetBoxTolerance(box_tolerance)
        alg.SetCellTolerance(cell_tolerance)
        alg.SetNumberOfCellsPerNode(n_cells_per_node)
        alg.SetCollisionMode(contact_mode)
        alg.SetGenerateScalars(generate_scalars)
        _update_alg(alg, progress_bar=progress_bar, message='Computing collisions')

        output = _get_output(alg)

        if generate_scalars:
            # must rename array as VTK sets the cell scalars array name to
            # a nullptr.
            # See https://github.com/pyvista/pyvista/pull/1540
            #
            # Note: Since all other cell arrays are destroyed when
            # generate_scalars is True, we can always index the first cell
            # array.
            output.cell_data.GetAbstractArray(0).SetName('collision_rgba')

        return output, alg.GetNumberOfContacts()

    @_deprecate_positional_args(allowed=['n_contours'])
    def contour_banded(  # type: ignore[misc]  # noqa: PLR0917
        self: PolyData,
        n_contours,
        rng=None,
        scalars=None,
        component=0,
        clip_tolerance=1e-6,
        generate_contour_edges: bool = True,  # noqa: FBT001, FBT002
        scalar_mode='value',
        clipping: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Generate filled contours.

        Generates filled contours for :vtk:`vtkPolyData`. Filled contours are
        bands of cells that all have the same cell scalar value, and can
        therefore be colored the same. The method is also referred to as
        filled contour generation.

        This filter implements :vtk:`vtkBandedPolyDataContourFilter`.

        Parameters
        ----------
        n_contours : int
            Number of contours.

        rng : Sequence, optional
            Range of the scalars. Optional and defaults to the minimum and
            maximum of the active scalars of ``scalars``.

        scalars : str, optional
            The name of the scalar array to use for contouring.  If ``None``,
            the active scalar array will be used.

        component : int, default: 0
            The component to use of an input scalars array with more than one
            component.

        clip_tolerance : float, default: 1e-6
            Set/Get the clip tolerance.  Warning: setting this too large will
            certainly cause numerical issues. Change from the default value at
            your own risk. The actual internal clip tolerance is computed by
            multiplying ``clip_tolerance`` by the scalar range.

        generate_contour_edges : bool, default: True
            Controls whether contour edges are generated.  Contour edges are
            the edges between bands. If enabled, they are generated from
            polygons/triangle strips and returned as a second output.

        scalar_mode : str, default: 'value'
            Control whether the cell scalars are output as an integer index or
            a scalar value.  If ``'index'``, the index refers to the bands
            produced by the clipping range. If ``'value'``, then a scalar value
            which is a value between clip values is used.

        clipping : bool, default: True
            Indicate whether to clip outside ``rng`` and only return cells with
            values within ``rng``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        output : pyvista.PolyData
            Surface containing the contour surface.

        edges : pyvista.PolyData
            Optional edges when ``generate_contour_edges`` is ``True``.

        Examples
        --------
        Plot the random hills dataset and with 8 contour lines. Note how we use 7
        colors here (``n_contours - 1``).

        >>> import pyvista as pv
        >>> from pyvista import examples

        >>> mesh = examples.load_random_hills()
        >>> n_contours = 8
        >>> _, edges = mesh.contour_banded(n_contours)

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(
        ...     edges,
        ...     line_width=5,
        ...     render_lines_as_tubes=True,
        ...     color='k',
        ... )
        >>> _ = pl.add_mesh(mesh, n_colors=n_contours - 1, cmap='Set3')
        >>> pl.show()

        Extract the surface from the uniform grid dataset and plot its contours
        alongside the output from the banded contour filter.

        >>> surf = examples.load_uniform().extract_surface()
        >>> n_contours = 5
        >>> rng = [200, 500]
        >>> output, edges = surf.contour_banded(n_contours, rng=rng)

        >>> dargs = dict(n_colors=n_contours - 1, clim=rng)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(
        ...     edges,
        ...     line_width=5,
        ...     render_lines_as_tubes=True,
        ...     color='k',
        ... )
        >>> _ = pl.add_mesh(surf, opacity=0.3, **dargs)
        >>> _ = pl.add_mesh(output, **dargs)
        >>> pl.show()

        """
        if scalars is None:
            set_default_active_scalars(self)
            if self.point_data.active_scalars_name is None:
                msg = 'No point scalars to contour.'
                raise MissingDataError(msg)
            scalars = self.active_scalars_name
        arr = get_array(self, scalars, preference='point', err=False)
        if arr is None:
            msg = 'No arrays present to contour.'
            raise ValueError(msg)
        field = get_array_association(self, scalars, preference='point')
        if field != FieldAssociation.POINT:
            msg = 'Only point data can be contoured.'
            raise ValueError(msg)

        if rng is None:
            rng = self.get_data_range(self.active_scalars)

        alg = _vtk.vtkBandedPolyDataContourFilter()
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.GenerateValues(n_contours, rng[0], rng[1])
        alg.SetInputDataObject(self)
        alg.SetClipping(clipping)
        if scalar_mode == 'value':
            alg.SetScalarModeToValue()
        elif scalar_mode == 'index':
            alg.SetScalarModeToIndex()
        else:
            msg = f'Invalid scalar mode "{scalar_mode}". Should be either "value" or "index".'
            raise ValueError(msg)
        alg.SetGenerateContourEdges(generate_contour_edges)
        alg.SetClipTolerance(clip_tolerance)
        alg.SetComponent(component)
        _update_alg(alg, progress_bar=progress_bar, message='Contouring Mesh')
        mesh = _get_output(alg)

        # Must rename array as VTK sets the active scalars array name to a nullptr.
        # Please note this was fixed upstream in https://gitlab.kitware.com/vtk/vtk/-/merge_requests/9840
        for i in range(mesh.GetPointData().GetNumberOfArrays()):
            array = mesh.GetPointData().GetAbstractArray(i)
            name = array.GetName()
            if name is None:
                array.SetName(self.point_data.active_scalars_name)
        for i in range(mesh.GetCellData().GetNumberOfArrays()):
            array = mesh.GetCellData().GetAbstractArray(i)
            name = array.GetName()
            if name is None:
                array.SetName(self.cell_data.active_scalars_name)

        if generate_contour_edges:
            return mesh, wrap(alg.GetContourEdgesOutput())
        return mesh

    @_deprecate_positional_args
    def reconstruct_surface(self, nbr_sz=None, sample_spacing=None, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Reconstruct a surface from the points in this dataset.

        This filter takes a list of points assumed to lie on the
        surface of a solid 3D object. A signed measure of the distance
        to the surface is computed and sampled on a regular grid. The
        grid can then be contoured at zero to extract the surface. The
        default values for neighborhood size and sample spacing should
        give reasonable results for most uses but can be set if
        desired.

        This is helpful when generating surfaces from point clouds and
        is more reliable than :func:`DataSetFilters.delaunay_3d`.

        Parameters
        ----------
        nbr_sz : int, optional
            Specify the number of neighbors each point has, used for
            estimating the local surface orientation.

            The default value of 20 should be fine for most
            applications, higher values can be specified if the spread
            of points is uneven. Values as low as 10 may yield
            adequate results for some surfaces. Higher values cause
            the algorithm to take longer and will cause
            errors on sharp boundaries.

        sample_spacing : float, optional
            The spacing of the 3D sampling grid.  If not set, a
            reasonable guess will be made.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Reconstructed surface.

        Examples
        --------
        Create a point cloud out of a sphere and reconstruct a surface
        from it.

        >>> import pyvista as pv
        >>> points = pv.wrap(pv.Sphere().points)
        >>> surf = points.reconstruct_surface()

        >>> pl = pv.Plotter(shape=(1, 2))
        >>> _ = pl.add_mesh(points)
        >>> _ = pl.add_title('Point Cloud of 3D Surface')
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(surf, color=True, show_edges=True)
        >>> _ = pl.add_title('Reconstructed Surface')
        >>> pl.show()

        See :ref:`surface_reconstruction_example` for more examples
        using this filter.

        """
        alg = _vtk.vtkSurfaceReconstructionFilter()
        alg.SetInputDataObject(self)
        if nbr_sz is not None:
            alg.SetNeighborhoodSize(nbr_sz)
        if sample_spacing is not None:
            alg.SetSampleSpacing(sample_spacing)

        # connect using ports as this will be slightly faster
        mc = _vtk.vtkMarchingCubes()
        mc.SetComputeNormals(False)
        mc.SetComputeScalars(False)
        mc.SetComputeGradients(False)
        mc.SetInputConnection(alg.GetOutputPort())
        mc.SetValue(0, 0.0)
        _update_alg(mc, progress_bar=progress_bar, message='Reconstructing surface')
        return wrap(mc.GetOutput())

    @_deprecate_positional_args
    def triangulate_contours(self, display_errors: bool = False, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Triangulate and fill all 2D contours to create polygons.

        .. versionadded:: 0.44.0

        This filter will generate triangles to fill all of the 2D contours
        in its input. The input to the filter is a set of lines (not polylines)
        which when joined form loops. The contours may be concave, and may even
        contain holes i.e. a contour may contain an internal contour that is
        wound in the opposite direction (as compared to the outer polygon
        normal) to indicate that it is a hole.

        .. note::

            This filter will assume that the input polygons lie in the same
            plane and will not perform any projection or transformation of the
            input data. You may need to project your data to a plane before
            using this filter.

        .. warning::

            The triangulation of is done in O(n) time for simple convex inputs,
            but for non-convex inputs the worst-case time is O(n^2*m^2) where n
            is the number of points and m is the number of holes. The best
            triangulation algorithms, in contrast, are O(n log n). The
            resulting triangles may be quite narrow, the algorithm does not
            attempt to produce high-quality triangles.

        Parameters
        ----------
        display_errors : bool, default: False
            Generate errors when the triangulation fails. Note that
            triangulation failures are often minor, because they involve tiny
            triangles that are too small to see.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Triangulated mesh with the filled contours.

        Examples
        --------
        Create banded contour and fill.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> image = examples.download_st_helens()
        >>> contours = image.contour([1302.3334, 1922.6666])
        >>> filled = contours.triangulate_contours()

        >>> pl = pv.Plotter(shape=(1, 2))
        >>> _ = pl.add_mesh(image, show_scalar_bar=False)
        >>> _ = pl.add_mesh(contours, color='black')
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(contours, color='black')
        >>> _ = pl.add_mesh(filled, color='red')
        >>> pl.link_views()
        >>> pl.view_xy()
        >>> pl.show()

        """
        alg = _vtk.vtkContourTriangulator()
        alg.SetInputDataObject(self)
        alg.SetTriangulationErrorDisplay(display_errors)
        _update_alg(alg, progress_bar=progress_bar, message='Triangulating Contours')
        return _get_output(alg)

    @_deprecate_positional_args
    def protein_ribbon(self, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Generate protein ribbon.

        Parameters
        ----------
        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Generated protein ribbon.

        Examples
        --------
        Generate protein ribbon.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> tgqp = examples.download_3gqp()
        >>> ribbon = tgqp.protein_ribbon()
        >>> ribbon.plot()

        """
        alg = _vtk.vtkRibbonFilter()
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Generating Protein Ribbons')
        return _get_output(alg)

    def ruled_surface(
        self, *, resolution: VectorLike[int] | None = None, progress_bar: bool = False
    ):
        """Create a ruled surface from a polyline.

        .. versionadded:: 0.45.0

        This filter is a filter that generates a surface from
        a set of lines. The lines are assumed to be "parallel"
        in the sense that they do not intersect and remain
        somewhat close to one another. A surface is generated
        by connecting the points defining each pair of lines
        with straight lines. This creates a strip for each pair
        of lines (i.e., a triangulation is created from two
        generating lines). The filter can handle an arbitrary
        number of lines, with lines i and i+1 assumed
        connected. Note that there are several different
        approaches for creating the ruled surface, the method
        for creating the surface can either use the input points
        or resample from the polylines (using a user-specified
        resolution).

        This filter implements :vtk:`vtkRuledSurfaceFilter`.

        Parameters
        ----------
        resolution : int, default: (1, 1)
            Set the number of points in the output polyline.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Ruled surface.

        Examples
        --------
        Create a ruled surface from a polyline.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> poly = pv.PolyData(
        ...     [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
        ...     lines=[[2, 0, 1], [2, 2, 3]],
        ...     force_float=False,
        ... )
        >>> surface = poly.ruled_surface(resolution=(21, 21))
        >>> _ = pl.add_mesh(surface, show_edges=True)
        >>> pl.show()

        """
        alg = _vtk.vtkRuledSurfaceFilter()
        alg.SetInputData(self)
        if resolution is not None:
            _validation.validate_array(resolution, must_have_shape=2, must_have_dtype=int)
            alg.SetResolution(resolution)  # type: ignore[arg-type]
        _update_alg(alg, progress_bar=progress_bar, message='Generating ruled surface')
        return _get_output(alg)

    def remove_unused_points(  # type: ignore[misc]
        self: _PolyDataType,
        *,
        inplace: bool = False,
    ) -> _PolyDataType:
        """Remove points which are not used by any cells.

        This filter is similar to :meth:`clean` but does `not` merge points or convert cells.
        The point order is also unchanged by this filter.

        .. versionadded:: 0.46

        Parameters
        ----------
        inplace : bool, default: False
            If ``True`` the mesh is updated in-place, otherwise a copy is returned.

        See Also
        --------
        pyvista.UnstructuredGridFilters.remove_unused_points

        Returns
        -------
        PolyData
            Mesh with unused points removed.

        Examples
        --------
        Create :class:`~pyvista.PolyData` with three points. The first two points are coincident
        and associated with :attr:`~pyvista.CellType.VERTEX` cells, and the third point is
        "unused" and not associated with any cells.

        >>> import pyvista as pv
        >>> points = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        >>> faces = [1, 0, 1, 1]
        >>> poly = pv.PolyData(points, faces)
        >>> poly
        PolyData (...)
          N Cells:    2
          N Points:   3
          N Strips:   0
          X Bounds:   0.000e+00, 1.000e+00
          Y Bounds:   0.000e+00, 1.000e+00
          Z Bounds:   0.000e+00, 1.000e+00
          N Arrays:   0

        Since the third point is unused, we can remove it. Note that coincident points are `not`
        merged by this filter, so the two vertex points are kept as-is.

        >>> poly = poly.remove_unused_points()
        >>> poly
        PolyData (...)
          N Cells:    2
          N Points:   2
          N Strips:   0
          X Bounds:   0.000e+00, 0.000e+00
          Y Bounds:   0.000e+00, 0.000e+00
          Z Bounds:   0.000e+00, 0.000e+00
          N Arrays:   0

        """
        removed = self.cast_to_unstructured_grid().remove_unused_points().extract_geometry()
        out = self if inplace else type(self)()
        out.copy_from(removed, deep=not inplace)
        return out
