"""Filters module with a class to manage filters/algorithms for polydata datasets."""
import collections.abc
import logging

import numpy as np

import pyvista
from pyvista import (
    NORMALS,
    _vtk,
    abstract_class,
    assert_empty_kwargs,
    generate_plane,
    get_array_association,
    vtk_id_list_to_array,
)
from pyvista.core.errors import DeprecationError, NotAllTrianglesError, VTKVersionError
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters


@abstract_class
class PolyDataFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for polydata datasets."""

    def edge_mask(self, angle, progress_bar=False):
        """Return a mask of the points of a surface mesh that has a surface angle greater than angle.

        Parameters
        ----------
        angle : float
            Angle to consider an edge.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        numpy.ndarray
            Mask of points with an angle greater than ``angle``.

        Examples
        --------
        Plot the mask of points that exceed 45 degrees.

        >>> import pyvista
        >>> mesh = pyvista.Cube().triangulate().subdivide(4)
        >>> mask = mesh.edge_mask(45)
        >>> mask  # doctest:+SKIP
        array([ True,  True,  True, ..., False, False, False])
        >>> mesh.plot(scalars=mask)

        """
        poly_data = self
        if not isinstance(poly_data, pyvista.PolyData):  # pragma: no cover
            poly_data = pyvista.PolyData(poly_data)
        poly_data.point_data['point_ind'] = np.arange(poly_data.n_points)
        featureEdges = _vtk.vtkFeatureEdges()
        featureEdges.SetInputData(poly_data)
        featureEdges.FeatureEdgesOn()
        featureEdges.BoundaryEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.SetFeatureAngle(angle)
        _update_alg(featureEdges, progress_bar, 'Computing Edges')
        edges = _get_output(featureEdges)
        orig_id = pyvista.point_array(edges, 'point_ind')

        return np.in1d(poly_data.point_data['point_ind'], orig_id,
                       assume_unique=True)

    def _boolean(self, btype, other_mesh, tolerance, progress_bar=False):
        """Perform boolean operation."""
        if not isinstance(other_mesh, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")
        if not self.is_all_triangles or not other_mesh.is_all_triangles:
            raise NotAllTrianglesError("Make sure both the input and output are triangulated.")

        bfilter = _vtk.vtkBooleanOperationPolyDataFilter()
        if btype == 'union':
            bfilter.SetOperationToUnion()
        elif btype == 'intersection':
            bfilter.SetOperationToIntersection()
        elif btype == 'difference':
            bfilter.SetOperationToDifference()
        else:  # pragma: no cover
            raise ValueError(f'Invalid btype {btype}')
        bfilter.SetInputData(0, self)
        bfilter.SetInputData(1, other_mesh)
        bfilter.ReorientDifferenceCellsOn()  # this is already default
        bfilter.SetTolerance(tolerance)
        _update_alg(bfilter, progress_bar, 'Performing Boolean Operation')

        return _get_output(bfilter)

    def boolean_cut(self, *args, **kwargs):  # pragma: no cover
        """Cut two meshes.

        .. deprecated:: 0.32.0
           Use :func:`PolyDataFilters.boolean_difference` instead.

        """
        raise DeprecationError('``boolean_cut`` has been deprecated.  '
                               'Please use ``boolean_difference``.')

    def boolean_add(self, *args, **kwargs):  # pragma: no cover
        """Merge two meshes together.

        .. deprecated:: 0.32.0
           Use :func:`PolyDataFilters.merge` instead.

        """
        raise DeprecationError('``boolean_add`` has been deprecated.  '
                               'Please use ``merge``.')

    def boolean_union(self, other_mesh, tolerance=1E-5, progress_bar=False):
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

        Parameters
        ----------
        other_mesh : pyvista.PolyData
            Mesh operating on the source mesh.

        tolerance : float, optional
            Tolerance used to determine when a point's absolute
            distance is considered to be zero.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The result of the boolean operation.

        Examples
        --------
        Demonstrate a boolean union with two spheres.  Note how the
        final mesh includes both spheres.

        >>> import pyvista
        >>> sphere_a = pyvista.Sphere()
        >>> sphere_b = pyvista.Sphere(center=(0.5, 0, 0))
        >>> result = sphere_a.boolean_union(sphere_b)
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(sphere_a, color='r', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(sphere_b, color='b', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(result, color='tan')
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        See :ref:`boolean_example` for more examples using this filter.

        """
        return self._boolean('union', other_mesh, tolerance, progress_bar=progress_bar)

    def boolean_intersection(self, other_mesh, tolerance=1E-5, progress_bar=False):
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

        Parameters
        ----------
        other_mesh : pyvista.PolyData
            Mesh operating on the source mesh.

        tolerance : float, optional
            Tolerance used to determine when a point's absolute
            distance is considered to be zero.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The result of the boolean operation.

        Examples
        --------
        Demonstrate a boolean intersection with two spheres.  Note how
        the final mesh only includes the intersection of the two.

        >>> import pyvista
        >>> sphere_a = pyvista.Sphere()
        >>> sphere_b = pyvista.Sphere(center=(0.5, 0, 0))
        >>> result = sphere_a.boolean_intersection(sphere_b)
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(sphere_a, color='r', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(sphere_b, color='b', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(result, color='tan')
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        See :ref:`boolean_example` for more examples using this filter.

        """
        return self._boolean('intersection', other_mesh, tolerance, progress_bar=progress_bar)

    def boolean_difference(self, other_mesh, tolerance=1E-5, progress_bar=False):
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

        tolerance : float, optional
            Tolerance used to determine when a point's absolute
            distance is considered to be zero.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The result of the boolean operation.

        Examples
        --------
        Demonstrate a boolean difference with two spheres.  Note how
        the final mesh only includes ``sphere_a``.

        >>> import pyvista
        >>> sphere_a = pyvista.Sphere()
        >>> sphere_b = pyvista.Sphere(center=(0.5, 0, 0))
        >>> result = sphere_a.boolean_difference(sphere_b)
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(sphere_a, color='r', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(sphere_b, color='b', style='wireframe', line_width=3)
        >>> _ = pl.add_mesh(result, color='tan')
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        See :ref:`boolean_example` for more examples using this filter.

        """
        return self._boolean('difference', other_mesh, tolerance, progress_bar=progress_bar)

    def __add__(self, dataset):
        """Merge these two meshes."""
        return self.merge(dataset)

    def __iadd__(self, dataset):
        """Merge another mesh into this one if possible.

        "If possible" means that ``dataset`` is also a :class:`PolyData`.
        Otherwise we have to return a :class:`pyvista.UnstructuredGrid`,
        so the in-place merge attempt will raise.

        """
        try:
            merged = self.merge(dataset, inplace=True)
        except TypeError:
            raise TypeError(
                'In-place merge only possible if the other mesh '
                'is also a PolyData.\nPlease use `mesh + other_mesh` '
                'instead, which returns a new UnstructuredGrid.'
            ) from None
        return merged

    def merge(self, dataset, merge_points=True, inplace=False,
              main_has_priority=True, progress_bar=False):
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

        Parameters
        ----------
        dataset : pyvista.DataSet
            PyVista dataset to merge this mesh with.

        merge_points : bool, optional
            Merge equivalent points when ``True``.

        inplace : bool, optional
            Updates grid inplace when ``True`` if the input type is a
            :class:`pyvista.PolyData`. For other input meshes the
            result is a :class:`pyvista.UnstructuredGrid` which makes
            in-place operation impossible.

        main_has_priority : bool, optional
            When this parameter is ``True`` and ``merge_points=True``,
            the arrays of the merging grids will be overwritten
            by the original main mesh.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            :class:`pyvista.PolyData` if ``dataset`` is a
            :class:`pyvista.PolyData`, otherwise a
            :class:`pyvista.UnstructuredGrid`.

        Examples
        --------
        >>> import pyvista
        >>> sphere_a = pyvista.Sphere()
        >>> sphere_b = pyvista.Sphere(center=(0.5, 0, 0))
        >>> merged = sphere_a.merge(sphere_b)
        >>> merged.plot(style='wireframe', color='tan')

        """
        # check if dataset or datasets are not polydata
        if isinstance(dataset, (list, tuple, pyvista.MultiBlock)):
            not_pd = any(not isinstance(data, pyvista.PolyData) for data in dataset)
        else:
            not_pd = not isinstance(dataset, pyvista.PolyData)

        # use dataset merge if not polydata
        if not_pd:
            return DataSetFilters.merge(self, dataset,
                                        merge_points=merge_points,
                                        main_has_priority=main_has_priority,
                                        inplace=inplace)

        append_filter = pyvista._vtk.vtkAppendPolyData()

        # note: unlike DataSetFilters.merge, we must put the
        # "to be preserved" dataset last due to the call to clean()
        if main_has_priority:
            append_filter.AddInputData(self)

        if isinstance(dataset, pyvista.DataSet):
            append_filter.AddInputData(dataset)
        else:
            for data in dataset:
                append_filter.AddInputData(data)

        if not main_has_priority:
            append_filter.AddInputData(self)

        _update_alg(append_filter, progress_bar, 'Merging')
        merged = _get_output(append_filter)
        if merge_points:
            merged = merged.clean(lines_to_points=False, polys_to_lines=False,
                                  strips_to_polys=False)

        if inplace:
            self.deep_copy(merged)
            return self

        return merged

    def intersection(self, mesh, split_first=True, split_second=True, progress_bar=False):
        """Compute the intersection between two meshes.

        .. note::
           This method returns the surface interection from two meshes
           (which often resolves as a line), whereas the
           :func:`PolyDataFilters.boolean_intersection` filter returns
           the "volume" intersection between two closed (manifold)
           meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to intersect with.

        split_first : bool, optional
            If ``True``, return the first input mesh split by the
            intersection with the second input mesh.

        split_second : bool, optional
            If ``True``, return the second input mesh split by the
            intersection with the first input mesh.

        progress_bar : bool, optional
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

        >>> intersection, _, s2_split = s1.intersection(s2,
        ...                                             split_first=False,
        ...                                             split_second=True)

        """
        intfilter = _vtk.vtkIntersectionPolyDataFilter()
        intfilter.SetInputDataObject(0, self)
        intfilter.SetInputDataObject(1, mesh)
        intfilter.SetComputeIntersectionPointArray(True)
        intfilter.SetSplitFirstOutput(split_first)
        intfilter.SetSplitSecondOutput(split_second)
        _update_alg(intfilter, progress_bar, 'Computing the intersection between two meshes')

        intersection = _get_output(intfilter, oport=0)
        first = _get_output(intfilter, oport=1)
        second = _get_output(intfilter, oport=2)

        return intersection, first, second

    def curvature(self, curv_type='mean', progress_bar=False):
        """Return the pointwise curvature of a mesh.

        Parameters
        ----------
        curv_type : str, optional
            Curvature type.  One of the following:

            * ``"Mean"``
            * ``"Gaussian"``
            * ``"Maximum"``
            * ``"Minimum"``

        progress_bar : bool, optional
            Display a progress bar to indicate progress. Default
            ``False``.

        Returns
        -------
        numpy.ndarray
            Array of curvature values.

        Examples
        --------
        Calculate the mean curvature of the hills example mesh.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> curv = hills.curvature()
        >>> curv   # doctest:+SKIP
        array([0.20587616, 0.06747695, ..., 0.11781171, 0.15988467])

        Plot it.

        >>> hills.plot(scalars=curv)

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
            raise ValueError('``curv_type`` must be either "Mean", '
                             '"Gaussian", "Maximum", or "Minimum".')
        _update_alg(curvefilter, progress_bar, 'Computing Curvature')

        # Compute and return curvature
        curv = _get_output(curvefilter)
        return _vtk.vtk_to_numpy(curv.GetPointData().GetScalars())

    def plot_curvature(self, curv_type='mean', **kwargs):
        """Plot the curvature.

        Parameters
        ----------
        curv_type : str, optional
            One of the following strings indicating curvature type:

            * ``'Mean'``
            * ``'Gaussian'``
            * ``'Maximum'``
            * ``'Minimum'``

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
        >>> hills.plot_curvature(curv_type='gaussian', smooth_shading=True,
        ...                      clim=[0, 1])

        """
        kwargs.setdefault('scalar_bar_args',
                          {'title': f'{curv_type.capitalize()} Curvature'})
        return self.plot(scalars=self.curvature(curv_type), **kwargs)

    def triangulate(self, inplace=False, progress_bar=False):
        """Return an all triangle mesh.

        More complex polygons will be broken down into triangles.

        Parameters
        ----------
        inplace : bool, optional
            Whether to update the mesh in-place.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing only triangles.

        Examples
        --------
        Generate a mesh with quadrilateral faces.

        >>> import pyvista
        >>> plane = pyvista.Plane()
        >>> plane.point_data.clear()
        >>> plane.plot(show_edges=True, line_width=5)

        Convert it to an all triangle mesh.

        >>> mesh = plane.triangulate()
        >>> mesh.plot(show_edges=True, line_width=5)

        """
        trifilter = _vtk.vtkTriangleFilter()
        trifilter.SetInputData(self)
        trifilter.PassVertsOff()
        trifilter.PassLinesOff()
        _update_alg(trifilter, progress_bar, 'Computing Triangle Mesh')

        mesh = _get_output(trifilter)
        if inplace:
            self.overwrite(mesh)
            return self
        return mesh

    def smooth(self, n_iter=20, relaxation_factor=0.01, convergence=0.0,
               edge_angle=15, feature_angle=45,
               boundary_smoothing=True, feature_smoothing=False, inplace=False,
               progress_bar=False):
        """Adjust point coordinates using Laplacian smoothing.

        The effect is to "relax" the mesh, making the cells better shaped and
        the vertices more evenly distributed.

        Parameters
        ----------
        n_iter : int
            Number of iterations for Laplacian smoothing.

        relaxation_factor : float, optional
            Relaxation factor controls the amount of displacement in a single
            iteration. Generally a lower relaxation factor and higher number of
            iterations is numerically more stable.

        convergence : float, optional
            Convergence criterion for the iteration process. Smaller numbers
            result in more smoothing iterations. Range from (0 to 1).

        edge_angle : float, optional
            Edge angle to control smoothing along edges (either interior or boundary).

        feature_angle : float, optional
            Feature angle for sharp edge identification.

        boundary_smoothing : bool, optional
            Boolean flag to control smoothing of boundary edges.

        feature_smoothing : bool, optional
            Boolean flag to control smoothing of feature edges.

        inplace : bool, optional
            Updates mesh in-place.

        progress_bar : bool, optional
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
        >>> smooth_cube = cube.smooth(1000, feature_smoothing=False)
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
        _update_alg(alg, progress_bar, 'Smoothing Mesh')

        mesh = _get_output(alg)
        if inplace:
            self.overwrite(mesh)
            return self

        return mesh

    def decimate_pro(self, reduction, feature_angle=45.0,
                     split_angle=75.0, splitting=True,
                     pre_split_mesh=False, preserve_topology=False,
                     inplace=False, progress_bar=False):
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

        feature_angle : float, optional
            Angle used to define what an edge is (i.e., if the surface
            normal between two adjacent triangles is >= ``feature_angle``,
            an edge exists).

        split_angle : float, optional
            Angle used to control the splitting of the mesh. A split
            line exists when the surface normals between two edge
            connected triangles are >= ``split_angle``.

        splitting : bool, optional
            Controls the splitting of the mesh at corners, along
            edges, at non-manifold points, or anywhere else a split is
            required. Turning splitting off will better preserve the
            original topology of the mesh, but may not necessarily
            give the exact requested decimation.

        pre_split_mesh : bool, optional
            Separates the mesh into semi-planar patches, which are
            disconnected from each other. This can give superior
            results in some cases. If ``pre_split_mesh`` is set to
            ``True``, the mesh is split with the specified
            ``split_angle``. Otherwise mesh splitting is deferred as
            long as possible.

        preserve_topology : bool, optional
            Controls topology preservation. If on, mesh splitting and
            hole elimination will not occur. This may limit the
            maximum reduction that may be achieved.

        inplace : bool, optional
            Whether to update the mesh in-place.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Decimated mesh.

        Examples
        --------
        Decimate a sphere.  First plot the sphere.

        >>> import pyvista
        >>> sphere = pyvista.Sphere(phi_resolution=60, theta_resolution=60)
        >>> sphere.plot(show_edges=True, line_width=2)

        Now decimate it and plot it.

        >>> decimated = sphere.decimate_pro(0.75)
        >>> decimated.plot(show_edges=True, line_width=2)

        See :ref:`decimate_example` for more examples using this filter.

        """
        alg = _vtk.vtkDecimatePro()
        alg.SetInputData(self)
        alg.SetTargetReduction(reduction)
        alg.SetPreserveTopology(preserve_topology)
        alg.SetFeatureAngle(feature_angle)
        alg.SetSplitting(splitting)
        alg.SetSplitAngle(split_angle)
        alg.SetPreSplitMesh(pre_split_mesh)
        _update_alg(alg, progress_bar, 'Decimating Mesh')

        mesh = _get_output(alg)
        if inplace:
            self.overwrite(mesh)
            return self

        return mesh

    def tube(self, radius=None, scalars=None, capping=True, n_sides=20,
             radius_factor=10, preference='point', inplace=False, progress_bar=False):
        """Generate a tube around each input line.

        The radius of the tube can be set to linearly vary with a
        scalar value.

        Parameters
        ----------
        radius : float
            Minimum tube radius (minimum because the tube radius may
            vary).

        scalars : str, optional
            Scalars array by which the radius varies.

        capping : bool, optional
            Turn on/off whether to cap the ends with polygons. Default
            ``True``.

        n_sides : int, optional
            Set the number of sides for the tube. Minimum of 3.

        radius_factor : float, optional
            Maximum tube radius in terms of a multiple of the minimum
            radius.

        preference : str, optional
            The field preference when searching for the scalars array by
            name.

        inplace : bool, optional
            Whether to update the mesh in-place.

        progress_bar : bool, optional
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
        >>> tube.plot(color='tan')

        See :ref:`ref_create_spline` for more examples using this filter.

        """
        poly_data = self
        if not isinstance(poly_data, pyvista.PolyData):
            poly_data = pyvista.PolyData(poly_data)
        if n_sides < 3:
            n_sides = 3
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
                raise TypeError('scalars array must be given as a string name')
            field = poly_data.get_array_association(scalars, preference=preference)
            # args: (idx, port, connection, field, name)
            tube.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
            tube.SetVaryRadiusToVaryRadiusByScalar()
        # Apply the filter
        _update_alg(tube, progress_bar, 'Creating Tube')

        mesh = _get_output(tube)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        return mesh

    def subdivide(self, nsub, subfilter='linear', inplace=False, progress_bar=False):
        """Increase the number of triangles in a single, connected triangular mesh.

        Uses one of the following vtk subdivision filters to subdivide a mesh:

        * ``vtkButterflySubdivisionFilter``
        * ``vtkLoopSubdivisionFilter``
        * ``vtkLinearSubdivisionFilter``

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

        subfilter : str, optional
            Can be one of the following:

            * ``'butterfly'``
            * ``'loop'``
            * ``'linear'``

        inplace : bool, optional
            Updates mesh in-place. Default ``False``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Subdivided mesh.

        Examples
        --------
        First, create an example coarse sphere mesh and plot it.

        >>> from pyvista import examples
        >>> import pyvista
        >>> mesh = pyvista.Sphere(phi_resolution=10, theta_resolution=10)
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
        subfilter = subfilter.lower()
        if subfilter == 'linear':
            sfilter = _vtk.vtkLinearSubdivisionFilter()
        elif subfilter == 'butterfly':
            sfilter = _vtk.vtkButterflySubdivisionFilter()
        elif subfilter == 'loop':
            sfilter = _vtk.vtkLoopSubdivisionFilter()
        else:
            raise ValueError("Subdivision filter must be one of the following: "
                             "'butterfly', 'loop', or 'linear'")

        # Subdivide
        sfilter.SetNumberOfSubdivisions(nsub)
        sfilter.SetInputData(self)
        _update_alg(sfilter, progress_bar, 'Subdividing Mesh')

        submesh = _get_output(sfilter)
        if inplace:
            self.overwrite(submesh)
            return self

        return submesh

    def subdivide_adaptive(self, max_edge_len=None, max_tri_area=None,
                           max_n_tris=None, max_n_passes=None, inplace=False,
                           progress_bar=False):
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

        inplace : bool, optional
            Updates mesh in-place.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Subdivided mesh.

        Examples
        --------
        First, load the example airplane mesh and plot it.

        >>> import pyvista
        >>> from pyvista import examples
        >>> mesh = pyvista.PolyData(examples.planefile)
        >>> mesh.plot(show_edges=True, line_width=3)

        Subdivide the mesh

        >>> submesh = mesh.subdivide_adaptive(max_n_passes=2)
        >>> submesh.plot(show_edges=True)

        """
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
        _update_alg(sfilter, progress_bar, 'Adaptively Subdividing Mesh')
        submesh = _get_output(sfilter)

        if inplace:
            self.overwrite(submesh)
            return self

        return submesh

    def decimate(self, target_reduction, volume_preservation=False,
                 attribute_error=False, scalars=True, vectors=True,
                 normals=False, tcoords=True, tensors=True, scalars_weight=0.1,
                 vectors_weight=0.1, normals_weight=0.1, tcoords_weight=0.1,
                 tensors_weight=0.1, inplace=False, progress_bar=False):
        """Reduce the number of triangles in a triangular mesh using ``vtkQuadricDecimation``.

        Parameters
        ----------
        target_reduction : float
            Fraction of the original mesh to remove.
            If ``target_reduction`` is set to 0.9, this filter will try
            to reduce the data set to 10% of its original size and will
            remove 90% of the input triangles.

        volume_preservation : bool, optional
            Decide whether to activate volume preservation which greatly
            reduces errors in triangle normal direction. If ``False``,
            volume preservation is disabled and if ``attribute_error``
            is active, these errors can be large. Defaults to ``False``.

        attribute_error : bool, optional
            Decide whether to include data attributes in the error
            metric. If ``False``, then only geometric error is used to
            control the decimation. Defaults to ``False``. If ``True``,
            the following flags are used to specify which attributes
            are to be included in the error calculation.

        scalars : bool, optional
            If attribute errors are to be included in the metric (i.e.,
            ``attribute_error`` is ``True``), then these flags control
            which attributes are to be included in the error
            calculation. Defaults to ``True``.

        vectors : bool, optional
            See ``scalars`` parameter. Defaults to ``True``.

        normals : bool, optional
            See ``scalars`` parameter. Defaults to ``False``.

        tcoords : bool, optional
            See ``scalars`` parameter. Defaults to ``True``.

        tensors : bool, optional
            See ``scalars`` parameter. Defaults to ``True``.

        scalars_weight : float, optional
            The scaling weight contribution of the scalar attribute.
            These values are used to weight the contribution of the
            attributes towards the error metric. Defaults to 0.1.

        vectors_weight : float, optional
            See ``scalars_weight`` parameter. Defaults to 0.1.

        normals_weight : float, optional
            See ``scalars_weight`` parameter. Defaults to 0.1.

        tcoords_weight : float, optional
            See ``scalars_weight`` parameter. Defaults to 0.1.

        tensors_weight : float, optional
            See ``scalars_weight`` parameter. Defaults to 0.1.

        inplace : bool, optional
            Whether to update the mesh in-place.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Decimated mesh.

        Notes
        -----
        If you encounter a segmentation fault or other error,
        consider using :func:`pyvista.clean` to remove any invalid
        cells before using this filter.

        Examples
        --------
        Decimate a sphere.  First plot the sphere.

        >>> import pyvista
        >>> sphere = pyvista.Sphere(phi_resolution=60, theta_resolution=60)
        >>> sphere.plot(show_edges=True, line_width=2)

        Now decimate it by 75% and plot it.

        >>> decimated = sphere.decimate(0.75)
        >>> decimated.plot(show_edges=True, line_width=2)

        See :ref:`decimate_example` for more examples using this filter.

        """
        # create decimation filter
        alg = _vtk.vtkQuadricDecimation()

        alg.SetVolumePreservation(volume_preservation)
        alg.SetAttributeErrorMetric(attribute_error)
        alg.SetScalarsAttribute(scalars)
        alg.SetVectorsAttribute(vectors)
        alg.SetNormalsAttribute(normals)
        alg.SetTCoordsAttribute(tcoords)
        alg.SetTensorsAttribute(tensors)
        alg.SetScalarsWeight(scalars_weight)
        alg.SetVectorsWeight(vectors_weight)
        alg.SetNormalsWeight(normals_weight)
        alg.SetTCoordsWeight(tcoords_weight)
        alg.SetTensorsWeight(tensors_weight)
        alg.SetTargetReduction(target_reduction)

        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Decimating Mesh')

        mesh = _get_output(alg)
        if inplace:
            self.overwrite(mesh)
            return self

        return mesh

    def compute_normals(self, cell_normals=True,
                        point_normals=True, split_vertices=False,
                        flip_normals=False, consistent_normals=True,
                        auto_orient_normals=False,
                        non_manifold_traversal=True,
                        feature_angle=30.0,
                        inplace=False,
                        progress_bar=False):
        """Compute point and/or cell normals for a mesh.

        The filter can reorder polygons to insure consistent
        orientation across polygon neighbors. Sharp edges can be split
        and points duplicated with separate normals to give crisp
        (rendered) surface definition. It is also possible to globally
        flip the normal orientation.

        The algorithm works by determining normals for each polygon
        and then averaging them at shared points. When sharp edges are
        present, the edges are split and new points generated to
        prevent blurry edges (due to Gouraud shading).

        Parameters
        ----------
        cell_normals : bool, optional
            Calculation of cell normals. Defaults to ``True``.

        point_normals : bool, optional
            Calculation of point normals. Defaults to ``True``.

        split_vertices : bool, optional
            Splitting of sharp edges. Defaults to ``False``.

        flip_normals : bool, optional
            Set global flipping of normal orientation. Flipping
            modifies both the normal direction and the order of a
            cell's points. Defaults to ``False``.

        consistent_normals : bool, optional
            Enforcement of consistent polygon ordering. Defaults to ``True``.

        auto_orient_normals : bool, optional
            Turn on/off the automatic determination of correct normal
            orientation. NOTE: This assumes a completely closed
            surface (i.e. no boundary edges) and no non-manifold
            edges. If these constraints do not hold, all bets are
            off. This option adds some computational complexity, and
            is useful if you do not want to have to inspect the
            rendered image to determine whether to turn on the
            ``flip_normals`` flag.  However, this flag can work with
            the ``flip_normals`` flag, and if both are set, all the
            normals in the output will point "inward". Defaults to
            ``False``.

        non_manifold_traversal : bool, optional
            Turn on/off traversal across non-manifold edges. Changing
            this may prevent problems where the consistency of
            polygonal ordering is corrupted due to topological
            loops. Defaults to ``True``.

        feature_angle : float, optional
            The angle that defines a sharp edge. If the difference in
            angle across neighboring polygons is greater than this
            value, the shared edge is considered "sharp". Defaults to
            30.0.

        inplace : bool, optional
            Updates mesh in-place. Defaults to ``False``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Updated mesh with cell and point normals.

        Notes
        -----
        Previous arrays named ``"Normals"`` will be overwritten.

        Normals are computed only for polygons and triangle
        strips. Normals are not computed for lines or vertices.

        Triangle strips are broken up into triangle polygons. You may
        want to restrip the triangles.

        It may be easier to run
        :func:`pyvista.PolyData.point_normals` or
        :func:`pyvista.PolyData.cell_normals` if you would just
        like the array of point or cell normals.

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

        See :ref:`surface_normal_example` for more examples using this filter.

        """
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
        _update_alg(normal, progress_bar, 'Computing Normals')

        mesh = _get_output(normal)
        if point_normals:
            mesh.GetPointData().SetActiveNormals('Normals')
        if cell_normals:
            mesh.GetCellData().SetActiveNormals('Normals')

        if inplace:
            self.overwrite(mesh)
            return self

        return mesh

    def clip_closed_surface(self, normal='x', origin=None,
                            tolerance=1e-06, inplace=False, progress_bar=False):
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

        inplace : bool, optional
            Updates mesh in-place. Defaults to ``False``.

        progress_bar : bool, optional
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
        if self.n_open_edges > 0:
            raise ValueError("This surface appears to be non-manifold.")
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = self.center

        # create the plane for clipping
        plane = generate_plane(normal, origin)
        collection = _vtk.vtkPlaneCollection()
        collection.AddItem(plane)

        alg = _vtk.vtkClipClosedSurface()
        alg.SetGenerateFaces(True)
        alg.SetInputDataObject(self)
        alg.SetTolerance(tolerance)
        alg.SetClippingPlanes(collection)
        _update_alg(alg, progress_bar, 'Clipping Closed Surface')
        result = _get_output(alg)

        if inplace:
            self.overwrite(result)
            return self
        else:
            return result

    def fill_holes(self, hole_size, inplace=False, progress_bar=False):  # pragma: no cover
        """Fill holes in a pyvista.PolyData or vtk.vtkPolyData object.

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

        inplace : bool, optional
            Return new mesh or overwrite input.

        progress_bar : bool, optional
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
        >>> edges = sphere.extract_feature_edges(feature_edges=False,
        ...                                      manifold_edges=False)  # doctest:+SKIP
        >>> assert edges.n_cells == 0  # doctest:+SKIP

        """
        logging.warning('pyvista.PolyData.fill_holes is known to segfault. '
                        'Use at your own risk')
        alg = _vtk.vtkFillHolesFilter()
        alg.SetHoleSize(hole_size)
        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Filling Holes')

        mesh = _get_output(alg)
        if inplace:
            self.overwrite(mesh)
            return self
        return mesh

    def clean(self, point_merging=True, tolerance=None, lines_to_points=True,
              polys_to_lines=True, strips_to_polys=True, inplace=False,
              absolute=True, progress_bar=False, **kwargs):
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

        inplace : bool, optional
            Updates mesh in-place. Default ``False``.

        absolute : bool, optional
            Control if ``tolerance`` is an absolute distance or a
            fraction.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Accepts for ``merge_tol`` to replace the ``tolerance``
            keyword argument.  This may be deprecated in future.

        Returns
        -------
        pyvista.PolyData
            Cleaned mesh.

        Examples
        --------
        Create a mesh with a degenerate face and then clean it,
        removing the degenerate face

        >>> import pyvista as pv
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
        >>> faces = np.array([3, 0, 1, 2, 3, 0, 3, 3])
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
        _update_alg(alg, progress_bar, 'Cleaning')
        output = _get_output(alg)

        # Check output so no segfaults occur
        if output.n_points < 1:
            raise ValueError('Clean tolerance is too high. Empty mesh returned.')

        if inplace:
            self.overwrite(output)
            return self
        return output

    def geodesic(self, start_vertex, end_vertex, inplace=False,
                 keep_order=True, progress_bar=False):
        """Calculate the geodesic path between two vertices using Dijkstra's algorithm.

        This will add an array titled ``'vtkOriginalPointIds'`` of the input
        mesh's point ids to the output mesh. The default behavior of the
        underlying ``vtkDijkstraGraphGeodesicPath`` filter is that the
        geodesic path is reversed in the resulting mesh. This is overridden
        in PyVista by default.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        inplace : bool, optional
            Whether the input mesh should be replaced with the path. The
            geodesic path is always returned.

        keep_order : bool, optional
            If ``True``, the points of the returned path are guaranteed
            to start with the start vertex (as opposed to the end vertex).

            .. versionadded:: 0.32.0

        progress_bar : bool, optional
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
        if not (0 <= start_vertex < self.n_points and
                0 <= end_vertex < self.n_points):
            raise IndexError('Invalid point indices.')
        if not self.is_all_triangles:
            raise NotAllTrianglesError("Input mesh for geodesic path must be all triangles.")

        dijkstra = _vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(self)
        dijkstra.SetStartVertex(start_vertex)
        dijkstra.SetEndVertex(end_vertex)
        _update_alg(dijkstra, progress_bar, 'Calculating the Geodesic Path')
        original_ids = vtk_id_list_to_array(dijkstra.GetIdList())

        output = _get_output(dijkstra)
        output["vtkOriginalPointIds"] = original_ids

        # Do not copy textures from input
        output.clear_textures()

        # ensure proper order if requested
        if keep_order and original_ids[0] == end_vertex:
            output.points[...] = output.points[::-1, :]
            output["vtkOriginalPointIds"] = output["vtkOriginalPointIds"][::-1]

        if inplace:
            self.overwrite(output)
            return self

        return output

    def geodesic_distance(self, start_vertex, end_vertex, progress_bar=False):
        """Calculate the geodesic distance between two vertices using Dijkstra's algorithm.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        progress_bar : bool, optional
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
        path = self.geodesic(start_vertex, end_vertex)
        sizes = path.compute_cell_sizes(length=True, area=False, volume=False, progress_bar=progress_bar)
        distance = np.sum(sizes['Length'])
        del path
        del sizes
        return distance

    def ray_trace(self, origin, end_point, first_point=False, plot=False,
                  off_screen=None):
        """Perform a single ray trace calculation.

        This requires a mesh and a line segment defined by an origin
        and end_point.

        Parameters
        ----------
        origin : np.ndarray or list
            Start of the line segment.

        end_point : np.ndarray or list
            End of the line segment.

        first_point : bool, optional
            Returns intersection of first point only.

        plot : bool, optional
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
        self.obbTree.IntersectWithLine(np.array(origin),
                                       np.array(end_point),
                                       points, cell_ids)

        intersection_points = _vtk.vtk_to_numpy(points.GetData())
        if first_point and intersection_points.shape[0] >= 1:
            intersection_points = intersection_points[0]

        intersection_cells = []
        if intersection_points.any():
            if first_point:
                ncells = 1
            else:
                ncells = cell_ids.GetNumberOfIds()
            for i in range(ncells):
                intersection_cells.append(cell_ids.GetId(i))
        intersection_cells = np.array(intersection_cells)

        if plot:
            plotter = pyvista.Plotter(off_screen=off_screen)
            plotter.add_mesh(self, label='Test Mesh')
            segment = np.array([origin, end_point])
            plotter.add_lines(segment, 'b', label='Ray Segment')
            plotter.add_mesh(intersection_points, 'r', point_size=10,
                             label='Intersection Points')
            plotter.add_legend()
            plotter.add_axes()
            plotter.show()

        return intersection_points, intersection_cells

    def multi_ray_trace(self, origins, directions, first_point=False, retry=False):
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
        origins : sequence
            Starting point for each trace.

        directions : sequence
            Direction vector for each trace.

        first_point : bool, optional
            Returns intersection of first point only.

        retry : bool, optional
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
        >>> points, rays, cells = sphere.multi_ray_trace([[0, 0, 0]]*3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], first_point=True)  # doctest:+SKIP
        >>> string = ", ".join([f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})" for point in points]) # doctest:+SKIP
        >>> f'Rays intersected at {string}' # doctest:+SKIP
        'Rays intersected at (0.499, 0.000, 0.000), (0.000, 0.497, 0.000), (0.000, 0.000, 0.500)'

        """
        if not self.is_all_triangles:
            raise NotAllTrianglesError

        try:
            import pyembree  # noqa
            import rtree  # noqa
            import trimesh  # noqa
        except (ModuleNotFoundError, ImportError):
            raise ImportError(
                "To use multi_ray_trace please install trimesh, rtree and pyembree with:\n"
                "\tconda install trimesh rtree pyembree"
            )

        origins = np.asarray(origins)
        directions = np.asarray(directions)
        faces_as_array = self.faces.reshape((self.n_faces, 4))[:, 1:]
        tmesh = trimesh.Trimesh(self.points, faces_as_array)
        locations, index_ray, index_tri = tmesh.ray.intersects_location(
            origins, directions, multiple_hits=not first_point
        )
        if retry:
            # gather intersecting rays in lists
            loc_lst, ray_lst, tri_lst = [arr.tolist() for arr in [locations, index_ray, index_tri]]

            # find indices that trimesh failed on
            all_ray_indices = np.arange(len(origins))
            retry_ray_indices = np.setdiff1d(all_ray_indices, index_ray, assume_unique=True)

            # compute ray points for all failed rays at once
            origins_retry = origins[retry_ray_indices, :]  # shape (n_retry, 3)
            directions_retry = directions[retry_ray_indices, :]
            unit_directions = directions_retry / np.linalg.norm(directions_retry,
                                                                axis=1, keepdims=True)
            second_points = origins_retry + unit_directions * self.length  # shape (n_retry, 3)

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

    def plot_boundaries(self, edge_color="red", line_width=None,
                        progress_bar=False, **kwargs):
        """Plot boundaries of a mesh.

        Parameters
        ----------
        edge_color : str, optional
            The color of the edges when they are added to the plotter.

        line_width : int, optional
            Width of the boundary lines.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            All additional keyword arguments will be passed to
            :func:`pyvista.BasePlotter.add_mesh`.

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

        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', None),
                                  notebook=kwargs.pop('notebook', None))
        plotter.add_mesh(edges, color=edge_color, style='wireframe', label='Edges',
                         line_width=line_width)
        plotter.add_mesh(self, label='Mesh', **kwargs)
        plotter.add_legend()
        return plotter.show()

    def plot_normals(self, show_mesh=True, mag=1.0, flip=False,
                     use_every=1, faces=False, color=None, **kwargs):
        """Plot the point normals of a mesh.

        Parameters
        ----------
        show_mesh : bool, optional
            Plot the mesh itself.  Defaults to ``True``.

        mag : float, optional
            Size magnitude of the normal arrows.  Defaults to 1.0.

        flip : bool, optional
            Flip the normal direction when ``True``.  Default
            ``False``.

        use_every : int, optional
            Display every nth normal.  By default every normal is
            displayed.  Display every 10th normal by setting this
            parameter to 10.

        faces : bool, optional
            Plot face normals instead of the default point normals.

        color : str, optional
            Color of the arrows.  Defaults to
            :attr:`pyvista.themes.DefaultTheme.edge_color`.

        **kwargs : dict, optional
            All additional keyword arguments will be passed to
            :func:`pyvista.BasePlotter.add_mesh`.

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
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', None),
                                  notebook=kwargs.pop('notebook', None))
        if show_mesh:
            plotter.add_mesh(self, **kwargs)

        if color is None:
            color = pyvista.global_theme.edge_color

        if faces:
            centers = self.cell_centers().points[::use_every]
            normals = self.cell_normals
        else:
            centers = self.points[::use_every]
            normals = self.point_normals

        if flip:
            normals *= -1

        plotter.add_arrows(centers, normals[::use_every],
                           mag=mag, color=color, show_scalar_bar=False)

        return plotter.show()

    def remove_points(self, remove, mode='any', keep_scalars=True, inplace=False):
        """Rebuild a mesh by removing points.

        Only valid for all-triangle meshes.

        Parameters
        ----------
        remove : np.ndarray
            If remove is a bool array, points that are ``True`` will
            be removed.  Otherwise, it is treated as a list of
            indices.

        mode : str, optional
            When ``'all'``, only faces containing all points flagged
            for removal will be removed.  Default ``'any'``.

        keep_scalars : bool, optional
            When ``True``, point and cell scalars will be passed on to
            the new mesh.

        inplace : bool, optional
            Updates mesh in-place.  Defaults to ``False``.

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
            raise TypeError('Remove must be either a mask or an integer array-like')

        if remove.dtype == np.bool_:
            if remove.size != self.n_points:
                raise ValueError('Mask different size than n_points')
            remove_mask = remove
        else:
            remove_mask = np.zeros(self.n_points, np.bool_)
            remove_mask[remove] = True

        if not self.is_all_triangles:
            raise NotAllTrianglesError

        f = self.faces.reshape(-1, 4)[:, 1:]
        vmask = remove_mask.take(f)
        if mode == 'all':
            fmask = ~(vmask).all(1)
        else:
            fmask = ~(vmask).any(1)

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
            for key in self.point_data:
                newmesh.point_data[key] = self.point_data[key][ridx]

            for key in self.cell_data:
                try:
                    newmesh.cell_data[key] = self.cell_data[key][fmask]
                except:
                    logging.warning(f'Unable to pass cell key {key} onto reduced mesh')

        # Return vtk surface and reverse indexing array
        if inplace:
            self.overwrite(newmesh)
            return self, ridx
        return newmesh, ridx

    def flip_normals(self):
        """Flip normals of a triangular mesh by reversing the point ordering.

        Examples
        --------
        Flip the normals of a sphere and plot the normals before and
        after the flip.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.plot_normals(mag=0.1)
        >>> sphere.flip_normals()
        >>> sphere.plot_normals(mag=0.1, opacity=0.5)

        """
        if not self.is_all_triangles:
            raise NotAllTrianglesError('Can only flip normals on an all triangle mesh.')

        if _vtk.VTK9:
            # use new connectivity API
            f = self._connectivity_array

            # swap first and last point index in-place
            # See: https://stackoverflow.com/a/33362288/3369879
            f[::3], f[2::3] = f[2::3], f[::3].copy()

        else:  # pragma: no cover
            f = self.faces
            f[1::4], f[3::4] = f[3::4], f[1::4].copy()
            self.faces[:] = f

    def delaunay_2d(self, tol=1e-05, alpha=0.0, offset=1.0, bound=False,
                    inplace=False, edge_source=None, progress_bar=False):
        """Apply a 2D Delaunay filter along the best fitting plane.

        This filter can be used to generate a 2d surface from a set of
        points on a plane.  If you want to create a surface from a
        point cloud, see :func:`pyvista.PolyDataFilters.reconstruct_surface`.

        Parameters
        ----------
        tol : float, optional
            Specify a tolerance to control discarding of closely
            spaced points. This tolerance is specified as a fraction
            of the diagonal length of the bounding box of the points.
            Defaults to ``1e-05``.

        alpha : float, optional
            Specify alpha (or distance) value to control output of
            this filter. For a non-zero alpha value, only edges or
            triangles contained within a sphere centered at mesh
            vertices will be output. Otherwise, only triangles will be
            output. Defaults to ``0.0``.

        offset : float, optional
            Specify a multiplier to control the size of the initial,
            bounding Delaunay triangulation. Defaults to ``1.0``.

        bound : bool, optional
            Boolean controls whether bounding triangulation points
            and associated triangles are included in the
            output. These are introduced as an initial triangulation
            to begin the triangulation process. This feature is nice
            for debugging output. Default ``False``.

        inplace : bool, optional
            If ``True``, overwrite this mesh with the triangulated
            mesh. Default ``False``.

        edge_source : pyvista.PolyData, optional
            Specify the source object used to specify constrained
            edges and loops. If set, and lines/polygons are defined, a
            constrained triangulation is created. The lines/polygons
            are assumed to reference points in the input point set
            (i.e. point ids are identical in the input and
            source).

        progress_bar : bool, optional
            Display a progress bar to indicate progress. Default
            ``False``.

        Returns
        -------
        pyvista.PolyData
            Mesh from the 2D delaunay filter.

        Examples
        --------
        First, generate 30 points on circle and plot them.

        >>> import pyvista
        >>> points = pyvista.Polygon(n_sides=30).points
        >>> circle = pyvista.PolyData(points)
        >>> circle.plot(show_edges=True, point_size=15)

        Use :func:`delaunay_2d` to fill the interior of the circle.

        >>> filled_circle = circle.delaunay_2d()
        >>> filled_circle.plot(show_edges=True, line_width=5)

        See :ref:`triangulated_surface` for more examples using this filter.

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
        _update_alg(alg, progress_bar, 'Computing 2D Triangulation')

        # Sometimes lines are given in the output. The
        # `.triangulate()` filter cleans those
        mesh = _get_output(alg).triangulate()
        if inplace:
            self.overwrite(mesh)
            return self
        return mesh

    def compute_arc_length(self, progress_bar=False):
        """Compute the arc length over the length of the probed line.

        It adds a new point-data array named ``"arc_length"`` with the
        computed arc length for each of the polylines in the
        input. For all other cell types, the arc length is set to 0.

        Parameters
        ----------
        progress_bar : bool, optional
            Display a progress bar to indicate progress. Default
            ``False``.

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
        >>> arc.plot(scalars="arc_length")

        """
        alg = _vtk.vtkAppendArcLength()
        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Computing the Arc Length')
        return _get_output(alg)

    def project_points_to_plane(self, origin=None, normal=(0, 0, 1),
                                inplace=False):
        """Project points of this mesh to a plane.

        Parameters
        ----------
        origin : numpy.ndarray or collections.abc.Sequence, optional
            Plane origin.  Defaults to the approximate center of the
            input mesh minus half the length of the input mesh in the
            direction of the normal.

        normal : numpy.ndarray or collections.abc.Sequence, optional
            Plane normal.  Defaults to +Z, i.e. ``[0, 0, 1]``.

        inplace : bool, optional
            Whether to overwrite the original mesh with the projected
            points.

        Returns
        -------
        pyvista.PolyData
            The points of this mesh projected onto a plane.

        Examples
        --------
        Flatten a sphere to the XY plane.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> projected = sphere.project_points_to_plane()
        >>> projected.plot(show_edges=True, line_width=3)

        """
        if not isinstance(normal, (np.ndarray, collections.abc.Sequence)) or len(normal) != 3:
            raise TypeError('Normal must be a length three vector')
        if origin is None:
            origin = np.array(self.center) - np.array(normal)*self.length/2.
        # choose what mesh to use
        if not inplace:
            mesh = self.copy()
        else:
            mesh = self
        # Make plane
        plane = generate_plane(normal, origin)
        # Perform projection in place on the copied mesh
        f = lambda p: plane.ProjectPoint(p, p)
        np.apply_along_axis(f, 1, mesh.points)
        return mesh

    def ribbon(self, width=None, scalars=None, angle=0.0, factor=2.0,
               normal=None, tcoords=False, preference='points', progress_bar=False):
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

        normal : tuple(float), optional
            Normal to use as default.

        tcoords : bool, str, optional
            If ``True``, generate texture coordinates along the
            ribbon. This can also be specified to generate the texture
            coordinates with either ``'length'`` or ``'normalized'``.

        preference : str, optional
            The field preference when searching for the scalars array by
            name.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Ribbon mesh.  Empty if there are no lines in the input dataset.

        Examples
        --------
        Convert a line to a ribbon and plot it.

        >>> import numpy as np
        >>> import pyvista
        >>> n = 1000
        >>> theta = np.linspace(-10 * np.pi, 10 * np.pi, n)
        >>> z = np.linspace(-2, 2, n)
        >>> r = z**2 + 1
        >>> x = r * np.sin(theta)
        >>> y = r * np.cos(theta)
        >>> points = np.column_stack((x, y, z))
        >>> pdata = pyvista.PolyData(points)
        >>> pdata.lines = np.hstack((n, range(n)))
        >>> pdata['distance'] = range(n)
        >>> ribbon = pdata.ribbon(width=0.2)
        >>> ribbon.plot(show_scalar_bar=False)

        """
        if scalars is not None:
            field = get_array_association(self, scalars, preference=preference)
        if width is None:
            width = self.length * 0.1
        alg = _vtk.vtkRibbonFilter()
        alg.SetInputDataObject(self)
        alg.SetWidth(width)
        if normal is not None:
            alg.SetUseDefaultNormal(True)
            alg.SetDefaultNormal(normal)
        alg.SetAngle(angle)
        if scalars is not None:
            alg.SetVaryWidth(True)
            alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars) # args: (idx, port, connection, field, name)
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
        _update_alg(alg, progress_bar, 'Creating a Ribbon')
        return _get_output(alg)

    def extrude(self, vector, capping=False, inplace=False, progress_bar=False):
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

        Parameters
        ----------
        vector : numpy.ndarray or sequence
            Direction and length to extrude the mesh in.

        capping : bool, optional
            Control if the sweep of a 2D object is capped.

        inplace : bool, optional
            Overwrites the original mesh in-place.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Extruded mesh.

        Examples
        --------
        Extrude a half circle arc.

        >>> import pyvista
        >>> arc = pyvista.CircularArc([-1, 0, 0], [1, 0, 0], [0, 0, 0])
        >>> mesh = arc.extrude([0, 0, 1])
        >>> mesh.plot(color='tan')

        Extrude and cap an 8 sided polygon.

        >>> poly = pyvista.Polygon(n_sides=8)
        >>> mesh = poly.extrude((0, 0, 1.5), capping=True)
        >>> mesh.plot(line_width=5, show_edges=True)

        """
        alg = _vtk.vtkLinearExtrusionFilter()
        alg.SetExtrusionTypeToVectorExtrusion()
        alg.SetVector(*vector)
        alg.SetInputData(self)
        alg.SetCapping(capping)
        _update_alg(alg, progress_bar, 'Extruding')
        output = pyvista.wrap(alg.GetOutput())
        if inplace:
            self.overwrite(output)
            return self
        return output

    def extrude_rotate(self, resolution=30, inplace=False,
                       translation=0.0, dradius=0.0, angle=360.0,
                       capping=False, progress_bar=False):
        """Sweep polygonal data creating "skirt" from free edges and lines, and lines from vertices.

        This takes polygonal data as input and generates polygonal
        data on output. The input dataset is swept around the z-axis
        to create new polygonal primitives. These primitives form a
        "skirt" or swept surface. For example, sweeping a line results
        in a cylindrical shell, and sweeping a circle creates a torus.

        There are a number of control parameters for this filter.  You
        can control whether the sweep of a 2D object (i.e., polygon or
        triangle strip) is capped with the generating geometry via the
        ``capping`` parameter. Also, you can control the angle of
        rotation, and whether translation along the z-axis is
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

        Parameters
        ----------
        resolution : int, optional
            Number of pieces to divide line into.

        inplace : bool, optional
            Overwrites the original mesh inplace.

        translation : float, optional
            Total amount of translation along the z-axis.

        dradius : float, optional
            Change in radius during sweep process.

        angle : float, optional
            The angle of rotation in degrees.

        capping : bool, optional
            Control if the sweep of a 2D object is capped.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Rotationally extruded mesh.

        Examples
        --------
        Create a "spring" using the rotational extrusion filter.

        >>> import pyvista
        >>> profile = pyvista.Polygon(center=[1.25, 0.0, 0.0], radius=0.2,
        ...                           normal=(0, 1, 0), n_sides=30)
        >>> extruded = profile.extrude_rotate(resolution=360, translation=4.0,
        ...                                   dradius=.5, angle=1500.0)
        >>> extruded.plot(smooth_shading=True)

        Create a "wine glass" using the rotational extrusion filter.

        >>> import numpy as np
        >>> points = np.array([[-0.18, 0, 0],
        ...                    [-0.18, 0, 0.01],
        ...                    [-0.18, 0, 0.02],
        ...                    [-0.01, 0, 0.03],
        ...                    [-0.01, 0, 0.04],
        ...                    [-0.02, 0, 0.5],
        ...                    [-0.05, 0, 0.75],
        ...                    [-0.1, 0, 0.8],
        ...                    [-0.2, 0, 1.0]])
        >>> spline = pyvista.Spline(points, 30)
        >>> extruded = spline.extrude_rotate(resolution=20)
        >>> extruded.plot(color='tan')

        """
        if resolution <= 0:
            raise ValueError('`resolution` should be positive')
        alg = _vtk.vtkRotationalExtrusionFilter()
        alg.SetInputData(self)
        alg.SetResolution(resolution)
        alg.SetTranslation(translation)
        alg.SetDeltaRadius(dradius)
        alg.SetCapping(capping)
        alg.SetAngle(angle)
        _update_alg(alg, progress_bar, 'Extruding')
        output = pyvista.wrap(alg.GetOutput())
        if inplace:
            self.overwrite(output)
            return self
        return output

    def strip(self, join=False, max_length=1000, pass_cell_data=False,
              pass_cell_ids=False, pass_point_ids=False, progress_bar=False):
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

        Parameters
        ----------
        join : bool, optional
            If ``True``, the output polygonal segments will be joined
            if they are contiguous. This is useful after slicing a
            surface. The default is ``False``.

        max_length : int, optional
            Specify the maximum number of triangles in a triangle
            strip, and/or the maximum number of lines in a poly-line.

        pass_cell_data : bool, optional
            Enable/Disable passing of the CellData in the input to the
            output as FieldData. Note the field data is transformed.
            Default is ``False``.

        pass_cell_ids : bool, optional
            If ``True``, the output polygonal dataset will have a
            celldata array that holds the cell index of the original
            3D cell that produced each output cell. This is useful for
            picking. The default is ``False`` to conserve memory.

        pass_point_ids : bool, optional
            If ``True``, the output polygonal dataset will have a
            pointdata array that holds the point index of the original
            vertex that produced each output vertex. This is useful
            for picking. The default is ``False`` to conserve memory.

        progress_bar : bool, optional
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
        _update_alg(alg, progress_bar, 'Stripping Mesh')
        return _get_output(alg)

    def collision(self, other_mesh, contact_mode=0, box_tolerance=0.001,
                  cell_tolerance=0.0, n_cells_per_node=2, generate_scalars=False,
                  progress_bar=False):
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

        contact_mode : int, optional
            Contact mode.  One of the following:

            * 0 - All contacts. Find all the contacting cell pairs
              with two points per collision
            * 1 - First contact. Quickly find the first contact point.
            * 2 - Half contacts. Find all the contacting cell pairs
              with one point per collision.

        box_tolerance : float, optional
             Oriented bounding box (OBB) tree tolerance in world coordinates.

        cell_tolerance : float, optional
            Cell tolerance (squared value).

        n_cells_per_node : int, optional
            Number of cells in each OBB.

        generate_scalars : bool, optional
            Flag to visualize the contact cells.  If ``True``, the
            contacting cells will be colored from red through blue,
            with collisions first determined colored red.  This array
            is stored as ``"collision_rgba"``.

            .. note::
               This will remove any other cell arrays in the mesh.

        progress_bar : bool, optional
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
        Due to the nature of the `vtk.vtkCollisionDetectionFilter
        <https://vtk.org/doc/nightly/html/classvtkCollisionDetectionFilter.html>`_,
        repeated uses of this method will be slower that using the
        ``vtk.vtkCollisionDetectionFilter`` directly.  The first
        update of the filter creates two instances of `vtkOBBTree
        <https://vtk.org/doc/nightly/html/classvtkOBBTree.html>`_,
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
        >>> import pyvista
        >>> mesh_a = pyvista.Sphere(radius=0.5)
        >>> mesh_b = pyvista.Cube((0.5, 0.5, 0.5)).extract_cells([0, 2, 4])
        >>> collision, ncol = mesh_a.collision(mesh_b, cell_tolerance=1)
        >>> collision['ContactCells'][:10]
        array([471, 471, 468, 468, 469, 469, 466, 466, 467, 467])

        Plot the collisions by creating a collision mask with the
        ``"ContactCells"`` field data.  Cells with a collision are
        colored red.

        >>> scalars = np.zeros(collision.n_cells, dtype=bool)
        >>> scalars[collision.field_data['ContactCells']] = True
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(collision, scalars=scalars, show_scalar_bar=False,
        ...                 cmap='bwr')
        >>> _ = pl.add_mesh(mesh_b, color='tan', line_width=5, opacity=0.7,
        ...                 show_edges=True)
        >>> pl.show()

        Alternatively, simply plot the collisions using the default
        ``'collision_rgba'`` array after enabling ``generate_scalars``.

        >>> collision, ncol = mesh_a.collision(mesh_b, cell_tolerance=1,
        ...                                    generate_scalars=True)
        >>> collision.plot()

        See :ref:`collision_example` for more examples using this filter.

        """
        if not pyvista._vtk.VTK9:  # pragma: no cover
            raise VTKVersionError('The collision filter requires VTK 9 or newer')

        # other mesh must be a polydata
        if not isinstance(other_mesh, pyvista.PolyData):
            other_mesh = other_mesh.extract_surface()

        # according to VTK limitations
        poly_data = self
        if not poly_data.is_all_triangles:
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
        _update_alg(alg, progress_bar, 'Computing collisions')

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

    def reconstruct_surface(self, nbr_sz=None, sample_spacing=None,
                            progress_bar=False):
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

        progress_bar : bool, optional
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

        >>> pl = pv.Plotter(shape=(1,2))
        >>> _ = pl.add_mesh(points)
        >>> _ = pl.add_title('Point Cloud of 3D Surface')
        >>> pl.subplot(0,1)
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
        _update_alg(mc, progress_bar, 'Reconstructing surface')
        surf = pyvista.wrap(mc.GetOutput())
        return surf
