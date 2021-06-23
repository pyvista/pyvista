"""Filters module with a class to manage filters/algorithms for polydata datasets."""
import collections.abc
import logging

import numpy as np

import pyvista
from pyvista import (
    abstract_class, _vtk, NORMALS, generate_plane, assert_empty_kwargs, vtk_id_list_to_array, get_array
)
from pyvista.core.errors import NotAllTrianglesError
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters


@abstract_class
class PolyDataFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for polydata datasets."""

    def edge_mask(poly_data, angle):
        """Return a mask of the points of a surface mesh that has a surface angle greater than angle.

        Parameters
        ----------
        angle : float
            Angle to consider an edge.

        """
        if not isinstance(poly_data, pyvista.PolyData):  # pragma: no cover
            poly_data = pyvista.PolyData(poly_data)
        poly_data.point_arrays['point_ind'] = np.arange(poly_data.n_points)
        featureEdges = _vtk.vtkFeatureEdges()
        featureEdges.SetInputData(poly_data)
        featureEdges.FeatureEdgesOn()
        featureEdges.BoundaryEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.SetFeatureAngle(angle)
        featureEdges.Update()
        edges = _get_output(featureEdges)
        orig_id = pyvista.point_array(edges, 'point_ind')

        return np.in1d(poly_data.point_arrays['point_ind'], orig_id,
                       assume_unique=True)

    def boolean_cut(poly_data, cut, tolerance=1E-5, inplace=False):
        """Perform a Boolean cut using another mesh.

        Parameters
        ----------
        cut : pyvista.PolyData
            Mesh making the cut

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        mesh : pyvista.PolyData
            The cut mesh.

        """
        if not isinstance(cut, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")
        if not poly_data.is_all_triangles() or not cut.is_all_triangles():
            raise NotAllTrianglesError("Make sure both the input and output are triangulated.")

        bfilter = _vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToIntersection()
        # bfilter.SetOperationToDifference()

        bfilter.SetInputData(1, cut)
        bfilter.SetInputData(0, poly_data)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.SetTolerance(tolerance)
        bfilter.Update()

        mesh = _get_output(bfilter)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def boolean_add(poly_data, mesh, inplace=False):
        """Add a mesh to the current mesh.

        Does not attempt to "join" the meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to add.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        joinedmesh : pyvista.PolyData
            The joined mesh.

        """
        if not isinstance(mesh, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")

        vtkappend = _vtk.vtkAppendPolyData()
        vtkappend.AddInputData(poly_data)
        vtkappend.AddInputData(mesh)
        vtkappend.Update()

        mesh = _get_output(vtkappend)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def __add__(poly_data, mesh):
        """Merge these two meshes."""
        if not isinstance(mesh, _vtk.vtkPolyData):
            return DataSetFilters.__add__(poly_data, mesh)
        return PolyDataFilters.boolean_add(poly_data, mesh)

    def boolean_union(poly_data, mesh, inplace=False):
        """Combine two meshes and attempts to create a manifold mesh.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to perform a union against.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        union : pyvista.PolyData
            The union mesh.

        """
        if not isinstance(mesh, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")

        bfilter = _vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToUnion()
        bfilter.SetInputData(1, mesh)
        bfilter.SetInputData(0, poly_data)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.Update()

        mesh = _get_output(bfilter)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def boolean_difference(poly_data, mesh, inplace=False):
        """Combine two meshes and retains only the volume in common between the meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to perform a union against.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        union : pyvista.PolyData
            The union mesh.

        """
        if not isinstance(mesh, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")

        bfilter = _vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToDifference()
        bfilter.SetInputData(1, mesh)
        bfilter.SetInputData(0, poly_data)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.Update()

        mesh = _get_output(bfilter)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def intersection(poly_data, mesh, split_first=True, split_second=True):
        """Compute the intersection between two meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to intersect with.

        split_first : bool, optional
            If `True`, return the first input mesh split by the intersection with the
            second input mesh.

        split_second : bool, optional
            If `True`, return the second input mesh split by the intersection with the
            first input mesh.

        Returns
        -------
        intersection: pyvista.PolyData
            The intersection line.

        first_split: pyvista.PolyData
            The first mesh split along the intersection. Returns the original first mesh
            if `split_first` is False.

        second_split: pyvista.PolyData
            The second mesh split along the intersection. Returns the original second mesh
            if `split_second` is False.

        Examples
        --------
        Intersect two spheres, returning the intersection and both spheres
        which have new points/cells along the intersection line.

        >>> import pyvista as pv
        >>> s1 = pv.Sphere()
        >>> s2 = pv.Sphere(center=(0.25, 0, 0))
        >>> intersection, s1_split, s2_split = s1.intersection(s2)

        The mesh splitting takes additional time and can be turned
        off for either mesh individually.

        >>> intersection, _, s2_split = s1.intersection(s2, \
                                                        split_first=False, \
                                                        split_second=True)

        """
        intfilter = _vtk.vtkIntersectionPolyDataFilter()
        intfilter.SetInputDataObject(0, poly_data)
        intfilter.SetInputDataObject(1, mesh)
        intfilter.SetComputeIntersectionPointArray(True)
        intfilter.SetSplitFirstOutput(split_first)
        intfilter.SetSplitSecondOutput(split_second)
        intfilter.Update()

        intersection = _get_output(intfilter, oport=0)
        first = _get_output(intfilter, oport=1)
        second = _get_output(intfilter, oport=2)

        return intersection, first, second

    def curvature(poly_data, curv_type='mean'):
        """Return the pointwise curvature of a mesh.

        Parameters
        ----------
        mesh : vtk.polydata
            vtk polydata mesh

        curvature string, optional
            One of the following strings
            Mean
            Gaussian
            Maximum
            Minimum

        Returns
        -------
        curvature : np.ndarray
            Curvature values

        """
        curv_type = curv_type.lower()

        # Create curve filter and compute curvature
        curvefilter = _vtk.vtkCurvatures()
        curvefilter.SetInputData(poly_data)
        if curv_type == 'mean':
            curvefilter.SetCurvatureTypeToMean()
        elif curv_type == 'gaussian':
            curvefilter.SetCurvatureTypeToGaussian()
        elif curv_type == 'maximum':
            curvefilter.SetCurvatureTypeToMaximum()
        elif curv_type == 'minimum':
            curvefilter.SetCurvatureTypeToMinimum()
        else:
            raise ValueError('Curv_Type must be either "Mean", '
                             '"Gaussian", "Maximum", or "Minimum"')
        curvefilter.Update()

        # Compute and return curvature
        curv = _get_output(curvefilter)
        return _vtk.vtk_to_numpy(curv.GetPointData().GetScalars())

    def plot_curvature(poly_data, curv_type='mean', **kwargs):
        """Plot the curvature.

        Parameters
        ----------
        curvtype : str, optional
            One of the following strings indicating curvature type:

            * ``'Mean'``
            * ``'Gaussian'``
            * ``'Maximum'``
            * ``'Minimum'``

        **kwargs : optional
            See :func:`pyvista.plot`

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up.

        Examples
        --------
        Plot the mean curvature of an example mesh.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> cpos = hills.plot_curvature(smooth_shading=True)

        """
        kwargs.setdefault('scalar_bar_args',
                          {'title': f'{curv_type.capitalize()} Curvature'})
        return poly_data.plot(scalars=poly_data.curvature(curv_type),
                              **kwargs)

    def triangulate(poly_data, inplace=False):
        """Return an all triangle mesh.

        More complex polygons will be broken down into tetrahedrals.

        Parameters
        ----------
        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh containing only triangles.

        """
        trifilter = _vtk.vtkTriangleFilter()
        trifilter.SetInputData(poly_data)
        trifilter.PassVertsOff()
        trifilter.PassLinesOff()
        trifilter.Update()

        mesh = _get_output(trifilter)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def smooth(poly_data, n_iter=20, relaxation_factor=0.01, convergence=0.0,
               edge_angle=15, feature_angle=45,
               boundary_smoothing=True, feature_smoothing=False, inplace=False):
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

        Returns
        -------
        mesh : pyvista.PolyData
            Smoothed mesh.

        Examples
        --------
        Smooth the edges of an all triangular cube

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(5).clean()
        >>> smooth_cube = cube.smooth(1000, feature_smoothing=False)
        >>> n_edge_cells = cube.extract_feature_edges().n_cells
        >>> n_smooth_cells = smooth_cube.extract_feature_edges().n_cells
        >>> print(f'Sharp Edges on Cube:        {n_edge_cells}')
        Sharp Edges on Cube:        384
        >>> print(f'Sharp Edges on Smooth Cube: {n_smooth_cells}')
        Sharp Edges on Smooth Cube: 12

        """
        alg = _vtk.vtkSmoothPolyDataFilter()
        alg.SetInputData(poly_data)
        alg.SetNumberOfIterations(n_iter)
        alg.SetConvergence(convergence)
        alg.SetFeatureEdgeSmoothing(feature_smoothing)
        alg.SetFeatureAngle(feature_angle)
        alg.SetEdgeAngle(edge_angle)
        alg.SetBoundarySmoothing(boundary_smoothing)
        alg.SetRelaxationFactor(relaxation_factor)
        alg.Update()

        mesh = _get_output(alg)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def decimate_pro(poly_data, reduction, feature_angle=45.0, split_angle=75.0, splitting=True,
                     pre_split_mesh=False, preserve_topology=False, inplace=False):
        """Reduce the number of triangles in a triangular mesh.

        It forms a good approximation to the original geometry. Based on the algorithm
        originally described in "Decimation of Triangle Meshes", Proc Siggraph 92.

        Parameters
        ----------
        reduction : float
            Reduction factor. A value of 0.9 will leave 10 % of the original number
            of vertices.

        feature_angle : float, optional
            Angle used to define what an edge is (i.e., if the surface normal between
            two adjacent triangles is >= feature_angle, an edge exists).

        split_angle : float, optional
            Angle used to control the splitting of the mesh. A split line exists
            when the surface normals between two edge connected triangles are >= split_angle.

        splitting : bool, optional
            Controls the splitting of the mesh at corners, along edges, at non-manifold
            points, or anywhere else a split is required. Turning splitting off
            will better preserve the original topology of the mesh, but may not
            necessarily give the exact requested decimation.

        pre_split_mesh : bool, optional
            Separates the mesh into semi-planar patches, which are disconnected
            from each other. This can give superior results in some cases. If pre_split_mesh
            is set to True, the mesh is split with the specified split_angle. Otherwise
            mesh splitting is deferred as long as possible.

        preserve_topology : bool, optional
            Controls topology preservation. If on, mesh splitting and hole elimination
            will not occur. This may limit the maximum reduction that may be achieved.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        mesh : pyvista.PolyData
            Decimated mesh.

        """
        alg = _vtk.vtkDecimatePro()
        alg.SetInputData(poly_data)
        alg.SetTargetReduction(reduction)
        alg.SetPreserveTopology(preserve_topology)
        alg.SetFeatureAngle(feature_angle)
        alg.SetSplitting(splitting)
        alg.SetSplitAngle(split_angle)
        alg.SetPreSplitMesh(pre_split_mesh)
        alg.Update()

        mesh = _get_output(alg)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def tube(poly_data, radius=None, scalars=None, capping=True, n_sides=20,
             radius_factor=10, preference='point', inplace=False):
        """Generate a tube around each input line.

        The radius of the tube can be set to linearly vary with a scalar value.

        Parameters
        ----------
        radius : float
            Minimum tube radius (minimum because the tube radius may vary).

        scalars : str, optional
            scalars array by which the radius varies

        capping : bool, optional
            Turn on/off whether to cap the ends with polygons. Default ``True``.

        n_sides : int, optional
            Set the number of sides for the tube. Minimum of 3.

        radius_factor : float, optional
            Maximum tube radius in terms of a multiple of the minimum radius.

        preference : str, optional
            The field preference when searching for the scalars array by name.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        mesh : pyvista.PolyData
            Tube-filtered mesh.

        Examples
        --------
        Convert a single line to a tube

        >>> import pyvista as pv
        >>> line = pv.Line()
        >>> tube = line.tube(radius=0.02)
        >>> print('Line Cells:', line.n_cells)
        Line Cells: 1
        >>> print('Tube Cells:', tube.n_cells)
        Tube Cells: 22

        """
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
            _, field = poly_data.get_array(scalars, preference=preference, info=True)
            # args: (idx, port, connection, field, name)
            tube.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
            tube.SetVaryRadiusToVaryRadiusByScalar()
        # Apply the filter
        tube.Update()

        mesh = _get_output(tube)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def subdivide(poly_data, nsub, subfilter='linear', inplace=False):
        """Increase the number of triangles in a single, connected triangular mesh.

        Uses one of the following vtk subdivision filters to subdivide a mesh.
        vtkButterflySubdivisionFilter
        vtkLoopSubdivisionFilter
        vtkLinearSubdivisionFilter

        Linear subdivision results in the fastest mesh subdivision,
        but it does not smooth mesh edges, but rather splits each
        triangle into 4 smaller triangles.

        Butterfly and loop subdivision perform smoothing when
        dividing, and may introduce artifacts into the mesh when
        dividing.

        Subdivision filter appears to fail for multiple part meshes.
        Should be one single mesh.

        Parameters
        ----------
        nsub : int
            Number of subdivisions.  Each subdivision creates 4 new
            triangles, so the number of resulting triangles is
            ``nface*4**nsub`` where ``nface`` is the current number of
            faces.

        subfilter : string, optional
            Can be one of the following: 'butterfly', 'loop', 'linear'.

        inplace : bool, optional
            Updates mesh in-place. Default ``False``.

        Returns
        -------
        mesh : Polydata object
            ``pyvista`` polydata object.

        Examples
        --------
        >>> from pyvista import examples
        >>> import pyvista
        >>> mesh = pyvista.PolyData(examples.planefile)
        >>> submesh = mesh.subdivide(1, 'loop')

        Alternatively, update the mesh in-place.

        >>> submesh = mesh.subdivide(1, 'loop', inplace=True)

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
        sfilter.SetInputData(poly_data)
        sfilter.Update()

        submesh = _get_output(sfilter)
        if inplace:
            poly_data.overwrite(submesh)
            return poly_data
        else:
            return submesh

    def subdivide_adaptive(poly_data, max_edge_len=None, max_tri_area=None,
                           max_n_tris=None, max_n_passes=None, inplace=False):
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

        Returns
        -------
        :class:`pyvista.PolyData`
            Subdivided mesh

        Examples
        --------
        >>> from pyvista import examples
        >>> import pyvista
        >>> mesh = pyvista.PolyData(examples.planefile)
        >>> submesh = mesh.subdivide_adaptive(max_n_passes=2)

        Alternatively, update the mesh in-place.

        >>> submesh = mesh.subdivide_adaptive(max_n_passes=2, inplace=True)

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

        sfilter.SetInputData(poly_data)
        sfilter.Update()
        submesh = _get_output(sfilter)

        if inplace:
            poly_data.overwrite(submesh)
            return poly_data
        else:
            return submesh

    def decimate(poly_data, target_reduction, volume_preservation=False,
                 attribute_error=False, scalars=True, vectors=True,
                 normals=False, tcoords=True, tensors=True, scalars_weight=0.1,
                 vectors_weight=0.1, normals_weight=0.1, tcoords_weight=0.1,
                 tensors_weight=0.1, inplace=False, progress_bar=False):
        """Reduce the number of triangles in a triangular mesh using vtkQuadricDecimation.

        Parameters
        ----------
        mesh : vtk.PolyData
            Mesh to decimate

        target_reduction : float
            Fraction of the original mesh to remove.
            TargetReduction is set to 0.9, this filter will try to reduce
            the data set to 10% of its original size and will remove 90%
            of the input triangles.

        volume_preservation : bool, optional
            Decide whether to activate volume preservation which greatly reduces
            errors in triangle normal direction. If off, volume preservation is
            disabled and if AttributeErrorMetric is active, these errors can be
            large. Defaults to False.

        attribute_error : bool, optional
            Decide whether to include data attributes in the error metric. If
            off, then only geometric error is used to control the decimation.
            Defaults to False.

        scalars : bool, optional
            If attribute errors are to be included in the metric (i.e.,
            AttributeErrorMetric is on), then the following flags control which
            attributes are to be included in the error calculation. Defaults to
            True.

        vectors : bool, optional
            See scalars parameter. Defaults to True.

        normals : bool, optional
            See scalars parameter. Defaults to False.

        tcoords : bool, optional
            See scalars parameter. Defaults to True.

        tensors : bool, optional
            See scalars parameter. Defaults to True.

        scalars_weight : float, optional
            The scaling weight contribution of the scalar attribute. These
            values are used to weight the contribution of the attributes towards
            the error metric. Defaults to 0.1.

        vectors_weight : float, optional
            See scalars weight parameter. Defaults to 0.1.

        normals_weight : float, optional
            See scalars weight parameter. Defaults to 0.1.

        tcoords_weight : float, optional
            See scalars weight parameter. Defaults to 0.1.

        tensors_weight : float, optional
            See scalars weight parameter. Defaults to 0.1.

        inplace : bool, optional
            Updates mesh in-place.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        outmesh : pyvista.PolyData
            Decimated mesh.

        Examples
        --------
        Decimate a sphere while preserving its volume

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=90, phi_resolution=90)
        >>> print(sphere.n_cells)
        15840
        >>> dec_sphere = sphere.decimate(0.9, volume_preservation=True)
        >>> print(dec_sphere.n_cells)
        1584

        Notes
        -----
        If you encounter a segmentation fault or other error, consider
        using ``clean`` to remove any invalid cells before using this
        filter.

        """
        # create decimation filter
        alg = _vtk.vtkQuadricDecimation()  # vtkDecimatePro as well

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

        alg.SetInputData(poly_data)
        _update_alg(alg, progress_bar, 'Decimating')

        mesh = _get_output(alg)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def compute_normals(poly_data, cell_normals=True,
                        point_normals=True, split_vertices=False,
                        flip_normals=False, consistent_normals=True,
                        auto_orient_normals=False,
                        non_manifold_traversal=True,
                        feature_angle=30.0, inplace=False):
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

        Returns
        -------
        mesh : pyvista.PolyData
            Updated mesh with cell and point normals.

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
        >>> sphere_with_norm.point_arrays['Normals'].shape
        (842, 3)
        >>> sphere_with_norm.cell_arrays['Normals'].shape
        (1680, 3)

        Notes
        -----
        Previous arrays named "Normals" will be overwritten.

        Normals are computed only for polygons and triangle
        strips. Normals are not computed for lines or vertices.

        Triangle strips are broken up into triangle polygons. You may
        want to restrip the triangles.

        May be easier to run ``mesh.point_normals`` or ``mesh.cell_normals``.

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
        normal.SetInputData(poly_data)
        normal.Update()

        mesh = _get_output(normal)
        if point_normals:
            mesh.GetPointData().SetActiveNormals('Normals')
        if cell_normals:
            mesh.GetCellData().SetActiveNormals('Normals')

        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def clip_closed_surface(poly_data, normal='x', origin=None,
                            tolerance=1e-06, inplace=False):
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

        Returns
        -------
        clipped_mesh : pyvista.PolyData
            The clipped mesh.

        Examples
        --------
        Clip a sphere in the X direction centered at the origin.  This
        will leave behind half a sphere in the positive X direction.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> clipped_mesh = sphere.clip_closed_surface()

        Clip the sphere at the xy plane and leave behind half the
        sphere in the positive Z direction.  Shift the clip upwards to
        leave a smaller mesh behind.

        >>> clipped_mesh = sphere.clip_closed_surface('z', origin=[0, 0, 0.3])

        """
        # verify it is manifold
        if poly_data.n_open_edges > 0:
            raise ValueError("This surface appears to be non-manifold.")
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = poly_data.center

        # create the plane for clipping
        plane = generate_plane(normal, origin)
        collection = _vtk.vtkPlaneCollection()
        collection.AddItem(plane)

        alg = _vtk.vtkClipClosedSurface()
        alg.SetGenerateFaces(True)
        alg.SetInputDataObject(poly_data)
        alg.SetTolerance(tolerance)
        alg.SetClippingPlanes(collection)
        alg.Update()  # Perform the Cut
        result = _get_output(alg)

        if inplace:
            poly_data.overwrite(result)
            return poly_data
        else:
            return result

    def fill_holes(poly_data, hole_size, inplace=False, progress_bar=False):  # pragma: no cover
        """
        Fill holes in a pyvista.PolyData or vtk.vtkPolyData object.

        Holes are identified by locating boundary edges, linking them together
        into loops, and then triangulating the resulting loops. Note that you
        can specify an approximate limit to the size of the hole that can be
        filled.

        Parameters
        ----------
        hole_size : float
            Specifies the maximum hole size to fill. This is represented as a
            radius to the bounding circumsphere containing the hole. Note that
            this is an approximate area; the actual area cannot be computed
            without first triangulating the hole.

        inplace : bool, optional
            Return new mesh or overwrite input.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh with holes filled.

        Examples
        --------
        Create a partial sphere with a hole and then fill it

        >>> import pyvista as pv
        >>> sphere_with_hole = pv.Sphere(end_theta=330)
        >>> sphere = sphere_with_hole.fill_holes(1000)
        >>> edges = sphere.extract_feature_edges(feature_edges=False,
        ...                                      manifold_edges=False)
        >>> assert edges.n_cells == 0

        """
        logging.warning('pyvista.PolyData.fill_holes is known to segfault. '
                        'Use at your own risk')
        alg = _vtk.vtkFillHolesFilter()
        alg.SetHoleSize(hole_size)
        alg.SetInputData(poly_data)
        _update_alg(alg, progress_bar, 'Filling Holes')

        mesh = _get_output(alg)
        if inplace:
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def clean(poly_data, point_merging=True, tolerance=None, lines_to_points=True,
              polys_to_lines=True, strips_to_polys=True, inplace=False,
              absolute=True, progress_bar=False, **kwargs):
        """Clean the mesh.

        This merges duplicate points, removes unused points, and/or
        removes degenerate cells.

        Parameters
        ----------
        point_merging : bool, optional
            Enables point merging.  On by default.

        tolerance : float, optional
            Set merging tolerance.  When enabled merging is set to
            absolute distance. If ``absolute`` is ``False``, then the
            merging tolerance is a fraction of the bounding box
            length. The alias ``merge_tol`` is also excepted.

        lines_to_points : bool, optional
            Turn on/off conversion of degenerate lines to points.
            Enabled by default.

        polys_to_lines : bool, optional
            Turn on/off conversion of degenerate polys to lines.
            Enabled by default.

        strips_to_polys : bool, optional
            Turn on/off conversion of degenerate strips to polys.

        inplace : bool, optional
            Updates mesh in-place. Default ``False``.

        absolute : bool, optional
            Control if ``tolerance`` is an absolute distance or a
            fraction.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        mesh : pyvista.PolyData
            Cleaned mesh.

        Examples
        --------
        Create a mesh with a degenerate face and then clean it,
        removing the degenerate face

        >>> import pyvista as pv
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        >>> faces = np.array([3, 0, 1, 2, 3, 0, 3, 3])
        >>> mesh = pv.PolyData(points, faces)
        >>> mout = mesh.clean()
        >>> print(mout.faces)  # doctest:+SKIP
        [3 0 1 2]

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
        alg.SetInputData(poly_data)
        _update_alg(alg, progress_bar, 'Cleaning')
        output = _get_output(alg)

        # Check output so no segfaults occur
        if output.n_points < 1:
            raise ValueError('Clean tolerance is too high. Empty mesh returned.')

        if inplace:
            poly_data.overwrite(output)
            return poly_data
        else:
            return output

    def geodesic(poly_data, start_vertex, end_vertex, inplace=False,
                 keep_order=True):
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

        Returns
        -------
        output : pyvista.PolyData
            ``PolyData`` object consisting of the line segment between the
            two given vertices. If ``inplace`` is ``True`` this is the
            same object as the input mesh.

        Examples
        --------
        Plot the path between two points on a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> path = sphere.geodesic(0, 100)
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(sphere)
        >>> actor = pl.add_mesh(path, line_width=5, color='k')
        >>> cpos = pl.show()

        """
        if not (0 <= start_vertex < poly_data.n_points and
                0 <= end_vertex < poly_data.n_points):
            raise IndexError('Invalid point indices.')
        if not poly_data.is_all_triangles():
            raise NotAllTrianglesError("Input mesh for geodesic path must be all triangles.")

        dijkstra = _vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(poly_data)
        dijkstra.SetStartVertex(start_vertex)
        dijkstra.SetEndVertex(end_vertex)
        dijkstra.Update()
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
            poly_data.overwrite(output)
            return poly_data
        else:
            return output

    def geodesic_distance(poly_data, start_vertex, end_vertex):
        """Calculate the geodesic distance between two vertices using Dijkstra's algorithm.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        Returns
        -------
        length : float
            Length of the geodesic segment.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> length = sphere.geodesic_distance(0, 100)
        >>> print(f'Length is {length:.3f}')
        Length is 0.812

        """
        path = poly_data.geodesic(start_vertex, end_vertex)
        sizes = path.compute_cell_sizes(length=True, area=False, volume=False)
        distance = np.sum(sizes['Length'])
        del path
        del sizes
        return distance

    def ray_trace(poly_data, origin, end_point, first_point=False, plot=False,
                  off_screen=False):
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
            Plots ray trace results

        off_screen : bool, optional
            Plots off screen when ``plot=True``.  Used for unit testing.

        Returns
        -------
        intersection_points : np.ndarray
            Location of the intersection points.  Empty array if no
            intersections.

        intersection_cells : np.ndarray
            Indices of the intersection cells.  Empty array if no
            intersections.

        Examples
        --------
        Compute the intersection between a ray from the origin and
        [1, 0, 0] and a sphere with radius 0.5 centered at the origin

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> point, cell = sphere.ray_trace([0, 0, 0], [1, 0, 0], first_point=True)
        >>> print(f'Intersected at {point[0]:.3f} {point[1]:.3f} {point[2]:.3f}')
        Intersected at 0.499 0.000 0.000

        """
        points = _vtk.vtkPoints()
        cell_ids = _vtk.vtkIdList()
        poly_data.obbTree.IntersectWithLine(np.array(origin),
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
            plotter.add_mesh(poly_data, label='Test Mesh')
            segment = np.array([origin, end_point])
            plotter.add_lines(segment, 'b', label='Ray Segment')
            plotter.add_mesh(intersection_points, 'r', point_size=10,
                             label='Intersection Points')
            plotter.add_legend()
            plotter.add_axes()
            plotter.show()

        return intersection_points, intersection_cells

    def multi_ray_trace(poly_data, origins, directions, first_point=False, retry=False):
        """Perform multiple ray trace calculations.

        This requires a mesh with only triangular faces,
        an array of origin points and an equal sized array of
        direction vectors to trace along.

        The embree library used for vectorisation of the ray traces is known to occasionally
        return no intersections where the VTK implementation would return an intersection.
        If the result appears to be missing some intersection points, set retry=True to run a second pass over rays
        that returned no intersections, using the VTK ray_trace implementation.


        Parameters
        ----------
        origins : np.ndarray or list
            Starting point for each trace.

        directions : np.ndarray or list
            Direction vector for each trace.

        first_point : bool, optional
            Returns intersection of first point only.

        retry : bool, optional
            Will retry rays that return no intersections using the ray_trace

        Returns
        -------
        intersection_points : np.ndarray
            Location of the intersection points.  Empty array if no
            intersections.

        intersection_rays : np.ndarray
            Indices of the ray for each intersection point. Empty array if no
            intersections.

        intersection_cells : np.ndarray
            Indices of the intersection cells.  Empty array if no
            intersections.

        Examples
        --------
        Compute the intersection between rays from the origin in
        directions ``[1, 0, 0]``, ``[0, 1, 0]`` and ``[0, 0, 1]``, and
        a sphere with radius 0.5 centered at the origin

        >>> import pyvista as pv # doctest: +SKIP
        ... sphere = pv.Sphere()
        ... points, rays, cells = sphere.multi_ray_trace([[0, 0, 0]]*3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], first_point=True)
        ... string = ", ".join([f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})" for point in points])
        ... print(f'Rays intersected at {string}')
        Rays intersected at (0.499, 0.000, 0.000), (0.000, 0.497, 0.000), (0.000, 0.000, 0.500)

        """
        if not poly_data.is_all_triangles():
            raise NotAllTrianglesError

        try:
            import trimesh, rtree, pyembree
        except (ModuleNotFoundError, ImportError):
            raise ImportError(
                "To use multi_ray_trace please install trimesh, rtree and pyembree with:\n"
                "\tconda install trimesh rtree pyembree"
            )

        origins = np.asarray(origins)
        directions = np.asarray(directions)
        faces_as_array = poly_data.faces.reshape((poly_data.n_faces, 4))[:, 1:]
        tmesh = trimesh.Trimesh(poly_data.points, faces_as_array)
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
            second_points = origins_retry + unit_directions * poly_data.length  # shape (n_retry, 3)

            for id_r, origin, second_point in zip(retry_ray_indices, origins_retry, second_points):
                locs, indices = poly_data.ray_trace(origin, second_point, first_point=first_point)
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

    def plot_boundaries(poly_data, edge_color="red", **kwargs):
        """Plot boundaries of a mesh.

        Parameters
        ----------
        edge_color : str, optional
            The color of the edges when they are added to the plotter.

        kwargs : optional
            All additional keyword arguments will be passed to
            :func:`pyvista.BasePlotter.add_mesh`

        """
        edges = DataSetFilters.extract_feature_edges(poly_data)

        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', None),
                                  notebook=kwargs.pop('notebook', None))
        plotter.add_mesh(edges, color=edge_color, style='wireframe', label='Edges')
        plotter.add_mesh(poly_data, label='Mesh', **kwargs)
        plotter.add_legend()
        return plotter.show()

    def plot_normals(poly_data, show_mesh=True, mag=1.0, flip=False,
                     use_every=1, **kwargs):
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

        Examples
        --------
        Plot the normals of a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> cpos = sphere.plot_normals(mag=0.1)

        """
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', None),
                                  notebook=kwargs.pop('notebook', None))
        if show_mesh:
            plotter.add_mesh(poly_data, **kwargs)

        normals = poly_data.point_normals
        if flip:
            normals *= -1
        plotter.add_arrows(poly_data.points[::use_every],
                           normals[::use_every], mag=mag, show_scalar_bar=False)
        return plotter.show()

    def remove_points(poly_data, remove, mode='any', keep_scalars=True, inplace=False):
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
        mesh : pyvista.PolyData
            Mesh without the points flagged for removal.

        ridx : np.ndarray
            Indices of new points relative to the original mesh.

        Examples
        --------
        Remove the first 100 points from a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> reduced_sphere, ridx = sphere.remove_points(range(100))

        """
        remove = np.asarray(remove)

        # np.asarray will eat anything, so we have to weed out bogus inputs
        if not issubclass(remove.dtype.type, (np.bool_, np.integer)):
            raise TypeError('Remove must be either a mask or an integer array-like')

        if remove.dtype == np.bool_:
            if remove.size != poly_data.n_points:
                raise ValueError('Mask different size than n_points')
            remove_mask = remove
        else:
            remove_mask = np.zeros(poly_data.n_points, np.bool_)
            remove_mask[remove] = True

        if not poly_data.is_all_triangles():
            raise NotAllTrianglesError

        f = poly_data.faces.reshape(-1, 4)[:, 1:]
        vmask = remove_mask.take(f)
        if mode == 'all':
            fmask = ~(vmask).all(1)
        else:
            fmask = ~(vmask).any(1)

        # Regenerate face and point arrays
        uni = np.unique(f.compress(fmask, 0), return_inverse=True)
        new_points = poly_data.points.take(uni[0], 0)

        nfaces = fmask.sum()
        faces = np.empty((nfaces, 4), dtype=pyvista.ID_TYPE)
        faces[:, 0] = 3
        faces[:, 1:] = np.reshape(uni[1], (nfaces, 3))

        newmesh = pyvista.PolyData(new_points, faces, deep=True)
        ridx = uni[0]

        # Add scalars back to mesh if requested
        if keep_scalars:
            for key in poly_data.point_arrays:
                newmesh.point_arrays[key] = poly_data.point_arrays[key][ridx]

            for key in poly_data.cell_arrays:
                try:
                    newmesh.cell_arrays[key] = poly_data.cell_arrays[key][fmask]
                except:
                    logging.warning(f'Unable to pass cell key {key} onto reduced mesh')

        # Return vtk surface and reverse indexing array
        if inplace:
            poly_data.overwrite(newmesh)
            return poly_data, ridx
        else:
            return newmesh, ridx

    def flip_normals(poly_data):
        """Flip normals of a triangular mesh by reversing the point ordering.

        Examples
        --------
        Flip the normals of a sphere and plot the normals before and
        after the flip.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> cpos = sphere.plot_normals(mag=0.1)
        >>> sphere.flip_normals()
        >>> cpos = sphere.plot_normals(mag=0.1)

        """
        if not poly_data.is_all_triangles:
            raise NotAllTrianglesError('Can only flip normals on an all triangle mesh')

        f = poly_data.faces.reshape((-1, 4))
        f[:, 1:] = f[:, 1:][:, ::-1]
        poly_data.faces = f

    def delaunay_2d(poly_data, tol=1e-05, alpha=0.0, offset=1.0, bound=False,
                    inplace=False, edge_source=None, progress_bar=False):
        """Apply a delaunay 2D filter along the best fitting plane.

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

        Examples
        --------
        Extract the points of a sphere and then convert the point
        cloud to a surface mesh.  Note that only the bottom half is
        converted to a mesh.

        >>> import pyvista as pv
        >>> points = pv.PolyData(pv.Sphere().points)
        >>> mesh = points.delaunay_2d()
        >>> mesh.is_all_triangles()
        True

        """
        alg = _vtk.vtkDelaunay2D()
        alg.SetProjectionPlaneMode(_vtk.VTK_BEST_FITTING_PLANE)
        alg.SetInputDataObject(poly_data)
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
            poly_data.overwrite(mesh)
            return poly_data
        else:
            return mesh

    def compute_arc_length(poly_data):
        """Compute the arc length over the length of the probed line.

        It adds a new point-data array named ``"arc_length"`` with the
        computed arc length for each of the polylines in the
        input. For all other cell types, the arc length is set to 0.

        Returns
        -------
        arc_length : float
            Arc length of the length of the probed line.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> path = sphere.geodesic(0, 100)
        >>> length = path.compute_arc_length()['arc_length'][-1]
        >>> print(f'Length is {length:.3f}')
        Length is 0.812

        This is identical to the geodesic_distance.

        >>> length = sphere.geodesic_distance(0, 100)
        >>> print(f'Length is {length:.3f}')
        Length is 0.812

        You can also plot the arc_length

        >>> arc = path.compute_arc_length()
        >>> cpos = arc.plot(scalars="arc_length")

        """
        alg = _vtk.vtkAppendArcLength()
        alg.SetInputData(poly_data)
        alg.Update()
        return _get_output(alg)

    def project_points_to_plane(poly_data, origin=None, normal=(0, 0, 1),
                                inplace=False):
        """Project points of this mesh to a plane.

        Parameters
        ----------
        origin : np.ndarray or collections.abc.Sequence, optional
            Plane origin.  Defaults the approximate center of the
            input mesh minus half the length of the input mesh in the
            direction of the normal.

        normal : np.ndarray or collections.abc.Sequence, optional
            Plane normal.  Defaults to +Z ``[0, 0, 1]``

        inplace : bool, optional
            Overwrite the original mesh with the projected points

        Examples
        --------
        Flatten a sphere to the XY plane

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> projected = sphere.project_points_to_plane([0, 0, 0])

        """
        if not isinstance(normal, (np.ndarray, collections.abc.Sequence)) or len(normal) != 3:
            raise TypeError('Normal must be a length three vector')
        if origin is None:
            origin = np.array(poly_data.center) - np.array(normal)*poly_data.length/2.
        # choose what mesh to use
        if not inplace:
            mesh = poly_data.copy()
        else:
            mesh = poly_data
        # Make plane
        plane = generate_plane(normal, origin)
        # Perform projection in place on the copied mesh
        f = lambda p: plane.ProjectPoint(p, p)
        np.apply_along_axis(f, 1, mesh.points)
        return mesh

    def ribbon(poly_data, width=None, scalars=None, angle=0.0, factor=2.0,
               normal=None, tcoords=False, preference='points'):
        """Create a ribbon of the lines in this dataset.

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
            minimum width. The default is 2.0

        normal : tuple(float), optional
            Normal to use as default.

        tcoords : bool, str, optional
            If ``True``, generate texture coordinates along the
            ribbon. This can also be specified to generate the texture
            coordinates with either ``'length'`` or ``'normalized'``.

        Examples
        --------
        Convert a line to a ribbon and plot it.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> path = sphere.geodesic(0, 100)
        >>> ribbon = path.ribbon()
        >>> cpos = pv.plot([sphere, ribbon])

        Notes
        -----
        If there are no lines in the input dataset, then the output
        will be an empty ``pyvista.PolyData`` mesh.

        """
        if scalars is not None:
            arr, field = get_array(poly_data, scalars, preference=preference, info=True)
        if width is None:
            width = poly_data.length * 0.1
        alg = _vtk.vtkRibbonFilter()
        alg.SetInputDataObject(poly_data)
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
        alg.Update()
        return _get_output(alg)

    def extrude(poly_data, vector, inplace=False, progress_bar=False):
        """Sweep polygonal data creating a "skirt" from free edges.

        This will create a line from vertices.

        This takes polygonal data as input and generates polygonal
        data on output. The input dataset is swept according to some
        extrusion function and creates new polygonal primitives. These
        primitives form a "skirt" or swept surface. For example,
        sweeping a line results in a quadrilateral, and sweeping a
        triangle creates a "wedge".

        There are a number of control parameters for this filter. You
        can control whether the sweep of a 2D object (i.e., polygon or
        triangle strip) is capped with the generating geometry via the
        "Capping" parameter.

        The skirt is generated by locating certain topological
        features. Free edges (edges of polygons or triangle strips
        only used by one polygon or triangle strips) generate
        surfaces. This is true also of lines or polylines. Vertices
        generate lines.

        This filter can be used to create 3D fonts, 3D irregular bar
        charts, or to model 2 1/2D objects like punched plates. It
        also can be used to create solid objects from 2D polygonal
        meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            Mesh to extrude.

        vector : np.ndarray or list
            Direction and length to extrude the mesh in.

        inplace : bool, optional
            Overwrites the original mesh in-place.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Examples
        --------
        Extrude a half arc circle

        >>> import pyvista
        >>> arc = pyvista.CircularArc([-1, 0, 0], [1, 0, 0], [0, 0, 0])
        >>> mesh = arc.extrude([0, 0, 1])
        >>> cpos = mesh.plot()

        """
        alg = _vtk.vtkLinearExtrusionFilter()
        alg.SetExtrusionTypeToVectorExtrusion()
        alg.SetVector(*vector)
        alg.SetInputData(poly_data)
        _update_alg(alg, progress_bar, 'Extruding')
        output = pyvista.wrap(alg.GetOutput())
        if inplace:
            poly_data.overwrite(output)
            return poly_data
        else:
            return output


    def extrude_rotate(poly_data, resolution=30, inplace=False,
                       translation=0.0, dradius=0.0, angle=360.0, progress_bar=False):
        """Sweep polygonal data creating "skirt" from free edges and lines, and lines from vertices.

        This is a modeling filter.

        This takes polygonal data as input and generates polygonal
        data on output. The input dataset is swept around the z-axis
        to create new polygonal primitives. These primitives form a
        "skirt" or swept surface. For example, sweeping a line
        results in a cylindrical shell, and sweeping a circle
        creates a torus.

        There are a number of control parameters for this filter.
        You can control whether the sweep of a 2D object (i.e.,
        polygon or triangle strip) is capped with the generating
        geometry via the "Capping" instance variable. Also, you can
        control the angle of rotation, and whether translation along
        the z-axis is performed along with the rotation.
        (Translation is useful for creating "springs".) You also can
        adjust the radius of the generating geometry using the
        "DeltaRotation" instance variable.

        The skirt is generated by locating certain topological
        features. Free edges (edges of polygons or triangle strips
        only used by one polygon or triangle strips) generate
        surfaces. This is true also of lines or polylines. Vertices
        generate lines.

        This filter can be used to model axisymmetric objects like
        cylinders, bottles, and wine glasses; or translational/
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
            The angle of rotation.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Examples
        --------
        >>> import pyvista
        >>> line = pyvista.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))
        >>> mesh = line.extrude_rotate(resolution = 4)
        >>> cpos = mesh.plot()

        """
        if resolution <= 0:
            raise ValueError('`resolution` should be positive')
        alg = _vtk.vtkRotationalExtrusionFilter()
        alg.SetInputData(poly_data)
        alg.SetResolution(resolution)
        alg.SetTranslation(translation)
        alg.SetDeltaRadius(dradius)
        alg.SetAngle(angle)
        _update_alg(alg, progress_bar, 'Extruding')
        output = pyvista.wrap(alg.GetOutput())
        if inplace:
            poly_data.overwrite(output)
            return poly_data
        else:
            return output

    def strip(poly_data, join=False, max_length=1000, pass_cell_data=False,
              pass_cell_ids=False, pass_point_ids=False):
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

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> slc = mesh.slice(normal='z', origin=(0,0,-10))
        >>> stripped = slc.strip()
        >>> stripped.n_cells
        1

        """
        alg = _vtk.vtkStripper()
        alg.SetInputDataObject(poly_data)
        alg.SetJoinContiguousSegments(join)
        alg.SetMaximumLength(max_length)
        alg.SetPassCellDataAsFieldData(pass_cell_data)
        alg.SetPassThroughCellIds(pass_cell_ids)
        alg.SetPassThroughPointIds(pass_point_ids)
        alg.Update()
        return _get_output(alg)
