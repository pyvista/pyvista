"""Filters module with a class of common filters that can be applied to any vtkDataSet."""

from __future__ import annotations

from collections.abc import Iterable
import contextlib
import functools
from typing import TYPE_CHECKING
from typing import Literal
from typing import Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np

import pyvista
import pyvista.core._vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError
from pyvista.core.errors import MissingDataError
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.utilities import transformations
from pyvista.core.utilities.arrays import FieldAssociation
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import get_array_association
from pyvista.core.utilities.arrays import set_default_active_scalars
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from pyvista.core.utilities.cells import numpy_to_idarr
from pyvista.core.utilities.geometric_objects import NORMALS
from pyvista.core.utilities.helpers import generate_plane
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.misc import assert_empty_kwargs

if TYPE_CHECKING:  # pragma: no cover
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike


@abstract_class
class DataSetFilters:
    """A set of common filters that can be applied to any vtkDataSet."""

    def _clip_with_function(
        self,
        function,
        invert=True,
        value=0.0,
        return_clipped=False,
        progress_bar=False,
        crinkle=False,
    ):
        """Clip using an implicit function (internal helper)."""
        if crinkle:
            # Add Cell IDs
            self.cell_data['cell_ids'] = np.arange(self.n_cells)

        if isinstance(self, _vtk.vtkPolyData):
            alg = _vtk.vtkClipPolyData()
        # elif isinstance(self, vtk.vtkImageData):
        #     alg = vtk.vtkClipVolume()
        #     alg.SetMixed3DCellGeneration(True)
        else:
            alg = _vtk.vtkTableBasedClipDataSet()
        alg.SetInputDataObject(self)  # Use the grid as the data we desire to cut
        alg.SetValue(value)
        alg.SetClipFunction(function)  # the implicit function
        alg.SetInsideOut(invert)  # invert the clip if needed
        alg.SetGenerateClippedOutput(return_clipped)
        _update_alg(alg, progress_bar, 'Clipping with Function')

        if return_clipped:
            a = _get_output(alg, oport=0)
            b = _get_output(alg, oport=1)
            if crinkle:
                set_a = set(a.cell_data['cell_ids'])
                set_b = set(b.cell_data['cell_ids']) - set_a
                a = self.extract_cells(list(set_a))
                b = self.extract_cells(list(set_b))
            return a, b
        clipped = _get_output(alg)
        if crinkle:
            clipped = self.extract_cells(np.unique(clipped.cell_data['cell_ids']))
        return clipped

    def align(
        self,
        target,
        max_landmarks=100,
        max_mean_distance=1e-5,
        max_iterations=500,
        check_mean_distance=True,
        start_by_matching_centroids=True,
        return_matrix=False,
    ):
        """Align a dataset to another.

        Uses the iterative closest point algorithm to align the points of the
        two meshes.  See the VTK class `vtkIterativeClosestPointTransform
        <https://vtk.org/doc/nightly/html/classvtkIterativeClosestPointTransform.html>`_

        Parameters
        ----------
        target : pyvista.DataSet
            The target dataset to align to.

        max_landmarks : int, default: 100
            The maximum number of landmarks.

        max_mean_distance : float, default: 1e-5
            The maximum mean distance for convergence.

        max_iterations : int, default: 500
            The maximum number of iterations.

        check_mean_distance : bool, default: True
            Whether to check the mean distance for convergence.

        start_by_matching_centroids : bool, default: True
            Whether to start the alignment by matching centroids. Default is True.

        return_matrix : bool, default: False
            Return the transform matrix as well as the aligned mesh.

        Returns
        -------
        aligned : pyvista.DataSet
            The dataset aligned to the target mesh.

        matrix : numpy.ndarray
            Transform matrix to transform the input dataset to the target dataset.

        Examples
        --------
        Create a cylinder, translate it, and use iterative closest point to
        align mesh to its original position.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> source = pv.Cylinder(resolution=30).triangulate().subdivide(1)
        >>> transformed = source.rotate_y(20).translate([-0.75, -0.5, 0.5])
        >>> aligned = transformed.align(source)
        >>> _, closest_points = aligned.find_closest_cell(
        ...     source.points, return_closest_point=True
        ... )
        >>> dist = np.linalg.norm(source.points - closest_points, axis=1)

        Visualize the source, transformed, and aligned meshes.

        >>> pl = pv.Plotter(shape=(1, 2))
        >>> _ = pl.add_text('Before Alignment')
        >>> _ = pl.add_mesh(
        ...     source, style='wireframe', opacity=0.5, line_width=2
        ... )
        >>> _ = pl.add_mesh(transformed)
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_text('After Alignment')
        >>> _ = pl.add_mesh(
        ...     source, style='wireframe', opacity=0.5, line_width=2
        ... )
        >>> _ = pl.add_mesh(
        ...     aligned,
        ...     scalars=dist,
        ...     scalar_bar_args={
        ...         'title': 'Distance to Source',
        ...         'fmt': '%.1E',
        ...     },
        ... )
        >>> pl.show()

        Show that the mean distance between the source and the target is
        nearly zero.

        >>> np.abs(dist).mean()  # doctest:+SKIP
        9.997635192915073e-05

        """
        icp = _vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(self)
        icp.SetTarget(target)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfLandmarks(max_landmarks)
        icp.SetMaximumMeanDistance(max_mean_distance)
        icp.SetMaximumNumberOfIterations(max_iterations)
        icp.SetCheckMeanDistance(check_mean_distance)
        icp.SetStartByMatchingCentroids(start_by_matching_centroids)
        icp.Update()
        matrix = pyvista.array_from_vtkmatrix(icp.GetMatrix())
        if return_matrix:
            return self.transform(matrix, inplace=False), matrix
        return self.transform(matrix, inplace=False)

    def clip(
        self,
        normal='x',
        origin=None,
        invert=True,
        value=0.0,
        inplace=False,
        return_clipped=False,
        progress_bar=False,
        crinkle=False,
    ):
        """Clip a dataset by a plane by specifying the origin and normal.

        If no parameters are given the clip will occur in the center
        of that dataset.

        Parameters
        ----------
        normal : tuple(float) or str, default: 'x'
            Length 3 tuple for the normal vector direction. Can also
            be specified as a string conventional direction such as
            ``'x'`` for ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.

        origin : sequence[float], optional
            The center ``(x, y, z)`` coordinate of the plane on which the clip
            occurs. The default is the center of the dataset.

        invert : bool, default: True
            Flag on whether to flip/invert the clip.

        value : float, default: 0.0
            Set the clipping value along the normal direction.

        inplace : bool, default: False
            Updates mesh in-place.

        return_clipped : bool, default: False
            Return both unclipped and clipped parts of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        crinkle : bool, default: False
            Crinkle the clip by extracting the entire cells along the
            clip. This adds the ``"cell_ids"`` array to the ``cell_data``
            attribute that tracks the original cell IDs of the original
            dataset.

        Returns
        -------
        pyvista.PolyData or tuple[pyvista.PolyData]
            Clipped mesh when ``return_clipped=False``,
            otherwise a tuple containing the unclipped and clipped datasets.

        Examples
        --------
        Clip a cube along the +X direction.  ``triangulate`` is used as
        the cube is initially composed of quadrilateral faces and
        subdivide only works on triangles.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip()
        >>> clipped_cube.plot()

        Clip a cube in the +Z direction.  This leaves half a cube
        below the XY plane.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip('z')
        >>> clipped_cube.plot()

        See :ref:`clip_with_surface_example` for more examples using this filter.

        """
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = self.center
        # create the plane for clipping
        function = generate_plane(normal, origin)
        # run the clip
        result = DataSetFilters._clip_with_function(
            self,
            function,
            invert=invert,
            value=value,
            return_clipped=return_clipped,
            progress_bar=progress_bar,
            crinkle=crinkle,
        )
        if inplace:
            if return_clipped:
                self.copy_from(result[0], deep=False)
                return self, result[1]
            else:
                self.copy_from(result, deep=False)
                return self
        return result

    def clip_box(
        self,
        bounds=None,
        invert=True,
        factor=0.35,
        progress_bar=False,
        merge_points=True,
        crinkle=False,
    ):
        """Clip a dataset by a bounding box defined by the bounds.

        If no bounds are given, a corner of the dataset bounds will be removed.

        Parameters
        ----------
        bounds : sequence[float], optional
            Length 6 sequence of floats: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
            Length 3 sequence of floats: distances from the min coordinate of
            of the input mesh. Single float value: uniform distance from the
            min coordinate. Length 12 sequence of length 3 sequence of floats:
            a plane collection (normal, center, ...).
            :class:`pyvista.PolyData`: if a poly mesh is passed that represents
            a box with 6 faces that all form a standard box, then planes will
            be extracted from the box to define the clipping region.

        invert : bool, default: True
            Flag on whether to flip/invert the clip.

        factor : float, default: 0.35
            If bounds are not given this is the factor along each axis to
            extract the default box.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        merge_points : bool, default: True
            If ``True``, coinciding points of independently defined mesh
            elements will be merged.

        crinkle : bool, default: False
            Crinkle the clip by extracting the entire cells along the
            clip. This adds the ``"cell_ids"`` array to the ``cell_data``
            attribute that tracks the original cell IDs of the original
            dataset.

        Returns
        -------
        pyvista.UnstructuredGrid
            Clipped dataset.

        Examples
        --------
        Clip a corner of a cube.  The bounds of a cube are normally
        ``[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]``, and this removes 1/8 of
        the cube's surface.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip_box([0, 1, 0, 1, 0, 1])
        >>> clipped_cube.plot()

        See :ref:`clip_with_plane_box_example` for more examples using this filter.

        """
        if bounds is None:

            def _get_quarter(dmin, dmax):
                """Get a section of the given range (internal helper)."""
                return dmax - ((dmax - dmin) * factor)

            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            xmin = _get_quarter(xmin, xmax)
            ymin = _get_quarter(ymin, ymax)
            zmin = _get_quarter(zmin, zmax)
            bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
        if isinstance(bounds, (float, int)):
            bounds = [bounds, bounds, bounds]
        elif isinstance(bounds, pyvista.PolyData):
            poly = bounds
            if poly.n_cells != 6:
                raise ValueError("The bounds mesh must have only 6 faces.")
            bounds = []
            poly.compute_normals(inplace=True)
            for cid in range(6):
                cell = poly.extract_cells(cid)
                normal = cell["Normals"][0]
                bounds.append(normal)
                bounds.append(cell.center)
        if not isinstance(bounds, (np.ndarray, Sequence)):
            raise TypeError('Bounds must be a sequence of floats with length 3, 6 or 12.')
        if len(bounds) not in [3, 6, 12]:
            raise ValueError('Bounds must be a sequence of floats with length 3, 6 or 12.')
        if len(bounds) == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            bounds = (xmin, xmin + bounds[0], ymin, ymin + bounds[1], zmin, zmin + bounds[2])
        if crinkle:
            self.cell_data['cell_ids'] = np.arange(self.n_cells)
        alg = _vtk.vtkBoxClipDataSet()
        if not merge_points:
            # vtkBoxClipDataSet uses vtkMergePoints by default
            alg.SetLocator(_vtk.vtkNonMergingPointLocator())
        alg.SetInputDataObject(self)
        alg.SetBoxClip(*bounds)
        port = 0
        if invert:
            # invert the clip if needed
            port = 1
            alg.GenerateClippedOutputOn()
        _update_alg(alg, progress_bar, 'Clipping a Dataset by a Bounding Box')
        clipped = _get_output(alg, oport=port)
        if crinkle:
            clipped = self.extract_cells(np.unique(clipped.cell_data['cell_ids']))
        return clipped

    def compute_implicit_distance(self, surface, inplace=False):
        """Compute the implicit distance from the points to a surface.

        This filter will compute the implicit distance from all of the
        nodes of this mesh to a given surface. This distance will be
        added as a point array called ``'implicit_distance'``.

        Nodes of this mesh which are interior to the input surface
        geometry have a negative distance, and nodes on the exterior
        have a positive distance. Nodes which intersect the input
        surface has a distance of zero.

        Parameters
        ----------
        surface : pyvista.DataSet
            The surface used to compute the distance.

        inplace : bool, default: False
            If ``True``, a new scalar array will be added to the
            ``point_data`` of this mesh and the modified mesh will
            be returned. Otherwise a copy of this mesh is returned
            with that scalar field added.

        Returns
        -------
        pyvista.DataSet
            Dataset containing the ``'implicit_distance'`` array in
            ``point_data``.

        Examples
        --------
        Compute the distance between all the points on a sphere and a
        plane.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(radius=0.35)
        >>> plane = pv.Plane()
        >>> _ = sphere.compute_implicit_distance(plane, inplace=True)
        >>> dist = sphere['implicit_distance']
        >>> type(dist)
        <class 'pyvista.core.pyvista_ndarray.pyvista_ndarray'>

        Plot these distances as a heatmap. Note how distances above the
        plane are positive, and distances below the plane are negative.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(
        ...     sphere, scalars='implicit_distance', cmap='bwr'
        ... )
        >>> _ = pl.add_mesh(plane, color='w', style='wireframe')
        >>> pl.show()

        We can also compute the distance from all the points on the
        plane to the sphere.

        >>> _ = plane.compute_implicit_distance(sphere, inplace=True)

        Again, we can plot these distances as a heatmap. Note how
        distances inside the sphere are negative and distances outside
        the sphere are positive.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(
        ...     plane,
        ...     scalars='implicit_distance',
        ...     cmap='bwr',
        ...     clim=[-0.35, 0.35],
        ... )
        >>> _ = pl.add_mesh(sphere, color='w', style='wireframe')
        >>> pl.show()

        See :ref:`clip_with_surface_example` and
        :ref:`voxelize_surface_mesh_example` for more examples using
        this filter.

        """
        function = _vtk.vtkImplicitPolyDataDistance()
        function.SetInput(surface)
        points = pyvista.convert_array(self.points)
        dists = _vtk.vtkDoubleArray()
        function.FunctionValue(points, dists)
        if inplace:
            self.point_data['implicit_distance'] = pyvista.convert_array(dists)
            return self
        result = self.copy()
        result.point_data['implicit_distance'] = pyvista.convert_array(dists)
        return result

    def clip_scalar(
        self,
        scalars=None,
        invert=True,
        value=0.0,
        inplace=False,
        progress_bar=False,
        both=False,
    ):
        """Clip a dataset by a scalar.

        Parameters
        ----------
        scalars : str, optional
            Name of scalars to clip on.  Defaults to currently active scalars.

        invert : bool, default: True
            Flag on whether to flip/invert the clip.  When ``True``,
            only the mesh below ``value`` will be kept.  When
            ``False``, only values above ``value`` will be kept.

        value : float, default: 0.0
            Set the clipping value.

        inplace : bool, default: False
            Update mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        both : bool, default: False
            If ``True``, also returns the complementary clipped mesh.

        Returns
        -------
        pyvista.PolyData or tuple
            Clipped dataset if ``both=False``.  If ``both=True`` then
            returns a tuple of both clipped datasets.

        Examples
        --------
        Remove the part of the mesh with "sample_point_scalars" above 100.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.load_hexbeam()
        >>> clipped = dataset.clip_scalar(
        ...     scalars="sample_point_scalars", value=100
        ... )
        >>> clipped.plot()

        Get clipped meshes corresponding to the portions of the mesh above and below 100.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.load_hexbeam()
        >>> _below, _above = dataset.clip_scalar(
        ...     scalars="sample_point_scalars", value=100, both=True
        ... )

        Remove the part of the mesh with "sample_point_scalars" below 100.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.load_hexbeam()
        >>> clipped = dataset.clip_scalar(
        ...     scalars="sample_point_scalars", value=100, invert=False
        ... )
        >>> clipped.plot()

        """
        if isinstance(self, _vtk.vtkPolyData):
            alg = _vtk.vtkClipPolyData()
        else:
            alg = _vtk.vtkTableBasedClipDataSet()

        alg.SetInputDataObject(self)
        alg.SetValue(value)
        if scalars is None:
            set_default_active_scalars(self)
        else:
            self.set_active_scalars(scalars)

        alg.SetInsideOut(invert)  # invert the clip if needed
        alg.SetGenerateClippedOutput(both)

        _update_alg(alg, progress_bar, 'Clipping by a Scalar')
        result0 = _get_output(alg)

        if inplace:
            self.copy_from(result0, deep=False)
            result0 = self

        if both:
            result1 = _get_output(alg, oport=1)
            if isinstance(self, _vtk.vtkPolyData):
                # For some reason vtkClipPolyData with SetGenerateClippedOutput on leaves unreferenced vertices
                result0, result1 = (r.clean() for r in (result0, result1))
            return result0, result1
        return result0

    def clip_surface(
        self,
        surface,
        invert=True,
        value=0.0,
        compute_distance=False,
        progress_bar=False,
        crinkle=False,
    ):
        """Clip any mesh type using a :class:`pyvista.PolyData` surface mesh.

        This will return a :class:`pyvista.UnstructuredGrid` of the clipped
        mesh. Geometry of the input dataset will be preserved where possible.
        Geometries near the clip intersection will be triangulated/tessellated.

        Parameters
        ----------
        surface : pyvista.PolyData
            The ``PolyData`` surface mesh to use as a clipping
            function.  If this input mesh is not a :class`pyvista.PolyData`,
            the external surface will be extracted.

        invert : bool, default: True
            Flag on whether to flip/invert the clip.

        value : float, default: 0.0
            Set the clipping value of the implicit function (if
            clipping with implicit function) or scalar value (if
            clipping with scalars).

        compute_distance : bool, default: False
            Compute the implicit distance from the mesh onto the input
            dataset.  A new array called ``'implicit_distance'`` will
            be added to the output clipped mesh.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        crinkle : bool, default: False
            Crinkle the clip by extracting the entire cells along the
            clip. This adds the ``"cell_ids"`` array to the ``cell_data``
            attribute that tracks the original cell IDs of the original
            dataset.

        Returns
        -------
        pyvista.PolyData
            Clipped surface.

        Examples
        --------
        Clip a cube with a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(center=(-0.4, -0.4, -0.4))
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped = cube.clip_surface(sphere)
        >>> clipped.plot(show_edges=True, cpos='xy', line_width=3)

        See :ref:`clip_with_surface_example` for more examples using
        this filter.

        """
        if not isinstance(surface, _vtk.vtkPolyData):
            surface = DataSetFilters.extract_geometry(surface)
        function = _vtk.vtkImplicitPolyDataDistance()
        function.SetInput(surface)
        if compute_distance:
            points = pyvista.convert_array(self.points)
            dists = _vtk.vtkDoubleArray()
            function.FunctionValue(points, dists)
            self['implicit_distance'] = pyvista.convert_array(dists)
        # run the clip
        return DataSetFilters._clip_with_function(
            self,
            function,
            invert=invert,
            value=value,
            progress_bar=progress_bar,
            crinkle=crinkle,
        )

    def slice_implicit(
        self,
        implicit_function,
        generate_triangles=False,
        contour=False,
        progress_bar=False,
    ):
        """Slice a dataset by a VTK implicit function.

        Parameters
        ----------
        implicit_function : vtk.vtkImplicitFunction
            Specify the implicit function to perform the cutting.

        generate_triangles : bool, default: False
            If this is enabled (``False`` by default), the output will
            be triangles. Otherwise the output will be the intersection
            polygons. If the cutting function is not a plane, the
            output will be 3D polygons, which might be nice to look at
            but hard to compute with downstream.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        Examples
        --------
        Slice the surface of a sphere.

        >>> import pyvista as pv
        >>> import vtk
        >>> sphere = vtk.vtkSphere()
        >>> sphere.SetRadius(10)
        >>> mesh = pv.Wavelet()
        >>> slice = mesh.slice_implicit(sphere)
        >>> slice.plot(show_edges=True, line_width=5)

        >>> cylinder = vtk.vtkCylinder()
        >>> cylinder.SetRadius(10)
        >>> mesh = pv.Wavelet()
        >>> slice = mesh.slice_implicit(cylinder)
        >>> slice.plot(show_edges=True, line_width=5)

        """
        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(self)  # Use the grid as the data we desire to cut
        alg.SetCutFunction(implicit_function)  # the cutter to use the function
        alg.SetGenerateTriangles(generate_triangles)
        _update_alg(alg, progress_bar, 'Slicing')
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output

    def slice(
        self,
        normal='x',
        origin=None,
        generate_triangles=False,
        contour=False,
        progress_bar=False,
    ):
        """Slice a dataset by a plane at the specified origin and normal vector orientation.

        If no origin is specified, the center of the input dataset will be used.

        Parameters
        ----------
        normal : sequence[float] | str, default: 'x'
            Length 3 tuple for the normal vector direction. Can also be
            specified as a string conventional direction such as ``'x'`` for
            ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.

        origin : sequence[float], optional
            The center ``(x, y, z)`` coordinate of the plane on which
            the slice occurs.

        generate_triangles : bool, default: False
            If this is enabled (``False`` by default), the output will
            be triangles. Otherwise the output will be the intersection
            polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        Examples
        --------
        Slice the surface of a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> slice_x = sphere.slice(normal='x')
        >>> slice_y = sphere.slice(normal='y')
        >>> slice_z = sphere.slice(normal='z')
        >>> slices = slice_x + slice_y + slice_z
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = self.center
        # create the plane for clipping
        plane = generate_plane(normal, origin)
        return DataSetFilters.slice_implicit(
            self,
            plane,
            generate_triangles=generate_triangles,
            contour=contour,
            progress_bar=progress_bar,
        )

    def slice_orthogonal(
        self,
        x=None,
        y=None,
        z=None,
        generate_triangles=False,
        contour=False,
        progress_bar=False,
    ):
        """Create three orthogonal slices through the dataset on the three cartesian planes.

        Yields a MutliBlock dataset of the three slices.

        Parameters
        ----------
        x : float, optional
            The X location of the YZ slice.

        y : float, optional
            The Y location of the XZ slice.

        z : float, optional
            The Z location of the XY slice.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        Examples
        --------
        Slice the random hills dataset with three orthogonal planes.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_orthogonal(contour=False)
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        # Create the three slices
        if x is None:
            x = self.center[0]
        if y is None:
            y = self.center[1]
        if z is None:
            z = self.center[2]
        output = pyvista.MultiBlock()
        if isinstance(self, pyvista.MultiBlock):
            for i in range(self.n_blocks):
                output.append(
                    self[i].slice_orthogonal(
                        x=x,
                        y=y,
                        z=z,
                        generate_triangles=generate_triangles,
                        contour=contour,
                    ),
                )
            return output
        output.append(
            self.slice(
                normal='x',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'YZ',
        )
        output.append(
            self.slice(
                normal='y',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'XZ',
        )
        output.append(
            self.slice(
                normal='z',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'XY',
        )
        return output

    def slice_along_axis(
        self,
        n=5,
        axis='x',
        tolerance=None,
        generate_triangles=False,
        contour=False,
        bounds=None,
        center=None,
        progress_bar=False,
    ):
        """Create many slices of the input dataset along a specified axis.

        Parameters
        ----------
        n : int, default: 5
            The number of slices to create.

        axis : str | int, default: 'x'
            The axis to generate the slices along. Perpendicular to the
            slices. Can be string name (``'x'``, ``'y'``, or ``'z'``) or
            axis index (``0``, ``1``, or ``2``).

        tolerance : float, optional
            The tolerance to the edge of the dataset bounds to create
            the slices. The ``n`` slices are placed equidistantly with
            an absolute padding of ``tolerance`` inside each side of the
            ``bounds`` along the specified axis. Defaults to 1% of the
            ``bounds`` along the specified axis.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        bounds : sequence[float], optional
            A 6-length sequence overriding the bounds of the mesh.
            The bounds along the specified axis define the extent
            where slices are taken.

        center : sequence[float], optional
            A 3-length sequence specifying the position of the line
            along which slices are taken. Defaults to the center of
            the mesh.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        Examples
        --------
        Slice the random hills dataset in the X direction.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_along_axis(n=10)
        >>> slices.plot(line_width=5)

        Slice the random hills dataset in the Z direction.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_along_axis(n=10, axis='z')
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        # parse axis input
        labels = ['x', 'y', 'z']
        label_to_index = {label: index for index, label in enumerate(labels)}
        if isinstance(axis, int):
            ax_index = axis
            ax_label = labels[ax_index]
        elif isinstance(axis, str):
            try:
                ax_index = label_to_index[axis.lower()]
            except KeyError:
                raise ValueError(
                    f'Axis ({axis!r}) not understood. Choose one of {labels}.',
                ) from None
            ax_label = axis
        # get the locations along that axis
        if bounds is None:
            bounds = self.bounds
        if center is None:
            center = self.center
        if tolerance is None:
            tolerance = (bounds[ax_index * 2 + 1] - bounds[ax_index * 2]) * 0.01
        rng = np.linspace(bounds[ax_index * 2] + tolerance, bounds[ax_index * 2 + 1] - tolerance, n)
        center = list(center)
        # Make each of the slices
        output = pyvista.MultiBlock()
        if isinstance(self, pyvista.MultiBlock):
            for i in range(self.n_blocks):
                output.append(
                    self[i].slice_along_axis(
                        n=n,
                        axis=ax_label,
                        tolerance=tolerance,
                        generate_triangles=generate_triangles,
                        contour=contour,
                        bounds=bounds,
                        center=center,
                    ),
                )
            return output
        for i in range(n):
            center[ax_index] = rng[i]
            slc = DataSetFilters.slice(
                self,
                normal=ax_label,
                origin=center,
                generate_triangles=generate_triangles,
                contour=contour,
                progress_bar=progress_bar,
            )
            output.append(slc, f'slice{i}')
        return output

    def slice_along_line(self, line, generate_triangles=False, contour=False, progress_bar=False):
        """Slice a dataset using a polyline/spline as the path.

        This also works for lines generated with :func:`pyvista.Line`.

        Parameters
        ----------
        line : pyvista.PolyData
            A PolyData object containing one single PolyLine cell.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        Examples
        --------
        Slice the random hills dataset along a circular arc.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> center = np.array(hills.center)
        >>> point_a = center + np.array([5, 0, 0])
        >>> point_b = center + np.array([-5, 0, 0])
        >>> arc = pv.CircularArc(point_a, point_b, center, resolution=100)
        >>> line_slice = hills.slice_along_line(arc)

        Plot the circular arc and the hills mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(hills, smooth_shading=True, style='wireframe')
        >>> _ = pl.add_mesh(
        ...     line_slice,
        ...     line_width=10,
        ...     render_lines_as_tubes=True,
        ...     color='k',
        ... )
        >>> _ = pl.add_mesh(arc, line_width=10, color='grey')
        >>> pl.show()

        See :ref:`slice_example` for more examples using this filter.

        """
        # check that we have a PolyLine cell in the input line
        if line.GetNumberOfCells() != 1:
            raise ValueError('Input line must have only one cell.')
        polyline = line.GetCell(0)
        if not isinstance(polyline, _vtk.vtkPolyLine):
            raise TypeError(f'Input line must have a PolyLine cell, not ({type(polyline)})')
        # Generate PolyPlane
        polyplane = _vtk.vtkPolyPlane()
        polyplane.SetPolyLine(polyline)
        # Create slice
        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(self)  # Use the grid as the data we desire to cut
        alg.SetCutFunction(polyplane)  # the cutter to use the poly planes
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        _update_alg(alg, progress_bar, 'Slicing along Line')
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output

    def threshold(
        self,
        value=None,
        scalars=None,
        invert=False,
        continuous=False,
        preference='cell',
        all_scalars=False,
        component_mode='all',
        component=0,
        method='upper',
        progress_bar=False,
    ):
        """Apply a ``vtkThreshold`` filter to the input dataset.

        This filter will apply a ``vtkThreshold`` filter to the input
        dataset and return the resulting object. This extracts cells
        where the scalar value in each cell satisfies the threshold
        criterion.  If ``scalars`` is ``None``, the input's active
        scalars array is used.

        .. warning::
           Thresholding is inherently a cell operation, even though it can use
           associated point data for determining whether to keep a cell. In
           other words, whether or not a given point is included after
           thresholding depends on whether that point is part of a cell that
           is kept after thresholding.

           Please also note the default ``preference`` choice for CELL data
           over POINT data. This is contrary to most other places in PyVista's
           API where the preference typically defaults to POINT data. We chose
           to prefer CELL data here so that if thresholding by a named array
           that exists for both the POINT and CELL data, this filter will
           default to the CELL data array while performing the CELL-wise
           operation.

        Parameters
        ----------
        value : float | sequence[float], optional
            Single value or ``(min, max)`` to be used for the data threshold. If
            a sequence, then length must be 2. If no value is specified, the
            non-NaN data range will be used to remove any NaN values.
            Please reference the ``method`` parameter for how single values
            are handled.

        scalars : str, optional
            Name of scalars to threshold on. Defaults to currently active scalars.

        invert : bool, default: False
            Invert the threshold results. That is, cells that would have been
            in the output with this option off are excluded, while cells that
            would have been excluded from the output are included.

        continuous : bool, default: False
            When True, the continuous interval [minimum cell scalar,
            maximum cell scalar] will be used to intersect the threshold bound,
            rather than the set of discrete scalar values from the vertices.

        preference : str, default: 'cell'
            When ``scalars`` is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``. Throughout PyVista, the preference
            is typically ``'point'`` but since the threshold filter is a
            cell-wise operation, we prefer cell data for thresholding
            operations.

        all_scalars : bool, default: False
            If using scalars from point data, all
            points in a cell must satisfy the threshold when this
            value is ``True``.  When ``False``, any point of the cell
            with a scalar value satisfying the threshold criterion
            will extract the cell. Has no effect when using cell data.

        component_mode : {'selected', 'all', 'any'}
            The method to satisfy the criteria for the threshold of
            multicomponent scalars.  'selected' (default)
            uses only the ``component``.  'all' requires all
            components to meet criteria.  'any' is when
            any component satisfies the criteria.

        component : int, default: 0
            When using ``component_mode='selected'``, this sets
            which component to threshold on.

        method : str, default: 'upper'
            Set the threshold method for single-values, defining which
            threshold bounds to use. If the ``value`` is a range, this
            parameter will be ignored, extracting data between the two
            values. For single values, ``'lower'`` will extract data
            lower than the  ``value``. ``'upper'`` will extract data
            larger than the ``value``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        See Also
        --------
        threshold_percent, :meth:`~pyvista.ImageDataFilters.image_threshold`, extract_values

        Returns
        -------
        pyvista.UnstructuredGrid
            Dataset containing geometry that meets the threshold requirements.

        Examples
        --------
        >>> import pyvista as pv
        >>> import numpy as np
        >>> volume = np.zeros([10, 10, 10])
        >>> volume[:3] = 1
        >>> vol = pv.wrap(volume)
        >>> threshed = vol.threshold(0.1)
        >>> threshed
        UnstructuredGrid (...)
          N Cells:    243
          N Points:   400
          X Bounds:   0.000e+00, 3.000e+00
          Y Bounds:   0.000e+00, 9.000e+00
          Z Bounds:   0.000e+00, 9.000e+00
          N Arrays:   1

        Apply the threshold filter to Perlin noise.  First generate
        the structured grid.

        >>> import pyvista as pv
        >>> noise = pv.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))
        >>> grid = pv.sample_function(
        ...     noise, [0, 1.0, -0, 1.0, 0, 1.0], dim=(20, 20, 20)
        ... )
        >>> grid.plot(
        ...     cmap='gist_earth_r',
        ...     show_scalar_bar=True,
        ...     show_edges=False,
        ... )

        Next, apply the threshold.

        >>> import pyvista as pv
        >>> noise = pv.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))
        >>> grid = pv.sample_function(
        ...     noise, [0, 1.0, -0, 1.0, 0, 1.0], dim=(20, 20, 20)
        ... )
        >>> threshed = grid.threshold(value=0.02)
        >>> threshed.plot(
        ...     cmap='gist_earth_r',
        ...     show_scalar_bar=False,
        ...     show_edges=True,
        ... )

        See :ref:`common_filter_example` for more examples using this filter.

        """
        # set the scalars to threshold on
        if scalars is None:
            set_default_active_scalars(self)
            _, scalars = self.active_scalars_info
        arr = get_array(self, scalars, preference=preference, err=False)
        if arr is None:
            raise ValueError('No arrays present to threshold.')

        field = get_array_association(self, scalars, preference=preference)

        # Run a standard threshold algorithm
        alg = _vtk.vtkThreshold()
        alg.SetAllScalars(all_scalars)
        alg.SetInputDataObject(self)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        # set thresholding parameters
        alg.SetUseContinuousCellRange(continuous)
        # use valid range if no value given
        if value is None:
            value = self.get_data_range(scalars)

        _set_threshold_limit(alg, value, method, invert)

        if component_mode == "component":
            alg.SetComponentModeToUseSelected()
            dim = arr.shape[1]
            if not isinstance(component, (int, np.integer)):
                raise TypeError("component must be int")
            if component > (dim - 1) or component < 0:
                raise ValueError(
                    f"scalars has {dim} components: supplied component {component} not in range",
                )
            alg.SetSelectedComponent(component)
        elif component_mode == "all":
            alg.SetComponentModeToUseAll()
        elif component_mode == "any":
            alg.SetComponentModeToUseAny()
        else:
            raise ValueError(
                f"component_mode must be 'component', 'all', or 'any' got: {component_mode}",
            )

        # Run the threshold
        _update_alg(alg, progress_bar, 'Thresholding')
        return _get_output(alg)

    def threshold_percent(
        self,
        percent=0.50,
        scalars=None,
        invert=False,
        continuous=False,
        preference='cell',
        method='upper',
        progress_bar=False,
    ):
        """Threshold the dataset by a percentage of its range on the active scalars array.

        .. warning::
           Thresholding is inherently a cell operation, even though it can use
           associated point data for determining whether to keep a cell. In
           other words, whether or not a given point is included after
           thresholding depends on whether that point is part of a cell that
           is kept after thresholding.

        Parameters
        ----------
        percent : float | sequence[float], optional
            The percentage in the range ``(0, 1)`` to threshold. If value is
            out of 0 to 1 range, then it will be divided by 100 and checked to
            be in that range.

        scalars : str, optional
            Name of scalars to threshold on. Defaults to currently active scalars.

        invert : bool, default: False
            Invert the threshold results. That is, cells that would have been
            in the output with this option off are excluded, while cells that
            would have been excluded from the output are included.

        continuous : bool, default: False
            When True, the continuous interval [minimum cell scalar,
            maximum cell scalar] will be used to intersect the threshold bound,
            rather than the set of discrete scalar values from the vertices.

        preference : str, default: 'cell'
            When ``scalars`` is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``. Throughout PyVista, the preference
            is typically ``'point'`` but since the threshold filter is a
            cell-wise operation, we prefer cell data for thresholding
            operations.

        method : str, default: 'upper'
            Set the threshold method for single-values, defining which
            threshold bounds to use. If the ``value`` is a range, this
            parameter will be ignored, extracting data between the two
            values. For single values, ``'lower'`` will extract data
            lower than the  ``value``. ``'upper'`` will extract data
            larger than the ``value``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UnstructuredGrid
            Dataset containing geometry that meets the threshold requirements.

        Examples
        --------
        Apply a 50% threshold filter.

        >>> import pyvista as pv
        >>> noise = pv.perlin_noise(0.1, (2, 2, 2), (0, 0, 0))
        >>> grid = pv.sample_function(
        ...     noise, [0, 1.0, -0, 1.0, 0, 1.0], dim=(30, 30, 30)
        ... )
        >>> threshed = grid.threshold_percent(0.5)
        >>> threshed.plot(
        ...     cmap='gist_earth_r',
        ...     show_scalar_bar=False,
        ...     show_edges=True,
        ... )

        Apply a 80% threshold filter.

        >>> threshed = grid.threshold_percent(0.8)
        >>> threshed.plot(
        ...     cmap='gist_earth_r',
        ...     show_scalar_bar=False,
        ...     show_edges=True,
        ... )

        See :ref:`common_filter_example` for more examples using a similar filter.

        """
        if scalars is None:
            set_default_active_scalars(self)
            _, tscalars = self.active_scalars_info
        else:
            tscalars = scalars
        dmin, dmax = self.get_data_range(arr_var=tscalars, preference=preference)

        def _check_percent(percent):
            """Make sure percent is between 0 and 1 or fix if between 0 and 100."""
            if percent >= 1:
                percent = float(percent) / 100.0
                if percent > 1:
                    raise ValueError(f'Percentage ({percent}) is out of range (0, 1).')
            if percent < 1e-10:
                raise ValueError(f'Percentage ({percent}) is too close to zero or negative.')
            return percent

        def _get_val(percent, dmin, dmax):
            """Get the value from a percentage of a range."""
            percent = _check_percent(percent)
            return dmin + float(percent) * (dmax - dmin)

        # Compute the values
        if isinstance(percent, (np.ndarray, Sequence)):
            # Get two values
            value = [_get_val(percent[0], dmin, dmax), _get_val(percent[1], dmin, dmax)]
        elif isinstance(percent, Iterable):
            raise TypeError('Percent must either be a single scalar or a sequence.')
        else:
            # Compute one value to threshold
            value = _get_val(percent, dmin, dmax)
        # Use the normal thresholding function on these values
        return DataSetFilters.threshold(
            self,
            value=value,
            scalars=scalars,
            invert=invert,
            continuous=continuous,
            preference=preference,
            method=method,
            progress_bar=progress_bar,
        )

    def outline(self, generate_faces=False, progress_bar=False):
        """Produce an outline of the full extent for the input dataset.

        Parameters
        ----------
        generate_faces : bool, default: False
            Generate solid faces for the box. This is disabled by default.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing an outline of the original dataset.

        Examples
        --------
        Generate and plot the outline of a sphere.  This is
        effectively the ``(x, y, z)`` bounds of the mesh.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> outline = sphere.outline()
        >>> pv.plot([sphere, outline], line_width=5)

        See :ref:`common_filter_example` for more examples using this filter.

        """
        alg = _vtk.vtkOutlineFilter()
        alg.SetInputDataObject(self)
        alg.SetGenerateFaces(generate_faces)
        _update_alg(alg, progress_bar, 'Producing an outline')
        return wrap(alg.GetOutputDataObject(0))

    def outline_corners(self, factor=0.2, progress_bar=False):
        """Produce an outline of the corners for the input dataset.

        Parameters
        ----------
        factor : float, default: 0.2
            Controls the relative size of the corners to the length of
            the corresponding bounds.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing outlined corners.

        Examples
        --------
        Generate and plot the corners of a sphere.  This is
        effectively the ``(x, y, z)`` bounds of the mesh.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> corners = sphere.outline_corners(factor=0.1)
        >>> pv.plot([sphere, corners], line_width=5)

        """
        alg = _vtk.vtkOutlineCornerFilter()
        alg.SetInputDataObject(self)
        alg.SetCornerFactor(factor)
        _update_alg(alg, progress_bar, 'Producing an Outline of the Corners')
        return wrap(alg.GetOutputDataObject(0))

    def extract_geometry(self, extent: Sequence[float] | None = None, progress_bar=False):
        """Extract the outer surface of a volume or structured grid dataset.

        This will extract all 0D, 1D, and 2D cells producing the
        boundary faces of the dataset.

        .. note::
            This tends to be less efficient than :func:`extract_surface`.

        Parameters
        ----------
        extent : sequence[float], optional
            Specify a ``(xmin, xmax, ymin, ymax, zmin, zmax)`` bounding box to
            clip data.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Surface of the dataset.

        Examples
        --------
        Extract the surface of a sample unstructured grid.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hex_beam = pv.read(examples.hexbeamfile)
        >>> hex_beam.extract_geometry()
        PolyData (...)
          N Cells:    88
          N Points:   90
          N Strips:   0
          X Bounds:   0.000e+00, 1.000e+00
          Y Bounds:   0.000e+00, 1.000e+00
          Z Bounds:   0.000e+00, 5.000e+00
          N Arrays:   3

        See :ref:`surface_smoothing_example` for more examples using this filter.

        """
        alg = _vtk.vtkGeometryFilter()
        alg.SetInputDataObject(self)
        if extent is not None:
            alg.SetExtent(extent)
            alg.SetExtentClipping(True)
        _update_alg(alg, progress_bar, 'Extracting Geometry')
        return _get_output(alg)

    def extract_all_edges(self, use_all_points=False, clear_data=False, progress_bar=False):
        """Extract all the internal/external edges of the dataset as PolyData.

        This produces a full wireframe representation of the input dataset.

        Parameters
        ----------
        use_all_points : bool, default: False
            Indicates whether all of the points of the input mesh should exist
            in the output. When ``True``, point numbering does not change and
            a threaded approach is used, which avoids the use of a point locator
            and is quicker.

            By default this is set to ``False``, and unused points are omitted
            from the output.

            This parameter can only be set to ``True`` with ``vtk==9.1.0`` or newer.

        clear_data : bool, default: False
            Clear any point, cell, or field data. This is useful
            if wanting to strictly extract the edges.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Edges extracted from the dataset.

        Examples
        --------
        Extract the edges of a sample unstructured grid and plot the edges.
        Note how it plots interior edges.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hex_beam = pv.read(examples.hexbeamfile)
        >>> edges = hex_beam.extract_all_edges()
        >>> edges.plot(line_width=5, color='k')

        See :ref:`cell_centers_example` for more examples using this filter.

        """
        alg = _vtk.vtkExtractEdges()
        alg.SetInputDataObject(self)
        if use_all_points:
            try:
                alg.SetUseAllPoints(use_all_points)
            except AttributeError:  # pragma: no cover
                raise VTKVersionError(
                    'This version of VTK does not support `use_all_points=True`. '
                    'VTK v9.1 or newer is required.',
                )
        # Suppress improperly used INFO for debugging messages in vtkExtractEdges
        verbosity = _vtk.vtkLogger.GetCurrentVerbosityCutoff()
        _vtk.vtkLogger.SetStderrVerbosity(_vtk.vtkLogger.VERBOSITY_OFF)
        _update_alg(alg, progress_bar, 'Extracting All Edges')
        # Restore the original vtkLogger verbosity level
        _vtk.vtkLogger.SetStderrVerbosity(verbosity)
        output = _get_output(alg)
        if clear_data:
            output.clear_data()
        return output

    def elevation(
        self,
        low_point=None,
        high_point=None,
        scalar_range=None,
        preference='point',
        set_active=True,
        progress_bar=False,
    ):
        """Generate scalar values on a dataset.

        The scalar values lie within a user specified range, and are
        generated by computing a projection of each dataset point onto
        a line.  The line can be oriented arbitrarily.  A typical
        example is to generate scalars based on elevation or height
        above a plane.

        .. warning::
           This will create a scalars array named ``'Elevation'`` on the
           point data of the input dataset and overwrite the array
           named ``'Elevation'`` if present.

        Parameters
        ----------
        low_point : sequence[float], optional
            The low point of the projection line in 3D space. Default is bottom
            center of the dataset. Otherwise pass a length 3 sequence.

        high_point : sequence[float], optional
            The high point of the projection line in 3D space. Default is top
            center of the dataset. Otherwise pass a length 3 sequence.

        scalar_range : str | sequence[float], optional
            The scalar range to project to the low and high points on the line
            that will be mapped to the dataset. If None given, the values will
            be computed from the elevation (Z component) range between the
            high and low points. Min and max of a range can be given as a length
            2 sequence. If ``str``, name of scalar array present in the
            dataset given, the valid range of that array will be used.

        preference : str, default: "point"
            When an array name is specified for ``scalar_range``, this is the
            preferred array type to search for in the dataset.
            Must be either ``'point'`` or ``'cell'``.

        set_active : bool, default: True
            A boolean flag on whether or not to set the new
            ``'Elevation'`` scalar as the active scalars array on the
            output dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset containing elevation scalars in the
            ``"Elevation"`` array in ``point_data``.

        Examples
        --------
        Generate the "elevation" scalars for a sphere mesh.  This is
        simply the height in Z from the XY plane.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere_elv = sphere.elevation()
        >>> sphere_elv.plot(smooth_shading=True)

        Access the first 4 elevation scalars.  This is a point-wise
        array containing the "elevation" of each point.

        >>> sphere_elv['Elevation'][:4]  # doctest:+SKIP
        array([-0.5       ,  0.5       , -0.49706897, -0.48831028], dtype=float32)

        See :ref:`common_filter_example` for more examples using this filter.

        """
        # Fix the projection line:
        if low_point is None:
            low_point = list(self.center)
            low_point[2] = self.bounds[4]
        if high_point is None:
            high_point = list(self.center)
            high_point[2] = self.bounds[5]
        # Fix scalar_range:
        if scalar_range is None:
            scalar_range = (low_point[2], high_point[2])
        elif isinstance(scalar_range, str):
            scalar_range = self.get_data_range(arr_var=scalar_range, preference=preference)
        elif isinstance(scalar_range, (np.ndarray, Sequence)):
            if len(scalar_range) != 2:
                raise ValueError('scalar_range must have a length of two defining the min and max')
        else:
            raise TypeError(f'scalar_range argument ({scalar_range}) not understood.')
        # Construct the filter
        alg = _vtk.vtkElevationFilter()
        alg.SetInputDataObject(self)
        # Set the parameters
        alg.SetScalarRange(scalar_range)
        alg.SetLowPoint(low_point)
        alg.SetHighPoint(high_point)
        _update_alg(alg, progress_bar, 'Computing Elevation')
        # Decide on updating active scalars array
        output = _get_output(alg)
        if not set_active:
            # 'Elevation' is automatically made active by the VTK filter
            output.point_data.active_scalars_name = self.point_data.active_scalars_name
        return output

    def contour(
        self,
        isosurfaces=10,
        scalars=None,
        compute_normals=False,
        compute_gradients=False,
        compute_scalars=True,
        rng=None,
        preference='point',
        method='contour',
        progress_bar=False,
    ):
        """Contour an input self by an array.

        ``isosurfaces`` can be an integer specifying the number of
        isosurfaces in the data range or a sequence of values for
        explicitly setting the isosurfaces.

        Parameters
        ----------
        isosurfaces : int | sequence[float], optional
            Number of isosurfaces to compute across valid data range or a
            sequence of float values to explicitly use as the isosurfaces.

        scalars : str | array_like[float], optional
            Name or array of scalars to threshold on. If this is an array, the
            output of this filter will save them as ``"Contour Data"``.
            Defaults to currently active scalars.

        compute_normals : bool, default: False
            Compute normals for the dataset.

        compute_gradients : bool, default: False
            Compute gradients for the dataset.

        compute_scalars : bool, default: True
            Preserves the scalar values that are being contoured.

        rng : sequence[float], optional
            If an integer number of isosurfaces is specified, this is
            the range over which to generate contours. Default is the
            scalars array's full data range.

        preference : str, default: "point"
            When ``scalars`` is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        method : str, default:  "contour"
            Specify to choose which vtk filter is used to create the contour.
            Must be one of ``'contour'``, ``'marching_cubes'`` and
            ``'flying_edges'``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Contoured surface.

        Examples
        --------
        Generate contours for the random hills dataset.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> contours = hills.contour()
        >>> contours.plot(line_width=5)

        Generate the surface of a mobius strip using flying edges.

        >>> import pyvista as pv
        >>> a = 0.4
        >>> b = 0.1
        >>> def f(x, y, z):
        ...     xx = x * x
        ...     yy = y * y
        ...     zz = z * z
        ...     xyz = x * y * z
        ...     xx_yy = xx + yy
        ...     a_xx = a * xx
        ...     b_yy = b * yy
        ...     return (
        ...         (xx_yy + 1) * (a_xx + b_yy)
        ...         + zz * (b * xx + a * yy)
        ...         - 2 * (a - b) * xyz
        ...         - a * b * xx_yy
        ...     ) ** 2 - 4 * (xx + yy) * (a_xx + b_yy - xyz * (a - b)) ** 2
        ...
        >>> n = 100
        >>> x_min, y_min, z_min = -1.35, -1.7, -0.65
        >>> grid = pv.ImageData(
        ...     dimensions=(n, n, n),
        ...     spacing=(
        ...         abs(x_min) / n * 2,
        ...         abs(y_min) / n * 2,
        ...         abs(z_min) / n * 2,
        ...     ),
        ...     origin=(x_min, y_min, z_min),
        ... )
        >>> x, y, z = grid.points.T
        >>> values = f(x, y, z)
        >>> out = grid.contour(
        ...     1,
        ...     scalars=values,
        ...     rng=[0, 0],
        ...     method='flying_edges',
        ... )
        >>> out.plot(color='lightblue', smooth_shading=True)

        See :ref:`common_filter_example` or
        :ref:`marching_cubes_example` for more examples using this
        filter.

        """
        if method is None or method == 'contour':
            alg = _vtk.vtkContourFilter()
        elif method == 'marching_cubes':
            alg = _vtk.vtkMarchingCubes()
        elif method == 'flying_edges':
            alg = _vtk.vtkFlyingEdges3D()
        else:
            raise ValueError(f"Method '{method}' is not supported")

        if rng is not None:
            if not isinstance(rng, (np.ndarray, Sequence)):
                raise TypeError(f'Array-like rng expected, got {type(rng).__name__}.')
            rng_shape = np.shape(rng)
            if rng_shape != (2,):
                raise ValueError(f'rng must be a two-length array-like, not {rng}.')
            if rng[0] > rng[1]:
                raise ValueError(f'rng must be a sorted min-max pair, not {rng}.')

        if isinstance(scalars, str):
            scalars_name = scalars
        elif isinstance(scalars, (Sequence, np.ndarray)):
            scalars_name = 'Contour Data'
            self[scalars_name] = scalars
        elif scalars is not None:
            raise TypeError(
                f'Invalid type for `scalars` ({type(scalars)}). Should be either '
                'a numpy.ndarray, a string, or None.',
            )

        # Make sure the input has scalars to contour on
        if self.n_arrays < 1:
            raise ValueError('Input dataset for the contour filter must have scalar.')

        alg.SetInputDataObject(self)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        # set the array to contour on
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars_name = self.active_scalars_info
        else:
            field = get_array_association(self, scalars_name, preference=preference)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        if field != FieldAssociation.POINT:
            raise TypeError('Contour filter only works on point data.')
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars_name,
        )  # args: (idx, port, connection, field, name)
        # set the isosurfaces
        if isinstance(isosurfaces, int):
            # generate values
            if rng is None:
                rng = self.get_data_range(scalars_name)
            alg.GenerateValues(isosurfaces, rng)
        elif isinstance(isosurfaces, (np.ndarray, Sequence)):
            alg.SetNumberOfContours(len(isosurfaces))
            for i, val in enumerate(isosurfaces):
                alg.SetValue(i, val)
        else:
            raise TypeError('isosurfaces not understood.')
        _update_alg(alg, progress_bar, 'Computing Contour')
        output = _get_output(alg)

        # some of these filters fail to correctly name the array
        if scalars_name not in output.point_data:
            if 'Unnamed_0' in output.point_data:
                output.point_data[scalars_name] = output.point_data.pop('Unnamed_0')

        return output

    def texture_map_to_plane(
        self,
        origin=None,
        point_u=None,
        point_v=None,
        inplace=False,
        name='Texture Coordinates',
        use_bounds=False,
        progress_bar=False,
    ):
        """Texture map this dataset to a user defined plane.

        This is often used to define a plane to texture map an image
        to this dataset.  The plane defines the spatial reference and
        extent of that image.

        Parameters
        ----------
        origin : sequence[float], optional
            Length 3 iterable of floats defining the XYZ coordinates of the
            bottom left corner of the plane.

        point_u : sequence[float], optional
            Length 3 iterable of floats defining the XYZ coordinates of the
            bottom right corner of the plane.

        point_v : sequence[float], optional
            Length 3 iterable of floats defining the XYZ coordinates of the
            top left corner of the plane.

        inplace : bool, default: False
            If ``True``, the new texture coordinates will be added to this
            dataset. If ``False``, a new dataset is returned with the texture
            coordinates.

        name : str, default: "Texture Coordinates"
            The string name to give the new texture coordinates if applying
            the filter inplace.

        use_bounds : bool, default: False
            Use the bounds to set the mapping plane by default (bottom plane
            of the bounding box).

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Original dataset with texture coordinates if
            ``inplace=True``, otherwise a copied dataset.

        Examples
        --------
        See :ref:`topo_map_example`

        """
        if use_bounds:
            if isinstance(use_bounds, (int, bool)):
                b = self.GetBounds()
            origin = [b[0], b[2], b[4]]  # BOTTOM LEFT CORNER
            point_u = [b[1], b[2], b[4]]  # BOTTOM RIGHT CORNER
            point_v = [b[0], b[3], b[4]]  # TOP LEFT CORNER
        alg = _vtk.vtkTextureMapToPlane()
        if origin is None or point_u is None or point_v is None:
            alg.SetAutomaticPlaneGeneration(True)
        else:
            alg.SetOrigin(origin)  # BOTTOM LEFT CORNER
            alg.SetPoint1(point_u)  # BOTTOM RIGHT CORNER
            alg.SetPoint2(point_v)  # TOP LEFT CORNER
        alg.SetInputDataObject(self)
        _update_alg(alg, progress_bar, 'Texturing Map to Plane')
        output = _get_output(alg)
        if not inplace:
            return output
        texture_coordinates = output.GetPointData().GetTCoords()
        texture_coordinates.SetName(name)
        otc = self.GetPointData().GetTCoords()
        self.GetPointData().SetTCoords(texture_coordinates)
        self.GetPointData().AddArray(texture_coordinates)
        # CRITICAL:
        if otc and otc.GetName() != name:
            # Add old ones back at the end if different name
            self.GetPointData().AddArray(otc)
        return self

    def texture_map_to_sphere(
        self,
        center=None,
        prevent_seam=True,
        inplace=False,
        name='Texture Coordinates',
        progress_bar=False,
    ):
        """Texture map this dataset to a user defined sphere.

        This is often used to define a sphere to texture map an image
        to this dataset. The sphere defines the spatial reference and
        extent of that image.

        Parameters
        ----------
        center : sequence[float], optional
            Length 3 iterable of floats defining the XYZ coordinates of the
            center of the sphere. If ``None``, this will be automatically
            calculated.

        prevent_seam : bool, default: True
            Control how the texture coordinates are generated.  If
            set, the s-coordinate ranges from 0 to 1 and 1 to 0
            corresponding to the theta angle variation between 0 to
            180 and 180 to 0 degrees.  Otherwise, the s-coordinate
            ranges from 0 to 1 between 0 to 360 degrees.

        inplace : bool, default: False
            If ``True``, the new texture coordinates will be added to
            the dataset inplace. If ``False`` (default), a new dataset
            is returned with the texture coordinates.

        name : str, default: "Texture Coordinates"
            The string name to give the new texture coordinates if applying
            the filter inplace.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset containing the texture mapped to a sphere.  Return
            type matches input.

        Examples
        --------
        See :ref:`texture_example`.

        """
        alg = _vtk.vtkTextureMapToSphere()
        if center is None:
            alg.SetAutomaticSphereGeneration(True)
        else:
            alg.SetAutomaticSphereGeneration(False)
            alg.SetCenter(center)
        alg.SetPreventSeam(prevent_seam)
        alg.SetInputDataObject(self)
        _update_alg(alg, progress_bar, 'Mapping texture to sphere')
        output = _get_output(alg)
        if not inplace:
            return output
        texture_coordinates = output.GetPointData().GetTCoords()
        texture_coordinates.SetName(name)
        otc = self.GetPointData().GetTCoords()
        self.GetPointData().SetTCoords(texture_coordinates)
        self.GetPointData().AddArray(texture_coordinates)
        # CRITICAL:
        if otc and otc.GetName() != name:
            # Add old ones back at the end if different name
            self.GetPointData().AddArray(otc)
        return self

    def compute_cell_sizes(
        self,
        length=True,
        area=True,
        volume=True,
        progress_bar=False,
        vertex_count=False,
    ):
        """Compute sizes for 0D (vertex count), 1D (length), 2D (area) and 3D (volume) cells.

        Parameters
        ----------
        length : bool, default: True
            Specify whether or not to compute the length of 1D cells.

        area : bool, default: True
            Specify whether or not to compute the area of 2D cells.

        volume : bool, default: True
            Specify whether or not to compute the volume of 3D cells.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        vertex_count : bool, default: False
            Specify whether or not to compute sizes for vertex and polyvertex cells (0D cells).
            The computed value is the number of points in the cell.

        Returns
        -------
        pyvista.DataSet
            Dataset with `cell_data` containing the ``"VertexCount"``,
            ``"Length"``, ``"Area"``, and ``"Volume"`` arrays if set
            in the parameters.  Return type matches input.

        Notes
        -----
        If cells do not have a dimension (for example, the length of
        hexahedral cells), the corresponding array will be all zeros.

        Examples
        --------
        Compute the face area of the example airplane mesh.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf.plot(show_edges=True, scalars='Area')

        """
        alg = _vtk.vtkCellSizeFilter()
        alg.SetInputDataObject(self)
        alg.SetComputeArea(area)
        alg.SetComputeVolume(volume)
        alg.SetComputeLength(length)
        alg.SetComputeVertexCount(vertex_count)
        _update_alg(alg, progress_bar, 'Computing Cell Sizes')
        return _get_output(alg)

    def cell_centers(self, vertex=True, progress_bar=False):
        """Generate points at the center of the cells in this dataset.

        These points can be used for placing glyphs or vectors.

        Parameters
        ----------
        vertex : bool, default: True
            Enable or disable the generation of vertex cells.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Polydata where the points are the cell centers of the
            original dataset.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Plane()
        >>> mesh.point_data.clear()
        >>> centers = mesh.cell_centers()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, show_edges=True)
        >>> actor = pl.add_points(
        ...     centers,
        ...     render_points_as_spheres=True,
        ...     color='red',
        ...     point_size=20,
        ... )
        >>> pl.show()

        See :ref:`cell_centers_example` for more examples using this filter.

        """
        alg = _vtk.vtkCellCenters()
        alg.SetInputDataObject(self)
        alg.SetVertexCells(vertex)
        _update_alg(alg, progress_bar, 'Generating Points at the Center of the Cells')
        return _get_output(alg)

    def glyph(
        self,
        orient=True,
        scale=True,
        factor=1.0,
        geom=None,
        indices=None,
        tolerance=None,
        absolute=False,
        clamping=False,
        rng=None,
        color_mode='scale',
        progress_bar=False,
    ):
        """Copy a geometric representation (called a glyph) to the input dataset.

        The glyph may be oriented along the input vectors, and it may
        be scaled according to scalar data or vector
        magnitude. Passing a table of glyphs to choose from based on
        scalars or vector magnitudes is also supported.  The arrays
        used for ``orient`` and ``scale`` must be either both point data
        or both cell data.

        Parameters
        ----------
        orient : bool | str, default: True
            If ``True``, use the active vectors array to orient the glyphs.
            If string, the vector array to use to orient the glyphs.
            If ``False``, the glyphs will not be orientated.

        scale : bool | str | sequence[float], default: True
            If ``True``, use the active scalars to scale the glyphs.
            If string, the scalar array to use to scale the glyphs.
            If ``False``, the glyphs will not be scaled.

        factor : float, default: 1.0
            Scale factor applied to scaling array.

        geom : vtk.vtkDataSet or tuple(vtk.vtkDataSet), optional
            The geometry to use for the glyph. If missing, an arrow glyph
            is used. If a sequence, the datasets inside define a table of
            geometries to choose from based on scalars or vectors. In this
            case a sequence of numbers of the same length must be passed as
            ``indices``. The values of the range (see ``rng``) affect lookup
            in the table.

        indices : sequence[float], optional
            Specifies the index of each glyph in the table for lookup in case
            ``geom`` is a sequence. If given, must be the same length as
            ``geom``. If missing, a default value of ``range(len(geom))`` is
            used. Indices are interpreted in terms of the scalar range
            (see ``rng``). Ignored if ``geom`` has length 1.

        tolerance : float, optional
            Specify tolerance in terms of fraction of bounding box length.
            Float value is between 0 and 1. Default is None. If ``absolute``
            is ``True`` then the tolerance can be an absolute distance.
            If ``None``, points merging as a preprocessing step is disabled.

        absolute : bool, default: False
            Control if ``tolerance`` is an absolute distance or a fraction.

        clamping : bool, default: False
            Turn on/off clamping of "scalar" values to range.

        rng : sequence[float], optional
            Set the range of values to be considered by the filter
            when scalars values are provided.

        color_mode : str, optional, default: ``'scale'``
            If ``'scale'`` , color the glyphs by scale.
            If ``'scalar'`` , color the glyphs by scalar.
            If ``'vector'`` , color the glyphs by vector.

            .. versionadded:: 0.44

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Glyphs at either the cell centers or points.

        Examples
        --------
        Create arrow glyphs oriented by vectors and scaled by scalars.
        Factor parameter is used to reduce the size of the arrows.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.load_random_hills()
        >>> arrows = mesh.glyph(
        ...     scale="Normals", orient="Normals", tolerance=0.05
        ... )
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(arrows, color="black")
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     scalars="Elevation",
        ...     cmap="terrain",
        ...     show_scalar_bar=False,
        ... )
        >>> pl.show()

        See :ref:`glyph_example` and :ref:`glyph_table_example` for more
        examples using this filter.

        """
        dataset = self

        # Make glyphing geometry if necessary
        if geom is None:
            arrow = _vtk.vtkArrowSource()
            _update_alg(arrow, progress_bar, 'Making Arrow')
            geom = arrow.GetOutput()
        # Check if a table of geometries was passed
        if isinstance(geom, (np.ndarray, Sequence)):
            if indices is None:
                # use default "categorical" indices
                indices = np.arange(len(geom))
            if not isinstance(indices, (np.ndarray, Sequence)):
                raise TypeError(
                    'If "geom" is a sequence then "indices" must '
                    'also be a sequence of the same length.',
                )
            if len(indices) != len(geom) and len(geom) != 1:
                raise ValueError('The sequence "indices" must be the same length as "geom".')
        else:
            geom = [geom]
        if any(not isinstance(subgeom, _vtk.vtkPolyData) for subgeom in geom):
            raise TypeError('Only PolyData objects can be used as glyphs.')

        # Run the algorithm
        alg = _vtk.vtkGlyph3D()

        if len(geom) == 1:
            # use a single glyph, ignore indices
            alg.SetSourceData(geom[0])
        else:
            for index, subgeom in zip(indices, geom):
                alg.SetSourceData(index, subgeom)
            if dataset.active_scalars is not None:
                if dataset.active_scalars.ndim > 1:
                    alg.SetIndexModeToVector()
                else:
                    alg.SetIndexModeToScalar()
            else:
                alg.SetIndexModeToOff()

        if isinstance(scale, str):
            dataset.set_active_scalars(scale, preference='cell')
            scale = True
        elif isinstance(scale, bool) and scale:
            try:
                set_default_active_scalars(self)
            except MissingDataError:
                warnings.warn("No data to use for scale. scale will be set to False.")
                scale = False
            except AmbiguousDataError as err:
                warnings.warn(f"{err}\nIt is unclear which one to use. scale will be set to False.")
                scale = False

        if scale:
            if dataset.active_scalars is not None:
                if dataset.active_scalars.ndim > 1:
                    alg.SetScaleModeToScaleByVector()
                else:
                    alg.SetScaleModeToScaleByScalar()
        else:
            alg.SetScaleModeToDataScalingOff()

        if isinstance(orient, str):
            if scale and dataset.active_scalars_info.association == FieldAssociation.CELL:
                prefer = 'cell'
            else:
                prefer = 'point'
            dataset.set_active_vectors(orient, preference=prefer)
            orient = True

        if orient:
            try:
                pyvista.set_default_active_vectors(dataset)
            except MissingDataError:
                warnings.warn("No vector-like data to use for orient. orient will be set to False.")
                orient = False
            except AmbiguousDataError as err:
                warnings.warn(
                    f"{err}\nIt is unclear which one to use. orient will be set to False.",
                )
                orient = False

        if scale and orient:
            if dataset.active_vectors_info.association != dataset.active_scalars_info.association:
                raise ValueError("Both ``scale`` and ``orient`` must use point data or cell data.")

        source_data = dataset
        set_actives_on_source_data = False

        if (scale and dataset.active_scalars_info.association == FieldAssociation.CELL) or (
            orient and dataset.active_vectors_info.association == FieldAssociation.CELL
        ):
            source_data = dataset.cell_centers()
            set_actives_on_source_data = True

        # Clean the points before glyphing
        if tolerance is not None:
            small = pyvista.PolyData(source_data.points)
            small.point_data.update(source_data.point_data)
            source_data = small.clean(
                point_merging=True,
                merge_tol=tolerance,
                lines_to_points=False,
                polys_to_lines=False,
                strips_to_polys=False,
                inplace=False,
                absolute=absolute,
                progress_bar=progress_bar,
            )
            set_actives_on_source_data = True

        # upstream operations (cell to point conversion, point merging) may have unset the correct active
        # scalars/vectors, so set them again
        if set_actives_on_source_data:
            if scale:
                source_data.set_active_scalars(dataset.active_scalars_name, preference='point')
            if orient:
                source_data.set_active_vectors(dataset.active_vectors_name, preference='point')

        if color_mode == 'scale':
            alg.SetColorModeToColorByScale()
        elif color_mode == 'scalar':
            alg.SetColorModeToColorByScalar()
        elif color_mode == 'vector':
            alg.SetColorModeToColorByVector()
        else:
            raise ValueError(f"Invalid color mode '{color_mode}'")

        if rng is not None:
            alg.SetRange(rng)
        alg.SetOrient(orient)
        alg.SetInputData(source_data)
        alg.SetVectorModeToUseVector()
        alg.SetScaleFactor(factor)
        alg.SetClamping(clamping)
        _update_alg(alg, progress_bar, 'Computing Glyphs')

        output = _get_output(alg)

        # Storing geom on the algorithm, for later use in legends.
        output._glyph_geom = geom

        return output

    def connectivity(
        self,
        extraction_mode: Literal[
            'all',
            'largest',
            'specified',
            'cell_seed',
            'point_seed',
            'closest',
        ] = 'all',
        variable_input=None,
        scalar_range=None,
        scalars=None,
        label_regions=True,
        region_ids=None,
        point_ids=None,
        cell_ids=None,
        closest_point=None,
        inplace=False,
        progress_bar=False,
        **kwargs,
    ):
        """Find and label connected regions.

        This filter extracts cell regions based on a specified connectivity
        criterion. The extraction criterion can be controlled with
        ``extraction_mode`` to extract the largest region or the closest
        region to a seed point, for example.

        In general, cells are considered to be connected if they
        share a point. However, if a ``scalar_range`` is provided, cells
        must also have at least one point with scalar values in the
        specified range to be considered connected.

        See :ref:`connectivity_example` and :ref:`volumetric_example` for
        more examples using this filter.

        .. versionadded:: 0.43.0

           * New extraction modes: ``'specified'``, ``'cell_seed'``, ``'point_seed'``,
             and ``'closest'``.
           * Extracted regions are now sorted in descending order by
             cell count.
           * Region connectivity can be controlled using ``scalar_range``.

        .. deprecated:: 0.43.0
           Parameter ``largest`` is deprecated. Use ``'largest'`` or
           ``extraction_mode='largest'`` instead.

        Parameters
        ----------
        extraction_mode : str, default: "all"
            * ``'all'``: Extract all connected regions.
            * ``'largest'`` : Extract the largest connected region (by cell
              count).
            * ``'specified'``: Extract specific region IDs. Use ``region_ids``
              to specify the region IDs to extract.
            * ``'cell_seed'``: Extract all regions sharing the specified cell
              ids. Use ``cell_ids`` to specify the cell ids.
            * ``'point_seed'`` : Extract all regions sharing the specified
              point ids. Use ``point_ids`` to specify the point ids.
            * ``'closest'`` : Extract the region closest to the specified
              point. Use ``closest_point`` to specify the point.

        variable_input : float | sequence[float], optional
            The convenience parameter used for specifying any required input
            values for some values of ``extraction_mode``. Setting
            ``variable_input`` is equivalent to setting:

            * ``'region_ids'`` if mode is ``'specified'``.
            * ``'cell_ids'`` if mode is ``'cell_seed'``.
            * ``'point_ids'`` if mode is ``'point_seed'``.
            * ``'closest_point'`` if mode is ``'closest'``.

            It has no effect if the mode is ``'all'`` or ``'largest'``.

        scalar_range : sequence[float], optional
            Scalar range in the form ``[min, max]``. If set, the connectivity is
            restricted to cells with at least one point with scalar values in
            the specified range.

        scalars : str, optional
            Name of scalars to use if ``scalar_range`` is specified. Defaults
            to currently active scalars.

            .. note::
               This filter requires point scalars to determine region
               connectivity. If cell scalars are provided, they are first
               converted to point scalars with :func:`cell_data_to_point_data`
               before applying the filter. The converted point scalars are
               removed from the output after applying the filter.

        label_regions : bool, default: True
            If ``True``, ``'RegionId'`` point and cell scalar arrays are stored.
            Each region is assigned a unique ID. IDs are zero-indexed and are
            assigned by region cell count in descending order (i.e. the largest
            region has ID ``0``).

        region_ids : sequence[int], optional
            Region ids to extract. Only used if ``extraction_mode`` is
            ``specified``.

        point_ids : sequence[int], optional
            Point ids to use as seeds. Only used if ``extraction_mode`` is
            ``point_seed``.

        cell_ids : sequence[int], optional
            Cell ids to use as seeds. Only used if ``extraction_mode`` is
            ``cell_seed``.

        closest_point : sequence[int], optional
            Point coordinates in ``(x, y, z)``. Only used if
            ``extraction_mode`` is ``closest``.

        inplace : bool, default: False
            If ``True`` the mesh is updated in-place, otherwise a copy
            is returned. A copy is always returned if the input type is
            not ``pyvista.PolyData`` or ``pyvista.UnstructuredGrid``.

        progress_bar : bool, default: False
            Display a progress bar.

        **kwargs : dict, optional
            Used for handling deprecated parameters.

        Returns
        -------
        pyvista.DataSet
            Dataset with labeled connected regions. Return type is
            ``pyvista.PolyData`` if input type is ``pyvista.PolyData`` and
            ``pyvista.UnstructuredGrid`` otherwise.

        See Also
        --------
        extract_largest, split_bodies, threshold, extract_values

        Examples
        --------
        Create a single mesh with three disconnected regions where each
        region has a different cell count.

        >>> import pyvista as pv
        >>> large = pv.Sphere(
        ...     center=(-4, 0, 0), phi_resolution=40, theta_resolution=40
        ... )
        >>> medium = pv.Sphere(
        ...     center=(-2, 0, 0), phi_resolution=15, theta_resolution=15
        ... )
        >>> small = pv.Sphere(
        ...     center=(0, 0, 0), phi_resolution=7, theta_resolution=7
        ... )
        >>> mesh = large + medium + small

        Plot their connectivity.

        >>> conn = mesh.connectivity('all')
        >>> conn.plot(cmap=['red', 'green', 'blue'], show_edges=True)

        Restrict connectivity to a scalar range.

        >>> mesh['y_coordinates'] = mesh.points[:, 1]
        >>> conn = mesh.connectivity('all', scalar_range=[-1, 0])
        >>> conn.plot(cmap=['red', 'green', 'blue'], show_edges=True)

        Extract the region closest to the origin.

        >>> conn = mesh.connectivity('closest', (0, 0, 0))
        >>> conn.plot(color='blue', show_edges=True)

        Extract a region using a cell ID ``100`` as a seed.

        >>> conn = mesh.connectivity('cell_seed', 100)
        >>> conn.plot(color='green', show_edges=True)

        Extract the largest region.

        >>> conn = mesh.connectivity('largest')
        >>> conn.plot(color='red', show_edges=True)

        Extract the largest and smallest regions by specifying their
        region IDs. Note that the region IDs of the output differ from
        the specified IDs since the input has three regions but the output
        only has two.

        >>> large_id = 0  # largest always has ID '0'
        >>> small_id = 2  # smallest has ID 'N-1' with N=3 regions
        >>> conn = mesh.connectivity('specified', (small_id, large_id))
        >>> conn.plot(cmap=['red', 'blue'], show_edges=True)

        """
        # Deprecated on v0.43.0
        keep_largest = kwargs.pop('largest', False)
        if keep_largest:  # pragma: no cover
            warnings.warn(
                "Use of `largest=True` is deprecated. Use 'largest' or "
                "`extraction_mode='largest'` instead.",
                PyVistaDeprecationWarning,
            )
            extraction_mode = 'largest'

        def _unravel_and_validate_ids(ids):
            ids = np.asarray(ids).ravel()
            is_all_integers = np.issubdtype(ids.dtype, np.integer)
            is_all_positive = not np.any(ids < 0)
            if not (is_all_positive and is_all_integers):
                raise ValueError('IDs must be positive integer values.')
            return np.unique(ids)

        def _post_process_extract_values(before_extraction, extracted):
            # Output is UnstructuredGrid, so apply vtkRemovePolyData
            # to input to cast the output as PolyData type instead
            has_cells = extracted.n_cells != 0
            if isinstance(before_extraction, pyvista.PolyData):
                all_ids = set(range(before_extraction.n_cells))

                ids_to_keep = set()
                if has_cells:
                    ids_to_keep |= set(extracted['vtkOriginalCellIds'])
                ids_to_remove = list(all_ids - ids_to_keep)
                if len(ids_to_remove) != 0:
                    if pyvista.vtk_version_info < (9, 1, 0):
                        raise VTKVersionError(
                            '`connectivity` with PolyData requires vtk>=9.1.0',
                        )  # pragma: no cover
                    remove = _vtk.vtkRemovePolyData()
                    remove.SetInputData(before_extraction)
                    remove.SetCellIds(numpy_to_idarr(ids_to_remove))
                    _update_alg(remove, progress_bar, "Removing Cells.")
                    extracted = _get_output(remove)
                    extracted.clean(
                        point_merging=False,
                        inplace=True,
                        progress_bar=progress_bar,
                    )  # remove unused points
            if has_cells:
                extracted.point_data.remove('vtkOriginalPointIds')
                extracted.cell_data.remove('vtkOriginalCellIds')
            return extracted

        # Store active scalars info to restore later if needed
        active_field, active_name = self.active_scalars_info  # type: ignore[attr-defined]

        # Set scalars
        if scalar_range is None:
            input_mesh = self.copy(deep=False)  # type: ignore[attr-defined]
        else:
            if isinstance(scalar_range, np.ndarray):
                num_elements = scalar_range.size
            elif isinstance(scalar_range, Sequence):
                num_elements = len(scalar_range)
            else:
                raise TypeError('Scalar range must be a numpy array or a sequence.')
            if num_elements != 2:
                raise ValueError('Scalar range must have two elements defining the min and max.')
            if scalar_range[0] > scalar_range[1]:
                raise ValueError(
                    f"Lower value of scalar range {scalar_range[0]} cannot be greater than the upper value {scalar_range[0]}",
                )

            # Input will be modified, so copy first
            input_mesh = self.copy()  # type: ignore[attr-defined]
            if scalars is None:
                set_default_active_scalars(input_mesh)
            else:
                input_mesh.set_active_scalars(scalars)
            # Make sure we have point data (required by the filter)
            field, name = input_mesh.active_scalars_info
            if field == FieldAssociation.CELL:
                # Convert to point data with a unique name
                # The point array will be removed later
                point_data = input_mesh.cell_data_to_point_data(progress_bar=progress_bar)[name]
                input_mesh.point_data['__point_data'] = point_data
                input_mesh.set_active_scalars('__point_data')

            if extraction_mode in ['all', 'specified', 'closest']:
                # Scalar connectivity has no effect if SetExtractionModeToAllRegions
                # (which applies to 'all' and 'specified') and 'closest'
                # can sometimes fail for some datasets/scalar values.
                # So, we filter scalar values beforehand
                if scalar_range is not None:
                    # Use extract_values to ensure that cells with at least one
                    # point within the range are kept (this is consistent
                    # with how the filter operates for other modes)
                    extracted = DataSetFilters.extract_values(
                        input_mesh,
                        ranges=scalar_range,
                        progress_bar=progress_bar,
                    )
                    input_mesh = _post_process_extract_values(input_mesh, extracted)

        alg = _vtk.vtkConnectivityFilter()
        alg.SetInputDataObject(input_mesh)

        # Due to inconsistent/buggy output, always keep this on and
        # remove scalars later as needed
        alg.ColorRegionsOn()  # This will create 'RegionId' scalars

        # Sort region ids
        alg.SetRegionIdAssignmentMode(alg.CELL_COUNT_DESCENDING)

        if scalar_range is not None:
            alg.ScalarConnectivityOn()
            alg.SetScalarRange(*scalar_range)

        if extraction_mode == 'all':
            alg.SetExtractionModeToAllRegions()

        elif extraction_mode == 'largest':
            alg.SetExtractionModeToLargestRegion()

        elif extraction_mode == 'specified':
            if region_ids is None:
                if variable_input is None:
                    raise ValueError(
                        "`region_ids` must be specified when `extraction_mode='specified'`.",
                    )
                else:
                    region_ids = variable_input
            # this mode returns scalar data with shape that may not match
            # the number of cells/points, so we extract all and filter later
            # alg.SetExtractionModeToSpecifiedRegions()
            region_ids = _unravel_and_validate_ids(region_ids)
            # [alg.AddSpecifiedRegion(i) for i in region_ids]
            alg.SetExtractionModeToAllRegions()

        elif extraction_mode == 'cell_seed':
            if cell_ids is None:
                if variable_input is None:
                    raise ValueError(
                        "`cell_ids` must be specified when `extraction_mode='cell_seed'`.",
                    )
                else:
                    cell_ids = variable_input
            alg.SetExtractionModeToCellSeededRegions()
            alg.InitializeSeedList()
            for i in _unravel_and_validate_ids(cell_ids):
                alg.AddSeed(i)

        elif extraction_mode == 'point_seed':
            if point_ids is None:
                if variable_input is None:
                    raise ValueError(
                        "`point_ids` must be specified when `extraction_mode='point_seed'`.",
                    )
                else:
                    point_ids = variable_input
            alg.SetExtractionModeToPointSeededRegions()
            alg.InitializeSeedList()
            for i in _unravel_and_validate_ids(point_ids):
                alg.AddSeed(i)

        elif extraction_mode == 'closest':
            if closest_point is None:
                if variable_input is None:
                    raise ValueError(
                        "`closest_point` must be specified when `extraction_mode='closest'`.",
                    )
                else:
                    closest_point = variable_input
            alg.SetExtractionModeToClosestPointRegion()
            alg.SetClosestPoint(*closest_point)

        else:
            raise ValueError(
                f"Invalid value for `extraction_mode` '{extraction_mode}'. Expected one of the following: 'all', 'largest', 'specified', 'cell_seed', 'point_seed', or 'closest'",
            )

        _update_alg(alg, progress_bar, 'Finding and Labeling Connected Regions.')
        output = _get_output(alg)

        # Process output
        output_needs_fixing = False  # initialize flag if output needs to be fixed
        if extraction_mode == 'all':
            pass  # Output is good
        elif extraction_mode == 'specified':
            # All regions were initially extracted, so extract only the
            # specified regions
            extracted = DataSetFilters.extract_values(
                output,
                values=region_ids,
                progress_bar=progress_bar,
            )
            output = _post_process_extract_values(output, extracted)

            if label_regions:
                # Extracted regions may not be contiguous and zero-based
                # which will need to be fixed
                output_needs_fixing = True

        elif extraction_mode == 'largest' and isinstance(output, pyvista.PolyData):
            # PolyData with 'largest' mode generates bad output with unreferenced points
            output_needs_fixing = True

        else:
            # All other extraction modes / cases may generate incorrect scalar arrays
            # e.g. 'largest' may output scalars with shape that does not match output mesh
            # e.g. 'seed' method scalars may have one RegionId, yet may contain many
            # disconnected regions. Therefore, check for correct scalars size
            if label_regions:
                invalid_cell_scalars = output.n_cells != output.cell_data['RegionId'].size
                invalid_point_scalars = output.n_points != output.point_data['RegionId'].size
                if invalid_cell_scalars or invalid_point_scalars:
                    output_needs_fixing = True

        if output_needs_fixing and output.n_cells > 0:
            # Fix bad output recursively using 'all' mode which has known good output
            output.point_data.remove('RegionId')
            output.cell_data.remove('RegionId')
            output = output.connectivity('all', label_regions=True, inplace=inplace)

        # Remove temp point array
        with contextlib.suppress(KeyError):
            output.point_data.remove('__point_data')

        if not label_regions and output.n_cells > 0:
            output.point_data.remove('RegionId')
            output.cell_data.remove('RegionId')

            # restore previously active scalars
            output.set_active_scalars(active_name, preference=active_field)

        if inplace:
            try:
                self.copy_from(output, deep=False)  # type: ignore[attr-defined]
            except:
                pass
            else:
                return self
        return output

    def extract_largest(self, inplace=False, progress_bar=False):
        """Extract largest connected set in mesh.

        Can be used to reduce residues obtained when generating an
        isosurface.  Works only if residues are not connected (share
        at least one point with) the main component of the image.

        Parameters
        ----------
        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Largest connected set in the dataset.  Return type matches input.

        Examples
        --------
        Join two meshes together, extract the largest, and plot it.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere() + pv.Cube()
        >>> largest = mesh.extract_largest()
        >>> largest.plot()

        See :ref:`connectivity_example` and :ref:`volumetric_example` for
        more examples using this filter.

        .. seealso::
            :func:`pyvista.DataSetFilters.connectivity`

        """
        return DataSetFilters.connectivity(
            self,
            'largest',
            label_regions=False,
            inplace=inplace,
            progress_bar=progress_bar,
        )

    def split_bodies(self, label=False, progress_bar=False):
        """Find, label, and split connected bodies/volumes.

        This splits different connected bodies into blocks in a
        :class:`pyvista.MultiBlock` dataset.

        Parameters
        ----------
        label : bool, default: False
            A flag on whether to keep the ID arrays given by the
            ``connectivity`` filter.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        See Also
        --------
        extract_values, partition, connectivity

        Returns
        -------
        pyvista.MultiBlock
            MultiBlock with a split bodies.

        Examples
        --------
        Split a uniform grid thresholded to be non-connected.

        >>> from pyvista import examples
        >>> dataset = examples.load_uniform()
        >>> _ = dataset.set_active_scalars('Spatial Cell Data')
        >>> threshed = dataset.threshold_percent([0.15, 0.50], invert=True)
        >>> bodies = threshed.split_bodies()
        >>> len(bodies)
        2

        See :ref:`split_vol` for more examples using this filter.

        """
        # Get the connectivity and label different bodies
        labeled = DataSetFilters.connectivity(self)
        classifier = labeled.cell_data['RegionId']
        bodies = pyvista.MultiBlock()
        for vid in np.unique(classifier):
            # Now extract it:
            b = labeled.threshold(
                [vid - 0.5, vid + 0.5],
                scalars='RegionId',
                progress_bar=progress_bar,
            )
            if not label:
                # strange behavior:
                # must use this method rather than deleting from the point_data
                # or else object is collected.
                b.cell_data.remove('RegionId')
                b.point_data.remove('RegionId')
            bodies.append(b)

        return bodies

    def warp_by_scalar(
        self,
        scalars=None,
        factor=1.0,
        normal=None,
        inplace=False,
        progress_bar=False,
        **kwargs,
    ):
        """Warp the dataset's points by a point data scalars array's values.

        This modifies point coordinates by moving points along point
        normals by the scalar amount times the scale factor.

        Parameters
        ----------
        scalars : str, optional
            Name of scalars to warp by. Defaults to currently active scalars.

        factor : float, default: 1.0
            A scaling factor to increase the scaling effect. Alias
            ``scale_factor`` also accepted - if present, overrides ``factor``.

        normal : sequence, optional
            User specified normal. If given, data normals will be
            ignored and the given normal will be used to project the
            warp.

        inplace : bool, default: False
            If ``True``, the points of the given dataset will be updated.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Accepts ``scale_factor`` instead of ``factor``.

        Returns
        -------
        pyvista.DataSet
            Warped Dataset.  Return type matches input.

        Examples
        --------
        First, plot the unwarped mesh.

        >>> from pyvista import examples
        >>> mesh = examples.download_st_helens()
        >>> mesh.plot(cmap='gist_earth', show_scalar_bar=False)

        Now, warp the mesh by the ``'Elevation'`` scalars.

        >>> warped = mesh.warp_by_scalar('Elevation')
        >>> warped.plot(cmap='gist_earth', show_scalar_bar=False)

        See :ref:`surface_normal_example` for more examples using this filter.

        """
        factor = kwargs.pop('scale_factor', factor)
        assert_empty_kwargs(**kwargs)
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
        _ = get_array(self, scalars, preference='point', err=True)

        field = get_array_association(self, scalars, preference='point')
        if field != FieldAssociation.POINT:
            raise TypeError('Dataset can only by warped by a point data array.')
        # Run the algorithm
        alg = _vtk.vtkWarpScalar()
        alg.SetInputDataObject(self)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetScaleFactor(factor)
        if normal is not None:
            alg.SetNormal(normal)
            alg.SetUseNormal(True)
        _update_alg(alg, progress_bar, 'Warping by Scalar')
        output = _get_output(alg)
        if inplace:
            if isinstance(self, (_vtk.vtkImageData, _vtk.vtkRectilinearGrid)):
                raise TypeError("This filter cannot be applied inplace for this mesh type.")
            self.copy_from(output, deep=False)
            return self
        return output

    def warp_by_vector(self, vectors=None, factor=1.0, inplace=False, progress_bar=False):
        """Warp the dataset's points by a point data vectors array's values.

        This modifies point coordinates by moving points along point
        vectors by the local vector times the scale factor.

        A classical application of this transform is to visualize
        eigenmodes in mechanics.

        Parameters
        ----------
        vectors : str, optional
            Name of vector to warp by. Defaults to currently active vector.

        factor : float, default: 1.0
            A scaling factor that multiplies the vectors to warp by. Can
            be used to enhance the warping effect.

        inplace : bool, default: False
            If ``True``, the function will update the mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            The warped mesh resulting from the operation.

        Examples
        --------
        Warp a sphere by vectors.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> sphere = examples.load_sphere_vectors()
        >>> warped = sphere.warp_by_vector()
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> actor = pl.add_text("Before warp")
        >>> actor = pl.add_mesh(sphere, color='white')
        >>> pl.subplot(0, 1)
        >>> actor = pl.add_text("After warp")
        >>> actor = pl.add_mesh(warped, color='white')
        >>> pl.show()

        See :ref:`warp_by_vectors_example` and :ref:`eigenmodes_example` for
        more examples using this filter.

        """
        if vectors is None:
            pyvista.set_default_active_vectors(self)
            field, vectors = self.active_vectors_info
        arr = get_array(self, vectors, preference='point')
        field = get_array_association(self, vectors, preference='point')
        if arr is None:
            raise ValueError('No vectors present to warp by vector.')

        # check that this is indeed a vector field
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                'Dataset can only by warped by a 3D vector point data array. '
                'The values you provided do not satisfy this requirement',
            )
        alg = _vtk.vtkWarpVector()
        alg.SetInputDataObject(self)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, vectors)
        alg.SetScaleFactor(factor)
        _update_alg(alg, progress_bar, 'Warping by Vector')
        warped_mesh = _get_output(alg)
        if inplace:
            self.copy_from(warped_mesh, deep=False)
            return self
        else:
            return warped_mesh

    def cell_data_to_point_data(self, pass_cell_data=False, progress_bar=False):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        The method of transformation is based on averaging the data
        values of all cells using a particular point. Optionally, the
        input cell data can be passed through to the output as well.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            If enabled, pass the input cell data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset with the point data transformed into cell data.
            Return type matches input.

        See Also
        --------
        point_data_to_cell_data
            Similar transformation applied to point data.
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
            Re-mesh :class:`~pyvista.ImageData` to a points-based representation.

        Examples
        --------
        First compute the face area of the example airplane mesh and
        show the cell values.  This is to show discrete cell data.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf.plot(scalars='Area')

        These cell scalars can be applied to individual points to
        effectively smooth out the cell data onto the points.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf = surf.cell_data_to_point_data()
        >>> surf.plot(scalars='Area')

        """
        alg = _vtk.vtkCellDataToPointData()
        alg.SetInputDataObject(self)
        alg.SetPassCellData(pass_cell_data)
        _update_alg(alg, progress_bar, 'Transforming cell data into point data.')
        active_scalars = None
        if not isinstance(self, pyvista.MultiBlock):
            active_scalars = self.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    def ctp(self, pass_cell_data=False, progress_bar=False, **kwargs):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        This method is an alias for
        :func:`pyvista.DataSetFilters.cell_data_to_point_data`.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            If enabled, pass the input cell data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Deprecated keyword argument ``pass_cell_arrays``.

        Returns
        -------
        pyvista.DataSet
            Dataset with the cell data transformed into point data.
            Return type matches input.

        """
        return DataSetFilters.cell_data_to_point_data(
            self,
            pass_cell_data=pass_cell_data,
            progress_bar=progress_bar,
            **kwargs,
        )

    def point_data_to_cell_data(self, pass_point_data=False, progress_bar=False):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified within cells.
        Optionally, the input point data can be passed through to the output.

        Parameters
        ----------
        pass_point_data : bool, default: False
            If enabled, pass the input point data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset with the point data transformed into cell data.
            Return type matches input.

        See Also
        --------
        cell_data_to_point_data
            Similar transformation applied to cell data.
        :meth:`~pyvista.ImageDataFilters.points_to_cells`
            Re-mesh :class:`~pyvista.ImageData` to a cells-based representation.

        Examples
        --------
        Color cells by their z coordinates.  First, create point
        scalars based on z-coordinates of a sample sphere mesh.  Then
        convert this point data to cell data.  Use a low resolution
        sphere for emphasis of cell valued data.

        First, plot these values as point values to show the
        difference between point and cell data.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> sphere['Z Coordinates'] = sphere.points[:, 2]
        >>> sphere.plot()

        Now, convert these values to cell data and then plot it.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> sphere['Z Coordinates'] = sphere.points[:, 2]
        >>> sphere = sphere.point_data_to_cell_data()
        >>> sphere.plot()

        """
        alg = _vtk.vtkPointDataToCellData()
        alg.SetInputDataObject(self)
        alg.SetPassPointData(pass_point_data)
        _update_alg(alg, progress_bar, 'Transforming point data into cell data')
        active_scalars = None
        if not isinstance(self, pyvista.MultiBlock):
            active_scalars = self.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    def ptc(self, pass_point_data=False, progress_bar=False, **kwargs):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        This method is an alias for
        :func:`pyvista.DataSetFilters.point_data_to_cell_data`.

        Parameters
        ----------
        pass_point_data : bool, default: False
            If enabled, pass the input point data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Deprecated keyword argument ``pass_point_arrays``.

        Returns
        -------
        pyvista.DataSet
            Dataset with the point data transformed into cell data.
            Return type matches input.

        """
        return DataSetFilters.point_data_to_cell_data(
            self,
            pass_point_data=pass_point_data,
            progress_bar=progress_bar,
            **kwargs,
        )

    def triangulate(self, inplace=False, progress_bar=False):
        """Return an all triangle mesh.

        More complex polygons will be broken down into triangles.

        Parameters
        ----------
        inplace : bool, default: False
            Updates mesh in-place.

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
        alg = _vtk.vtkDataSetTriangleFilter()
        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Converting to triangle mesh')

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)
            return self
        return mesh

    def delaunay_3d(self, alpha=0.0, tol=0.001, offset=2.5, progress_bar=False):
        """Construct a 3D Delaunay triangulation of the mesh.

        This filter can be used to generate a 3D tetrahedral mesh from
        a surface or scattered points.  If you want to create a
        surface from a point cloud, see
        :func:`pyvista.PolyDataFilters.reconstruct_surface`.

        Parameters
        ----------
        alpha : float, default: 0.0
            Distance value to control output of this filter. For a
            non-zero alpha value, only vertices, edges, faces, or
            tetrahedra contained within the circumsphere (of radius
            alpha) will be output. Otherwise, only tetrahedra will be
            output.

        tol : float, default: 0.001
            Tolerance to control discarding of closely spaced points.
            This tolerance is specified as a fraction of the diagonal
            length of the bounding box of the points.

        offset : float, default: 2.5
            Multiplier to control the size of the initial, bounding
            Delaunay triangulation.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the Delaunay triangulation.

        Examples
        --------
        Generate a 3D Delaunay triangulation of a surface mesh of a
        sphere and plot the interior edges generated.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=5, phi_resolution=5)
        >>> grid = sphere.delaunay_3d()
        >>> edges = grid.extract_all_edges()
        >>> edges.plot(line_width=5, color='k')

        """
        alg = _vtk.vtkDelaunay3D()
        alg.SetInputData(self)
        alg.SetAlpha(alpha)
        alg.SetTolerance(tol)
        alg.SetOffset(offset)
        _update_alg(alg, progress_bar, 'Computing 3D Triangulation')
        return _get_output(alg)

    def select_enclosed_points(
        self,
        surface,
        tolerance=0.001,
        inside_out=False,
        check_surface=True,
        progress_bar=False,
    ):
        """Mark points as to whether they are inside a closed surface.

        This evaluates all the input points to determine whether they are in an
        enclosed surface. The filter produces a (0,1) mask
        (in the form of a vtkDataArray) that indicates whether points are
        outside (mask value=0) or inside (mask value=1) a provided surface.
        (The name of the output vtkDataArray is ``"SelectedPoints"``.)

        This filter produces and output data array, but does not modify the
        input dataset. If you wish to extract cells or poinrs, various
        threshold filters are available (i.e., threshold the output array).

        .. warning::
           The filter assumes that the surface is closed and
           manifold. A boolean flag can be set to force the filter to
           first check whether this is true. If ``False`` and not manifold,
           an error will be raised.

        Parameters
        ----------
        surface : pyvista.PolyData
            Set the surface to be used to test for containment. This must be a
            :class:`pyvista.PolyData` object.

        tolerance : float, default: 0.001
            The tolerance on the intersection. The tolerance is expressed as a
            fraction of the bounding box of the enclosing surface.

        inside_out : bool, default: False
            By default, points inside the surface are marked inside or sent
            to the output. If ``inside_out`` is ``True``, then the points
            outside the surface are marked inside.

        check_surface : bool, default: True
            Specify whether to check the surface for closure. When ``True``, the
            algorithm first checks to see if the surface is closed and
            manifold. If the surface is not closed and manifold, a runtime
            error is raised.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing the ``point_data['SelectedPoints']`` array.

        Examples
        --------
        Determine which points on a plane are inside a manifold sphere
        surface mesh.  Extract these points using the
        :func:`DataSetFilters.extract_points` filter and then plot them.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> plane = pv.Plane()
        >>> selected = plane.select_enclosed_points(sphere)
        >>> pts = plane.extract_points(
        ...     selected['SelectedPoints'].view(bool),
        ...     adjacent_cells=False,
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere, style='wireframe')
        >>> _ = pl.add_points(pts, color='r')
        >>> pl.show()

        """
        if not isinstance(surface, pyvista.PolyData):
            raise TypeError("`surface` must be `pyvista.PolyData`")
        if check_surface and surface.n_open_edges > 0:
            raise RuntimeError(
                "Surface is not closed. Please read the warning in the "
                "documentation for this function and either pass "
                "`check_surface=False` or repair the surface.",
            )
        alg = _vtk.vtkSelectEnclosedPoints()
        alg.SetInputData(self)
        alg.SetSurfaceData(surface)
        alg.SetTolerance(tolerance)
        alg.SetInsideOut(inside_out)
        _update_alg(alg, progress_bar, 'Selecting Enclosed Points')
        result = _get_output(alg)
        out = self.copy()
        bools = result['SelectedPoints'].astype(np.uint8)
        if len(bools) < 1:
            bools = np.zeros(out.n_points, dtype=np.uint8)
        out['SelectedPoints'] = bools
        return out

    def sample(
        self,
        target,
        tolerance=None,
        pass_cell_data=True,
        pass_point_data=True,
        categorical=False,
        progress_bar=False,
        locator=None,
        pass_field_data=True,
        mark_blank=True,
        snap_to_closest_point=False,
    ):
        """Resample array data from a passed mesh onto this mesh.

        For `mesh1.sample(mesh2)`, the arrays from `mesh2` are sampled onto
        the points of `mesh1`.  This function interpolates within an
        enclosing cell.  This contrasts with
        :function`pyvista.DataSetFilters.interpolate` that uses a distance
        weighting for nearby points.  If there is cell topology, `sample` is
        usually preferred.

        The point data 'vtkValidPointMask' stores whether the point could be sampled
        with a value of 1 meaning successful sampling. And a value of 0 means
        unsuccessful.

        This uses :class:`vtk.vtkResampleWithDataSet`.

        Parameters
        ----------
        target : pyvista.DataSet
            The vtk data object to sample from - point and cell arrays from
            this object are sampled onto the nodes of the ``dataset`` mesh.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        pass_cell_data : bool, default: True
            Preserve source mesh's original cell data arrays.

        pass_point_data : bool, default: True
            Preserve source mesh's original point data arrays.

        categorical : bool, default: False
            Control whether the source point data is to be treated as
            categorical. If the data is categorical, then the resultant data
            will be determined by a nearest neighbor interpolation scheme.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        locator : vtkAbstractCellLocator or str, optional
            Prototype cell locator to perform the ``FindCell()``
            operation.  Default uses the DataSet ``FindCell`` method.
            Valid strings with mapping to vtk cell locators are

                * 'cell' - vtkCellLocator
                * 'cell_tree' - vtkCellTreeLocator
                * 'obb_tree' - vtkOBBTree
                * 'static_cell' - vtkStaticCellLocator

        pass_field_data : bool, default: True
            Preserve source mesh's original field data arrays.

        mark_blank : bool, default: True
            Whether to mark blank points and cells in "vtkGhostType".

        snap_to_closest_point : bool, default: False
            Whether to snap to cell with closest point if no cell is found. Useful
            when sampling from data with vertex cells. Requires vtk >=9.3.0.

            .. versionadded:: 0.43

        Returns
        -------
        pyvista.DataSet
            Dataset containing resampled data.

        See Also
        --------
        pyvista.DataSetFilters.interpolate

        Examples
        --------
        Resample data from another dataset onto a sphere.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
        >>> data_to_probe = examples.load_uniform()
        >>> result = mesh.sample(data_to_probe)
        >>> result.plot(scalars="Spatial Point Data")

        If sampling from a set of points represented by a ``(n, 3)``
        shaped ``numpy.ndarray``, they need to be converted to a
        PyVista DataSet, e.g. :class:`pyvista.PolyData`, first.

        >>> import numpy as np
        >>> points = np.array([[1.5, 5.0, 6.2], [6.7, 4.2, 8.0]])
        >>> mesh = pv.PolyData(points)
        >>> result = mesh.sample(data_to_probe)
        >>> result["Spatial Point Data"]
        pyvista_ndarray([ 46.5 , 225.12])

        See :ref:`resampling_example` for more examples using this filter.

        """
        if not pyvista.is_pyvista_dataset(target):
            raise TypeError('`target` must be a PyVista mesh type.')
        alg = _vtk.vtkResampleWithDataSet()  # Construct the ResampleWithDataSet object
        alg.SetInputData(self)  # Set the Input data (actually the source i.e. where to sample from)
        # Set the Source data (actually the target, i.e. where to sample to)
        alg.SetSourceData(target)
        alg.SetPassCellArrays(pass_cell_data)
        alg.SetPassPointArrays(pass_point_data)
        alg.SetPassFieldArrays(pass_field_data)

        alg.SetMarkBlankPointsAndCells(mark_blank)
        alg.SetCategoricalData(categorical)

        if tolerance is not None:
            alg.SetComputeTolerance(False)
            alg.SetTolerance(tolerance)
        if locator:
            if isinstance(locator, str):
                locator_map = {
                    "cell": _vtk.vtkCellLocator(),
                    "cell_tree": _vtk.vtkCellTreeLocator(),
                    "obb_tree": _vtk.vtkOBBTree(),
                    "static_cell": _vtk.vtkStaticCellLocator(),
                }
                try:
                    locator = locator_map[locator]
                except KeyError as err:
                    raise ValueError(
                        f"locator must be a string from {locator_map.keys()}, got {locator}",
                    ) from err
            alg.SetCellLocatorPrototype(locator)

        if snap_to_closest_point:
            try:
                alg.SnapToCellWithClosestPointOn()
            except AttributeError:  # pragma: no cover
                raise VTKVersionError("`snap_to_closest_point=True` requires vtk 9.3.0 or newer")
        _update_alg(alg, progress_bar, 'Resampling array Data from a Passed Mesh onto Mesh')
        return _get_output(alg)

    def interpolate(
        self,
        target,
        sharpness=2.0,
        radius=1.0,
        strategy='null_value',
        null_value=0.0,
        n_points=None,
        pass_cell_data=True,
        pass_point_data=True,
        progress_bar=False,
    ):
        """Interpolate values onto this mesh from a given dataset.

        The ``target`` dataset is typically a point cloud. Only point data from
        the ``target`` mesh will be interpolated onto points of this mesh. Whether
        preexisting point and cell data of this mesh are preserved in the
        output can be customized with the ``pass_point_data`` and
        ``pass_cell_data`` parameters.

        This uses a Gaussian interpolation kernel. Use the ``sharpness`` and
        ``radius`` parameters to adjust this kernel. You can also switch this
        kernel to use an N closest points approach.

        If the cell topology is more useful for interpolating, e.g. from a
        discretized FEM or CFD simulation, use
        :func:`pyvista.DataSetFilters.sample` instead.

        Parameters
        ----------
        target : pyvista.DataSet
            The vtk data object to sample from. Point and cell arrays from
            this object are interpolated onto this mesh.

        sharpness : float, default: 2.0
            Set the sharpness (i.e., falloff) of the Gaussian kernel. As the
            sharpness increases the effects of distant points are reduced.

        radius : float, optional
            Specify the radius within which the basis points must lie.

        strategy : str, default: "null_value"
            Specify a strategy to use when encountering a "null" point during
            the interpolation process. Null points occur when the local
            neighborhood (of nearby points to interpolate from) is empty. If
            the strategy is set to ``'mask_points'``, then an output array is
            created that marks points as being valid (=1) or null (invalid =0)
            (and the NullValue is set as well). If the strategy is set to
            ``'null_value'``, then the output data value(s) are set to the
            ``null_value`` (specified in the output point data). Finally, the
            strategy ``'closest_point'`` is to simply use the closest point to
            perform the interpolation.

        null_value : float, default: 0.0
            Specify the null point value. When a null point is encountered
            then all components of each null tuple are set to this value.

        n_points : int, optional
            If given, specifies the number of the closest points used to form
            the interpolation basis. This will invalidate the radius argument
            in favor of an N closest points approach. This typically has poorer
            results.

        pass_cell_data : bool, default: True
            Preserve input mesh's original cell data arrays.

        pass_point_data : bool, default: True
            Preserve input mesh's original point data arrays.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Interpolated dataset.  Return type matches input.

        See Also
        --------
        pyvista.DataSetFilters.sample

        Examples
        --------
        Interpolate the values of 5 points onto a sample plane.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> rng = np.random.default_rng(7)
        >>> point_cloud = rng.random((5, 3))
        >>> point_cloud[:, 2] = 0
        >>> point_cloud -= point_cloud.mean(0)
        >>> pdata = pv.PolyData(point_cloud)
        >>> pdata['values'] = rng.random(5)
        >>> plane = pv.Plane()
        >>> plane.clear_data()
        >>> plane = plane.interpolate(pdata, sharpness=3)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(
        ...     pdata, render_points_as_spheres=True, point_size=50
        ... )
        >>> _ = pl.add_mesh(plane, style='wireframe', line_width=5)
        >>> pl.show()

        See :ref:`interpolate_example` for more examples using this filter.

        """
        if not pyvista.is_pyvista_dataset(target):
            raise TypeError('`target` must be a PyVista mesh type.')

        # Must cast to UnstructuredGrid in some cases (e.g. vtkImageData/vtkRectilinearGrid)
        # I believe the locator and the interpolator call `GetPoints` and not all mesh types have that method
        if isinstance(target, (pyvista.ImageData, pyvista.RectilinearGrid)):
            target = target.cast_to_unstructured_grid()

        gaussian_kernel = _vtk.vtkGaussianKernel()
        gaussian_kernel.SetSharpness(sharpness)
        gaussian_kernel.SetRadius(radius)
        gaussian_kernel.SetKernelFootprintToRadius()
        if n_points:
            gaussian_kernel.SetNumberOfPoints(n_points)
            gaussian_kernel.SetKernelFootprintToNClosest()

        locator = _vtk.vtkStaticPointLocator()
        locator.SetDataSet(target)
        locator.BuildLocator()

        interpolator = _vtk.vtkPointInterpolator()
        interpolator.SetInputData(self)
        interpolator.SetSourceData(target)
        interpolator.SetKernel(gaussian_kernel)
        interpolator.SetLocator(locator)
        interpolator.SetNullValue(null_value)
        if strategy == 'null_value':
            interpolator.SetNullPointsStrategyToNullValue()
        elif strategy == 'mask_points':
            interpolator.SetNullPointsStrategyToMaskPoints()
        elif strategy == 'closest_point':
            interpolator.SetNullPointsStrategyToClosestPoint()
        else:
            raise ValueError(f'strategy `{strategy}` not supported.')
        interpolator.SetPassPointArrays(pass_point_data)
        interpolator.SetPassCellArrays(pass_cell_data)
        _update_alg(interpolator, progress_bar, 'Interpolating')
        return _get_output(interpolator)

    def streamlines(
        self,
        vectors=None,
        source_center=None,
        source_radius=None,
        n_points=100,
        start_position=None,
        return_source=False,
        pointa=None,
        pointb=None,
        progress_bar=False,
        **kwargs,
    ):
        """Integrate a vector field to generate streamlines.

        The default behavior uses a sphere as the source - set its
        location and radius via the ``source_center`` and
        ``source_radius`` keyword arguments.  ``n_points`` defines the
        number of starting points on the sphere surface.
        Alternatively, a line source can be used by specifying
        ``pointa`` and ``pointb``.  ``n_points`` again defines the
        number of points on the line.

        You can retrieve the source by specifying
        ``return_source=True``.

        Optional keyword parameters from
        :func:`pyvista.DataSetFilters.streamlines_from_source` can be
        used here to control the generation of streamlines.

        Parameters
        ----------
        vectors : str, optional
            The string name of the active vector field to integrate across.

        source_center : sequence[float], optional
            Length 3 tuple of floats defining the center of the source
            particles. Defaults to the center of the dataset.

        source_radius : float, optional
            Float radius of the source particle cloud. Defaults to one-tenth of
            the diagonal of the dataset's spatial extent.

        n_points : int, default: 100
            Number of particles present in source sphere or line.

        start_position : sequence[float], optional
            A single point.  This will override the sphere point source.

        return_source : bool, default: False
            Return the source particles as :class:`pyvista.PolyData` as well as the
            streamlines. This will be the second value returned if ``True``.

        pointa, pointb : sequence[float], optional
            The coordinates of a start and end point for a line source. This
            will override the sphere and start_position point source.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            See :func:`pyvista.DataSetFilters.streamlines_from_source`.

        Returns
        -------
        streamlines : pyvista.PolyData
            This produces polylines as the output, with each cell
            (i.e., polyline) representing a streamline. The attribute values
            associated with each streamline are stored in the cell data, whereas
            those associated with streamline-points are stored in the point data.

        source : pyvista.PolyData
            The points of the source are the seed points for the streamlines.
            Only returned if ``return_source=True``.

        Examples
        --------
        See the :ref:`streamlines_example` example.

        """
        if source_center is None:
            source_center = self.center
        if source_radius is None:
            source_radius = self.length / 10.0

        # A single point at start_position
        if start_position is not None:
            source_center = start_position
            source_radius = 0.0
            n_points = 1

        if (pointa is not None and pointb is None) or (pointa is None and pointb is not None):
            raise ValueError("Both pointa and pointb must be provided")
        elif pointa is not None and pointb is not None:
            source = _vtk.vtkLineSource()
            source.SetPoint1(pointa)
            source.SetPoint2(pointb)
            source.SetResolution(n_points)
        else:
            source = _vtk.vtkPointSource()
            source.SetCenter(source_center)
            source.SetRadius(source_radius)
            source.SetNumberOfPoints(n_points)
        source.Update()
        input_source = wrap(source.GetOutput())
        output = self.streamlines_from_source(
            input_source,
            vectors,
            progress_bar=progress_bar,
            **kwargs,
        )
        if return_source:
            return output, input_source
        return output

    def streamlines_from_source(
        self,
        source,
        vectors=None,
        integrator_type=45,
        integration_direction='both',
        surface_streamlines=False,
        initial_step_length=0.5,
        step_unit='cl',
        min_step_length=0.01,
        max_step_length=1.0,
        max_steps=2000,
        terminal_speed=1e-12,
        max_error=1e-6,
        max_time=None,
        compute_vorticity=True,
        rotation_scale=1.0,
        interpolator_type='point',
        progress_bar=False,
    ):
        """Generate streamlines of vectors from the points of a source mesh.

        The integration is performed using a specified integrator, by default
        Runge-Kutta2. This supports integration through any type of dataset.
        If the dataset contains 2D cells like polygons or triangles and the
        ``surface_streamlines`` parameter is used, the integration is constrained
        to lie on the surface defined by 2D cells.

        Parameters
        ----------
        source : pyvista.DataSet
            The points of the source provide the starting points of the
            streamlines.  This will override both sphere and line sources.

        vectors : str, optional
            The string name of the active vector field to integrate across.

        integrator_type : {45, 2, 4}, default: 45
            The integrator type to be used for streamline generation.
            The default is Runge-Kutta45. The recognized solvers are:
            RUNGE_KUTTA2 (``2``),  RUNGE_KUTTA4 (``4``), and RUNGE_KUTTA45
            (``45``). Options are ``2``, ``4``, or ``45``.

        integration_direction : str, default: "both"
            Specify whether the streamline is integrated in the upstream or
            downstream directions (or both). Options are ``'both'``,
            ``'backward'``, or ``'forward'``.

        surface_streamlines : bool, default: False
            Compute streamlines on a surface.

        initial_step_length : float, default: 0.5
            Initial step size used for line integration, expressed ib length
            unitsL or cell length units (see ``step_unit`` parameter).
            either the starting size for an adaptive integrator, e.g., RK45, or
            the constant / fixed size for non-adaptive ones, i.e., RK2 and RK4).

        step_unit : {'cl', 'l'}, default: "cl"
            Uniform integration step unit. The valid unit is now limited to
            only LENGTH_UNIT (``'l'``) and CELL_LENGTH_UNIT (``'cl'``).
            Default is CELL_LENGTH_UNIT.

        min_step_length : float, default: 0.01
            Minimum step size used for line integration, expressed in length or
            cell length units. Only valid for an adaptive integrator, e.g., RK45.

        max_step_length : float, default: 1.0
            Maximum step size used for line integration, expressed in length or
            cell length units. Only valid for an adaptive integrator, e.g., RK45.

        max_steps : int, default: 2000
            Maximum number of steps for integrating a streamline.

        terminal_speed : float, default: 1e-12
            Terminal speed value, below which integration is terminated.

        max_error : float, 1e-6
            Maximum error tolerated throughout streamline integration.

        max_time : float, optional
            Specify the maximum length of a streamline expressed in LENGTH_UNIT.

        compute_vorticity : bool, default: True
            Vorticity computation at streamline points. Necessary for generating
            proper stream-ribbons using the ``vtkRibbonFilter``.

        rotation_scale : float, default: 1.0
            This can be used to scale the rate with which the streamribbons
            twist.

        interpolator_type : str, default: "point"
            Set the type of the velocity field interpolator to locate cells
            during streamline integration either by points or cells.
            The cell locator is more robust then the point locator. Options
            are ``'point'`` or ``'cell'`` (abbreviations of ``'p'`` and ``'c'``
            are also supported).

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Streamlines. This produces polylines as the output, with
            each cell (i.e., polyline) representing a streamline. The
            attribute values associated with each streamline are
            stored in the cell data, whereas those associated with
            streamline-points are stored in the point data.

        Examples
        --------
        See the :ref:`streamlines_example` example.

        """
        integration_direction = str(integration_direction).strip().lower()
        if integration_direction not in ['both', 'back', 'backward', 'forward']:
            raise ValueError(
                "Integration direction must be one of:\n 'backward', "
                f"'forward', or 'both' - not '{integration_direction}'.",
            )
        if integrator_type not in [2, 4, 45]:
            raise ValueError('Integrator type must be one of `2`, `4`, or `45`.')
        if interpolator_type not in ['c', 'cell', 'p', 'point']:
            raise ValueError("Interpolator type must be either 'cell' or 'point'")
        if step_unit not in ['l', 'cl']:
            raise ValueError("Step unit must be either 'l' or 'cl'")
        step_unit = {
            'cl': _vtk.vtkStreamTracer.CELL_LENGTH_UNIT,
            'l': _vtk.vtkStreamTracer.LENGTH_UNIT,
        }[step_unit]
        if isinstance(vectors, str):
            self.set_active_scalars(vectors)
            self.set_active_vectors(vectors)
        elif vectors is None:
            pyvista.set_default_active_vectors(self)

        if max_time is None:
            max_velocity = self.get_data_range()[-1]
            max_time = 4.0 * self.GetLength() / max_velocity
        if not isinstance(source, pyvista.DataSet):
            raise TypeError("source must be a pyvista.DataSet")

        # vtk throws error with two Structured Grids
        # See: https://github.com/pyvista/pyvista/issues/1373
        if isinstance(self, pyvista.StructuredGrid) and isinstance(source, pyvista.StructuredGrid):
            source = source.cast_to_unstructured_grid()

        # Build the algorithm
        alg = _vtk.vtkStreamTracer()
        # Inputs
        alg.SetInputDataObject(self)
        alg.SetSourceData(source)

        # general parameters
        alg.SetComputeVorticity(compute_vorticity)
        alg.SetInitialIntegrationStep(initial_step_length)
        alg.SetIntegrationStepUnit(step_unit)
        alg.SetMaximumError(max_error)
        alg.SetMaximumIntegrationStep(max_step_length)
        alg.SetMaximumNumberOfSteps(max_steps)
        alg.SetMaximumPropagation(max_time)
        alg.SetMinimumIntegrationStep(min_step_length)
        alg.SetRotationScale(rotation_scale)
        alg.SetSurfaceStreamlines(surface_streamlines)
        alg.SetTerminalSpeed(terminal_speed)
        # Model parameters
        if integration_direction == 'forward':
            alg.SetIntegrationDirectionToForward()
        elif integration_direction in ['backward', 'back']:
            alg.SetIntegrationDirectionToBackward()
        else:
            alg.SetIntegrationDirectionToBoth()
        # set integrator type
        if integrator_type == 2:
            alg.SetIntegratorTypeToRungeKutta2()
        elif integrator_type == 4:
            alg.SetIntegratorTypeToRungeKutta4()
        else:
            alg.SetIntegratorTypeToRungeKutta45()
        # set interpolator type
        if interpolator_type in ['c', 'cell']:
            alg.SetInterpolatorTypeToCellLocator()
        else:
            alg.SetInterpolatorTypeToDataSetPointLocator()
        # run the algorithm
        _update_alg(alg, progress_bar, 'Generating Streamlines')
        return _get_output(alg)

    def streamlines_evenly_spaced_2D(
        self,
        vectors=None,
        start_position=None,
        integrator_type=2,
        step_length=0.5,
        step_unit='cl',
        max_steps=2000,
        terminal_speed=1e-12,
        interpolator_type='point',
        separating_distance=10,
        separating_distance_ratio=0.5,
        closed_loop_maximum_distance=0.5,
        loop_angle=20,
        minimum_number_of_loop_points=4,
        compute_vorticity=True,
        progress_bar=False,
    ):
        """Generate evenly spaced streamlines on a 2D dataset.

        This filter only supports datasets that lie on the xy plane, i.e. ``z=0``.
        Particular care must be used to choose a `separating_distance`
        that do not result in too much memory being utilized.  The
        default unit is cell length.

        Parameters
        ----------
        vectors : str, optional
            The string name of the active vector field to integrate across.

        start_position : sequence[float], optional
            The seed point for generating evenly spaced streamlines.
            If not supplied, a random position in the dataset is chosen.

        integrator_type : {2, 4}, default: 2
            The integrator type to be used for streamline generation.
            The default is Runge-Kutta2. The recognized solvers are:
            RUNGE_KUTTA2 (``2``) and RUNGE_KUTTA4 (``4``).

        step_length : float, default: 0.5
            Constant Step size used for line integration, expressed in length
            units or cell length units (see ``step_unit`` parameter).

        step_unit : {'cl', 'l'}, default: "cl"
            Uniform integration step unit. The valid unit is now limited to
            only LENGTH_UNIT (``'l'``) and CELL_LENGTH_UNIT (``'cl'``).
            Default is CELL_LENGTH_UNIT.

        max_steps : int, default: 2000
            Maximum number of steps for integrating a streamline.

        terminal_speed : float, default: 1e-12
            Terminal speed value, below which integration is terminated.

        interpolator_type : str, optional
            Set the type of the velocity field interpolator to locate cells
            during streamline integration either by points or cells.
            The cell locator is more robust then the point locator. Options
            are ``'point'`` or ``'cell'`` (abbreviations of ``'p'`` and ``'c'``
            are also supported).

        separating_distance : float, default: 10
            The distance between streamlines expressed in ``step_unit``.

        separating_distance_ratio : float, default: 0.5
            Streamline integration is stopped if streamlines are closer than
            ``SeparatingDistance*SeparatingDistanceRatio`` to other streamlines.

        closed_loop_maximum_distance : float, default: 0.5
            The distance between points on a streamline to determine a
            closed loop.

        loop_angle : float, default: 20
            The maximum angle in degrees between points to determine a closed loop.

        minimum_number_of_loop_points : int, default: 4
            The minimum number of points before which a closed loop will
            be determined.

        compute_vorticity : bool, default: True
            Vorticity computation at streamline points. Necessary for generating
            proper stream-ribbons using the ``vtkRibbonFilter``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            This produces polylines as the output, with each cell
            (i.e., polyline) representing a streamline. The attribute
            values associated with each streamline are stored in the
            cell data, whereas those associated with streamline-points
            are stored in the point data.

        Examples
        --------
        Plot evenly spaced streamlines for cylinder in a crossflow.
        This dataset is a multiblock dataset, and the fluid velocity is in the
        first block.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_cylinder_crossflow()
        >>> streams = mesh[0].streamlines_evenly_spaced_2D(
        ...     start_position=(4, 0.1, 0.0),
        ...     separating_distance=3,
        ...     separating_distance_ratio=0.2,
        ... )
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(
        ...     streams.tube(radius=0.02), scalars="vorticity_mag"
        ... )
        >>> plotter.view_xy()
        >>> plotter.show()

        See :ref:`2d_streamlines_example` for more examples using this filter.
        """
        if integrator_type not in [2, 4]:
            raise ValueError('Integrator type must be one of `2` or `4`.')
        if interpolator_type not in ['c', 'cell', 'p', 'point']:
            raise ValueError("Interpolator type must be either 'cell' or 'point'")
        if step_unit not in ['l', 'cl']:
            raise ValueError("Step unit must be either 'l' or 'cl'")
        step_unit = {
            'cl': _vtk.vtkStreamTracer.CELL_LENGTH_UNIT,
            'l': _vtk.vtkStreamTracer.LENGTH_UNIT,
        }[step_unit]
        if isinstance(vectors, str):
            self.set_active_scalars(vectors)
            self.set_active_vectors(vectors)
        elif vectors is None:
            pyvista.set_default_active_vectors(self)

        loop_angle = loop_angle * np.pi / 180

        # Build the algorithm
        alg = _vtk.vtkEvenlySpacedStreamlines2D()
        # Inputs
        alg.SetInputDataObject(self)

        # Seed for starting position
        if start_position is not None:
            alg.SetStartPosition(start_position)

        # Integrator controls
        if integrator_type == 2:
            alg.SetIntegratorTypeToRungeKutta2()
        else:
            alg.SetIntegratorTypeToRungeKutta4()
        alg.SetInitialIntegrationStep(step_length)
        alg.SetIntegrationStepUnit(step_unit)
        alg.SetMaximumNumberOfSteps(max_steps)

        # Stopping criteria
        alg.SetTerminalSpeed(terminal_speed)
        alg.SetClosedLoopMaximumDistance(closed_loop_maximum_distance)
        alg.SetLoopAngle(loop_angle)
        alg.SetMinimumNumberOfLoopPoints(minimum_number_of_loop_points)

        # Separation criteria
        alg.SetSeparatingDistance(separating_distance)
        if separating_distance_ratio is not None:
            alg.SetSeparatingDistanceRatio(separating_distance_ratio)

        alg.SetComputeVorticity(compute_vorticity)

        # Set interpolator type
        if interpolator_type in ['c', 'cell']:
            alg.SetInterpolatorTypeToCellLocator()
        else:
            alg.SetInterpolatorTypeToDataSetPointLocator()

        # Run the algorithm
        _update_alg(alg, progress_bar, 'Generating Evenly Spaced Streamlines on a 2D Dataset')
        return _get_output(alg)

    def decimate_boundary(self, target_reduction=0.5, progress_bar=False):
        """Return a decimated version of a triangulation of the boundary.

        Only the outer surface of the input dataset will be considered.

        Parameters
        ----------
        target_reduction : float, default: 0.5
            Fraction of the original mesh to remove.
            TargetReduction is set to ``0.9``, this filter will try to reduce
            the data set to 10% of its original size and will remove 90%
            of the input triangles.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Decimated boundary.

        Examples
        --------
        See the :ref:`linked_views_example` example.

        """
        return (
            self.extract_geometry(progress_bar=progress_bar)
            .triangulate()
            .decimate(target_reduction)
        )

    def sample_over_line(self, pointa, pointb, resolution=None, tolerance=None, progress_bar=False):
        """Sample a dataset onto a line.

        Parameters
        ----------
        pointa : sequence[float]
            Location in ``[x, y, z]``.

        pointb : sequence[float]
            Location in ``[x, y, z]``.

        resolution : int, optional
            Number of pieces to divide line into. Defaults to number of cells
            in the input mesh. Must be a positive integer.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is in a
            cell of the input.  If not given, tolerance is automatically generated.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Line object with sampled data from dataset.

        Examples
        --------
        Sample over a plane that is interpolating a point cloud.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> rng = np.random.default_rng(12)
        >>> point_cloud = rng.random((5, 3))
        >>> point_cloud[:, 2] = 0
        >>> point_cloud -= point_cloud.mean(0)
        >>> pdata = pv.PolyData(point_cloud)
        >>> pdata['values'] = rng.random(5)
        >>> plane = pv.Plane()
        >>> plane.clear_data()
        >>> plane = plane.interpolate(pdata, sharpness=3.5)
        >>> sample = plane.sample_over_line((-0.5, -0.5, 0), (0.5, 0.5, 0))
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(
        ...     pdata, render_points_as_spheres=True, point_size=50
        ... )
        >>> _ = pl.add_mesh(sample, scalars='values', line_width=10)
        >>> _ = pl.add_mesh(plane, scalars='values', style='wireframe')
        >>> pl.show()

        """
        if resolution is None:
            resolution = int(self.n_cells)
        # Make a line and sample the dataset
        line = pyvista.Line(pointa, pointb, resolution=resolution)
        return line.sample(self, tolerance=tolerance, progress_bar=progress_bar)

    def plot_over_line(
        self,
        pointa,
        pointb,
        resolution=None,
        scalars=None,
        title=None,
        ylabel=None,
        figsize=None,
        figure=True,
        show=True,
        tolerance=None,
        fname=None,
        progress_bar=False,
    ):
        """Sample a dataset along a high resolution line and plot.

        Plot the variables of interest in 2D using matplotlib where the
        X-axis is distance from Point A and the Y-axis is the variable
        of interest. Note that this filter returns ``None``.

        Parameters
        ----------
        pointa : sequence[float]
            Location in ``[x, y, z]``.

        pointb : sequence[float]
            Location in ``[x, y, z]``.

        resolution : int, optional
            Number of pieces to divide line into. Defaults to number of cells
            in the input mesh. Must be a positive integer.

        scalars : str, optional
            The string name of the variable in the input dataset to probe. The
            active scalar is used by default.

        title : str, optional
            The string title of the matplotlib figure.

        ylabel : str, optional
            The string label of the Y-axis. Defaults to variable name.

        figsize : tuple(int), optional
            The size of the new figure.

        figure : bool, default: True
            Flag on whether or not to create a new figure.

        show : bool, default: True
            Shows the matplotlib figure.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is in a
            cell of the input.  If not given, tolerance is automatically generated.

        fname : str, optional
            Save the figure this file name when set.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Examples
        --------
        See the :ref:`plot_over_line_example` example.

        """
        # Sample on line
        sampled = DataSetFilters.sample_over_line(
            self,
            pointa,
            pointb,
            resolution,
            tolerance,
            progress_bar=progress_bar,
        )

        # Get variable of interest
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
        values = sampled.get_array(scalars)
        distance = sampled['Distance']

        # Remainder is plotting
        if figure:
            plt.figure(figsize=figsize)
        # Plot it in 2D
        if values.ndim > 1:
            for i in range(values.shape[1]):
                plt.plot(distance, values[:, i], label=f'Component {i}')
            plt.legend()
        else:
            plt.plot(distance, values)
        plt.xlabel('Distance')
        if ylabel is None:
            plt.ylabel(scalars)
        else:
            plt.ylabel(ylabel)
        if title is None:
            plt.title(f'{scalars} Profile')
        else:
            plt.title(title)
        if fname:
            plt.savefig(fname)
        if show:  # pragma: no cover
            plt.show()

    def sample_over_multiple_lines(self, points, tolerance=None, progress_bar=False):
        """Sample a dataset onto a multiple lines.

        Parameters
        ----------
        points : array_like[float]
            List of points defining multiple lines.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is in a
            cell of the input.  If not given, tolerance is automatically generated.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Line object with sampled data from dataset.

        Examples
        --------
        Sample over a plane that is interpolating a point cloud.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> rng = np.random.default_rng(12)
        >>> point_cloud = rng.random((5, 3))
        >>> point_cloud[:, 2] = 0
        >>> point_cloud -= point_cloud.mean(0)
        >>> pdata = pv.PolyData(point_cloud)
        >>> pdata['values'] = rng.random(5)
        >>> plane = pv.Plane()
        >>> plane.clear_data()
        >>> plane = plane.interpolate(pdata, sharpness=3.5)
        >>> sample = plane.sample_over_multiple_lines(
        ...     [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0]]
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(
        ...     pdata, render_points_as_spheres=True, point_size=50
        ... )
        >>> _ = pl.add_mesh(sample, scalars='values', line_width=10)
        >>> _ = pl.add_mesh(plane, scalars='values', style='wireframe')
        >>> pl.show()

        """
        # Make a multiple lines and sample the dataset
        multiple_lines = pyvista.MultipleLines(points=points)
        return multiple_lines.sample(self, tolerance=tolerance, progress_bar=progress_bar)

    def sample_over_circular_arc(
        self,
        pointa,
        pointb,
        center,
        resolution=None,
        tolerance=None,
        progress_bar=False,
    ):
        """Sample a dataset over a circular arc.

        Parameters
        ----------
        pointa : sequence[float]
            Location in ``[x, y, z]``.

        pointb : sequence[float]
            Location in ``[x, y, z]``.

        center : sequence[float]
            Location in ``[x, y, z]``.

        resolution : int, optional
            Number of pieces to divide circular arc into. Defaults to
            number of cells in the input mesh. Must be a positive
            integer.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Arc containing the sampled data.

        Examples
        --------
        Sample a dataset over a circular arc and plot it.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> uniform = examples.load_uniform()
        >>> uniform["height"] = uniform.points[:, 2]
        >>> pointa = [
        ...     uniform.bounds[1],
        ...     uniform.bounds[2],
        ...     uniform.bounds[5],
        ... ]
        >>> pointb = [
        ...     uniform.bounds[1],
        ...     uniform.bounds[3],
        ...     uniform.bounds[4],
        ... ]
        >>> center = [
        ...     uniform.bounds[1],
        ...     uniform.bounds[2],
        ...     uniform.bounds[4],
        ... ]
        >>> sampled_arc = uniform.sample_over_circular_arc(
        ...     pointa, pointb, center
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(uniform, style='wireframe')
        >>> _ = pl.add_mesh(sampled_arc, line_width=10)
        >>> pl.show_axes()
        >>> pl.show()

        """
        if resolution is None:
            resolution = int(self.n_cells)
        # Make a circular arc and sample the dataset
        circular_arc = pyvista.CircularArc(pointa, pointb, center, resolution=resolution)
        return circular_arc.sample(self, tolerance=tolerance, progress_bar=progress_bar)

    def sample_over_circular_arc_normal(
        self,
        center,
        resolution=None,
        normal=None,
        polar=None,
        angle=None,
        tolerance=None,
        progress_bar=False,
    ):
        """Sample a dataset over a circular arc defined by a normal and polar vector and plot it.

        The number of segments composing the polyline is controlled by
        setting the object resolution.

        Parameters
        ----------
        center : sequence[float]
            Location in ``[x, y, z]``.

        resolution : int, optional
            Number of pieces to divide circular arc into. Defaults to
            number of cells in the input mesh. Must be a positive
            integer.

        normal : sequence[float], optional
            The normal vector to the plane of the arc.  By default it
            points in the positive Z direction.

        polar : sequence[float], optional
            Starting point of the arc in polar coordinates.  By
            default it is the unit vector in the positive x direction.

        angle : float, optional
            Arc length (in degrees), beginning at the polar vector.  The
            direction is counterclockwise.  By default it is 360.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sampled Dataset.

        Examples
        --------
        Sample a dataset over a circular arc.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> uniform = examples.load_uniform()
        >>> uniform["height"] = uniform.points[:, 2]
        >>> normal = [0, 0, 1]
        >>> polar = [0, 9, 0]
        >>> center = [
        ...     uniform.bounds[1],
        ...     uniform.bounds[2],
        ...     uniform.bounds[5],
        ... ]
        >>> arc = uniform.sample_over_circular_arc_normal(
        ...     center, normal=normal, polar=polar
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(uniform, style='wireframe')
        >>> _ = pl.add_mesh(arc, line_width=10)
        >>> pl.show_axes()
        >>> pl.show()

        """
        if resolution is None:
            resolution = int(self.n_cells)
        # Make a circular arc and sample the dataset
        circular_arc = pyvista.CircularArcFromNormal(
            center,
            resolution=resolution,
            normal=normal,
            polar=polar,
            angle=angle,
        )
        return circular_arc.sample(self, tolerance=tolerance, progress_bar=progress_bar)

    def plot_over_circular_arc(
        self,
        pointa,
        pointb,
        center,
        resolution=None,
        scalars=None,
        title=None,
        ylabel=None,
        figsize=None,
        figure=True,
        show=True,
        tolerance=None,
        fname=None,
        progress_bar=False,
    ):
        """Sample a dataset along a circular arc and plot it.

        Plot the variables of interest in 2D where the X-axis is
        distance from Point A and the Y-axis is the variable of
        interest. Note that this filter returns ``None``.

        Parameters
        ----------
        pointa : sequence[float]
            Location in ``[x, y, z]``.

        pointb : sequence[float]
            Location in ``[x, y, z]``.

        center : sequence[float]
            Location in ``[x, y, z]``.

        resolution : int, optional
            Number of pieces to divide the circular arc into. Defaults
            to number of cells in the input mesh. Must be a positive
            integer.

        scalars : str, optional
            The string name of the variable in the input dataset to
            probe. The active scalar is used by default.

        title : str, optional
            The string title of the ``matplotlib`` figure.

        ylabel : str, optional
            The string label of the Y-axis. Defaults to the variable name.

        figsize : tuple(int), optional
            The size of the new figure.

        figure : bool, default: True
            Flag on whether or not to create a new figure.

        show : bool, default: True
            Shows the ``matplotlib`` figure when ``True``.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        fname : str, optional
            Save the figure this file name when set.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Examples
        --------
        Sample a dataset along a high resolution circular arc and plot.

        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> a = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[5]]
        >>> b = [mesh.bounds[1], mesh.bounds[2], mesh.bounds[4]]
        >>> center = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
        >>> mesh.plot_over_circular_arc(
        ...     a, b, center, resolution=1000, show=False
        ... )  # doctest:+SKIP

        """
        # Sample on circular arc
        sampled = DataSetFilters.sample_over_circular_arc(
            self,
            pointa,
            pointb,
            center,
            resolution,
            tolerance,
            progress_bar=progress_bar,
        )

        # Get variable of interest
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
        values = sampled.get_array(scalars)
        distance = sampled['Distance']

        # create the matplotlib figure
        if figure:
            plt.figure(figsize=figsize)
        # Plot it in 2D
        if values.ndim > 1:
            for i in range(values.shape[1]):
                plt.plot(distance, values[:, i], label=f'Component {i}')
            plt.legend()
        else:
            plt.plot(distance, values)
        plt.xlabel('Distance')
        if ylabel is None:
            plt.ylabel(scalars)
        else:
            plt.ylabel(ylabel)
        if title is None:
            plt.title(f'{scalars} Profile')
        else:
            plt.title(title)
        if fname:
            plt.savefig(fname)
        if show:  # pragma: no cover
            plt.show()

    def plot_over_circular_arc_normal(
        self,
        center,
        resolution=None,
        normal=None,
        polar=None,
        angle=None,
        scalars=None,
        title=None,
        ylabel=None,
        figsize=None,
        figure=True,
        show=True,
        tolerance=None,
        fname=None,
        progress_bar=False,
    ):
        """Sample a dataset along a resolution circular arc defined by a normal and polar vector and plot it.

        Plot the variables of interest in 2D where the X-axis is
        distance from Point A and the Y-axis is the variable of
        interest. Note that this filter returns ``None``.

        Parameters
        ----------
        center : sequence[int]
            Location in ``[x, y, z]``.

        resolution : int, optional
            Number of pieces to divide circular arc into. Defaults to
            number of cells in the input mesh. Must be a positive
            integer.

        normal : sequence[float], optional
            The normal vector to the plane of the arc.  By default it
            points in the positive Z direction.

        polar : sequence[float], optional
            Starting point of the arc in polar coordinates.  By
            default it is the unit vector in the positive x direction.

        angle : float, optional
            Arc length (in degrees), beginning at the polar vector.  The
            direction is counterclockwise.  By default it is 360.

        scalars : str, optional
            The string name of the variable in the input dataset to
            probe. The active scalar is used by default.

        title : str, optional
            The string title of the `matplotlib` figure.

        ylabel : str, optional
            The string label of the Y-axis. Defaults to variable name.

        figsize : tuple(int), optional
            The size of the new figure.

        figure : bool, optional
            Flag on whether or not to create a new figure.

        show : bool, default: True
            Shows the matplotlib figure.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        fname : str, optional
            Save the figure this file name when set.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Examples
        --------
        Sample a dataset along a high resolution circular arc and plot.

        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> normal = normal = [0, 0, 1]
        >>> polar = [0, 9, 0]
        >>> angle = 90
        >>> center = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
        >>> mesh.plot_over_circular_arc_normal(
        ...     center, polar=polar, angle=angle
        ... )  # doctest:+SKIP

        """
        # Sample on circular arc
        sampled = DataSetFilters.sample_over_circular_arc_normal(
            self,
            center,
            resolution,
            normal,
            polar,
            angle,
            tolerance,
            progress_bar=progress_bar,
        )

        # Get variable of interest
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
        values = sampled.get_array(scalars)
        distance = sampled['Distance']

        # create the matplotlib figure
        if figure:
            plt.figure(figsize=figsize)
        # Plot it in 2D
        if values.ndim > 1:
            for i in range(values.shape[1]):
                plt.plot(distance, values[:, i], label=f'Component {i}')
            plt.legend()
        else:
            plt.plot(distance, values)
        plt.xlabel('Distance')
        if ylabel is None:
            plt.ylabel(scalars)
        else:
            plt.ylabel(ylabel)
        if title is None:
            plt.title(f'{scalars} Profile')
        else:
            plt.title(title)
        if fname:
            plt.savefig(fname)
        if show:  # pragma: no cover
            plt.show()

    def extract_cells(self, ind, invert=False, progress_bar=False):
        """Return a subset of the grid.

        Parameters
        ----------
        ind : sequence[int]
            Numpy array of cell indices to be extracted.

        invert : bool, default: False
            Invert the selection.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        See Also
        --------
        extract_points, extract_values

        Returns
        -------
        pyvista.UnstructuredGrid
            Subselected grid.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = pv.read(examples.hexbeamfile)
        >>> subset = grid.extract_cells(range(20))
        >>> subset.n_cells
        20
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(
        ...     grid, style='wireframe', line_width=5, color='black'
        ... )
        >>> actor = pl.add_mesh(subset, color='grey')
        >>> pl.show()

        """
        if invert:
            _, ind = numpy_to_idarr(ind, return_ind=True)
            ind = [i for i in range(self.n_cells) if i not in ind]

        # Create selection objects
        selectionNode = _vtk.vtkSelectionNode()
        selectionNode.SetFieldType(_vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(_vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(numpy_to_idarr(ind))

        selection = _vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = _vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, self)
        extract_sel.SetInputData(1, selection)
        _update_alg(extract_sel, progress_bar, 'Extracting Cells')
        subgrid = _get_output(extract_sel)

        # extracts only in float32
        if subgrid.n_points:
            if self.points.dtype != np.dtype('float32'):
                ind = subgrid.point_data['vtkOriginalPointIds']
                subgrid.points = self.points[ind]

        return subgrid

    def extract_points(self, ind, adjacent_cells=True, include_cells=True, progress_bar=False):
        """Return a subset of the grid (with cells) that contains any of the given point indices.

        Parameters
        ----------
        ind : sequence[int]
            Sequence of point indices to be extracted.

        adjacent_cells : bool, default: True
            If ``True``, extract the cells that contain at least one of
            the extracted points. If ``False``, extract the cells that
            contain exclusively points from the extracted points list.
            Has no effect if ``include_cells`` is ``False``.

        include_cells : bool, default: True
            Specifies if the cells shall be returned or not.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        See Also
        --------
        extract_cells, extract_values

        Returns
        -------
        pyvista.UnstructuredGrid
            Subselected grid.

        Examples
        --------
        Extract all the points of a sphere with a Z coordinate greater than 0

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> extracted = sphere.extract_points(
        ...     sphere.points[:, 2] > 0, include_cells=False
        ... )
        >>> extracted.clear_data()  # clear for plotting
        >>> extracted.plot()

        """
        ind = np.array(ind)
        # Create selection objects
        selectionNode = _vtk.vtkSelectionNode()
        selectionNode.SetFieldType(_vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(_vtk.vtkSelectionNode.INDICES)
        if not include_cells:
            adjacent_cells = True
        if not adjacent_cells:
            # Build array of point indices to be removed.
            ind_rem = np.ones(self.n_points, dtype='bool')
            ind_rem[ind] = False
            ind = np.arange(self.n_points)[ind_rem]
            # Invert selection
            selectionNode.GetProperties().Set(_vtk.vtkSelectionNode.INVERSE(), 1)
        selectionNode.SetSelectionList(numpy_to_idarr(ind))
        if include_cells:
            selectionNode.GetProperties().Set(_vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)

        selection = _vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = _vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, self)
        extract_sel.SetInputData(1, selection)
        _update_alg(extract_sel, progress_bar, 'Extracting Points')
        return _get_output(extract_sel)

    def split_values(
        self,
        values: None
        | (
            float | VectorLike[float] | MatrixLike[float] | dict[str, float] | dict[float, str]
        ) = None,
        *,
        ranges: None
        | (
            VectorLike[float]
            | MatrixLike[float]
            | dict[str, VectorLike[float]]
            | dict[tuple[float, float], str]
        ) = None,
        scalars: str | None = None,
        preference: Literal['point', 'cell'] = 'point',
        component_mode: Literal['any', 'all', 'multi'] | int = 'all',
        **kwargs,
    ):
        """Split mesh into separate sub-meshes using point or cell data.

        By default, this filter generates a separate mesh for each unique value in the
        data array and combines them as blocks in a :class:`~pyvista.MultiBlock`
        dataset. Optionally, specific values and/or ranges of values may be specified to
        control which values to split from the input.

        This filter is a convenience method for :meth:`~pyvista.DataSetFilter.extract_values`
        with ``split`` set to ``True`` by default. Refer to that filter's documentation
        for more details.

        .. versionadded:: 0.44

        Parameters
        ----------
        values : number | array_like | dict, optional
            Value(s) to extract. Can be a number, an iterable of numbers, or a dictionary
            with numeric entries. For ``dict`` inputs, either its keys or values may be
            numeric, and the other field must be strings. The numeric field is used as
            the input for this parameter, and if ``split`` is ``True``, the string field
            is used to set the block names of the returned :class:`~pyvista.MultiBlock`.

            .. note::
                When extracting multi-component values with ``component_mode=multi``,
                each value is specified as a multi-component scalar. In this case,
                ``values`` can be a single vector or an array of row vectors.

        ranges : array_like | dict, optional
            Range(s) of values to extract. Can be a single range (i.e. a sequence of
            two numbers in the form ``[lower, upper]``), a sequence of ranges, or a
            dictionary with range entries. Any combination of ``values`` and ``ranges``
            may be specified together. The endpoints of the ranges are included in the
            extraction. Ranges cannot be set when ``component_mode=multi``.

            For ``dict`` inputs, either its keys or values may be numeric, and the other
            field must be strings. The numeric field is used as the input for this
            parameter, and if ``split`` is ``True``, the string field is used to set the
            block names of the returned :class:`~pyvista.MultiBlock`.

            .. note::
                Use ``+/-`` infinity to specify an unlimited bound, e.g.:

                - ``[0, float('inf')]`` to extract values greater than or equal to zero.
                - ``[float('-inf'), 0]`` to extract values less than or equal to zero.

        scalars : str, optional
            Name of scalars to extract with. Defaults to currently active scalars.

        preference : str, default: 'point'
            When ``scalars`` is specified, this is the preferred array type to search
            for in the dataset.  Must be either ``'point'`` or ``'cell'``.

        component_mode : int | 'any' | 'all' | 'multi', default: 'all'
            Specify the component(s) to use when ``scalars`` is a multi-component array.
            Has no effect when the scalars have a single component. Must be one of:

            - number: specify the component number as a 0-indexed integer. The selected
              component must have the specified value(s).
            - ``'any'``: any single component can have the specified value(s).
            - ``'all'``: all individual components must have the specified values(s).
            - ``'multi'``: the entire multi-component item must have the specified value.

        **kwargs : dict, optional
            Additional keyword arguments passed to :meth:`~pyvista.DataSetFilter.extract_values`.

        See Also
        --------
        extract_values, split_bodies, partition

        Returns
        -------
        pyvista.MultiBlock
            Composite of split meshes with :class:`pyvista.UnstructuredGrid` blocks.

        Examples
        --------
        Load image with labeled regions.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> image = examples.load_channels()
        >>> np.unique(image.active_scalars)
        pyvista_ndarray([0, 1, 2, 3, 4])

        Split the image into its separate regions. Here, we also remove the first
        region for visualization.

        >>> multiblock = image.split_values()
        >>> _ = multiblock.pop(0)  # Remove first region

        Plot the regions.

        >>> plot = pv.Plotter()
        >>> _ = plot.add_composite(multiblock, multi_colors=True)
        >>> _ = plot.show_grid()
        >>> plot.show()

        Note that the block names are generic by default.

        >>> multiblock.keys()
        ['Block-01', 'Block-02', 'Block-03', 'Block-04']

        To name the output blocks, use a dictionary as input instead.

        Here, we also explicitly omit the region with ``0`` values from the input
        instead of removing it from the output.

        >>> labels = dict(region1=1, region2=2, region3=3, region4=4)
        >>>
        >>> multiblock = image.split_values(labels)
        >>> multiblock.keys()
        ['region1', 'region2', 'region3', 'region4']

        Plot the regions as separate meshes using the labels instead of plotting
        the MultiBlock directly.

        Clear scalar data so we can color each mesh using a single color
        >>> _ = [block.clear_data() for block in multiblock]
        >>>
        >>> plot = pv.Plotter()
        >>> plot.set_color_cycler('default')
        >>> _ = [
        ...     plot.add_mesh(block, label=label)
        ...     for block, label in zip(multiblock, labels)
        ... ]
        >>> _ = plot.add_legend()
        >>> plot.show()

        """
        if values is None and ranges is None:
            values = '_unique'  # type: ignore[assignment]
        return self.extract_values(
            values=values,
            ranges=ranges,
            scalars=scalars,
            preference=preference,
            component_mode=component_mode,
            split=True,
            **kwargs,
        )

    def extract_values(
        self,
        values: None
        | (
            float | VectorLike[float] | MatrixLike[float] | dict[str, float] | dict[float, str]
        ) = None,
        *,
        ranges: None
        | (
            VectorLike[float]
            | MatrixLike[float]
            | dict[str, VectorLike[float]]
            | dict[tuple[float, float], str]
        ) = None,
        scalars: str | None = None,
        preference: Literal['point', 'cell'] = 'point',
        component_mode: Literal['any', 'all', 'multi'] | int = 'all',
        invert: bool = False,
        adjacent_cells: bool = True,
        include_cells: bool | None = None,
        split: bool = False,
        pass_point_ids: bool = True,
        pass_cell_ids: bool = True,
        progress_bar: bool = False,
    ):
        """Return a subset of the mesh based on the value(s) of point or cell data.

        Points and cells may be extracted with a single value, multiple values, a range
        of values, or any mix of values and ranges. This enables threshold-like
        filtering of data in a discontinuous manner to extract a single label or groups
        of labels from categorical data, or to extract multiple regions from continuous
        data. Extracted values may optionally be split into separate meshes.

        This filter operates on point data and cell data distinctly:

        **Point data**

            All cells with at least one point with the specified value(s) are returned.
            Optionally, set ``adjacent_cells`` to ``False`` to only extract points from
            cells where all points in the cell strictly have the specified value(s).
            In these cases, a point is only included in the output if that point is part
            of an extracted cell.

            Alternatively, set ``include_cells`` to ``False`` to exclude cells from
            the operation completely and extract all points with a specified value.

        **Cell Data**

            Only the cells (and their points) with the specified values(s) are included
            in the output.

        Internally, :meth:`~pyvista.DataSetFilters.extract_points` is called to extract
        points for point data, and :meth:`~pyvista.DataSetFilters.extract_cells` is
        called to extract cells for cell data.

        By default, two arrays are included with the output: ``'vtkOriginalPointIds'``
        and ``'vtkOriginalCellIds'``. These arrays can be used to link the filtered
        points or cells directly to the input.

        .. versionadded:: 0.44

        Parameters
        ----------
        values : number | array_like | dict, optional
            Value(s) to extract. Can be a number, an iterable of numbers, or a dictionary
            with numeric entries. For ``dict`` inputs, either its keys or values may be
            numeric, and the other field must be strings. The numeric field is used as
            the input for this parameter, and if ``split`` is ``True``, the string field
            is used to set the block names of the returned :class:`~pyvista.MultiBlock`.

            .. note::
                When extracting multi-component values with ``component_mode=multi``,
                each value is specified as a multi-component scalar. In this case,
                ``values`` can be a single vector or an array of row vectors.

        ranges : array_like | dict, optional
            Range(s) of values to extract. Can be a single range (i.e. a sequence of
            two numbers in the form ``[lower, upper]``), a sequence of ranges, or a
            dictionary with range entries. Any combination of ``values`` and ``ranges``
            may be specified together. The endpoints of the ranges are included in the
            extraction. Ranges cannot be set when ``component_mode=multi``.

            For ``dict`` inputs, either its keys or values may be numeric, and the other
            field must be strings. The numeric field is used as the input for this
            parameter, and if ``split`` is ``True``, the string field is used to set the
            block names of the returned :class:`~pyvista.MultiBlock`.

            .. note::
                Use ``+/-`` infinity to specify an unlimited bound, e.g.:

                - ``[0, float('inf')]`` to extract values greater than or equal to zero.
                - ``[float('-inf'), 0]`` to extract values less than or equal to zero.

        scalars : str, optional
            Name of scalars to extract with. Defaults to currently active scalars.

        preference : str, default: 'point'
            When ``scalars`` is specified, this is the preferred array type to search
            for in the dataset.  Must be either ``'point'`` or ``'cell'``.

        component_mode : int | 'any' | 'all' | 'multi', default: 'all'
            Specify the component(s) to use when ``scalars`` is a multi-component array.
            Has no effect when the scalars have a single component. Must be one of:

            - number: specify the component number as a 0-indexed integer. The selected
              component must have the specified value(s).
            - ``'any'``: any single component can have the specified value(s).
            - ``'all'``: all individual components must have the specified values(s).
            - ``'multi'``: the entire multi-component item must have the specified value.

        invert : bool, default: False
            Invert the extraction values. If ``True`` extract the points (with cells)
            which do *not* have the specified values.

        adjacent_cells : bool, default: True
            If ``True``, include cells (and their points) that contain at least one of
            the extracted points. If ``False``, only include cells that contain
            exclusively points from the extracted points list. Has no effect if
            ``include_cells`` is ``False``. Has no effect when extracting values from
            cell data.

        include_cells : bool, default: None
            Specify if cells shall be used for extraction or not. If ``False``, points
            with the specified values are extracted regardless of their cell
            connectivity, and all cells at the output will be vertex cells (one for each
            point.) Has no effect when extracting values from cell data.

            By default, this value is ``True`` if the input has at least one cell and
            ``False`` otherwise.

        split : bool, default: False
            If ``True``, each value in ``values`` and each range in ``range`` is
            extracted independently and returned as a :class:`~pyvista.MultiBlock`.
            The number of blocks returned equals the number of input values and ranges.
            The blocks may be named if a dictionary is used as input. See ``values``
            and ``ranges`` for details.

            .. note::
                Output blocks may contain empty meshes if no values meet the extraction
                criteria. This can impact plotting since empty meshes cannot be plotted
                by default. Use :meth:`pyvista.MultiBlock.clean` on the output to remove
                empty meshes, or set ``pv.global_theme.allow_empty_mesh = True`` to
                enable plotting empty meshes.

        pass_point_ids : bool, default: True
            Add a point array ``'vtkOriginalPointIds'`` that identifies the original
            points the extracted points correspond to.

        pass_cell_ids : bool, default: True
            Add a cell array ``'vtkOriginalCellIds'`` that identifies the original cells
            the extracted cells correspond to.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        See Also
        --------
        split_values, extract_points, extract_cells, threshold, partition

        Returns
        -------
        pyvista.UnstructuredGrid or pyvista.MultiBlock
            An extracted mesh or a composite of extracted meshes, depending on ``split``.

        Examples
        --------
        Extract a single value from a grid's point data.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> extracted = mesh.extract_values(0)

        Plot extracted values. Since adjacent cells are included by default, points with
        values other than ``0`` are included in the output.

        >>> extracted.get_data_range()
        (np.float64(0.0), np.float64(81.0))
        >>> extracted.plot()

        Set ``include_cells=False`` to only extract points. The output scalars now
        strictly contain zeros.

        >>> extracted = mesh.extract_values(0, include_cells=False)
        >>> extracted.get_data_range()
        (np.float64(0.0), np.float64(0.0))
        >>> extracted.plot(render_points_as_spheres=True, point_size=100)

        Use ``ranges`` to extract values from a grid's point data in range.

        Here, we use ``+/-`` infinity to extract all values of ``100`` or less.

        >>> extracted = mesh.extract_values(ranges=[-np.inf, 100])
        >>> extracted.plot()

        Extract every third cell value from cell data.

        >>> mesh = examples.load_hexbeam()
        >>> lower, upper = mesh.get_data_range()
        >>> step = 3
        >>> extracted = mesh.extract_values(
        ...     range(lower, upper, step)  # values 0, 3, 6, ...
        ... )

        Plot result and show an outline of the input for context.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(extracted)
        >>> _ = pl.add_mesh(mesh.extract_all_edges())
        >>> pl.show()

        Any combination of values and ranges may be specified.

        E.g. extract a single value and two ranges, and split the result into separate
        blocks of a MultiBlock.

        >>> extracted = mesh.extract_values(
        ...     values=18, ranges=[[0, 8], [29, 40]], split=True
        ... )
        >>> extracted
        MultiBlock (...)
          N Blocks    3
          X Bounds    0.000, 1.000
          Y Bounds    0.000, 1.000
          Z Bounds    0.000, 5.000
        >>> extracted.plot(multi_colors=True)

        Extract values from multi-component scalars.

        First, create a point cloud with a 3-component RGB color array.

        >>> rng = np.random.default_rng(seed=1)
        >>> points = rng.random((30, 3))
        >>> colors = rng.random((30, 3))
        >>> point_cloud = pv.PointSet(points)
        >>> point_cloud['colors'] = colors
        >>> plot_kwargs = dict(
        ...     render_points_as_spheres=True, point_size=50, rgb=True
        ... )
        >>> point_cloud.plot(**plot_kwargs)

        Extract values from a single component.

        E.g. extract points with a strong red component (i.e. > 0.8).

        >>> extracted = point_cloud.extract_values(
        ...     ranges=[0.8, 1.0], component_mode=0
        ... )
        >>> extracted.plot(**plot_kwargs)

        Extract values from all components.

        E.g. extract points where all RGB components are dark (i.e. < 0.5).

        >>> extracted = point_cloud.extract_values(
        ...     ranges=[0.0, 0.5], component_mode='all'
        ... )
        >>> extracted.plot(**plot_kwargs)

        Extract specific multi-component values.

        E.g. round the scalars to create binary RGB components, and extract only green
        and blue components.

        >>> point_cloud['colors'] = np.round(point_cloud['colors'])
        >>> green = [0, 1, 0]
        >>> blue = [0, 0, 1]
        >>>
        >>> extracted = point_cloud.extract_values(
        ...     values=[blue, green],
        ...     component_mode='multi',
        ... )
        >>> extracted.plot(**plot_kwargs)

        Use the original IDs returned by the extraction to modify the original point
        cloud.

        For example, change the color of the blue and green points to yellow.

        >>> point_ids = extracted['vtkOriginalPointIds']
        >>> yellow = [1, 1, 0]
        >>> point_cloud['colors'][point_ids] = yellow
        >>> point_cloud.plot(**plot_kwargs)

        """

        def _validate_scalar_array(scalars_, preference_):
            # Get the scalar array and field association to use for extraction
            try:
                if scalars_ is None:
                    set_default_active_scalars(self)
                    _, scalars_ = self.active_scalars_info
                array_ = get_array(self, scalars_, preference=preference_, err=True)
            except MissingDataError:
                raise ValueError(
                    'No point data or cell data found. Scalar data is required to use this filter.',
                )
            except KeyError:
                raise ValueError(
                    f'Array name \'{scalars_}\' is not valid and does not exist with this dataset.',
                )
            association_ = get_array_association(self, scalars_, preference=preference_)
            return array_, association_

        def _validate_component_mode(array_, component_mode_):
            # Validate component mode and return logic function
            num_components = 1 if array_.ndim == 1 else array_.shape[1]
            if isinstance(component_mode_, (int, np.integer)) or component_mode_ in ['0', '1', '2']:
                component_mode_ = int(component_mode_)
                if component_mode_ > num_components - 1 or component_mode_ < 0:
                    raise ValueError(
                        f"Invalid component index '{component_mode_}' specified for scalars with {num_components} component(s). Value must be one of: {tuple(range(num_components))}.",
                    )
                array_ = array_[:, component_mode_] if num_components > 1 else array_
                component_logic_function = None
            elif isinstance(component_mode_, str) and component_mode_ in ['any', 'all', 'multi']:
                if array_.ndim == 1:
                    component_logic_function = None
                elif component_mode_ == 'any':
                    component_logic_function = functools.partial(np.any, axis=1)
                elif component_mode_ in ['all', 'multi']:
                    component_logic_function = functools.partial(np.all, axis=1)
            else:
                raise ValueError(
                    f"Invalid component '{component_mode_}'. Must be an integer, 'any', 'all', or 'multi'.",
                )
            return array_, num_components, component_logic_function

        def _get_inputs_from_dict(input_):
            # Get extraction values from dict if applicable.
            # If dict, also validate names/labels mapped to the values
            if not isinstance(input_, dict):
                return None, input_
            else:
                dict_keys, dict_values = list(input_.keys()), list(input_.values())
                if all(isinstance(key, str) for key in dict_keys):
                    return dict_keys, dict_values
                elif all(isinstance(val, str) for val in dict_values):
                    return dict_values, dict_keys
                else:
                    raise TypeError(
                        "Invalid dict mapping. The dict's keys or values must contain strings.",
                    )

        def _validate_values_and_ranges(array_, values_, ranges_, num_components_, component_mode_):
            # Make sure we have input values to extract
            is_multi_mode = component_mode_ == 'multi'
            if values_ is None:
                if ranges_ is None:
                    raise TypeError(
                        'No ranges or values were specified. At least one must be specified.',
                    )
                elif is_multi_mode:
                    raise TypeError(
                        f"Ranges cannot be extracted using component mode '{component_mode_}'. Expected {None}, got {ranges_}.",
                    )
            elif (
                isinstance(values_, str) and values_ == '_unique'
            ):  # Private flag used by `split_values` to use unique values
                axis = 0 if is_multi_mode else None
                values_ = np.unique(array_, axis=axis)

            # Validate values
            if values_ is not None:
                if is_multi_mode:
                    values_ = np.atleast_2d(values_)
                    if values_.ndim > 2:
                        raise ValueError(
                            f'Component values cannot be more than 2 dimensions. Got {values_.ndim}.',
                        )
                    if not values_.shape[1] == num_components_:
                        raise ValueError(
                            f'Num components in values array ({values_.shape[1]}) must match num components in data array ({num_components_}).',
                        )
                else:
                    values_ = np.atleast_1d(values_)
                    if values_.ndim > 1:
                        raise ValueError(
                            f'Values must be one-dimensional. Got {values_.ndim}d values.',
                        )
                if not (
                    np.issubdtype(dtype := values_.dtype, np.floating)
                    or np.issubdtype(dtype, np.integer)
                ):
                    raise TypeError('Values must be numeric.')

            # Validate ranges
            if ranges_ is not None:
                ranges_ = np.atleast_2d(ranges_)
                if (ndim := ranges_.ndim) > 2:
                    raise ValueError(f'Ranges must be 2 dimensional. Got {ndim}.')
                if not (
                    np.issubdtype(dtype := ranges_.dtype, np.floating)
                    or np.issubdtype(dtype, np.integer)
                ):
                    raise TypeError('Ranges must be numeric.')
                is_valid_range = ranges_[:, 0] <= ranges_[:, 1]
                not_valid = np.invert(is_valid_range)
                if np.any(not_valid):
                    invalid_ranges = ranges_[not_valid]
                    raise ValueError(
                        f'Invalid range {invalid_ranges[0]} specified. Lower value cannot be greater than upper value.',
                    )
            return values_, ranges_

        # Return empty mesh if input is empty mesh
        if self.n_points == 0:  # type: ignore[attr-defined]
            return self.copy()  # type: ignore[attr-defined]

        array, association = _validate_scalar_array(scalars, preference)
        array, num_components, component_logic = _validate_component_mode(array, component_mode)
        value_names, values = _get_inputs_from_dict(values)
        range_names, ranges = _get_inputs_from_dict(ranges)
        valid_values, valid_ranges = _validate_values_and_ranges(
            array,
            values,
            ranges,
            num_components,
            component_mode,
        )

        # Set default for include cells
        if include_cells is None:
            include_cells = self.n_cells > 0  # type: ignore[attr-defined]

        kwargs = dict(
            array=array,
            association=association,
            component_logic=component_logic,
            invert=invert,
            adjacent_cells=adjacent_cells,
            include_cells=include_cells,
            pass_point_ids=pass_point_ids,
            pass_cell_ids=pass_cell_ids,
            progress_bar=progress_bar,
        )

        if split:
            multi = pyvista.MultiBlock()
            # Split values and ranges separately and combine into single multiblock
            if values is not None:
                value_names = value_names if value_names else [None] * len(valid_values)
                for (
                    name,
                    val,
                ) in zip(value_names, valid_values):
                    multi.append(self._extract_values(values=[val], **kwargs), name)
            if ranges is not None:
                range_names = range_names if range_names else [None] * len(valid_ranges)
                for (
                    name,
                    rng,
                ) in zip(range_names, valid_ranges):
                    multi.append(self._extract_values(ranges=[rng], **kwargs), name)
            return multi

        return DataSetFilters._extract_values(
            self,
            values=valid_values,
            ranges=valid_ranges,
            **kwargs,
        )

    def _extract_values(
        self,
        values=None,
        ranges=None,
        *,
        array,
        association,
        component_logic,
        invert,
        adjacent_cells,
        include_cells,
        progress_bar,
        pass_point_ids,
        pass_cell_ids,
    ):
        """Extract values using validated input.

        Internal method for extract_values filter to avoid repeated calls to input
        validation methods.
        """

        def _update_id_mask(logic_):
            """Apply component logic and update the id mask."""
            logic_ = component_logic(logic_) if component_logic else logic_
            id_mask[logic_] = True

        # Determine which ids to keep
        id_mask = np.zeros((len(array),), dtype=np.bool_)
        if values is not None:
            for val in values:
                logic = array == val
                _update_id_mask(logic)

        if ranges is not None:
            for lower, upper in ranges:
                finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
                if finite_lower and finite_upper:
                    logic = np.logical_and(array >= lower, array <= upper)
                elif not finite_lower and finite_upper:
                    logic = array <= upper
                elif finite_lower and not finite_upper:
                    logic = array >= lower
                else:
                    # Extract all
                    logic = np.ones_like(array, dtype=np.bool_)
                _update_id_mask(logic)

        id_mask = np.invert(id_mask) if invert else id_mask

        # Extract point or cell ids
        if association == FieldAssociation.POINT:
            output = self.extract_points(
                id_mask,
                adjacent_cells=adjacent_cells,
                include_cells=include_cells,
                progress_bar=progress_bar,
            )
        else:
            output = self.extract_cells(
                id_mask,
                progress_bar=progress_bar,
            )

        # Process output arrays
        if (POINT_IDS := 'vtkOriginalPointIds') in output.point_data and not pass_point_ids:
            output.point_data.remove(POINT_IDS)
        if (CELL_IDS := 'vtkOriginalCellIds') in output.cell_data and not pass_cell_ids:
            output.cell_data.remove(CELL_IDS)

        return output

    def extract_surface(
        self,
        pass_pointid=True,
        pass_cellid=True,
        nonlinear_subdivision=1,
        progress_bar=False,
    ):
        """Extract surface mesh of the grid.

        Parameters
        ----------
        pass_pointid : bool, default: True
            Adds a point array ``"vtkOriginalPointIds"`` that
            identifies which original points these surface points
            correspond to.

        pass_cellid : bool, default: True
            Adds a cell array ``"vtkOriginalCellIds"`` that
            identifies which original cells these surface cells
            correspond to.

        nonlinear_subdivision : int, default: 1
            If the input is an unstructured grid with nonlinear faces,
            this parameter determines how many times the face is
            subdivided into linear faces.

            If 0, the output is the equivalent of its linear
            counterpart (and the midpoints determining the nonlinear
            interpolation are discarded). If 1 (the default), the
            nonlinear face is triangulated based on the midpoints. If
            greater than 1, the triangulated pieces are recursively
            subdivided to reach the desired subdivision. Setting the
            value to greater than 1 may cause some point data to not
            be passed even if no nonlinear faces exist. This option
            has no effect if the input is not an unstructured grid.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Surface mesh of the grid.

        Warnings
        --------
        Both ``"vtkOriginalPointIds"`` and ``"vtkOriginalCellIds"`` may be
        affected by other VTK operations. See `issue 1164
        <https://github.com/pyvista/pyvista/issues/1164>`_ for
        recommendations on tracking indices across operations.

        Examples
        --------
        Extract the surface of an UnstructuredGrid.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_hexbeam()
        >>> surf = grid.extract_surface()
        >>> type(surf)
        <class 'pyvista.core.pointset.PolyData'>
        >>> surf["vtkOriginalPointIds"]
        pyvista_ndarray([ 0,  2, 36, 27,  7,  8, 81,  1, 18,  4, 54,  3,  6, 45,
                         72,  5, 63,  9, 35, 44, 11, 16, 89, 17, 10, 26, 62, 13,
                         12, 53, 80, 15, 14, 71, 19, 37, 55, 20, 38, 56, 21, 39,
                         57, 22, 40, 58, 23, 41, 59, 24, 42, 60, 25, 43, 61, 28,
                         82, 29, 83, 30, 84, 31, 85, 32, 86, 33, 87, 34, 88, 46,
                         73, 47, 74, 48, 75, 49, 76, 50, 77, 51, 78, 52, 79, 64,
                         65, 66, 67, 68, 69, 70])
        >>> surf["vtkOriginalCellIds"]
        pyvista_ndarray([ 0,  0,  0,  1,  1,  1,  3,  3,  3,  2,  2,  2, 36, 36,
                         36, 37, 37, 37, 39, 39, 39, 38, 38, 38,  5,  5,  9,  9,
                         13, 13, 17, 17, 21, 21, 25, 25, 29, 29, 33, 33,  4,  4,
                          8,  8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28, 32, 32,
                          7,  7, 11, 11, 15, 15, 19, 19, 23, 23, 27, 27, 31, 31,
                         35, 35,  6,  6, 10, 10, 14, 14, 18, 18, 22, 22, 26, 26,
                         30, 30, 34, 34])

        Note that in the "vtkOriginalCellIds" array, the same original cells
        appears multiple times since this array represents the original cell of
        each surface cell extracted.

        See the :ref:`extract_surface_example` for more examples using this filter.

        """
        surf_filter = _vtk.vtkDataSetSurfaceFilter()
        surf_filter.SetInputData(self)
        surf_filter.SetPassThroughPointIds(pass_pointid)
        surf_filter.SetPassThroughCellIds(pass_cellid)

        if nonlinear_subdivision != 1:
            surf_filter.SetNonlinearSubdivisionLevel(nonlinear_subdivision)

        # available in 9.0.2
        # surf_filter.SetDelegation(delegation)

        _update_alg(surf_filter, progress_bar, 'Extracting Surface')
        return _get_output(surf_filter)

    def surface_indices(self, progress_bar=False):
        """Return the surface indices of a grid.

        Parameters
        ----------
        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        numpy.ndarray
            Indices of the surface points.

        Examples
        --------
        Return the first 10 surface indices of an UnstructuredGrid.

        >>> from pyvista import examples
        >>> grid = examples.load_hexbeam()
        >>> ind = grid.surface_indices()
        >>> ind[:10]  # doctest:+SKIP
        pyvista_ndarray([ 0,  2, 36, 27,  7,  8, 81,  1, 18,  4])

        """
        surf = DataSetFilters.extract_surface(self, pass_cellid=True, progress_bar=progress_bar)
        return surf.point_data['vtkOriginalPointIds']

    def extract_feature_edges(
        self,
        feature_angle=30.0,
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=True,
        manifold_edges=True,
        clear_data=False,
        progress_bar=False,
    ):
        """Extract edges from the surface of the mesh.

        If the given mesh is not PolyData, the external surface of the given
        mesh is extracted and used.

        From vtk documentation, the edges are one of the following:

            1) Boundary (used by one polygon) or a line cell.
            2) Non-manifold (used by three or more polygons).
            3) Feature edges (edges used by two triangles and whose
               dihedral angle > feature_angle).
            4) Manifold edges (edges used by exactly two polygons).

        Parameters
        ----------
        feature_angle : float, default: 30.0
            Feature angle (in degrees) used to detect sharp edges on
            the mesh. Used only when ``feature_edges=True``.

        boundary_edges : bool, default: True
            Extract the boundary edges.

        non_manifold_edges : bool, default: True
            Extract non-manifold edges.

        feature_edges : bool, default: True
            Extract edges exceeding ``feature_angle``.

        manifold_edges : bool, default: True
            Extract manifold edges.

        clear_data : bool, default: False
            Clear any point, cell, or field data. This is useful
            if wanting to strictly extract the edges.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Extracted edges.

        Examples
        --------
        Extract the edges from an unstructured grid.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hex_beam = pv.read(examples.hexbeamfile)
        >>> feat_edges = hex_beam.extract_feature_edges()
        >>> feat_edges.clear_data()  # clear array data for plotting
        >>> feat_edges.plot(line_width=10)

        See the :ref:`extract_edges_example` for more examples using this filter.

        """
        dataset = self
        if not isinstance(dataset, _vtk.vtkPolyData):
            dataset = DataSetFilters.extract_surface(dataset)
        featureEdges = _vtk.vtkFeatureEdges()
        featureEdges.SetInputData(dataset)
        featureEdges.SetFeatureAngle(feature_angle)
        featureEdges.SetManifoldEdges(manifold_edges)
        featureEdges.SetNonManifoldEdges(non_manifold_edges)
        featureEdges.SetBoundaryEdges(boundary_edges)
        featureEdges.SetFeatureEdges(feature_edges)
        featureEdges.SetColoring(False)
        _update_alg(featureEdges, progress_bar, 'Extracting Feature Edges')
        output = _get_output(featureEdges)
        if clear_data:
            output.clear_data()
        return output

    def merge(
        self,
        grid=None,
        merge_points=True,
        tolerance=0.0,
        inplace=False,
        main_has_priority=True,
        progress_bar=False,
    ):
        """Join one or many other grids to this grid.

        Grid is updated in-place by default.

        Can be used to merge points of adjacent cells when no grids
        are input.

        .. note::
           The ``+`` operator between two meshes uses this filter with
           the default parameters. When the target mesh is already a
           :class:`pyvista.UnstructuredGrid`, in-place merging via
           ``+=`` is similarly possible.

        Parameters
        ----------
        grid : vtk.UnstructuredGrid or list of vtk.UnstructuredGrids, optional
            Grids to merge to this grid.

        merge_points : bool, default: True
            Points in exactly the same location will be merged between
            the two meshes. Warning: this can leave degenerate point data.

        tolerance : float, default: 0.0
            The absolute tolerance to use to find coincident points when
            ``merge_points=True``.

        inplace : bool, default: False
            Updates grid inplace when True if the input type is an
            :class:`pyvista.UnstructuredGrid`.

        main_has_priority : bool, default: True
            When this parameter is true and merge_points is true,
            the arrays of the merging grids will be overwritten
            by the original main mesh.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UnstructuredGrid
            Merged grid.

        Notes
        -----
        When two or more grids are joined, the type and name of each
        array must match or the arrays will be ignored and not
        included in the final merged mesh.

        Examples
        --------
        Merge three separate spheres into a single mesh.

        >>> import pyvista as pv
        >>> sphere_a = pv.Sphere(center=(1, 0, 0))
        >>> sphere_b = pv.Sphere(center=(0, 1, 0))
        >>> sphere_c = pv.Sphere(center=(0, 0, 1))
        >>> merged = sphere_a.merge([sphere_b, sphere_c])
        >>> merged.plot()

        """
        append_filter = _vtk.vtkAppendFilter()
        append_filter.SetMergePoints(merge_points)
        append_filter.SetTolerance(tolerance)

        if not main_has_priority:
            append_filter.AddInputData(self)

        if isinstance(grid, pyvista.DataSet):
            append_filter.AddInputData(grid)
        elif isinstance(grid, (list, tuple, pyvista.MultiBlock)):
            grids = grid
            for grid in grids:
                append_filter.AddInputData(grid)

        if main_has_priority:
            append_filter.AddInputData(self)

        _update_alg(append_filter, progress_bar, 'Merging')
        merged = _get_output(append_filter)
        if inplace:
            if type(self) is type(merged):
                self.deep_copy(merged)
                return self
            else:
                raise TypeError(f"Mesh type {type(self)} cannot be overridden by output.")
        return merged

    def __add__(self, dataset):
        """Combine this mesh with another into a :class:`pyvista.UnstructuredGrid`."""
        return DataSetFilters.merge(self, dataset)

    def __iadd__(self, dataset):
        """Merge another mesh into this one if possible.

        "If possible" means that ``self`` is a :class:`pyvista.UnstructuredGrid`.
        Otherwise we have to return a new object, and the attempted in-place
        merge will raise.

        """
        try:
            merged = DataSetFilters.merge(self, dataset, inplace=True)
        except TypeError:
            raise TypeError(
                'In-place merge only possible if the target mesh '
                'is an UnstructuredGrid.\nPlease use `mesh + other_mesh` '
                'instead, which returns a new UnstructuredGrid.',
            ) from None
        return merged

    def compute_cell_quality(
        self,
        quality_measure='scaled_jacobian',
        null_value=-1.0,
        progress_bar=False,
    ):
        """Compute a function of (geometric) quality for each cell of a mesh.

        The per-cell quality is added to the mesh's cell data, in an
        array named ``"CellQuality"``. Cell types not supported by this
        filter or undefined quality of supported cell types will have an
        entry of -1.

        Defaults to computing the scaled Jacobian.

        Options for cell quality measure:

        - ``'area'``
        - ``'aspect_beta'``
        - ``'aspect_frobenius'``
        - ``'aspect_gamma'``
        - ``'aspect_ratio'``
        - ``'collapse_ratio'``
        - ``'condition'``
        - ``'diagonal'``
        - ``'dimension'``
        - ``'distortion'``
        - ``'jacobian'``
        - ``'max_angle'``
        - ``'max_aspect_frobenius'``
        - ``'max_edge_ratio'``
        - ``'med_aspect_frobenius'``
        - ``'min_angle'``
        - ``'oddy'``
        - ``'radius_ratio'``
        - ``'relative_size_squared'``
        - ``'scaled_jacobian'``
        - ``'shape'``
        - ``'shape_and_size'``
        - ``'shear'``
        - ``'shear_and_size'``
        - ``'skew'``
        - ``'stretch'``
        - ``'taper'``
        - ``'volume'``
        - ``'warpage'``

        Notes
        -----
        There is a `discussion about shape option <https://github.com/pyvista/pyvista/discussions/6143>`_.

        Parameters
        ----------
        quality_measure : str, default: 'scaled_jacobian'
            The cell quality measure to use.

        null_value : float, default: -1.0
            Float value for undefined quality. Undefined quality are qualities
            that could be addressed by this filter but is not well defined for
            the particular geometry of cell in question, e.g. a volume query
            for a triangle. Undefined quality will always be undefined.
            The default value is -1.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset with the computed mesh quality in the
            ``cell_data`` as the ``"CellQuality"`` array.

        Examples
        --------
        Compute and plot the minimum angle of a sample sphere mesh.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
        >>> cqual = sphere.compute_cell_quality('min_angle')
        >>> cqual.plot(show_edges=True)

        See the :ref:`mesh_quality_example` for more examples using this filter.

        """
        alg = _vtk.vtkCellQuality()
        possible_measure_setters = {
            'area': 'SetQualityMeasureToArea',
            'aspect_beta': 'SetQualityMeasureToAspectBeta',
            'aspect_frobenius': 'SetQualityMeasureToAspectFrobenius',
            'aspect_gamma': 'SetQualityMeasureToAspectGamma',
            'aspect_ratio': 'SetQualityMeasureToAspectRatio',
            'collapse_ratio': 'SetQualityMeasureToCollapseRatio',
            'condition': 'SetQualityMeasureToCondition',
            'diagonal': 'SetQualityMeasureToDiagonal',
            'dimension': 'SetQualityMeasureToDimension',
            'distortion': 'SetQualityMeasureToDistortion',
            'jacobian': 'SetQualityMeasureToJacobian',
            'max_angle': 'SetQualityMeasureToMaxAngle',
            'max_aspect_frobenius': 'SetQualityMeasureToMaxAspectFrobenius',
            'max_edge_ratio': 'SetQualityMeasureToMaxEdgeRatio',
            'med_aspect_frobenius': 'SetQualityMeasureToMedAspectFrobenius',
            'min_angle': 'SetQualityMeasureToMinAngle',
            'oddy': 'SetQualityMeasureToOddy',
            'radius_ratio': 'SetQualityMeasureToRadiusRatio',
            'relative_size_squared': 'SetQualityMeasureToRelativeSizeSquared',
            'scaled_jacobian': 'SetQualityMeasureToScaledJacobian',
            'shape': 'SetQualityMeasureToShape',
            'shape_and_size': 'SetQualityMeasureToShapeAndSize',
            'shear': 'SetQualityMeasureToShear',
            'shear_and_size': 'SetQualityMeasureToShearAndSize',
            'skew': 'SetQualityMeasureToSkew',
            'stretch': 'SetQualityMeasureToStretch',
            'taper': 'SetQualityMeasureToTaper',
            'volume': 'SetQualityMeasureToVolume',
            'warpage': 'SetQualityMeasureToWarpage',
        }

        # we need to check if these quality measures exist as VTK API changes
        measure_setters = {}
        for name, attr in possible_measure_setters.items():
            setter_candidate = getattr(alg, attr, None)
            if setter_candidate:
                measure_setters[name] = setter_candidate

        try:
            # Set user specified quality measure
            measure_setters[quality_measure]()
        except (KeyError, IndexError):
            options = ', '.join([f"'{s}'" for s in list(measure_setters.keys())])
            raise KeyError(
                f'Cell quality type ({quality_measure}) not available. Options are: {options}',
            )
        alg.SetInputData(self)
        alg.SetUndefinedQuality(null_value)
        _update_alg(alg, progress_bar, 'Computing Cell Quality')
        return _get_output(alg)

    def compute_boundary_mesh_quality(self, *, progress_bar=False):
        """Compute metrics on the boundary faces of a mesh.

        The metrics that can be computed on the boundary faces of the mesh and are:

        - Distance from cell center to face center
        - Distance from cell center to face plane
        - Angle of faces plane normal and cell center to face center vector

        Parameters
        ----------
        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset with the computed metrics on the boundary faces of a mesh.
            ``cell_data`` as the ``"CellQuality"`` array.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_can_crushed_vtu()
        >>> cqual = mesh.compute_boundary_mesh_quality()
        >>> plotter = pv.Plotter(shape=(2, 2))
        >>> _ = plotter.add_mesh(mesh, show_edges=True)
        >>> plotter.subplot(1, 0)
        >>> _ = plotter.add_mesh(
        ...     cqual, scalars="DistanceFromCellCenterToFaceCenter"
        ... )
        >>> plotter.subplot(0, 1)
        >>> _ = plotter.add_mesh(
        ...     cqual, scalars="DistanceFromCellCenterToFacePlane"
        ... )
        >>> plotter.subplot(1, 1)
        >>> _ = plotter.add_mesh(
        ...     cqual,
        ...     scalars="AngleFaceNormalAndCellCenterToFaceCenterVector",
        ... )
        >>> plotter.show()

        """
        if pyvista.vtk_version_info < (9, 3, 0):
            raise VTKVersionError(
                '`vtkBoundaryMeshQuality` requires vtk>=9.3.0',
            )  # pragma: no cover
        alg = _vtk.vtkBoundaryMeshQuality()
        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Compute Boundary Mesh Quality')
        return _get_output(alg)

    def compute_derivative(
        self,
        scalars=None,
        gradient=True,
        divergence=None,
        vorticity=None,
        qcriterion=None,
        faster=False,
        preference='point',
        progress_bar=False,
    ):
        """Compute derivative-based quantities of point/cell scalar field.

        Utilize ``vtkGradientFilter`` to compute derivative-based quantities,
        such as gradient, divergence, vorticity, and Q-criterion, of the
        selected point or cell scalar field.

        Parameters
        ----------
        scalars : str, optional
            String name of the scalars array to use when computing the
            derivative quantities.  Defaults to the active scalars in
            the dataset.

        gradient : bool | str, default: True
            Calculate gradient. If a string is passed, the string will be used
            for the resulting array name. Otherwise, array name will be
            ``'gradient'``. Default ``True``.

        divergence : bool | str, optional
            Calculate divergence. If a string is passed, the string will be
            used for the resulting array name. Otherwise, default array name
            will be ``'divergence'``.

        vorticity : bool | str, optional
            Calculate vorticity. If a string is passed, the string will be used
            for the resulting array name. Otherwise, default array name will be
            ``'vorticity'``.

        qcriterion : bool | str, optional
            Calculate qcriterion. If a string is passed, the string will be
            used for the resulting array name. Otherwise, default array name
            will be ``'qcriterion'``.

        faster : bool, default: False
            Use faster algorithm for computing derivative quantities. Result is
            less accurate and performs fewer derivative calculations,
            increasing computation speed. The error will feature smoothing of
            the output and possibly errors at boundaries. Option has no effect
            if DataSet is not :class:`pyvista.UnstructuredGrid`.

        preference : str, default: "point"
            Data type preference. Either ``'point'`` or ``'cell'``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset with calculated derivative.

        Examples
        --------
        First, plot the random hills dataset with the active elevation
        scalars.  These scalars will be used for the derivative
        calculations.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> hills.plot(smooth_shading=True)

        Compute and plot the gradient of the active scalars.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> deriv = hills.compute_derivative()
        >>> deriv.plot(scalars='gradient')

        See the :ref:`gradients_example` for more examples using this filter.

        """
        alg = _vtk.vtkGradientFilter()
        # Check if scalars array given
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
        if not isinstance(scalars, str):
            raise TypeError('scalars array must be given as a string name')
        if not any((gradient, divergence, vorticity, qcriterion)):
            raise ValueError(
                'must set at least one of gradient, divergence, vorticity, or qcriterion',
            )

            # bool(non-empty string/True) == True, bool(None/False) == False
        alg.SetComputeGradient(bool(gradient))
        if isinstance(gradient, bool):
            gradient = 'gradient'
        alg.SetResultArrayName(gradient)

        alg.SetComputeDivergence(bool(divergence))
        if isinstance(divergence, bool):
            divergence = 'divergence'
        alg.SetDivergenceArrayName(divergence)

        alg.SetComputeVorticity(bool(vorticity))
        if isinstance(vorticity, bool):
            vorticity = 'vorticity'
        alg.SetVorticityArrayName(vorticity)

        alg.SetComputeQCriterion(bool(qcriterion))
        if isinstance(qcriterion, bool):
            qcriterion = 'qcriterion'
        alg.SetQCriterionArrayName(qcriterion)

        alg.SetFasterApproximation(faster)
        field = get_array_association(self, scalars, preference=preference)
        # args: (idx, port, connection, field, name)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Computing Derivative')
        return _get_output(alg)

    def shrink(self, shrink_factor=1.0, progress_bar=False):
        """Shrink the individual faces of a mesh.

        This filter shrinks the individual faces of a mesh rather than
        scaling the entire mesh.

        Parameters
        ----------
        shrink_factor : float, default: 1.0
            Fraction of shrink for each cell. Default does not modify the
            faces.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset with shrunk faces.  Return type matches input.

        Examples
        --------
        First, plot the original cube.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.plot(show_edges=True, line_width=5)

        Now, plot the mesh with shrunk faces.

        >>> shrunk = mesh.shrink(0.5)
        >>> shrunk.clear_data()  # cleans up plot
        >>> shrunk.plot(show_edges=True, line_width=5)

        """
        if not (0.0 <= shrink_factor <= 1.0):
            raise ValueError('`shrink_factor` should be between 0.0 and 1.0')
        alg = _vtk.vtkShrinkFilter()
        alg.SetInputData(self)
        alg.SetShrinkFactor(shrink_factor)
        _update_alg(alg, progress_bar, 'Shrinking Mesh')
        output = _get_output(alg)
        if isinstance(self, _vtk.vtkPolyData):
            return output.extract_surface()
        return output

    def tessellate(self, max_n_subdivide=3, merge_points=True, progress_bar=False):
        """Tessellate a mesh.

        This filter approximates nonlinear FEM-like elements with linear
        simplices. The output mesh will have geometry and any fields specified
        as attributes in the input mesh's point data. The attribute's copy
        flags are honored, except for normals.

        For more details see `vtkTessellatorFilter <https://vtk.org/doc/nightly/html/classvtkTessellatorFilter.html#details>`_.

        Parameters
        ----------
        max_n_subdivide : int, default: 3
            Maximum number of subdivisions.

        merge_points : bool, default: True
            The adaptive tessellation will output vertices that are not shared among cells,
            even where they should be. This can be corrected to some extent.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset with tessellated mesh.  Return type matches input.

        Examples
        --------
        First, plot the high order FEM-like elements.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> points = np.array(
        ...     [
        ...         [0.0, 0.0, 0.0],
        ...         [2.0, 0.0, 0.0],
        ...         [1.0, 2.0, 0.0],
        ...         [1.0, 0.5, 0.0],
        ...         [1.5, 1.5, 0.0],
        ...         [0.5, 1.5, 0.0],
        ...     ]
        ... )
        >>> cells = np.array([6, 0, 1, 2, 3, 4, 5])
        >>> cell_types = np.array([69])
        >>> mesh = pv.UnstructuredGrid(cells, cell_types, points)
        >>> mesh.plot(show_edges=True, line_width=5)

        Now, plot the tessellated mesh.

        >>> tessellated = mesh.tessellate()
        >>> tessellated.clear_data()  # cleans up plot
        >>> tessellated.plot(show_edges=True, line_width=5)

        """
        if isinstance(self, _vtk.vtkPolyData):
            raise TypeError('Tessellate filter is not supported for PolyData objects.')
        alg = _vtk.vtkTessellatorFilter()
        alg.SetInputData(self)
        alg.SetMergePoints(merge_points)
        alg.SetMaximumNumberOfSubdivisions(max_n_subdivide)
        _update_alg(alg, progress_bar, 'Tessellating Mesh')
        return _get_output(alg)

    def transform(
        self: _vtk.vtkDataSet,
        trans: _vtk.vtkMatrix4x4 | _vtk.vtkTransform | NumpyArray[float],
        transform_all_input_vectors=False,
        inplace=True,
        progress_bar=False,
    ):
        """Transform this mesh with a 4x4 transform.

        .. warning::
            When using ``transform_all_input_vectors=True``, there is
            no distinction in VTK between vectors and arrays with
            three components.  This may be an issue if you have scalar
            data with three components (e.g. RGB data).  This will be
            improperly transformed as if it was vector data rather
            than scalar data.  One possible (albeit ugly) workaround
            is to store the three components as separate scalar
            arrays.

        .. warning::
            In general, transformations give non-integer results. This
            method converts integer-typed vector data to float before
            performing the transformation. This applies to the points
            array, as well as any vector-valued data that is affected
            by the transformation. To prevent subtle bugs arising from
            in-place transformations truncating the result to integers,
            this conversion always applies to the input mesh.

        Parameters
        ----------
        trans : vtk.vtkMatrix4x4, vtk.vtkTransform, or numpy.ndarray
            Accepts a vtk transformation object or a 4x4
            transformation matrix.

        transform_all_input_vectors : bool, default: False
            When ``True``, all arrays with three components are
            transformed. Otherwise, only the normals and vectors are
            transformed.  See the warning for more details.

        inplace : bool, default: False
            When ``True``, modifies the dataset inplace.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Transformed dataset.  Return type matches input unless
            input dataset is a :class:`pyvista.ImageData`, in which
            case the output datatype is a :class:`pyvista.StructuredGrid`.

        Examples
        --------
        Translate a mesh by ``(50, 100, 200)``.

        >>> import numpy as np
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()

        Here a 4x4 :class:`numpy.ndarray` is used, but
        ``vtk.vtkMatrix4x4`` and ``vtk.vtkTransform`` are also
        accepted.

        >>> transform_matrix = np.array(
        ...     [
        ...         [1, 0, 0, 50],
        ...         [0, 1, 0, 100],
        ...         [0, 0, 1, 200],
        ...         [0, 0, 0, 1],
        ...     ]
        ... )
        >>> transformed = mesh.transform(transform_matrix)
        >>> transformed.plot(show_edges=True)

        """
        if inplace and isinstance(self, pyvista.Grid):
            raise TypeError(f'Cannot transform a {self.__class__} inplace')

        if isinstance(trans, _vtk.vtkMatrix4x4):
            m = trans
            t = _vtk.vtkTransform()
            t.SetMatrix(m)
        elif isinstance(trans, _vtk.vtkTransform):
            t = trans
            m = trans.GetMatrix()
        elif isinstance(trans, np.ndarray):
            if trans.shape != (4, 4):
                raise ValueError('Transformation array must be 4x4')
            m = vtkmatrix_from_array(trans)
            t = _vtk.vtkTransform()
            t.SetMatrix(m)
        else:
            raise TypeError(
                'Input transform must be either:\n'
                '\tvtk.vtkMatrix4x4\n'
                '\tvtk.vtkTransform\n'
                '\t4x4 np.ndarray\n',
            )

        if m.GetElement(3, 3) == 0:
            raise ValueError("Transform element (3,3), the inverse scale term, is zero")

        # vtkTransformFilter truncates the result if the input is an integer type
        # so convert input points and relevant vectors to float
        # (creating a new copy would be harmful much more often)
        converted_ints = False
        if not np.issubdtype(self.points.dtype, np.floating):
            self.points = self.points.astype(np.float32)
            converted_ints = True
        if transform_all_input_vectors:
            # all vector-shaped data will be transformed
            point_vectors = [
                name for name, data in self.point_data.items() if data.shape == (self.n_points, 3)
            ]
            cell_vectors = [
                name for name, data in self.cell_data.items() if data.shape == (self.n_cells, 3)
            ]
        else:
            # we'll only transform active vectors and normals
            point_vectors = [
                self.point_data.active_vectors_name,
                self.point_data.active_normals_name,
            ]
            cell_vectors = [
                self.cell_data.active_vectors_name,
                self.cell_data.active_normals_name,
            ]
        # dynamically convert each self.point_data[name] etc. to float32
        all_vectors = [point_vectors, cell_vectors]
        all_dataset_attrs = [self.point_data, self.cell_data]
        for vector_names, dataset_attrs in zip(all_vectors, all_dataset_attrs):
            for vector_name in vector_names:
                if vector_name is None:
                    continue
                vector_arr = dataset_attrs[vector_name]
                if not np.issubdtype(vector_arr.dtype, np.floating):
                    dataset_attrs[vector_name] = vector_arr.astype(np.float32)
                    converted_ints = True
        if converted_ints:
            warnings.warn(
                'Integer points, vector and normal data (if any) of the input mesh '
                'have been converted to ``np.float32``. This is necessary in order '
                'to transform properly.',
            )

        # vtkTransformFilter doesn't respect active scalars.  We need to track this
        active_point_scalars_name = self.point_data.active_scalars_name
        active_cell_scalars_name = self.cell_data.active_scalars_name

        # vtkTransformFilter sometimes doesn't transform all vector arrays
        # when there are active point/cell scalars. Use this workaround
        self.active_scalars_name = None

        f = _vtk.vtkTransformFilter()
        f.SetInputDataObject(self)
        f.SetTransform(t)
        f.SetTransformAllInputVectors(transform_all_input_vectors)

        _update_alg(f, progress_bar, 'Transforming')
        res = pyvista.core.filters._get_output(f)

        # make the previously active scalars active again
        if active_point_scalars_name is not None:
            self.point_data.active_scalars_name = active_point_scalars_name
            res.point_data.active_scalars_name = active_point_scalars_name
        if active_cell_scalars_name is not None:
            self.cell_data.active_scalars_name = active_cell_scalars_name
            res.cell_data.active_scalars_name = active_cell_scalars_name

        if inplace:
            self.copy_from(res, deep=False)
            return self

        # The output from the transform filter contains a shallow copy
        # of the original dataset except for the point arrays.  Here
        # we perform a copy so the two are completely unlinked.
        if isinstance(self, pyvista.Grid):
            output: _vtk.vtkDataSet = pyvista.StructuredGrid()
        else:
            output = self.__class__()
        output.copy_from(res, deep=True)
        return output

    def reflect(
        self,
        normal,
        point=None,
        inplace=False,
        transform_all_input_vectors=False,
        progress_bar=False,
    ):
        """Reflect a dataset across a plane.

        Parameters
        ----------
        normal : array_like[float]
            Normal direction for reflection.

        point : array_like[float]
            Point which, along with ``normal``, defines the reflection
            plane. If not specified, this is the origin.

        inplace : bool, default: False
            When ``True``, modifies the dataset inplace.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise,
            only the points, normals and active vectors are transformed.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Reflected dataset.  Return type matches input.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh = mesh.reflect((0, 0, 1), point=(0, 0, -100))
        >>> mesh.plot(show_edges=True)

        See the :ref:`reflect_example` for more examples using this filter.

        """
        t = transformations.reflection(normal, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
            progress_bar=progress_bar,
        )

    def integrate_data(self, progress_bar=False):
        """Integrate point and cell data.

        Area or volume is also provided in point data.

        This filter uses the VTK `vtkIntegrateAttributes
        <https://vtk.org/doc/nightly/html/classvtkIntegrateAttributes.html>`_
        and requires VTK v9.1.0 or newer.

        Parameters
        ----------
        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UnstructuredGrid
            Mesh with 1 point and 1 vertex cell with integrated data in point
            and cell data.

        Examples
        --------
        Integrate data on a sphere mesh.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> sphere = pv.Sphere(theta_resolution=100, phi_resolution=100)
        >>> sphere.point_data["data"] = 2 * np.ones(sphere.n_points)
        >>> integrated = sphere.integrate_data()

        There is only 1 point and cell, so access the only value.

        >>> integrated["Area"][0]
        np.float64(3.14)
        >>> integrated["data"][0]
        np.float64(6.28)

        See the :ref:`integrate_example` for more examples using this filter.

        """
        if not hasattr(_vtk, 'vtkIntegrateAttributes'):  # pragma: no cover
            raise VTKVersionError('`integrate_data` requires VTK 9.1.0 or newer.')

        alg = _vtk.vtkIntegrateAttributes()
        alg.SetInputData(self)
        alg.SetDivideAllCellDataByVolume(False)
        _update_alg(alg, progress_bar, 'Integrating Variables')
        return _get_output(alg)

    def partition(self, n_partitions, generate_global_id=False, as_composite=True):
        """Break down input dataset into a requested number of partitions.

        Cells on boundaries are uniquely assigned to each partition without duplication.

        It uses a kdtree implementation that builds balances the cell
        centers among a requested number of partitions. The current implementation
        only supports power-of-2 target partition. If a non-power of two value
        is specified for ``n_partitions``, then the load balancing simply
        uses the power-of-two greater than the requested value

        For more details, see `vtkRedistributeDataSetFilter
        <https://vtk.org/doc/nightly/html/classvtkRedistributeDataSetFilter.html>`_.

        Parameters
        ----------
        n_partitions : int
            Specify the number of partitions to split the input dataset
            into. Current implementation results in a number of partitions equal
            to the power of 2 greater than or equal to the chosen value.

        generate_global_id : bool, default: False
            Generate global cell ids if ``None`` are present in the input.  If
            global cell ids are present in the input then this flag is
            ignored.

            This is stored as ``"vtkGlobalCellIds"`` within the ``cell_data``
            of the output dataset(s).

        as_composite : bool, default: False
            Return the partitioned dataset as a :class:`pyvista.MultiBlock`.

        See Also
        --------
        split_bodies, extract_values

        Returns
        -------
        pyvista.MultiBlock or pyvista.UnstructuredGrid
            UnStructuredGrid if ``as_composite=False`` and MultiBlock when ``True``.

        Examples
        --------
        Partition a simple ImageData into a :class:`pyvista.MultiBlock`
        containing each partition.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(5, 5, 5))
        >>> out = grid.partition(4, as_composite=True)
        >>> out.plot(multi_colors=True, show_edges=True)

        Partition of the Stanford bunny.

        >>> from pyvista import examples
        >>> mesh = examples.download_bunny()
        >>> out = mesh.partition(4, as_composite=True)
        >>> out.plot(multi_colors=True, cpos='xy')

        """
        # While vtkRedistributeDataSetFilter exists prior to 9.1.0, it doesn't
        # work correctly, returning the wrong number of partitions.
        if pyvista.vtk_version_info < (9, 1, 0):
            raise VTKVersionError('`partition` requires vtk>=9.1.0')  # pragma: no cover
        if not hasattr(_vtk, 'vtkRedistributeDataSetFilter'):
            raise VTKVersionError(
                '`partition` requires vtkRedistributeDataSetFilter, but it '
                f'was not found in VTK {pyvista.vtk_version_info}',
            )  # pragma: no cover

        alg = _vtk.vtkRedistributeDataSetFilter()
        alg.SetInputData(self)
        alg.SetNumberOfPartitions(n_partitions)
        alg.SetPreservePartitionsInOutput(True)
        alg.SetGenerateGlobalCellIds(generate_global_id)
        alg.Update()

        # pyvista does not yet support vtkPartitionedDataSet
        part = alg.GetOutput()
        datasets = [part.GetPartition(ii) for ii in range(part.GetNumberOfPartitions())]
        output = pyvista.MultiBlock(datasets)
        if not as_composite:
            # note, SetPreservePartitionsInOutput does not work correctly in
            # vtk 9.2.0, so instead we set it to True always and simply merge
            # the result. See:
            # https://gitlab.kitware.com/vtk/vtk/-/issues/18632
            return pyvista.merge(list(output), merge_points=False)
        return output

    def explode(self, factor=0.1):
        """Push each individual cell away from the center of the dataset.

        Parameters
        ----------
        factor : float, default: 0.1
            How much each cell will move from the center of the dataset
            relative to its distance from it. Increase this number to push the
            cells farther away.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the exploded cells.

        Notes
        -----
        This is similar to :func:`shrink <pyvista.DataSetFilters.shrink>`
        except that it does not change the size of the cells.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> xrng = np.linspace(0, 1, 3)
        >>> yrng = np.linspace(0, 2, 4)
        >>> zrng = np.linspace(0, 3, 5)
        >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
        >>> exploded = grid.explode()
        >>> exploded.plot(show_edges=True)

        """
        split = self.separate_cells()
        if not isinstance(split, pyvista.UnstructuredGrid):
            split = split.cast_to_unstructured_grid()

        vec = (split.cell_centers().points - split.center) * factor
        split.points += np.repeat(vec, np.diff(split.offset), axis=0)
        return split

    def separate_cells(self):
        """Return a copy of the dataset with separated cells with no shared points.

        This method may be useful when datasets have scalars that need to be
        associated to each point of each cell rather than either each cell or
        just the points of the dataset.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid with isolated cells.

        Examples
        --------
        Load the example hex beam and separate its cells. This increases the
        total number of points in the dataset since points are no longer
        shared.

        >>> from pyvista import examples
        >>> grid = examples.load_hexbeam()
        >>> grid.n_points
        99
        >>> sep_grid = grid.separate_cells()
        >>> sep_grid.n_points
        320

        See the :ref:`point_cell_scalars_example` for a more detailed example
        using this filter.

        """
        return self.shrink(1.0)

    def extract_cells_by_type(self, cell_types, progress_bar=False):
        """Extract cells of a specified type.

        Given an input dataset and a list of cell types, produce an output
        dataset containing only cells of the specified type(s). Note that if
        the input dataset is homogeneous (e.g., all cells are of the same type)
        and the cell type is one of the cells specified, then the input dataset
        is shallow copied to the output.

        The type of output dataset is always the same as the input type. Since
        structured types of data (i.e., :class:`pyvista.ImageData`,
        :class:`pyvista.StructuredGrid`, :class`pyvista.RectilnearGrid`)
        are all composed of a cell of the same
        type, the output is either empty, or a shallow copy of the input.
        Unstructured data (:class:`pyvista.UnstructuredGrid`,
        :class:`pyvista.PolyData`) input may produce a subset of the input data
        (depending on the selected cell types).

        Parameters
        ----------
        cell_types :  int | sequence[int]
            The cell types to extract. Must be a single or list of integer cell
            types. See :class:`pyvista.CellType`.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.DataSet
            Dataset with the extracted cells. Type is the same as the input.

        Notes
        -----
        Unlike :func:`pyvista.DataSetFilters.extract_cells` which always
        produces a :class:`pyvista.UnstructuredGrid` output, this filter
        produces the same output type as input type.

        Examples
        --------
        Create an unstructured grid with both hexahedral and tetrahedral
        cells and then extract each individual cell type.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> beam = examples.load_hexbeam()
        >>> beam = beam.translate([1, 0, 0])
        >>> ugrid = beam + examples.load_tetbeam()
        >>> hex_cells = ugrid.extract_cells_by_type(pv.CellType.HEXAHEDRON)
        >>> tet_cells = ugrid.extract_cells_by_type(pv.CellType.TETRA)
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> _ = pl.add_text('Extracted Hexahedron cells')
        >>> _ = pl.add_mesh(hex_cells, show_edges=True)
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_text('Extracted Tetrahedron cells')
        >>> _ = pl.add_mesh(tet_cells, show_edges=True)
        >>> pl.show()

        """
        alg = _vtk.vtkExtractCellsByType()
        alg.SetInputDataObject(self)
        if isinstance(cell_types, int):
            alg.AddCellType(cell_types)
        elif isinstance(cell_types, (np.ndarray, Sequence)):
            for cell_type in cell_types:
                alg.AddCellType(cell_type)
        else:
            raise TypeError(
                f'Invalid type {type(cell_types)} for `cell_types`. Expecting an int or a sequence.',
            )
        _update_alg(alg, progress_bar, 'Extracting cell types')
        return _get_output(alg)

    def sort_labels(
        self,
        scalars=None,
        preference='point',
        output_scalars=None,
        progress_bar=False,
        inplace=False,
    ):
        """Sort labeled data by number of points or cells.

        This filter renumbers scalar label data of any type with ``N`` labels
        such that the output labels are contiguous from ``[0, N)`` and
        sorted in descending order from largest to smallest (by label count).
        I.e., the largest label will have a value of ``0`` and the smallest
        label will have a value of ``N-1``.

        The filter is a convenience method for :func:`pyvista.DataSetFilters.pack_labels`
        with ``sort=True``.

        Parameters
        ----------
        scalars : str, optional
            Name of scalars to sort. Defaults to currently active scalars.

        preference : str, default: "point"
            When ``scalars`` is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        output_scalars : str, None
            Name of the sorted output scalars. By default, the output is
            saved to ``'packed_labels'``.

        progress_bar : bool, default: False
            If ``True``, display a progress bar. Has no effect if VTK
            version is lower than 9.3.

        inplace : bool, default: False
            If ``True``, the mesh is updated in-place.

        Returns
        -------
        pyvista.Dataset
            Dataset with sorted labels.

        Examples
        --------
        Sort segmented image labels.

        Load image labels

        >>> from pyvista import examples
        >>> import numpy as np
        >>> image_labels = examples.load_frog_tissues()

        Show label info for first four labels

        >>> label_number, label_size = np.unique(
        ...     image_labels['MetaImage'], return_counts=True
        ... )
        >>> label_number[:4]
        pyvista_ndarray([0, 1, 2, 3], dtype=uint8)
        >>> label_size[:4]
        array([30805713,    35279,    19172,    38129])

        Sort labels

        >>> sorted_labels = image_labels.sort_labels()

        Show sorted label info for the four largest labels. Note
        the difference in label size after sorting.

        >>> sorted_label_number, sorted_label_size = np.unique(
        ...     sorted_labels["packed_labels"], return_counts=True
        ... )
        >>> sorted_label_number[:4]
        pyvista_ndarray([0, 1, 2, 3], dtype=uint8)
        >>> sorted_label_size[:4]
        array([30805713,   438052,   204672,   133880])

        """
        return self.pack_labels(
            scalars=scalars,
            output_scalars=output_scalars,
            preference=preference,
            progress_bar=progress_bar,
            inplace=inplace,
            sort=True,
        )

    def pack_labels(
        self,
        sort=False,
        scalars=None,
        preference='point',
        output_scalars=None,
        progress_bar=False,
        inplace=False,
    ):
        """Renumber labeled data such that labels are contiguous.

        This filter renumbers scalar label data of any type with ``N`` labels
        such that the output labels are contiguous from ``[0, N)``. The
        output may optionally be sorted by label count.

        The output array ``'packed_labels'`` is added to the output by default,
        and is automatically set as the active scalars.

        See Also
        --------
        sort_labels
            Similar function with ``sort=True`` by default.

        Notes
        -----
        This filter uses ``vtkPackLabels`` as the underlying method which
        requires VTK version 9.3 or higher. If ``vtkPackLabels`` is not
        available, packing is done with ``NumPy`` instead which may be
        slower. For best performance, consider upgrading VTK.

        .. versionadded:: 0.43

        Parameters
        ----------
        sort : bool, default: False
            Whether to sort the output by label count in descending order
            (i.e. from largest to smallest).

        scalars : str, optional
            Name of scalars to pack. Defaults to currently active scalars.

        preference : str, default: "point"
            When ``scalars`` is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        output_scalars : str, None
            Name of the packed output scalars. By default, the output is
            saved to ``'packed_labels'``.

        progress_bar : bool, default: False
            If ``True``, display a progress bar. Has no effect if VTK
            version is lower than 9.3.

        inplace : bool, default: False
            If ``True``, the mesh is updated in-place.

        Returns
        -------
        pyvista.Dataset
            Dataset with packed labels.

        Examples
        --------
        Pack segmented image labels.

        Load non-contiguous image labels

        >>> from pyvista import examples
        >>> import numpy as np
        >>> image_labels = examples.load_frog_tissues()

        Show range of labels

        >>> image_labels.get_data_range()
        (np.uint8(0), np.uint8(29))

        Find 'gaps' in the labels

        >>> label_numbers = np.unique(image_labels.active_scalars)
        >>> label_max = np.max(label_numbers)
        >>> missing_labels = set(range(label_max)) - set(label_numbers)
        >>> len(missing_labels)
        4

        Pack labels to remove gaps

        >>> packed_labels = image_labels.pack_labels()

        Show range of packed labels

        >>> packed_labels.get_data_range()
        (np.uint8(0), np.uint8(25))

        """
        # Set a input scalars
        if scalars is None:
            set_default_active_scalars(self)
            _, scalars = self.active_scalars_info

        field = get_array_association(self, scalars, preference=preference)

        # Determine output scalars
        default_output_scalars = "packed_labels"
        if output_scalars is None:
            output_scalars = default_output_scalars
        if not isinstance(output_scalars, str):
            raise TypeError(f"Output scalars must be a string, got {type(output_scalars)} instead.")

        # Do packing
        if hasattr(_vtk, 'vtkPackLabels'):  # pragma: no cover
            alg = _vtk.vtkPackLabels()
            alg.SetInputDataObject(self)
            alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
            if sort:
                alg.SortByLabelCount()
            alg.PassFieldDataOn()
            alg.PassCellDataOn()
            alg.PassPointDataOn()
            _update_alg(alg, progress_bar, 'Packing labels')
            result = _get_output(alg)

            if output_scalars is not scalars:
                # vtkPackLabels does not pass un-packed labels through to the
                # output, so add it back here
                if field == FieldAssociation.POINT:
                    result.point_data[scalars] = self.point_data[scalars]
                else:
                    result.cell_data[scalars] = self.cell_data[scalars]
            result.rename_array("PackedLabels", output_scalars)

            if inplace:
                self.copy_from(result, deep=False)
                return self
            return result

        else:  # Use numpy
            # Get mapping from input ID to output ID
            arr = get_array(self, scalars, preference=preference, err=True)
            label_numbers_in, label_sizes = np.unique(arr, return_counts=True)
            if sort:
                label_numbers_in = label_numbers_in[np.argsort(label_sizes)[::-1]]
            label_range_in = np.arange(0, np.max(label_numbers_in))
            label_numbers_out = label_range_in[: len(label_numbers_in)]

            # Pack/sort array
            packed_array = np.zeros_like(arr)
            for num_in, num_out in zip(label_numbers_in, label_numbers_out):
                packed_array[arr == num_in] = num_out

            result = self if inplace else self.copy(deep=True)

            # Add output to mesh
            if field == FieldAssociation.POINT:
                result.point_data[output_scalars] = packed_array
            else:
                result.cell_data[output_scalars] = packed_array

            # vtkPackLabels sets active scalars by default, so do the same here
            result.set_active_scalars(output_scalars, preference=field)

            return result


def _set_threshold_limit(alg, value, method, invert):
    """Set vtkThreshold limits and function.

    Addresses VTK API deprecations and previous PyVista inconsistencies with ParaView. Reference:

    * https://github.com/pyvista/pyvista/issues/2850
    * https://github.com/pyvista/pyvista/issues/3610
    * https://discourse.vtk.org/t/unnecessary-vtk-api-change/9929

    """
    # Check value
    if isinstance(value, (np.ndarray, Sequence)):
        if len(value) != 2:
            raise ValueError(
                f'Value range must be length one for a float value or two for min/max; not ({value}).',
            )
        # Check range
        if value[0] > value[1]:
            raise ValueError(
                'Value sequence is invalid, please use (min, max). The provided first value is greater than the second.',
            )
    elif isinstance(value, Iterable):
        raise TypeError('Value must either be a single scalar or a sequence.')
    alg.SetInvert(invert)
    # Set values and function
    if pyvista.vtk_version_info >= (9, 1):
        if isinstance(value, (np.ndarray, Sequence)):
            alg.SetThresholdFunction(_vtk.vtkThreshold.THRESHOLD_BETWEEN)
            alg.SetLowerThreshold(value[0])
            alg.SetUpperThreshold(value[1])
        else:
            # Single value
            if method.lower() == 'lower':
                alg.SetLowerThreshold(value)
                alg.SetThresholdFunction(_vtk.vtkThreshold.THRESHOLD_LOWER)
            elif method.lower() == 'upper':
                alg.SetUpperThreshold(value)
                alg.SetThresholdFunction(_vtk.vtkThreshold.THRESHOLD_UPPER)
            else:
                raise ValueError('Invalid method choice. Either `lower` or `upper`')
    else:  # pragma: no cover
        # ThresholdByLower, ThresholdByUpper, ThresholdBetween
        if isinstance(value, (np.ndarray, Sequence)):
            alg.ThresholdBetween(value[0], value[1])
        else:
            # Single value
            if method.lower() == 'lower':
                alg.ThresholdByLower(value)
            elif method.lower() == 'upper':
                alg.ThresholdByUpper(value)
            else:
                raise ValueError('Invalid method choice. Either `lower` or `upper`')
