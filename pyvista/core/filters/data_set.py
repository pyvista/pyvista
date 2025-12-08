"""Filters module with a class of common filters that can be applied to any :vtk:`vtkDataSet`."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
import contextlib
import functools
import itertools
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core import _validation
import pyvista.core._vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError
from pyvista.core.errors import MissingDataError
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_object import DataObjectFilters
from pyvista.core.filters.data_object import _cast_output_to_match_input_type
from pyvista.core.utilities.arrays import FieldAssociation
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import get_array_association
from pyvista.core.utilities.arrays import set_default_active_scalars
from pyvista.core.utilities.arrays import set_default_active_vectors
from pyvista.core.utilities.cells import numpy_to_idarr
from pyvista.core.utilities.geometric_objects import NORMALS
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import _BoundsSizeMixin
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.misc import assert_empty_kwargs
from pyvista.core.utilities.transform import Transform

if TYPE_CHECKING:
    from pyvista import Color
    from pyvista import DataSet
    from pyvista import ImageData
    from pyvista import MultiBlock
    from pyvista import PolyData
    from pyvista import RectilinearGrid
    from pyvista import UnstructuredGrid
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike
    from pyvista.core._typing_core import _DataObjectType
    from pyvista.core._typing_core import _DataSetType
    from pyvista.plotting._typing import ColorLike
    from pyvista.plotting._typing import ColormapOptions


@abstract_class
class DataSetFilters(_BoundsSizeMixin, DataObjectFilters):
    """A set of common filters that can be applied to any :vtk:`vtkDataSet`."""

    @_deprecate_positional_args(allowed=['target'])
    def align(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        target: DataSet | _vtk.vtkDataSet,
        max_landmarks: int = 100,
        max_mean_distance: float = 1e-5,
        max_iterations: int = 500,
        check_mean_distance: bool = True,  # noqa: FBT001, FBT002
        start_by_matching_centroids: bool = True,  # noqa: FBT001, FBT002
        return_matrix: bool = False,  # noqa: FBT001, FBT002
    ):
        """Align a dataset to another.

        Uses the iterative closest point algorithm to align the points of the
        two meshes. See the VTK class :vtk:`vtkIterativeClosestPointTransform`.

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

        See Also
        --------
        align_xyz
            Align a dataset to the x-y-z axes.

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
        >>> _ = pl.add_mesh(source, style='wireframe', opacity=0.5, line_width=2)
        >>> _ = pl.add_mesh(transformed)
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_text('After Alignment')
        >>> _ = pl.add_mesh(source, style='wireframe', opacity=0.5, line_width=2)
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
        icp.SetTarget(wrap(target))
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfLandmarks(max_landmarks)
        icp.SetMaximumMeanDistance(max_mean_distance)
        icp.SetMaximumNumberOfIterations(max_iterations)
        icp.SetCheckMeanDistance(check_mean_distance)
        icp.SetStartByMatchingCentroids(start_by_matching_centroids)
        icp.Update()
        matrix = pv.array_from_vtkmatrix(icp.GetMatrix())
        if return_matrix:
            return self.transform(matrix, inplace=False), matrix
        return self.transform(matrix, inplace=False)

    def align_xyz(  # type: ignore[misc]
        self: _DataSetType,
        *,
        centered: bool = True,
        axis_0_direction: VectorLike[float] | str | None = None,
        axis_1_direction: VectorLike[float] | str | None = None,
        axis_2_direction: VectorLike[float] | str | None = None,
        return_matrix: bool = False,
    ):
        """Align a dataset to the x-y-z axes.

        This filter aligns a mesh's :func:`~pyvista.principal_axes` to the world x-y-z
        axes. The principal axes are effectively used as a rotation matrix to rotate
        the dataset for the alignment. The transformation matrix used for the alignment
        can optionally be returned.

        Note that the transformation is not unique, since the signs of the principal
        axes are arbitrary. Consequently, applying this filter to similar meshes
        may result in dissimilar alignment (e.g. one axis may point up instead of down).
        To address this, the sign of one or two axes may optionally be "seeded" with a
        vector which approximates the axis or axes of the input. This can be useful
        for cases where the orientation of the input has a clear physical meaning.

        .. versionadded:: 0.45

        Parameters
        ----------
        centered : bool, default: True
            Center the mesh at the origin. If ``False``, the aligned dataset has the
            same center as the input.

        axis_0_direction : VectorLike[float] | str, optional
            Approximate direction vector of this mesh's primary axis prior to
            alignment. If set, this axis is flipped such that it best aligns with
            the specified vector. Can be a vector or string specifying the axis by
            name (e.g. ``'x'`` or ``'-x'``, etc.).

        axis_1_direction : VectorLike[float] | str, optional
            Approximate direction vector of this mesh's secondary axis prior to
            alignment. If set, this axis is flipped such that it best aligns with
            the specified vector. Can be a vector or string specifying the axis by
            name (e.g. ``'x'`` or ``'-x'``, etc.).

        axis_2_direction : VectorLike[float] | str, optional
            Approximate direction vector of this mesh's third axis prior to
            alignment. If set, this axis is flipped such that it best aligns with
            the specified vector. Can be a vector or string specifying the axis by
            name (e.g. ``'x'`` or ``'-x'``, etc.).

        return_matrix : bool, default: False
            Return the transform matrix as well as the aligned mesh.

        Returns
        -------
        pyvista.DataSet
            The dataset aligned to the x-y-z axes.

        numpy.ndarray
            Transform matrix to transform the input dataset to the x-y-z axes if
            ``return_matrix`` is ``True``.

        See Also
        --------
        pyvista.principal_axes
            Best-fit axes used by this filter for the alignment.

        align
            Align a source mesh to a target mesh using iterative closest point (ICP).

        Examples
        --------
        Create a dataset and align it to the x-y-z axes.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_oblique_cone()
        >>> aligned = mesh.align_xyz()

        Plot the aligned mesh along with the original. Show axes at the origin for
        context.

        >>> axes = pv.AxesAssembly(scale=aligned.length)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(aligned)
        >>> _ = pl.add_mesh(mesh, style='wireframe', color='black', line_width=3)
        >>> _ = pl.add_actor(axes)
        >>> pl.show()

        Align the mesh but don't center it.

        >>> aligned = mesh.align_xyz(centered=False)

        Plot the result again. The aligned mesh has the same position as the input.

        >>> axes = pv.AxesAssembly(position=mesh.center, scale=aligned.length)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(aligned)
        >>> _ = pl.add_mesh(mesh, style='wireframe', color='black', line_width=3)
        >>> _ = pl.add_actor(axes)
        >>> pl.show()

        Note how the tip of the cone is pointing along the z-axis. This indicates that
        the cone's axis is the third principal axis. It is also pointing in the negative
        z-direction. To control the alignment so that the cone points upward, we can
        seed an approximate direction specifying what "up" means for the original mesh
        in world coordinates prior to the alignment.

        We can see that the cone is originally pointing downward, somewhat in the
        negative z-direction. Therefore, we can specify the ``'-z'`` vector
        as the "up" direction of the mesh's third axis prior to alignment.

        >>> aligned = mesh.align_xyz(axis_2_direction='-z')

        Plot the mesh. The cone is now pointing upward in the desired direction.

        >>> axes = pv.AxesAssembly(scale=aligned.length)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(aligned)
        >>> _ = pl.add_actor(axes)
        >>> pl.show()

        The specified direction only needs to be approximate. For example, we get the
        same result by specifying the ``'y'`` direction as the mesh's original "up"
        direction.

        >>> aligned, matrix = mesh.align_xyz(axis_2_direction='y', return_matrix=True)
        >>> axes = pv.AxesAssembly(scale=aligned.length)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(aligned)
        >>> _ = pl.add_actor(axes)
        >>> pl.show()

        We can optionally return the transformation matrix.

        >>> aligned, matrix = mesh.align_xyz(axis_2_direction='y', return_matrix=True)

        The matrix can be inverted, for example, to transform objects from the world
        axes back to the original mesh's local coordinate system.

        >>> inverse = pv.Transform(matrix).inverse_matrix

        Use the inverse to label the object's axes prior to alignment. For actors,
        we set the :attr:`~pyvista.Prop3D.user_matrix` as the inverse.

        >>> axes_local = pv.AxesAssembly(
        ...     scale=aligned.length,
        ...     user_matrix=inverse,
        ...     labels=["X'", "Y'", "Z'"],
        ... )

        Plot the original mesh with its local axes, along with the algned mesh and its
        axes.

        >>> axes_aligned = pv.AxesAssembly(scale=aligned.length)
        >>> pl = pv.Plotter()
        >>> # Add aligned mesh with axes
        >>> _ = pl.add_mesh(aligned)
        >>> _ = pl.add_actor(axes_aligned)
        >>> # Add original mesh with axes
        >>> _ = pl.add_mesh(mesh, style='wireframe', color='black', line_width=3)
        >>> _ = pl.add_actor(axes_local)
        >>> pl.show()

        """

        def _validate_vector(
            vector: VectorLike[float] | str | None, name: str
        ) -> NumpyArray[float] | None:
            if vector is None:
                vector_ = vector
            else:
                if isinstance(vector, str):
                    vector = vector.lower()
                    valid_strings = list(NORMALS.keys())
                    _validation.check_contains(valid_strings, must_contain=vector, name=name)
                    vector = NORMALS[vector]
                vector_ = _validation.validate_array3(vector, dtype_out=float, name=name)
            return vector_

        axes, std = pv.principal_axes(self.points, return_std=True)

        if axis_0_direction is None and axis_1_direction is None and axis_2_direction is None:
            # Set directions of first two axes to +X,+Y by default
            # Keep third axis as None (direction cannot be set if first two are set)
            axis_0_direction = (1.0, 0.0, 0.0)
            axis_1_direction = (0.0, 1.0, 0.0)
        else:
            axis_0_direction = _validate_vector(axis_0_direction, name='axis 0 direction')
            axis_1_direction = _validate_vector(axis_1_direction, name='axis 1 direction')
            axis_2_direction = _validate_vector(axis_2_direction, name='axis 2 direction')

        # Swap any axes which have equal std (e.g. so that we XYZ order instead of YXZ order)
        # Note: Swapping may create a left-handed coordinate frame. This is fixed later.
        axes = _swap_axes(axes, std)

        # Maybe flip directions of first two axes
        if axis_0_direction is not None and np.dot(axes[0], axis_0_direction) < 0:
            axes[0] *= -1
        if axis_1_direction is not None and np.dot(axes[1], axis_1_direction) < 0:
            axes[1] *= -1

        # Ensure axes form a right-handed coordinate frame
        if np.linalg.det(axes) < 0:
            axes[2] *= -1

        # Maybe flip direction of third axis
        if axis_2_direction is not None:
            if np.dot(axes[2], axis_2_direction) >= 0:
                pass  # nothing to do, sign is correct
            elif axis_0_direction is not None and axis_1_direction is not None:
                msg = (
                    f'Invalid `axis_2_direction` {axis_2_direction}. '
                    f'This direction results in a left-handed transformation.'
                )
                raise ValueError(msg)
            else:
                axes[2] *= -1
                # Need to also flip a second vector to keep system as right-handed
                if axis_1_direction is not None:
                    # Second axis has been set, so modify first axis
                    axes[0] *= -1
                else:
                    # First axis has been set, so modify second axis
                    axes[1] *= -1

        rotation = Transform().rotate(axes)
        aligned = self.transform(rotation, inplace=False)
        translation = Transform().translate(-np.array(aligned.center))
        if not centered:
            translation.translate(self.center)
        aligned.transform(translation, inplace=True)

        if return_matrix:
            return aligned, rotation.compose(translation).matrix
        return aligned

    @_deprecate_positional_args(allowed=['surface'])
    def compute_implicit_distance(  # type: ignore[misc]
        self: _DataSetType,
        surface: DataSet | _vtk.vtkDataSet,
        inplace: bool = False,  # noqa: FBT001, FBT002
    ):
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
        >>> _ = pl.add_mesh(sphere, scalars='implicit_distance', cmap='bwr')
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
        :ref:`voxelize_example` for more examples using
        this filter.

        """
        function = _vtk.vtkImplicitPolyDataDistance()
        function.SetInput(surface)
        points = pv.convert_array(self.points)
        dists = _vtk.vtkDoubleArray()
        function.FunctionValue(points, dists)
        if inplace:
            self.point_data['implicit_distance'] = pv.convert_array(dists)
            return self
        result = self.copy()
        result.point_data['implicit_distance'] = pv.convert_array(dists)
        return result

    @_deprecate_positional_args
    def clip_scalar(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        scalars: str | None = None,
        invert: bool = True,  # noqa: FBT001, FBT002
        value: float | VectorLike[float] = 0.0,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        both: bool = False,  # noqa: FBT001, FBT002
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

        value : float | VectorLike[float], default: 0.0
            Set the clipping value. Can also be set as a range of values.
            The range produces an output similar to an isovolume filter of Paraview.

        inplace : bool, default: False
            Update mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        both : bool, default: False
            If ``True``, also returns the complementary clipped mesh.

        Returns
        -------
        output : pyvista.PolyData | tuple
            Clipped dataset if ``both=False``.  If ``both=True`` then
            returns a tuple of both clipped datasets.

        Examples
        --------
        Remove the part of the mesh with "sample_point_scalars" above 100.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.load_hexbeam()
        >>> clipped = dataset.clip_scalar(scalars='sample_point_scalars', value=100)
        >>> clipped.plot()

        Get clipped meshes corresponding to the portions of the mesh above and below 100.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.load_hexbeam()
        >>> _below, _above = dataset.clip_scalar(
        ...     scalars='sample_point_scalars', value=100, both=True
        ... )

        Remove the part of the mesh with "sample_point_scalars" below 100.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.load_hexbeam()
        >>> clipped = dataset.clip_scalar(
        ...     scalars='sample_point_scalars', value=100, invert=False
        ... )
        >>> clipped.plot()

        """
        if isinstance(self, _vtk.vtkPolyData):
            alg: _vtk.vtkClipPolyData | _vtk.vtkTableBasedClipDataSet = _vtk.vtkClipPolyData()  # type: ignore[unreachable]
        else:
            alg = _vtk.vtkTableBasedClipDataSet()

        if is_single_value := isinstance(value, (float, int)):
            alg.SetValue(value)
        else:
            lower, upper = _validation.validate_data_range(value)
            alg.SetValue(upper)
            if not invert:
                msg = 'Cannot have invert=False for a range clip'
                raise ValueError(msg)
            if both:
                msg = 'Cannot have both=True for a range clip'
                raise ValueError(msg)
        alg.SetInputDataObject(self)
        if scalars is None:
            set_default_active_scalars(self)
        else:
            self.set_active_scalars(scalars)

        alg.SetInsideOut(invert)  # invert the clip if needed
        alg.SetGenerateClippedOutput(both)

        _update_alg(alg, progress_bar=progress_bar, message='Clipping by a Scalar')
        result0 = _get_output(alg)
        if inplace:
            if isinstance(self, pv.core.grid.ImageData):
                msg = 'Cannot use inplace argument for ImageData type input.'
                raise TypeError(msg)
            self.copy_from(result0, deep=False)
            result0 = self
        if not is_single_value:
            return result0.clip_scalar(scalars=scalars, invert=False, value=lower, inplace=inplace)
        if both:
            result1 = _get_output(alg, oport=1)
            if isinstance(self, _vtk.vtkPolyData):
                # For some reason vtkClipPolyData with SetGenerateClippedOutput on
                # leaves unreferenced vertices
                result0, result1 = (r.clean() for r in (result0, result1))  # type: ignore[unreachable]
            return result0, result1
        return result0

    @_deprecate_positional_args(allowed=['surface'])
    def clip_surface(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        surface: DataSet | _vtk.vtkDataSet,
        invert: bool = True,  # noqa: FBT001, FBT002
        value: float = 0.0,
        compute_distance: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        crinkle: bool = False,  # noqa: FBT001, FBT002
    ):
        """Clip any mesh type using a :class:`pyvista.PolyData` surface mesh.

        The clipped mesh type matches the input type for :class:`~pyvista.PointSet` and
        :class:`~pyvista.PolyData`, otherwise the output type is
        :class:`~pyvista.UnstructuredGrid`.
        Geometry of the input dataset will be preserved where possible.
        Geometries near the clip intersection will be triangulated/tessellated.

        Parameters
        ----------
        surface : pyvista.PolyData
            The ``PolyData`` surface mesh to use as a clipping
            function.  If this input mesh is not a :class:`pyvista.PolyData`,
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
        DataSet
            Clipped mesh. Output type matches input type for
            :class:`~pyvista.PointSet`, :class:`~pyvista.PolyData`, and
            :class:`~pyvista.MultiBlock`; otherwise the output type is
            :class:`~pyvista.UnstructuredGrid`.

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
            surface = wrap(surface).extract_geometry()
        function = _vtk.vtkImplicitPolyDataDistance()
        function.SetInput(surface)
        if compute_distance:
            points = pv.convert_array(self.points)
            dists = _vtk.vtkDoubleArray()
            function.FunctionValue(points, dists)
            self['implicit_distance'] = pv.convert_array(dists)
        # run the clip
        clipped = DataSetFilters._clip_with_function(
            self,
            function,
            invert=invert,
            value=value,
            progress_bar=progress_bar,
            crinkle=crinkle,
        )
        return _cast_output_to_match_input_type(clipped, self)

    @_deprecate_positional_args(allowed=['value'])
    def threshold(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        value: float | VectorLike[float] | None = None,
        scalars: str | None = None,
        invert: bool = False,  # noqa: FBT001, FBT002
        continuous: bool = False,  # noqa: FBT001, FBT002
        preference: Literal['point', 'cell'] = 'cell',
        all_scalars: bool = False,  # noqa: FBT001, FBT002
        component_mode: Literal['component', 'all', 'any'] = 'all',
        component: int = 0,
        method: Literal['upper', 'lower'] = 'upper',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Apply a :vtk:`vtkThreshold` filter to the input dataset.

        This filter will apply a :vtk:`vtkThreshold` filter to the input
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

        component_mode : {'component', 'all', 'any'}
            The method to satisfy the criteria for the threshold of
            multicomponent scalars.  'component' (default)
            uses only the ``component``.  'all' requires all
            components to meet criteria.  'any' is when
            any component satisfies the criteria.

        component : int, default: 0
            When using ``component_mode='component'``, this sets
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
        threshold_percent
            Threshold a dataset by a percentage of its scalar range.
        :meth:`~pyvista.DataSetFilters.extract_values`
            Threshold-like filter for extracting specific values and ranges.
        :meth:`~pyvista.ImageDataFilters.image_threshold`
            Similar method for thresholding :class:`~pyvista.ImageData`.
        :meth:`~pyvista.ImageDataFilters.select_values`
            Threshold-like filter for ``ImageData`` to keep some values and replace others.

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
        ...     noise, bounds=[0, 1.0, -0, 1.0, 0, 1.0], dim=(20, 20, 20)
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
        ...     noise, bounds=[0, 1.0, -0, 1.0, 0, 1.0], dim=(20, 20, 20)
        ... )
        >>> threshed = grid.threshold(value=0.02)
        >>> threshed.plot(
        ...     cmap='gist_earth_r',
        ...     show_scalar_bar=False,
        ...     show_edges=True,
        ... )

        See :ref:`using_filters_example` and :ref:`image_representations_example`
        for more examples using this filter.

        """
        # set the scalars to threshold on
        scalars_ = set_default_active_scalars(self).name if scalars is None else scalars
        arr = get_array(self, scalars_, preference=preference, err=False)
        if arr is None:
            msg = 'No arrays present to threshold.'
            raise ValueError(msg)

        field = get_array_association(self, scalars_, preference=preference)

        # Run a standard threshold algorithm
        alg = _vtk.vtkThreshold()
        alg.SetAllScalars(all_scalars)
        alg.SetInputDataObject(self)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars_,
        )  # args: (idx, port, connection, field, name)
        # set thresholding parameters
        alg.SetUseContinuousCellRange(continuous)
        # use valid range if no value given
        if value is None:
            value = self.get_data_range(scalars)

        _set_threshold_limit(alg, value=value, method=method, invert=invert)

        if component_mode == 'component':
            alg.SetComponentModeToUseSelected()
            dim = arr.shape[1]
            if not isinstance(component, (int, np.integer)):
                msg = 'component must be int'  # type: ignore[unreachable]
                raise TypeError(msg)
            if component > (dim - 1) or component < 0:
                msg = f'scalars has {dim} components: supplied component {component} not in range'
                raise ValueError(msg)
            alg.SetSelectedComponent(component)
        elif component_mode == 'all':
            alg.SetComponentModeToUseAll()
        elif component_mode == 'any':
            alg.SetComponentModeToUseAny()
        else:
            msg = f"component_mode must be 'component', 'all', or 'any' got: {component_mode}"  # type: ignore[unreachable]
            raise ValueError(msg)

        # Run the threshold
        _update_alg(alg, progress_bar=progress_bar, message='Thresholding')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['percent'])
    def threshold_percent(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        percent: float = 0.50,
        scalars: str | None = None,
        invert: bool = False,  # noqa: FBT001, FBT002
        continuous: bool = False,  # noqa: FBT001, FBT002
        preference: Literal['point', 'cell'] = 'cell',
        method: Literal['upper', 'lower'] = 'upper',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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

        See Also
        --------
        threshold
            Threshold a dataset by value.

        Examples
        --------
        Apply a 50% threshold filter.

        >>> import pyvista as pv
        >>> noise = pv.perlin_noise(0.1, (2, 2, 2), (0, 0, 0))
        >>> grid = pv.sample_function(
        ...     noise, bounds=[0, 1.0, -0, 1.0, 0, 1.0], dim=(30, 30, 30)
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

        See :ref:`using_filters_example` for more examples using a similar filter.

        """
        tscalars = set_default_active_scalars(self).name if scalars is None else scalars
        dmin, dmax = self.get_data_range(arr_var=tscalars, preference=preference)

        def _check_percent(percent):
            """Make sure percent is between 0 and 1 or fix if between 0 and 100."""
            if percent >= 1:
                percent = float(percent) / 100.0
                if percent > 1:
                    msg = f'Percentage ({percent}) is out of range (0, 1).'
                    raise ValueError(msg)
            if percent < 1e-10:
                msg = f'Percentage ({percent}) is too close to zero or negative.'
                raise ValueError(msg)
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
            msg = 'Percent must either be a single scalar or a sequence.'
            raise TypeError(msg)
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

    @_deprecate_positional_args
    def outline(  # type: ignore[misc]
        self: _DataObjectType,
        generate_faces: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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

        See Also
        --------
        bounding_box
            Similar filter with additional options.

        Examples
        --------
        Generate and plot the outline of a sphere.  This is
        effectively the ``(x, y, z)`` bounds of the mesh.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> outline = sphere.outline()
        >>> pv.plot([sphere, outline], line_width=5)

        See :ref:`using_filters_example` for more examples using this filter.

        """
        alg = _vtk.vtkOutlineFilter()
        alg.SetInputDataObject(self)
        alg.SetGenerateFaces(generate_faces)
        _update_alg(alg, progress_bar=progress_bar, message='Producing an outline')
        return wrap(alg.GetOutputDataObject(0))

    @_deprecate_positional_args
    def outline_corners(  # type: ignore[misc]
        self: _DataObjectType,
        factor: float = 0.2,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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
        _update_alg(alg, progress_bar=progress_bar, message='Producing an Outline of the Corners')
        return wrap(alg.GetOutputDataObject(0))

    def gaussian_splatting(  # type: ignore[misc]
        self: _DataSetType,
        *,
        radius: float = 0.1,
        dimensions: VectorLike[int] = (50, 50, 50),
        progress_bar: bool = False,
    ):
        """Splat points into a volume using a Gaussian distribution.

        This filter uses :vtk:`vtkGaussianSplatter` to splat points into a volume
        dataset. Each point is surrounded with a Gaussian distribution function
        weighted by input scalar data. The distribution function is volumetrically
        sampled to create a structured dataset.

        .. versionadded:: 0.46

        Parameters
        ----------
        radius : float, default: 0.1
            This value is expressed as a percentage of the length of the longest side of
            the sampling volume. This determines the "width" of the splatter in
            terms of the distribution. Smaller numbers greatly reduce execution time.

        dimensions : VectorLike[int], default: (50, 50, 50)
            Sampling dimensions of the structured point set. Higher values produce better
            results but are much slower. This is the :attr:`~pyvista.ImageData.dimensions`
            of the returned :class:`~pyvista.ImageData`.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Image data with scalar values representing the splatting
            of the points.

        See Also
        --------
        voxelize_binary_mask
            Alternative method for generating :class:`~pyvista.ImageData` from a dataset.

        Examples
        --------
        Create an image data volume from a point cloud using gaussian splatter.

        >>> import pyvista as pv

        Load the Stanford Bunny mesh.

        >>> bunny = pv.examples.download_bunny()

        Apply Gaussian splatter to generate a volumetric representation.

        >>> volume = bunny.gaussian_splatting(radius=0.01)

        Threshold the volume to filter out low-density regions.

        >>> threshed = volume.threshold(0.05)

        Visualize the thresholded volume with semi-transparency and no scalar bar.

        >>> threshed.plot(opacity=0.5, show_scalar_bar=False)

        """
        from pyvista.core import _validation  # noqa: PLC0415

        _validation.check_range(radius, [0.0, 1.0], name='radius')
        dimensions_ = _validation.validate_array3(dimensions, name='dimensions')
        alg = _vtk.vtkGaussianSplatter()
        alg.SetInputDataObject(self)
        alg.SetRadius(radius)
        alg.SetSampleDimensions(list(dimensions_))
        message = 'Splatting Points with Gaussian Distribution'
        _update_alg(alg, progress_bar=progress_bar, message=message)
        return _get_output(alg)

    @_deprecate_positional_args
    def extract_geometry(  # type: ignore[misc]
        self: _DataSetType,
        extent: VectorLike[float] | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> PolyData:
        """Extract the outer surface of a volume or structured grid dataset.

        This will extract all 0D, 1D, and 2D cells producing the
        boundary faces of the dataset.

        .. note::
            This tends to be less efficient than :func:`extract_surface`.

        Parameters
        ----------
        extent : VectorLike[float], optional
            Specify a ``(x_min, x_max, y_min, y_max, z_min, z_max)`` bounding box to
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
            extent_ = _validation.validate_arrayN(extent, must_have_length=6, to_list=True)
            alg.SetExtent(extent_)
            alg.SetExtentClipping(True)
        _update_alg(alg, progress_bar=progress_bar, message='Extracting Geometry')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['isosurfaces', 'scalars'])
    def contour(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        isosurfaces: int | Sequence[float] = 10,
        scalars: str | NumpyArray[float] | None = None,
        compute_normals: bool = False,  # noqa: FBT001, FBT002
        compute_gradients: bool = False,  # noqa: FBT001, FBT002
        compute_scalars: bool = True,  # noqa: FBT001, FBT002
        rng: VectorLike[float] | None = None,
        preference: Literal['point', 'cell'] = 'point',
        method: Literal['contour', 'marching_cubes', 'flying_edges'] = 'contour',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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

        See :ref:`using_filters_example` or
        :ref:`marching_cubes_example` for more examples using this
        filter.

        """
        if method == 'contour':
            alg = _vtk.vtkContourFilter()
        elif method == 'marching_cubes':
            alg = _vtk.vtkMarchingCubes()  # type: ignore[assignment]
        elif method == 'flying_edges':
            alg = _vtk.vtkFlyingEdges3D()  # type: ignore[assignment]
        else:
            msg = f"Method '{method}' is not supported"  # type: ignore[unreachable]
            raise ValueError(msg)

        if isinstance(scalars, str):
            scalars_name = scalars
        elif isinstance(scalars, (Sequence, np.ndarray)) and not isinstance(scalars, str):
            scalars_name = 'Contour Data'
            self[scalars_name] = scalars
        elif scalars is None:
            scalars_name = set_default_active_scalars(self).name
        else:
            msg = (
                f'Invalid type for `scalars` ({type(scalars)}). Should be either '
                'a numpy.ndarray, a string, or None.'
            )
            raise TypeError(msg)

        # Make sure the input has scalars to contour on
        if self.n_arrays < 1:
            msg = 'Input dataset for the contour filter must have scalar.'
            raise ValueError(msg)

        alg.SetInputDataObject(self)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        field = get_array_association(self, scalars_name, preference=preference)
        if field != FieldAssociation.POINT:
            msg = 'Contour filter only works on point data.'
            raise TypeError(msg)
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
                rng_: list[float] = list(self.get_data_range(scalars_name))
            else:
                rng_ = list(_validation.validate_data_range(rng, name='rng'))
            alg.GenerateValues(isosurfaces, rng_)
        else:
            isosurfaces_ = _validation.validate_arrayN(
                isosurfaces, dtype_out=float, name='isosurfaces'
            )

            alg.SetNumberOfContours(len(isosurfaces_))
            for i, val in enumerate(isosurfaces_):
                alg.SetValue(i, val)

        _update_alg(alg, progress_bar=progress_bar, message='Computing Contour')
        output = _get_output(alg)

        # some of these filters fail to correctly name the array
        if scalars_name not in output.point_data and 'Unnamed_0' in output.point_data:
            output.point_data[scalars_name] = output.point_data.pop('Unnamed_0')

        return output

    @_deprecate_positional_args
    def texture_map_to_plane(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        origin: VectorLike[float] | None = None,
        point_u: VectorLike[float] | None = None,
        point_v: VectorLike[float] | None = None,
        inplace: bool = False,  # noqa: FBT001, FBT002
        name: str = 'Texture Coordinates',
        use_bounds: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
            _validation.check_instance(use_bounds, bool, name='use_bounds')
            bounds = self.bounds
            origin = [bounds.x_min, bounds.y_min, bounds.z_min]  # BOTTOM LEFT CORNER
            point_u = [bounds.x_max, bounds.y_min, bounds.z_min]  # BOTTOM RIGHT CORNER
            point_v = [bounds.x_min, bounds.y_max, bounds.z_min]  # TOP LEFT CORNER
        alg = _vtk.vtkTextureMapToPlane()
        if origin is None or point_u is None or point_v is None:
            alg.SetAutomaticPlaneGeneration(True)
        else:
            alg.SetOrigin(*origin)  # BOTTOM LEFT CORNER
            alg.SetPoint1(*point_u)  # BOTTOM RIGHT CORNER
            alg.SetPoint2(*point_v)  # TOP LEFT CORNER
        alg.SetInputDataObject(self)
        _update_alg(alg, progress_bar=progress_bar, message='Texturing Map to Plane')
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

    @_deprecate_positional_args
    def texture_map_to_sphere(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        center: VectorLike[float] | None = None,
        prevent_seam: bool = True,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
        name: str = 'Texture Coordinates',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
            alg.SetCenter(*center)
        alg.SetPreventSeam(prevent_seam)
        alg.SetInputDataObject(self)
        _update_alg(alg, progress_bar=progress_bar, message='Mapping texture to sphere')
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

    @_deprecate_positional_args
    def glyph(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        orient: bool | str = True,  # noqa: FBT001, FBT002
        scale: bool | str = True,  # noqa: FBT001, FBT002
        factor: float = 1.0,
        geom: _vtk.vtkDataSet | DataSet | Sequence[_vtk.vtkDataSet | DataSet] | None = None,
        indices: VectorLike[int] | None = None,
        tolerance: float | None = None,
        absolute: bool = False,  # noqa: FBT001, FBT002
        clamping: bool = False,  # noqa: FBT001, FBT002
        rng: VectorLike[float] | None = None,
        color_mode: Literal['scale', 'scalar', 'vector'] = 'scale',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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

        geom : :vtk:`vtkDataSet` | tuple[:vtk:`vtkDataSet`], optional
            The geometry to use for the glyph. If missing, an arrow glyph
            is used. If a sequence, the datasets inside define a table of
            geometries to choose from based on scalars or vectors. In this
            case a sequence of numbers of the same length must be passed as
            ``indices``. The values of the range (see ``rng``) affect lookup
            in the table.

            .. note::

                The reference direction is relative to ``(1, 0, 0)`` on the
                provided geometry. That is, the provided geometry will be rotated
                from ``(1, 0, 0)`` to the direction of the ``orient`` vector at
                each point.

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
        >>> arrows = mesh.glyph(scale='Normals', orient='Normals', tolerance=0.05)
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(arrows, color='black')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     scalars='Elevation',
        ...     cmap='terrain',
        ...     show_scalar_bar=False,
        ... )
        >>> pl.show()

        See :ref:`glyph_example`, :ref:`movie_glyphs_example`, and
        :ref:`glyph_table_example` for more examples using this filter.

        """
        dataset = self

        # Make glyphing geometry if necessary
        if geom is None:
            arrow = _vtk.vtkArrowSource()
            _update_alg(arrow, progress_bar=progress_bar, message='Making Arrow')
            geoms: Sequence[_vtk.vtkDataSet] = [arrow.GetOutput()]
        # Check if a table of geometries was passed
        elif isinstance(geom, (np.ndarray, Sequence)):
            geoms = geom
        else:
            geoms = [geom]

        if indices is None:
            # use default "categorical" indices
            indices = np.arange(len(geoms))
        elif not isinstance(indices, (np.ndarray, Sequence)):
            msg = (  # type: ignore[unreachable]
                'If "geom" is a sequence then "indices" must also be a '
                'sequence of the same length.'
            )
            raise TypeError(msg)
        if len(indices) != len(geoms) and len(geoms) != 1:
            msg = 'The sequence "indices" must be the same length as "geom".'
            raise ValueError(msg)

        if any(not isinstance(subgeom, _vtk.vtkPolyData) for subgeom in geoms):
            msg = 'Only PolyData objects can be used as glyphs.'
            raise TypeError(msg)

        # Run the algorithm
        alg = _vtk.vtkGlyph3D()

        if len(geoms) == 1:
            # use a single glyph, ignore indices
            alg.SetSourceData(geoms[0])
        else:
            for index, subgeom in zip(indices, geoms, strict=True):
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
            do_scale = True
        elif scale:
            try:
                set_default_active_scalars(self)
            except MissingDataError:
                warn_external(
                    'No data to use for scale. scale will be set to False.'
                )  # pragma: no cover
                do_scale = False
            except AmbiguousDataError as err:
                warn_external(
                    f'{err}\nIt is unclear which one to use. scale will be set to False.'
                )
                do_scale = False
            else:
                do_scale = True
        else:
            do_scale = False

        if do_scale:
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
            dataset.set_active_vectors(orient, preference=prefer)  # type: ignore[arg-type]
            orient = True

        if orient:
            try:
                set_default_active_vectors(dataset)
            except MissingDataError:
                warn_external(
                    'No vector-like data to use for orient. orient will be set to False.'
                )
                orient = False
            except AmbiguousDataError as err:
                warn_external(
                    f'{err}\nIt is unclear which one to use. orient will be set to False.'
                )
                orient = False

        if (
            scale
            and orient
            and dataset.active_vectors_info.association != dataset.active_scalars_info.association
        ):
            msg = 'Both ``scale`` and ``orient`` must use point data or cell data.'
            raise ValueError(msg)

        source_data = dataset
        set_actives_on_source_data = False

        if (scale and dataset.active_scalars_info.association == FieldAssociation.CELL) or (
            orient and dataset.active_vectors_info.association == FieldAssociation.CELL
        ):
            source_data = dataset.cell_centers()
            set_actives_on_source_data = True

        # Clean the points before glyphing
        if tolerance is not None:
            small = pv.PolyData(source_data.points)
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

        # upstream operations (cell to point conversion, point merging) may have unset
        # the correct active scalars/vectors, so set them again
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
            msg = f"Invalid color mode '{color_mode}'"  # type: ignore[unreachable]
            raise ValueError(msg)

        if rng is not None:
            valid_range = _validation.validate_data_range(rng)
            alg.SetRange(valid_range)
        alg.SetOrient(orient)
        alg.SetInputData(source_data)
        alg.SetVectorModeToUseVector()
        alg.SetScaleFactor(factor)
        alg.SetClamping(clamping)
        _update_alg(alg, progress_bar=progress_bar, message='Computing Glyphs')

        output = _get_output(alg)

        # Storing geom on the algorithm, for later use in legends.
        output._glyph_geom = geoms

        return output

    @_deprecate_positional_args(allowed=['extraction_mode', 'variable_input'])
    def connectivity(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        extraction_mode: Literal[
            'all',
            'largest',
            'specified',
            'cell_seed',
            'point_seed',
            'closest',
        ] = 'all',
        variable_input: float | VectorLike[float] | None = None,
        scalar_range: VectorLike[float] | None = None,
        scalars: str | None = None,
        label_regions: bool = True,  # noqa: FBT001, FBT002
        region_assignment_mode: Literal['ascending', 'descending', 'unspecified'] = 'descending',
        region_ids: VectorLike[int] | None = None,
        point_ids: VectorLike[int] | None = None,
        cell_ids: VectorLike[int] | None = None,
        closest_point: VectorLike[float] | None = None,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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

        See :ref:`connectivity_example` and :ref:`volumetric_analysis_example` for
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
               converted to point scalars with
               :func:`~pyvista.DataObjectFilters.cell_data_to_point_data`
               before applying the filter. The converted point scalars are
               removed from the output after applying the filter.

        label_regions : bool, default: True
            If ``True``, ``'RegionId'`` point and cell scalar arrays are stored.
            Each region is assigned a unique ID. IDs are zero-indexed and are
            assigned by region cell count in descending order (i.e. the largest
            region has ID ``0``).

        region_assignment_mode : str, default: "descending"
            Strategy used to assign connected region IDs if ``label_regions`` is True.
            Can be either:

            - ``"ascending"``: IDs are sorted by increasing order of cell count
            - ``"descending"``: IDs are sorted by decreasing order of cell counts
            - ``"unspecified"``: no particular order

            .. versionadded:: 0.47

            .. admonition:: ParaView compatibility
                :class: note dropdown

                The default value ``"descending"`` differs from ParaView's, which
                is set to ``"unspecified"`` (verified for 5.11 and 6.0 versions).

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
        >>> small = pv.Sphere(center=(0, 0, 0), phi_resolution=7, theta_resolution=7)
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

        Extract a region using a cell ID ``3100`` as a seed.

        >>> conn = mesh.connectivity('cell_seed', 3100)
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
            warn_external(
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
                msg = 'IDs must be positive integer values.'
                raise ValueError(msg)
            return np.unique(ids)

        def _post_process_extract_values(before_extraction, extracted):
            # Output is UnstructuredGrid, so apply vtkRemovePolyData
            # to input to cast the output as PolyData type instead
            has_cells = extracted.n_cells != 0
            if isinstance(before_extraction, pv.PolyData):
                all_ids = set(range(before_extraction.n_cells))

                ids_to_keep = set()
                if has_cells:
                    ids_to_keep |= set(extracted['vtkOriginalCellIds'])
                ids_to_remove = list(all_ids - ids_to_keep)
                if len(ids_to_remove) != 0:
                    remove = _vtk.vtkRemovePolyData()
                    remove.SetInputData(before_extraction)
                    remove.SetCellIds(numpy_to_idarr(ids_to_remove))
                    _update_alg(remove, progress_bar=progress_bar, message='Removing Cells.')
                    extracted = _get_output(remove)
                    extracted.clean(
                        point_merging=False,
                        inplace=True,
                        progress_bar=progress_bar,
                    )  # remove unused points

            return extracted

        # Store active scalars info to restore later if needed
        active_field, active_name = self.active_scalars_info

        # Set scalars
        if scalar_range is None:
            input_mesh = self.copy(deep=False)
        else:
            if isinstance(scalar_range, np.ndarray):
                num_elements = scalar_range.size
            elif isinstance(scalar_range, Sequence):
                num_elements = len(scalar_range)
            else:
                msg = 'Scalar range must be a numpy array or a sequence.'  # type: ignore[unreachable]
                raise TypeError(msg)
            if num_elements != 2:
                msg = 'Scalar range must have two elements defining the min and max.'
                raise ValueError(msg)
            if scalar_range[0] > scalar_range[1]:
                msg = (
                    f'Lower value of scalar range {scalar_range[0]} cannot be greater '
                    f'than the upper value {scalar_range[0]}'
                )
                raise ValueError(msg)

            # Input will be modified, so copy first
            input_mesh = self.copy()
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
        modes = {
            'ascending': alg.CELL_COUNT_ASCENDING,
            'descending': alg.CELL_COUNT_DESCENDING,
            'unspecified': alg.UNSPECIFIED,
        }
        if region_assignment_mode not in modes:
            msg = f"Invalid `region_assignment_mode` '{region_assignment_mode}' . Must be in {list(modes.keys())}"  # noqa: E501
            raise ValueError(msg)

        if region_assignment_mode == 'unspecified' and extraction_mode == 'specified':
            warn_external(
                'Using the `unspecified` region assignment mode with the `specified` extraction mode can be unintuitive. Ignore this warning if this was intentional.',  # noqa: E501
                UserWarning,
            )

        alg.SetRegionIdAssignmentMode(modes[region_assignment_mode])

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
                    msg = "`region_ids` must be specified when `extraction_mode='specified'`."
                    raise ValueError(msg)
                else:
                    region_ids = cast('NumpyArray[int]', variable_input)
            # this mode returns scalar data with shape that may not match
            # the number of cells/points, so we extract all and filter later
            # alg.SetExtractionModeToSpecifiedRegions()
            region_ids = _unravel_and_validate_ids(region_ids)
            # [alg.AddSpecifiedRegion(i) for i in region_ids]
            alg.SetExtractionModeToAllRegions()

        elif extraction_mode == 'cell_seed':
            if cell_ids is None:
                if variable_input is None:
                    msg = "`cell_ids` must be specified when `extraction_mode='cell_seed'`."
                    raise ValueError(msg)
                else:
                    cell_ids = cast('NumpyArray[int]', variable_input)
            alg.SetExtractionModeToCellSeededRegions()
            alg.InitializeSeedList()
            for i in _unravel_and_validate_ids(cell_ids):
                alg.AddSeed(i)

        elif extraction_mode == 'point_seed':
            if point_ids is None:
                if variable_input is None:
                    msg = "`point_ids` must be specified when `extraction_mode='point_seed'`."
                    raise ValueError(msg)
                else:
                    point_ids = cast('NumpyArray[int]', variable_input)
            alg.SetExtractionModeToPointSeededRegions()
            alg.InitializeSeedList()
            for i in _unravel_and_validate_ids(point_ids):
                alg.AddSeed(i)

        elif extraction_mode == 'closest':
            if closest_point is None:
                if variable_input is None:
                    msg = "`closest_point` must be specified when `extraction_mode='closest'`."
                    raise ValueError(msg)
                else:
                    closest_point = cast('NumpyArray[float]', variable_input)
            alg.SetExtractionModeToClosestPointRegion()
            alg.SetClosestPoint(*closest_point)

        else:
            msg = (  # type: ignore[unreachable]
                f"Invalid value for `extraction_mode` '{extraction_mode}'. "
                f"Expected one of the following: 'all', 'largest', 'specified', "
                f"'cell_seed', 'point_seed', or 'closest'"
            )
            raise ValueError(msg)

        _update_alg(
            alg, progress_bar=progress_bar, message='Finding and Labeling Connected Regions.'
        )
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

        elif extraction_mode == 'largest' and isinstance(output, pv.PolyData):
            # PolyData with 'largest' mode generates bad output with unreferenced points
            output_needs_fixing = True

        # All other extraction modes / cases may generate incorrect scalar arrays
        # e.g. 'largest' may output scalars with shape that does not match output mesh
        # e.g. 'seed' method scalars may have one RegionId, yet may contain many
        # disconnected regions. Therefore, check for correct scalars size
        elif label_regions:
            invalid_cell_scalars = output.n_cells != output.cell_data['RegionId'].size
            invalid_point_scalars = output.n_points != output.point_data['RegionId'].size
            if invalid_cell_scalars or invalid_point_scalars:
                output_needs_fixing = True

        if output_needs_fixing and output.n_cells > 0:
            # Fix bad output recursively using 'all' mode which has known good output
            output.point_data.remove('RegionId')
            output.cell_data.remove('RegionId')
            output = output.connectivity(
                'all',
                label_regions=True,
                inplace=inplace,
                region_assignment_mode=region_assignment_mode,
            )

        # Remove temp point array
        with contextlib.suppress(KeyError):
            output.point_data.remove('__point_data')

        if not label_regions and output.n_cells > 0:
            output.point_data.remove('RegionId')
            output.cell_data.remove('RegionId')

            # restore previously active scalars
            output.set_active_scalars(active_name, preference=active_field)

        output.cell_data.pop('vtkOriginalCellIds', None)
        output.point_data.pop('vtkOriginalPointIds', None)

        if inplace:
            try:
                self.copy_from(output, deep=False)
            except TypeError:
                pass
            else:
                return self
        return output

    @_deprecate_positional_args
    def extract_largest(  # type: ignore[misc]
        self: _DataSetType,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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

        See :ref:`connectivity_example` and :ref:`volumetric_analysis_example` for
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

    @_deprecate_positional_args
    def split_bodies(  # type: ignore[misc]
        self: _DataSetType,
        label: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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
        bodies = pv.MultiBlock()
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

    @_deprecate_positional_args(allowed=['scalars'])
    def warp_by_scalar(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        scalars: str | None = None,
        factor: float = 1.0,
        normal: VectorLike[float] | None = None,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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

        See :ref:`compute_normals_example` for more examples using this filter.

        """
        factor = kwargs.pop('scale_factor', factor)
        assert_empty_kwargs(**kwargs)
        scalars_ = set_default_active_scalars(self).name if scalars is None else scalars
        _ = get_array(self, scalars_, preference='point', err=True)

        field = get_array_association(self, scalars_, preference='point')
        if field != FieldAssociation.POINT:
            msg = 'Dataset can only by warped by a point data array.'
            raise TypeError(msg)
        # Run the algorithm
        alg = _vtk.vtkWarpScalar()
        alg.SetInputDataObject(self)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars_,
        )  # args: (idx, port, connection, field, name)
        alg.SetScaleFactor(factor)
        if normal is not None:
            alg.SetNormal(*normal)
            alg.SetUseNormal(True)
        _update_alg(alg, progress_bar=progress_bar, message='Warping by Scalar')
        output = _get_output(alg)
        if inplace:
            if isinstance(self, (_vtk.vtkImageData, _vtk.vtkRectilinearGrid)):
                msg = 'This filter cannot be applied inplace for this mesh type.'  # type: ignore[unreachable]
                raise TypeError(msg)
            self.copy_from(output, deep=False)
            return self
        return output

    @_deprecate_positional_args(allowed=['vectors'])
    def warp_by_vector(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        vectors: str | None = None,
        factor: float = 1.0,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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
        >>> actor = pl.add_text('Before warp')
        >>> actor = pl.add_mesh(sphere, color='white')
        >>> pl.subplot(0, 1)
        >>> actor = pl.add_text('After warp')
        >>> actor = pl.add_mesh(warped, color='white')
        >>> pl.show()

        See :ref:`warp_by_vector_example` and :ref:`warp_by_vector_eigenmodes_example` for
        more examples using this filter.

        """
        vectors_ = set_default_active_vectors(self).name if vectors is None else vectors
        arr = get_array(self, vectors_, preference='point')
        field = get_array_association(self, vectors_, preference='point')
        if arr is None:
            msg = 'No vectors present to warp by vector.'
            raise ValueError(msg)

        # check that this is indeed a vector field
        if arr.ndim != 2 or arr.shape[1] != 3:
            msg = (
                'Dataset can only by warped by a 3D vector point data array. '
                'The values you provided do not satisfy this requirement'
            )
            raise ValueError(msg)
        alg = _vtk.vtkWarpVector()
        alg.SetInputDataObject(self)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, vectors_)
        alg.SetScaleFactor(factor)
        _update_alg(alg, progress_bar=progress_bar, message='Warping by Vector')
        warped_mesh = _get_output(alg)
        if inplace:
            self.copy_from(warped_mesh, deep=False)
            return self
        else:
            return warped_mesh

    @_deprecate_positional_args
    def delaunay_3d(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        alpha: float = 0.0,
        tol: float = 0.001,
        offset: float = 2.5,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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
        _update_alg(alg, progress_bar=progress_bar, message='Computing 3D Triangulation')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['surface'])
    def select_enclosed_points(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        surface: PolyData,
        tolerance: float = 0.001,
        inside_out: bool = False,  # noqa: FBT001, FBT002
        check_surface: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Mark points as to whether they are inside a closed surface.

        This evaluates all the input points to determine whether they are in an
        enclosed surface. The filter produces a (0,1) mask
        (in the form of a :vtk:`vtkDataArray`) that indicates whether points are
        outside (mask value=0) or inside (mask value=1) a provided surface.
        (The name of the output :vtk:`vtkDataArray` is ``"SelectedPoints"``.)

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

        See Also
        --------
        :ref:`extract_cells_inside_surface_example`

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
        if not isinstance(surface, pv.PolyData):
            msg = '`surface` must be `pyvista.PolyData`'  # type: ignore[unreachable]
            raise TypeError(msg)
        if check_surface and surface.n_open_edges > 0:
            msg = (
                'Surface is not closed. Please read the warning in the '
                'documentation for this function and either pass '
                '`check_surface=False` or repair the surface.'
            )
            raise RuntimeError(msg)
        alg = _vtk.vtkSelectEnclosedPoints()
        alg.SetInputData(self)
        alg.SetSurfaceData(surface)
        alg.SetTolerance(tolerance)
        alg.SetInsideOut(inside_out)
        _update_alg(alg, progress_bar=progress_bar, message='Selecting Enclosed Points')
        result = _get_output(alg)
        out = self.copy()
        bools = result['SelectedPoints'].astype(np.uint8)
        if len(bools) < 1:
            bools = np.zeros(out.n_points, dtype=np.uint8)
        out['SelectedPoints'] = bools
        return out

    @_deprecate_positional_args(allowed=['target'])
    def interpolate(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        target: DataSet | _vtk.vtkDataSet,
        sharpness: float = 2.0,
        radius: float = 1.0,
        strategy: Literal['null_value', 'mask_points', 'closest_point'] = 'null_value',
        null_value: float = 0.0,
        n_points: int | None = None,
        pass_cell_data: bool = True,  # noqa: FBT001, FBT002
        pass_point_data: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
        :func:`pyvista.DataObjectFilters.sample` instead.

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
        pyvista.DataObjectFilters.sample
            Resample array data from one mesh onto another.

        :meth:`pyvista.ImageDataFilters.resample`
            Resample image data to modify its dimensions and spacing.

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
        >>> _ = pl.add_mesh(pdata, render_points_as_spheres=True, point_size=50)
        >>> _ = pl.add_mesh(plane, style='wireframe', line_width=5)
        >>> pl.show()

        See :ref:`interpolate_example`, :ref:`interpolate_sample_example`,
        and :ref:`resampling_example` for more examples using this filter.

        """
        # Must cast to UnstructuredGrid in some cases (e.g. vtkImageData/vtkRectilinearGrid)
        # I believe the locator and the interpolator call `GetPoints` and not all mesh types
        # have that method
        target_ = wrap(target)
        target_ = (
            target_.cast_to_unstructured_grid()
            if isinstance(target_, (pv.ImageData, pv.RectilinearGrid))
            else target_
        )

        gaussian_kernel = _vtk.vtkGaussianKernel()
        gaussian_kernel.SetSharpness(sharpness)
        gaussian_kernel.SetRadius(radius)
        gaussian_kernel.SetKernelFootprintToRadius()
        if n_points:
            gaussian_kernel.SetNumberOfPoints(n_points)
            gaussian_kernel.SetKernelFootprintToNClosest()

        locator = _vtk.vtkStaticPointLocator()
        locator.SetDataSet(target_)
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
            msg = f'strategy `{strategy}` not supported.'  # type: ignore[unreachable]
            raise ValueError(msg)
        interpolator.SetPassPointArrays(pass_point_data)
        interpolator.SetPassCellArrays(pass_cell_data)
        _update_alg(interpolator, progress_bar=progress_bar, message='Interpolating')
        return _get_output(interpolator)

    @_deprecate_positional_args(allowed=['vectors'])
    def streamlines(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        vectors: str | None = None,
        source_center: VectorLike[float] | None = None,
        source_radius: float | None = None,
        n_points: int = 100,
        start_position: VectorLike[float] | None = None,
        return_source: bool = False,  # noqa: FBT001, FBT002
        pointa: VectorLike[float] | None = None,
        pointb: VectorLike[float] | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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

        alg: _vtk.vtkAlgorithm
        if (pointa is not None and pointb is None) or (pointa is None and pointb is not None):
            msg = 'Both pointa and pointb must be provided'
            raise ValueError(msg)
        elif pointa is not None and pointb is not None:
            line_source = _vtk.vtkLineSource()
            line_source.SetPoint1(*pointa)
            line_source.SetPoint2(*pointb)
            line_source.SetResolution(n_points)
            alg = line_source
        else:
            point_source = _vtk.vtkPointSource()
            point_source.SetCenter(*source_center)
            point_source.SetRadius(source_radius)
            point_source.SetNumberOfPoints(n_points)
            alg = point_source

        alg.Update()
        input_source = cast('pv.DataSet', wrap(alg.GetOutput()))

        output = self.streamlines_from_source(
            input_source,
            vectors,
            progress_bar=progress_bar,
            **kwargs,
        )
        if return_source:
            return output, input_source
        return output

    @_deprecate_positional_args(allowed=['source', 'vectors'])
    def streamlines_from_source(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        source: _vtk.vtkDataSet,
        vectors: str | None = None,
        integrator_type: Literal[45, 2, 4] = 45,
        integration_direction: Literal['both', 'backward', 'forward'] = 'both',
        surface_streamlines: bool = False,  # noqa: FBT001, FBT002
        initial_step_length: float = 0.5,
        step_unit: Literal['cl', 'l'] = 'cl',
        min_step_length: float = 0.01,
        max_step_length: float = 1.0,
        max_steps: int = 2000,
        terminal_speed: float = 1e-12,
        max_error: float = 1e-6,
        max_time: float | None = None,
        compute_vorticity: bool = True,  # noqa: FBT001, FBT002
        rotation_scale: float = 1.0,
        interpolator_type: Literal['point', 'cell', 'p', 'c'] = 'point',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        max_length: float | None = None,
    ):
        """Generate streamlines of vectors from the points of a source mesh.

        The integration is performed using a specified integrator, by default
        Runge-Kutta45. This supports integration through any type of dataset.
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
            Specify the maximum length of a streamline expressed in physical length.

            .. deprecated:: 0.45.0
               ``max_time`` parameter is deprecated. Use ``max_length`` instead.
                It will be removed in v0.48. Default for ``max_time`` changed in v0.45.0.

        compute_vorticity : bool, default: True
            Vorticity computation at streamline points. Necessary for generating
            proper stream-ribbons using the :vtk:`vtkRibbonFilter`.

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

        max_length : float, optional
            Specify the maximum length of a streamline expressed in physical length.
            Default is 4 times the diagonal length of the bounding box of the ``source``
            dataset.

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
        integration_direction_lower = str(integration_direction).strip().lower()
        if integration_direction_lower not in ['both', 'back', 'backward', 'forward']:
            msg = (
                "Integration direction must be one of:\n 'backward', "
                f"'forward', or 'both' - not '{integration_direction_lower}'."
            )
            raise ValueError(msg)
        else:
            integration_direction_ = cast(
                'Literal["both", "back", "backward", "forward"]', integration_direction
            )
        if integrator_type not in [2, 4, 45]:
            msg = 'Integrator type must be one of `2`, `4`, or `45`.'
            raise ValueError(msg)
        if interpolator_type not in ['c', 'cell', 'p', 'point']:
            msg = "Interpolator type must be either 'cell' or 'point'"
            raise ValueError(msg)
        if step_unit not in ['l', 'cl']:
            msg = "Step unit must be either 'l' or 'cl'"
            raise ValueError(msg)
        step_unit_val = {
            'cl': _vtk.vtkStreamTracer.CELL_LENGTH_UNIT,
            'l': _vtk.vtkStreamTracer.LENGTH_UNIT,
        }[step_unit]
        if isinstance(vectors, str):
            self.set_active_scalars(vectors)
            self.set_active_vectors(vectors)
        elif vectors is None:
            set_default_active_vectors(self)

        if max_time is not None:
            if max_length is not None:
                warn_external(
                    '``max_length`` and ``max_time`` provided. Ignoring deprecated ``max_time``.',
                    PyVistaDeprecationWarning,
                )
            else:
                warn_external(
                    '``max_time`` parameter is deprecated.  It will be removed in v0.48',
                    PyVistaDeprecationWarning,
                )
                max_length = max_time

        if max_length is None:
            max_length = 4.0 * self.GetLength()

        source = wrap(source)
        # vtk throws error with two Structured Grids
        # See: https://github.com/pyvista/pyvista/issues/1373
        if isinstance(self, pv.StructuredGrid) and isinstance(source, pv.StructuredGrid):
            source = source.cast_to_unstructured_grid()

        # Build the algorithm
        alg = _vtk.vtkStreamTracer()
        # Inputs
        alg.SetInputDataObject(self)
        alg.SetSourceData(source)

        # general parameters
        alg.SetComputeVorticity(compute_vorticity)
        alg.SetInitialIntegrationStep(initial_step_length)
        alg.SetIntegrationStepUnit(step_unit_val)
        alg.SetMaximumError(max_error)
        alg.SetMaximumIntegrationStep(max_step_length)
        alg.SetMaximumNumberOfSteps(max_steps)
        alg.SetMaximumPropagation(max_length)
        alg.SetMinimumIntegrationStep(min_step_length)
        alg.SetRotationScale(rotation_scale)
        alg.SetSurfaceStreamlines(surface_streamlines)
        alg.SetTerminalSpeed(terminal_speed)
        # Model parameters
        if integration_direction_ == 'forward':
            alg.SetIntegrationDirectionToForward()
        elif integration_direction_ in ['backward', 'back']:
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
        _update_alg(alg, progress_bar=progress_bar, message='Generating Streamlines')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['vectors'])
    def streamlines_evenly_spaced_2D(  # type: ignore[misc]  # noqa: N802, PLR0917
        self: _DataSetType,
        vectors: str | None = None,
        start_position: VectorLike[float] | None = None,
        integrator_type: Literal[2, 4] = 2,
        step_length: float = 0.5,
        step_unit: Literal['cl', 'l'] = 'cl',
        max_steps: int = 2000,
        terminal_speed: float = 1e-12,
        interpolator_type: Literal['point', 'cell', 'p', 'c'] = 'point',
        separating_distance: float = 10.0,
        separating_distance_ratio: float = 0.5,
        closed_loop_maximum_distance: float = 0.5,
        loop_angle: float = 20.0,
        minimum_number_of_loop_points: int = 4,
        compute_vorticity: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
            proper stream-ribbons using the :vtk:`vtkRibbonFilter`.

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
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(streams.tube(radius=0.02), scalars='vorticity_mag')
        >>> pl.view_xy()
        >>> pl.show()

        See :ref:`streamlines_2D_example` for more examples using this filter.

        """
        if integrator_type not in [2, 4]:
            msg = 'Integrator type must be one of `2` or `4`.'
            raise ValueError(msg)
        if interpolator_type not in ['c', 'cell', 'p', 'point']:
            msg = "Interpolator type must be either 'cell' or 'point'"
            raise ValueError(msg)
        if step_unit not in ['l', 'cl']:
            msg = "Step unit must be either 'l' or 'cl'"
            raise ValueError(msg)
        step_unit_ = {
            'cl': _vtk.vtkStreamTracer.CELL_LENGTH_UNIT,
            'l': _vtk.vtkStreamTracer.LENGTH_UNIT,
        }[step_unit]
        if isinstance(vectors, str):
            self.set_active_scalars(vectors)
            self.set_active_vectors(vectors)
        elif vectors is None:
            set_default_active_vectors(self)

        loop_angle = loop_angle * np.pi / 180

        # Build the algorithm
        alg = _vtk.vtkEvenlySpacedStreamlines2D()
        # Inputs
        alg.SetInputDataObject(self)

        # Seed for starting position
        if start_position is not None:
            alg.SetStartPosition(*start_position)

        # Integrator controls
        if integrator_type == 2:
            alg.SetIntegratorTypeToRungeKutta2()
        else:
            alg.SetIntegratorTypeToRungeKutta4()
        alg.SetInitialIntegrationStep(step_length)
        alg.SetIntegrationStepUnit(step_unit_)
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
        _update_alg(
            alg,
            progress_bar=progress_bar,
            message='Generating Evenly Spaced Streamlines on a 2D Dataset',
        )
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['target_reduction'])
    def decimate_boundary(  # type: ignore[misc]
        self: _DataSetType,
        target_reduction: float = 0.5,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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

    @_deprecate_positional_args(allowed=['pointa', 'pointb'])
    def sample_over_line(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        pointa: VectorLike[float],
        pointb: VectorLike[float],
        resolution: int | None = None,
        tolerance: float | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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
        >>> _ = pl.add_mesh(pdata, render_points_as_spheres=True, point_size=50)
        >>> _ = pl.add_mesh(sample, scalars='values', line_width=10)
        >>> _ = pl.add_mesh(plane, scalars='values', style='wireframe')
        >>> pl.show()

        """
        if resolution is None:
            resolution = int(self.n_cells)
        # Make a line and sample the dataset
        line = pv.Line(pointa, pointb, resolution=resolution)
        return line.sample(self, tolerance=tolerance, progress_bar=progress_bar)

    @_deprecate_positional_args(allowed=['pointa', 'pointb'])
    def plot_over_line(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        pointa: VectorLike[float],
        pointb: VectorLike[float],
        resolution: int | None = None,
        scalars: str | None = None,
        title: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[int, int] | None = None,
        figure: bool = True,  # noqa: FBT001, FBT002
        show: bool = True,  # noqa: FBT001, FBT002
        tolerance: float | None = None,
        fname: str | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
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

        figsize : tuple(int, int), optional
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
        from matplotlib import pyplot as plt  # noqa: PLC0415

        # Sample on line
        sampled = DataSetFilters.sample_over_line(
            self,
            pointa,
            pointb,
            resolution=resolution,
            tolerance=tolerance,
            progress_bar=progress_bar,
        )

        # Get variable of interest
        scalars_ = set_default_active_scalars(self).name if scalars is None else scalars
        values = sampled.get_array(scalars_)
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
            plt.ylabel(scalars_)
        else:
            plt.ylabel(ylabel)
        if title is None:
            plt.title(f'{scalars_} Profile')
        else:
            plt.title(title)
        if fname:
            plt.savefig(fname)
        if show:  # pragma: no cover
            plt.show()

    @_deprecate_positional_args(allowed=['points'])
    def sample_over_multiple_lines(  # type: ignore[misc]
        self: _DataSetType,
        points: MatrixLike[float],
        tolerance: float | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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
        ...     [
        ...         [-0.5, -0.5, 0],
        ...         [0.5, -0.5, 0],
        ...         [0.5, 0.5, 0],
        ...     ]
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pdata, render_points_as_spheres=True, point_size=50)
        >>> _ = pl.add_mesh(sample, scalars='values', line_width=10)
        >>> _ = pl.add_mesh(plane, scalars='values', style='wireframe')
        >>> pl.show()

        """
        # Make a multiple lines and sample the dataset
        multiple_lines = pv.MultipleLines(points=points)
        return multiple_lines.sample(self, tolerance=tolerance, progress_bar=progress_bar)

    @_deprecate_positional_args
    def sample_over_circular_arc(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        pointa: VectorLike[float],
        pointb: VectorLike[float],
        center: VectorLike[float],
        resolution: int | None = None,
        tolerance: float | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
        >>> uniform['height'] = uniform.points[:, 2]
        >>> pointa = [
        ...     uniform.bounds.x_max,
        ...     uniform.bounds.y_min,
        ...     uniform.bounds.z_max,
        ... ]
        >>> pointb = [
        ...     uniform.bounds.x_max,
        ...     uniform.bounds.y_max,
        ...     uniform.bounds.z_min,
        ... ]
        >>> center = [
        ...     uniform.bounds.x_max,
        ...     uniform.bounds.y_min,
        ...     uniform.bounds.z_min,
        ... ]
        >>> sampled_arc = uniform.sample_over_circular_arc(
        ...     pointa=pointa, pointb=pointb, center=center
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
        circular_arc = pv.CircularArc(
            pointa=pointa, pointb=pointb, center=center, resolution=resolution
        )
        return circular_arc.sample(self, tolerance=tolerance, progress_bar=progress_bar)

    @_deprecate_positional_args
    def sample_over_circular_arc_normal(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        center: VectorLike[float],
        resolution: int | None = None,
        normal: VectorLike[float] | None = None,
        polar: VectorLike[float] | None = None,
        angle: float | None = None,
        tolerance: float | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
        >>> uniform['height'] = uniform.points[:, 2]
        >>> normal = [0, 0, 1]
        >>> polar = [0, 9, 0]
        >>> center = [
        ...     uniform.bounds.x_max,
        ...     uniform.bounds.y_min,
        ...     uniform.bounds.z_max,
        ... ]
        >>> arc = uniform.sample_over_circular_arc_normal(
        ...     center=center, normal=normal, polar=polar
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
        circular_arc = pv.CircularArcFromNormal(
            center=center,
            resolution=resolution,
            normal=normal,
            polar=polar,
            angle=angle,
        )
        return circular_arc.sample(self, tolerance=tolerance, progress_bar=progress_bar)

    @_deprecate_positional_args
    def plot_over_circular_arc(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        pointa: VectorLike[float],
        pointb: VectorLike[float],
        center: VectorLike[float],
        resolution: int | None = None,
        scalars: str | None = None,
        title: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[int, int] | None = None,
        figure: bool = True,  # noqa: FBT001, FBT002
        show: bool = True,  # noqa: FBT001, FBT002
        tolerance: float | None = None,
        fname: str | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
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
        >>> a = [mesh.bounds.x_min, mesh.bounds.y_min, mesh.bounds.z_max]
        >>> b = [mesh.bounds.x_max, mesh.bounds.y_min, mesh.bounds.z_min]
        >>> center = [
        ...     mesh.bounds.x_min,
        ...     mesh.bounds.y_min,
        ...     mesh.bounds.z_min,
        ... ]
        >>> mesh.plot_over_circular_arc(
        ...     a, b, center, resolution=1000, show=False
        ... )  # doctest:+SKIP

        """
        from matplotlib import pyplot as plt  # noqa: PLC0415

        # Sample on circular arc
        sampled = DataSetFilters.sample_over_circular_arc(
            self,
            pointa=pointa,
            pointb=pointb,
            center=center,
            resolution=resolution,
            tolerance=tolerance,
            progress_bar=progress_bar,
        )

        # Get variable of interest
        scalars_ = set_default_active_scalars(self).name if scalars is None else scalars
        values = sampled.get_array(scalars_)
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
            plt.ylabel(scalars_)
        else:
            plt.ylabel(ylabel)
        if title is None:
            plt.title(f'{scalars_} Profile')
        else:
            plt.title(title)
        if fname:
            plt.savefig(fname)
        if show:  # pragma: no cover
            plt.show()

    @_deprecate_positional_args
    def plot_over_circular_arc_normal(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        center: VectorLike[float],
        resolution: int | None = None,
        normal: VectorLike[float] | None = None,
        polar: VectorLike[float] | None = None,
        angle: float | None = None,
        scalars: str | None = None,
        title: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[int, int] | None = None,
        figure: bool = True,  # noqa: FBT001, FBT002
        show: bool = True,  # noqa: FBT001, FBT002
        tolerance: float | None = None,
        fname: str | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Sample a dataset along a circular arc defined by a normal and polar vector and plot it.

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

        See Also
        --------
        :ref:`plot_over_circular_arc_example`

        Examples
        --------
        Sample a dataset along a high resolution circular arc and plot.

        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> normal = normal = [0, 0, 1]
        >>> polar = [0, 9, 0]
        >>> angle = 90
        >>> center = [
        ...     mesh.bounds.x_min,
        ...     mesh.bounds.y_min,
        ...     mesh.bounds.z_min,
        ... ]
        >>> mesh.plot_over_circular_arc_normal(
        ...     center, polar=polar, angle=angle
        ... )  # doctest:+SKIP

        """
        from matplotlib import pyplot as plt  # noqa: PLC0415

        # Sample on circular arc
        sampled = DataSetFilters.sample_over_circular_arc_normal(
            self,
            center=center,
            resolution=resolution,
            normal=normal,
            polar=polar,
            angle=angle,
            tolerance=tolerance,
            progress_bar=progress_bar,
        )

        # Get variable of interest
        scalars_ = set_default_active_scalars(self).name if scalars is None else scalars
        values = sampled.get_array(scalars_)
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
            plt.ylabel(scalars_)
        else:
            plt.ylabel(ylabel)
        if title is None:
            plt.title(f'{scalars_} Profile')
        else:
            plt.title(title)
        if fname:
            plt.savefig(fname)
        if show:  # pragma: no cover
            plt.show()

    @_deprecate_positional_args(allowed=['ind'])
    def extract_cells(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        ind: int | VectorLike[int],
        invert: bool = False,  # noqa: FBT001, FBT002
        pass_cell_ids: bool = True,  # noqa: FBT001, FBT002
        pass_point_ids: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Return a subset of the grid.

        Parameters
        ----------
        ind : sequence[int]
            Numpy array of cell indices to be extracted.

        invert : bool, default: False
            Invert the selection.

        pass_point_ids : bool, default: True
            Add a point array ``'vtkOriginalPointIds'`` that identifies the original
            points the extracted points correspond to.

            .. versionadded:: 0.47

        pass_cell_ids : bool, default: True
            Add a cell array ``'vtkOriginalCellIds'`` that identifies the original cells
            the extracted cells correspond to.

            .. versionadded:: 0.47

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
        >>> actor = pl.add_mesh(grid, style='wireframe', line_width=5, color='black')
        >>> actor = pl.add_mesh(subset, color='grey')
        >>> pl.show()

        """
        if invert:
            ind_: VectorLike[int]
            _, ind_ = numpy_to_idarr(ind, return_ind=True)  # type: ignore[misc]
            mask = np.ones(self.n_cells, bool)
            mask[ind_] = False
            ids = numpy_to_idarr(mask)
        else:
            ids = numpy_to_idarr(ind)

        # Create selection objects
        selectionNode = _vtk.vtkSelectionNode()
        selectionNode.SetFieldType(_vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(_vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(ids)

        selection = _vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # Extract using a shallow copy to avoid the side effect of creating the
        # vtkOriginalPointIds and vtkOriginalCellIds arrays in the input
        # dataset.
        #
        # See: https://github.com/pyvista/pyvista/pull/7946
        ds_copy = self.copy(deep=False)

        extract_sel = _vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, ds_copy)
        extract_sel.SetInputData(1, selection)
        _update_alg(extract_sel, progress_bar=progress_bar, message='Extracting Cells')
        subgrid = _get_output(extract_sel)

        # extracts only in float32
        if subgrid.n_points and self.points.dtype != np.dtype('float32'):
            ind = subgrid.point_data['vtkOriginalPointIds']
            subgrid.points = self.points[ind]

        # Process output arrays
        if (name := 'vtkOriginalPointIds') in (data := subgrid.point_data) and not pass_point_ids:
            del data[name]
        if (name := 'vtkOriginalCellIds') in (data := subgrid.cell_data) and not pass_cell_ids:
            del data[name]
        return subgrid

    @_deprecate_positional_args(allowed=['ind'])
    def extract_points(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        ind: int | VectorLike[int] | VectorLike[bool],
        adjacent_cells: bool = True,  # noqa: FBT001, FBT002
        include_cells: bool = True,  # noqa: FBT001, FBT002
        pass_cell_ids: bool = True,  # noqa: FBT001, FBT002
        pass_point_ids: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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

        pass_point_ids : bool, default: True
            Add a point array ``'vtkOriginalPointIds'`` that identifies the original
            points the extracted points correspond to.

            .. versionadded:: 0.47

        pass_cell_ids : bool, default: True
            Add a cell array ``'vtkOriginalCellIds'`` that identifies the original cells
            the extracted cells correspond to.

            .. versionadded:: 0.47

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
        extract_sel.SetInputData(0, self.copy(deep=False))
        extract_sel.SetInputData(1, selection)
        _update_alg(extract_sel, progress_bar=progress_bar, message='Extracting Points')
        output = _get_output(extract_sel)

        # Process output arrays
        if (name := 'vtkOriginalPointIds') in (data := output.point_data) and not pass_point_ids:
            del data[name]
        if (name := 'vtkOriginalCellIds') in (data := output.cell_data) and not pass_cell_ids:
            del data[name]
        return output

    def split_values(  # type: ignore[misc]
        self: _DataSetType,
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

        This filter is a convenience method for :meth:`~pyvista.DataSetFilters.extract_values`
        with ``split`` set to ``True`` by default. Refer to that filter's documentation
        for more details.

        .. versionadded:: 0.44

        Parameters
        ----------
        values : float | ArrayLike[float] | dict, optional
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
            Additional keyword arguments passed to :meth:`~pyvista.DataSetFilters.extract_values`.

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

    def extract_values(  # type: ignore[misc]
        self: _DataSetType,
        values: (
            float | VectorLike[float] | MatrixLike[float] | dict[str, float] | dict[float, str]
        )
        | None = None,
        *,
        ranges: (
            VectorLike[float]
            | MatrixLike[float]
            | dict[str, VectorLike[float]]
            | dict[tuple[float, float], str]
        )
        | None = None,
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
        values : float | ArrayLike[float] | dict, optional
            Value(s) to extract. Can be a number, an iterable of numbers, or a dictionary
            with numeric entries. For ``dict`` inputs, either its keys or values may be
            numeric, and the other field must be strings. The numeric field is used as
            the input for this parameter, and if ``split`` is ``True``, the string field
            is used to set the block names of the returned :class:`~pyvista.MultiBlock`.

            .. note::
                When extracting multi-component values with ``component_mode=multi``,
                each value is specified as a multi-component scalar. In this case,
                ``values`` can be a single vector or an array of row vectors.

        ranges : ArrayLike[float] | dict, optional
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
        split_values
            Wrapper around this filter to split values and return a :class:`~pyvista.MultiBlock`.
        :meth:`~pyvista.ImageDataFilters.select_values`
            Similar filter specialized for :class:`~pyvista.ImageData`.
        extract_points
            Extract a subset of a mesh's points.
        extract_cells
            Extract a subset of a mesh's cells.
        threshold
            Similar filter for thresholding a mesh by value.
        partition
            Split a mesh into a number of sub-parts.

        Returns
        -------
        output : pyvista.UnstructuredGrid | pyvista.MultiBlock
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
          N Blocks:   3
          X Bounds:   0.000e+00, 1.000e+00
          Y Bounds:   0.000e+00, 1.000e+00
          Z Bounds:   0.000e+00, 5.000e+00
        >>> extracted.plot(multi_colors=True)

        Extract values from multi-component scalars.

        First, create a point cloud with a 3-component RGB color array.

        >>> rng = np.random.default_rng(seed=1)
        >>> points = rng.random((30, 3))
        >>> colors = rng.random((30, 3))
        >>> point_cloud = pv.PointSet(points)
        >>> point_cloud['colors'] = colors
        >>> plot_kwargs = dict(render_points_as_spheres=True, point_size=50, rgb=True)
        >>> point_cloud.plot(**plot_kwargs)

        Extract values from a single component.

        E.g. extract points with a strong red component (i.e. > 0.8).

        >>> extracted = point_cloud.extract_values(ranges=[0.8, 1.0], component_mode=0)
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
        validated = self._validate_extract_values(
            values=values,
            ranges=ranges,
            scalars=scalars,
            preference=preference,
            component_mode=component_mode,
            split=split,
        )
        if isinstance(validated, tuple):
            (
                valid_values,
                valid_ranges,
                value_names,
                range_names,
                array,
                _,
                association,
                component_logic,
            ) = validated
        else:
            # Return empty dataset
            return validated

        # Set default for include cells
        if include_cells is None:
            include_cells = self.n_cells > 0

        kwargs = dict(
            values=valid_values,
            ranges=valid_ranges,
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
            return self._split_values(
                method=self._extract_values,
                value_names=value_names,
                range_names=range_names,
                **kwargs,
            )

        return self._extract_values(**kwargs)

    def _validate_extract_values(  # type: ignore[misc]
        self: _DataSetType,
        *,
        values,
        ranges,
        scalars,
        preference,
        component_mode,
        split,
        mesh_type=None,
    ):
        def _validate_scalar_array(scalars_, preference_):
            # Get the scalar array and field association to use for extraction
            scalars_ = set_default_active_scalars(self).name if scalars_ is None else scalars_
            array_ = get_array(self, scalars_, preference=preference_, err=True)
            association_ = get_array_association(self, scalars_, preference=preference_)
            return array_, scalars_, association_

        def _validate_component_mode(array_, component_mode_):
            # Validate component mode and return logic function
            num_components = 1 if array_.ndim == 1 else array_.shape[1]
            if isinstance(component_mode_, (int, np.integer)) or component_mode_ in [
                '0',
                '1',
                '2',
            ]:
                component_mode_ = int(component_mode_)
                if component_mode_ > num_components - 1 or component_mode_ < 0:
                    msg = (
                        f"Invalid component index '{component_mode_}' specified for "
                        f'scalars with {num_components} component(s). '
                        f'Value must be one of: {tuple(range(num_components))}.'
                    )
                    raise ValueError(msg)
                array_ = array_[:, component_mode_] if num_components > 1 else array_
                component_logic_function = None
            elif isinstance(component_mode_, str) and component_mode_ in [
                'any',
                'all',
                'multi',
            ]:
                if array_.ndim == 1:
                    component_logic_function = None
                elif component_mode_ == 'any':
                    component_logic_function = functools.partial(np.any, axis=1)
                elif component_mode_ in ['all', 'multi']:
                    component_logic_function = functools.partial(np.all, axis=1)
            else:
                msg = (
                    f"Invalid component '{component_mode_}'. "
                    f"Must be an integer, 'any', 'all', or 'multi'."
                )
                raise ValueError(msg)
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
                    msg = "Invalid dict mapping. The dict's keys or values must contain strings."
                    raise TypeError(msg)

        def _validate_values_and_ranges(
            array_, *, values_, ranges_, num_components_, component_mode_
        ):
            # Make sure we have input values to extract
            is_multi_mode = component_mode_ == 'multi'
            if values_ is None:
                if ranges_ is None:
                    msg = 'No ranges or values were specified. At least one must be specified.'
                    raise TypeError(msg)
                elif is_multi_mode:
                    msg = (
                        f"Ranges cannot be extracted using component mode '{component_mode_}'. "
                        f'Expected {None}, got {ranges_}.'
                    )
                    raise TypeError(msg)
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
                        msg = (
                            f'Component values cannot be more than 2 dimensions. '
                            f'Got {values_.ndim}.'
                        )
                        raise ValueError(msg)
                    if values_.shape[1] != num_components_:
                        msg = (
                            f'Num components in values array ({values_.shape[1]}) must match '
                            f'num components in data array ({num_components_}).'
                        )
                        raise ValueError(msg)
                else:
                    values_ = np.atleast_1d(values_)
                    if values_.ndim > 1:
                        msg = f'Values must be one-dimensional. Got {values_.ndim}d values.'
                        raise ValueError(msg)
                if not (
                    np.issubdtype(dtype := values_.dtype, np.floating)
                    or np.issubdtype(dtype, np.integer)
                ):
                    msg = 'Values must be numeric.'
                    raise TypeError(msg)

            # Validate ranges
            if ranges_ is not None:
                ranges_ = np.atleast_2d(ranges_)
                if (ndim := ranges_.ndim) > 2:
                    msg = f'Ranges must be 2 dimensional. Got {ndim}.'
                    raise ValueError(msg)
                if not (
                    np.issubdtype(dtype := ranges_.dtype, np.floating)
                    or np.issubdtype(dtype, np.integer)
                ):
                    msg = 'Ranges must be numeric.'
                    raise TypeError(msg)
                is_valid_range = ranges_[:, 0] <= ranges_[:, 1]
                not_valid = np.invert(is_valid_range)
                if np.any(not_valid):
                    invalid_ranges = ranges_[not_valid]
                    msg = (
                        f'Invalid range {invalid_ranges[0]} specified. '
                        f'Lower value cannot be greater than upper value.'
                    )
                    raise ValueError(msg)
            return values_, ranges_

        if self.is_empty:
            # Empty input, return empty output
            mesh_type = pv.UnstructuredGrid if mesh_type is None else mesh_type
            out = mesh_type()
            if split:
                # Do basic validation just to get num blocks for multiblock
                _, values_ = _get_inputs_from_dict(values)
                _, ranges_ = _get_inputs_from_dict(ranges)
                n_values = len(np.atleast_1d(values_)) if values_ is not None else 0
                n_ranges = len(np.atleast_2d(ranges_)) if ranges_ is not None else 0
                return pv.MultiBlock([out.copy() for _ in range(n_values + n_ranges)])
            return out

        array, array_name, association = _validate_scalar_array(scalars, preference)
        array, num_components, component_logic = _validate_component_mode(array, component_mode)
        value_names, values = _get_inputs_from_dict(values)
        range_names, ranges = _get_inputs_from_dict(ranges)
        valid_values, valid_ranges = _validate_values_and_ranges(
            array,
            values_=values,
            ranges_=ranges,
            num_components_=num_components,
            component_mode_=component_mode,
        )

        return (
            valid_values,
            valid_ranges,
            value_names,
            range_names,
            array,
            array_name,
            association,
            component_logic,
        )

    def _split_values(  # type:ignore[misc]
        self: _DataSetType,
        *,
        method,
        values,
        ranges,
        value_names,
        range_names,
        **kwargs,
    ):
        # Split values and ranges separately and combine into single multiblock
        multi = pv.MultiBlock()
        if values is not None:
            value_names = value_names or [None] * len(values)
            for name, val in zip(value_names, values, strict=True):
                multi.append(method(values=[val], ranges=None, **kwargs), name)
        if ranges is not None:
            range_names = range_names or [None] * len(ranges)
            for name, rng in zip(range_names, ranges, strict=True):
                multi.append(method(values=None, ranges=[rng], **kwargs), name)
        return multi

    def _apply_component_logic_to_array(  # type: ignore[misc]
        self: _DataSetType,
        *,
        values,
        ranges,
        array,
        component_logic,
        invert,
    ):
        """Extract values using validated input.

        Internal method for extract_values filter to avoid repeated calls to input
        validation methods.
        """

        def _update_id_mask(logic_) -> None:
            """Apply component logic and update the id mask."""
            logic_ = component_logic(logic_) if component_logic else logic_
            id_mask[logic_] = True

        # Determine which ids to keep
        id_mask = np.zeros((len(array),), dtype=bool)
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

        return np.invert(id_mask) if invert else id_mask

    def _extract_values(  # type: ignore[misc]
        self: _DataSetType,
        *,
        values,
        ranges,
        array,
        component_logic,
        invert,
        association,
        adjacent_cells,
        include_cells,
        progress_bar,
        pass_point_ids,
        pass_cell_ids,
    ):
        id_mask = self._apply_component_logic_to_array(
            values=values,
            ranges=ranges,
            array=array,
            component_logic=component_logic,
            invert=invert,
        )

        # Extract point or cell ids
        if association == FieldAssociation.POINT:
            output = self.extract_points(
                id_mask,
                adjacent_cells=adjacent_cells,
                include_cells=include_cells,
                pass_point_ids=pass_point_ids,
                pass_cell_ids=pass_cell_ids,
                progress_bar=progress_bar,
            )
        else:
            output = self.extract_cells(
                id_mask,
                pass_point_ids=pass_point_ids,
                pass_cell_ids=pass_cell_ids,
                progress_bar=progress_bar,
            )

        return output

    @_deprecate_positional_args
    def extract_surface(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        pass_pointid: bool = True,  # noqa: FBT001, FBT002
        pass_cellid: bool = True,  # noqa: FBT001, FBT002
        nonlinear_subdivision: int = 1,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
        >>> surf['vtkOriginalPointIds']
        pyvista_ndarray([ 0,  2, 36, 27,  7,  8, 81,  1, 18,  4, 54,  3,  6, 45,
                         72,  5, 63,  9, 35, 44, 11, 16, 89, 17, 10, 26, 62, 13,
                         12, 53, 80, 15, 14, 71, 19, 37, 55, 20, 38, 56, 21, 39,
                         57, 22, 40, 58, 23, 41, 59, 24, 42, 60, 25, 43, 61, 28,
                         82, 29, 83, 30, 84, 31, 85, 32, 86, 33, 87, 34, 88, 46,
                         73, 47, 74, 48, 75, 49, 76, 50, 77, 51, 78, 52, 79, 64,
                         65, 66, 67, 68, 69, 70])
        >>> surf['vtkOriginalCellIds']
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

        _update_alg(surf_filter, progress_bar=progress_bar, message='Extracting Surface')
        return _get_output(surf_filter)

    @_deprecate_positional_args
    def surface_indices(  # type: ignore[misc]
        self: _DataSetType,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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

    @_deprecate_positional_args(allowed=['feature_angle'])
    def extract_feature_edges(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        feature_angle: float = 30.0,
        boundary_edges: bool = True,  # noqa: FBT001, FBT002
        non_manifold_edges: bool = True,  # noqa: FBT001, FBT002
        feature_edges: bool = True,  # noqa: FBT001, FBT002
        manifold_edges: bool = True,  # noqa: FBT001, FBT002
        clear_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
        _update_alg(featureEdges, progress_bar=progress_bar, message='Extracting Feature Edges')
        output = _get_output(featureEdges)
        if clear_data:
            output.clear_data()
        return output

    @_deprecate_positional_args
    def merge_points(  # type: ignore[misc]
        self: _DataSetType,
        tolerance: float = 0.0,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Merge duplicate points in this mesh.

        .. versionadded:: 0.45

        Parameters
        ----------
        tolerance : float, optional
            Specify a tolerance to use when comparing points. Points within
            this tolerance will be merged.

        inplace : bool, default: False
            Overwrite the original mesh with the result. Only possible if the input
            is :class:`~pyvista.PolyData` or :class:`~pyvista.UnstructuredGrid`.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        output : pyvista.PolyData | pyvista.UnstructuredGrid
            Mesh with merged points. PolyData is returned only if the input is PolyData.

        Examples
        --------
        Merge duplicate points in a mesh.

        >>> import pyvista as pv
        >>> mesh = pv.Cylinder(resolution=4)
        >>> mesh.n_points
        16
        >>> _ = mesh.merge_points(inplace=True)
        >>> mesh.n_points
        8

        """
        # Create a second mesh with points. This is required for the merge
        # to work correctly. Additional points are not required for PolyData inputs
        other_points = None if isinstance(self, pv.PolyData) else self.points
        other_mesh = pv.PolyData(other_points)
        return self.merge(
            other_mesh,
            merge_points=True,
            tolerance=tolerance,
            inplace=inplace,
            main_has_priority=None,
            progress_bar=progress_bar,
        )

    @_deprecate_positional_args(allowed=['grid'])
    def merge(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        grid: DataSet
        | _vtk.vtkDataSet
        | MultiBlock
        | Sequence[DataSet | _vtk.vtkDataSet]
        | None = None,
        merge_points: bool = True,  # noqa: FBT001, FBT002
        tolerance: float = 0.0,
        inplace: bool = False,  # noqa: FBT001, FBT002
        main_has_priority: bool | None = None,  # noqa: FBT001
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Join one or many other grids to this grid.

        Can be used to merge points of adjacent cells when no grids
        are input.

        .. note::
           The ``+`` operator between two meshes uses this filter with
           the default parameters. When the target mesh is already a
           :class:`pyvista.UnstructuredGrid`, in-place merging via
           ``+=`` is similarly possible.


        .. warning::

            The merge order of this filter depends on the installed version
            of VTK. For example, if merging meshes ``a``, ``b``, and ``c``,
            the merged order is ``bca`` for VTK<9.5 and ``abc`` for VTK>=9.5.
            This may be a breaking change for some applications. If only
            merging two meshes, it may be possible to maintain `some` backwards
            compatibility by swapping the input order of the two meshes,
            though this may also affect the merged arrays and is therefore
            not fully backwards-compatible.

        Parameters
        ----------
        grid : :vtk:`vtkUnstructuredGrid` | list[:vtk:`vtkUnstructuredGrid`], optional
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

            .. deprecated:: 0.46

                This keyword will be removed in a future version. The main mesh
                always has priority with VTK 9.5.0 or later.

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
        vtk_at_least_95 = _vtk.vtk_version_info >= (9, 5, 0)
        if main_has_priority is not None:
            msg = (
                "The keyword 'main_has_priority' is deprecated and should not be used.\n"
                'The main mesh will always have priority in a future version, and this keyword '
                'will be removed.'
            )
            if main_has_priority is False and vtk_at_least_95:
                msg += '\nIts value cannot be False for vtk>=9.5.0.'
                raise ValueError(msg)
            else:
                warn_external(msg, pv.PyVistaDeprecationWarning)
        elif not vtk_at_least_95:
            # Set default for older VTK:
            main_has_priority = True

        append_filter = _vtk.vtkAppendFilter()
        append_filter.SetMergePoints(merge_points)
        append_filter.SetTolerance(tolerance)

        # For vtk < 9.5, the last appended mesh has priority.
        # For newer vtk, the first appended mesh has priority. We apply
        # logic accordingly to ensure the main mesh is appended in the
        # correct order
        append_main_first = (not main_has_priority) or vtk_at_least_95
        if append_main_first:
            append_filter.AddInputData(self)

        if isinstance(grid, _vtk.vtkDataSet):
            append_filter.AddInputData(grid)
        elif isinstance(grid, (list, tuple, pv.MultiBlock)):
            grids = grid
            for grid_ in grids:
                append_filter.AddInputData(grid_)

        if not append_main_first:
            append_filter.AddInputData(self)

        _update_alg(append_filter, progress_bar=progress_bar, message='Merging')
        merged = _get_output(append_filter)

        if not vtk_at_least_95:
            # Update field data
            priority = (
                grid if (isinstance(grid, pv.DataObject) and not main_has_priority) else self
            )
            for array in merged.field_data:
                merged.field_data[array] = priority.field_data[array]

        if inplace:
            if type(self) is type(merged):
                self.deep_copy(merged)
                return self
            else:
                msg = f'Mesh type {type(self)} cannot be overridden by output.'
                raise TypeError(msg)
        return merged

    def __add__(  # type: ignore[misc]
        self: _DataSetType, dataset
    ):
        """Combine this mesh with another into a :class:`pyvista.UnstructuredGrid`."""
        return DataSetFilters.merge(self, dataset)

    def __iadd__(  # type: ignore[misc]
        self: _DataSetType, dataset
    ):
        """Merge another mesh into this one if possible.

        "If possible" means that ``self`` is a :class:`pyvista.UnstructuredGrid`.
        Otherwise we have to return a new object, and the attempted in-place
        merge will raise.

        """
        try:
            merged = DataSetFilters.merge(self, dataset, inplace=True)
        except TypeError:
            msg = (
                'In-place merge only possible if the target mesh '
                'is an UnstructuredGrid.\nPlease use `mesh + other_mesh` '
                'instead, which returns a new UnstructuredGrid.'
            )
            raise TypeError(msg) from None
        return merged

    @_deprecate_positional_args(allowed=['quality_measure'])
    def compute_cell_quality(  # type: ignore[misc]
        self: _DataSetType,
        quality_measure: str = 'scaled_jacobian',
        null_value: float = -1.0,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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

        .. note::

            Refer to the `Verdict Library Reference Manual <https://public.kitware.com/Wiki/images/6/6b/VerdictManual-revA.pdf>`_
            for low-level technical information about how each metric is computed,
            which :class:`~pyvista.CellType` it applies to as well as the metric's
            full, normal, and acceptable range of values.

        .. deprecated:: 0.45

            Use :meth:`~pyvista.DataObjectFilters.cell_quality` instead. Note that
            this new filter does not include an array named ``'CellQuality'``.

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
        >>> cqual = sphere.compute_cell_quality('min_angle')  # doctest:+SKIP
        >>> cqual.plot(show_edges=True)  # doctest:+SKIP

        See the :ref:`mesh_quality_example` for more examples using this filter.

        """
        if pv.version_info >= (0, 48):  # pragma: no cover
            msg = 'Convert this deprecation warning into an error.'
            raise RuntimeError(msg)
        if pv.version_info >= (0, 49):  # pragma: no cover
            msg = 'Remove this filter.'
            raise RuntimeError(msg)

        msg = (
            'This filter is deprecated. Use `cell_quality` instead. Note that this\n'
            "new filter does not include an array named ``'CellQuality'`"
        )
        warn_external(msg, PyVistaDeprecationWarning)

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
            msg = f'Cell quality type ({quality_measure}) not available. Options are: {options}'
            raise KeyError(msg)
        alg.SetInputData(self)
        alg.SetUndefinedQuality(null_value)
        _update_alg(alg, progress_bar=progress_bar, message='Computing Cell Quality')
        return _get_output(alg)

    def compute_boundary_mesh_quality(  # type: ignore[misc]
        self: _DataSetType, *, progress_bar: bool = False
    ):
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
        >>> pl = pv.Plotter(shape=(2, 2))
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>> pl.subplot(1, 0)
        >>> _ = pl.add_mesh(cqual, scalars='DistanceFromCellCenterToFaceCenter')
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(cqual, scalars='DistanceFromCellCenterToFacePlane')
        >>> pl.subplot(1, 1)
        >>> _ = pl.add_mesh(
        ...     cqual,
        ...     scalars='AngleFaceNormalAndCellCenterToFaceCenterVector',
        ... )
        >>> pl.show()

        """
        if pv.vtk_version_info < (9, 3, 0):  # pragma: no cover
            msg = '`vtkBoundaryMeshQuality` requires vtk>=9.3.0'
            raise VTKVersionError(msg)
        alg = _vtk.vtkBoundaryMeshQuality()
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Compute Boundary Mesh Quality')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['scalars'])
    def compute_derivative(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        scalars: str | None = None,
        gradient: bool | str = True,  # noqa: FBT001, FBT002
        divergence: bool | str = False,  # noqa: FBT001, FBT002
        vorticity: bool | str = False,  # noqa: FBT001, FBT002
        qcriterion: bool | str = False,  # noqa: FBT001, FBT002
        faster: bool = False,  # noqa: FBT001, FBT002
        preference: Literal['point', 'cell'] = 'point',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Compute derivative-based quantities of point/cell scalar field.

        Utilize :vtk:`vtkGradientFilter` to compute derivative-based quantities,
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
        scalars_ = set_default_active_scalars(self).name if scalars is None else scalars
        if not isinstance(scalars_, str):
            msg = 'scalars array must be given as a string name'  # type: ignore[unreachable]
            raise TypeError(msg)
        if not any((gradient, divergence, vorticity, qcriterion)):
            msg = 'must set at least one of gradient, divergence, vorticity, or qcriterion'
            raise ValueError(msg)

            # bool(non-empty string/True) == True, bool(None/False) == False
        alg.SetComputeGradient(bool(gradient))
        alg.SetResultArrayName('gradient' if isinstance(gradient, bool) else gradient)

        alg.SetComputeDivergence(bool(divergence))
        alg.SetDivergenceArrayName('divergence' if isinstance(divergence, bool) else divergence)

        alg.SetComputeVorticity(bool(vorticity))
        alg.SetVorticityArrayName('vorticity' if isinstance(vorticity, bool) else vorticity)

        alg.SetComputeQCriterion(bool(qcriterion))
        alg.SetQCriterionArrayName('qcriterion' if isinstance(qcriterion, bool) else qcriterion)

        alg.SetFasterApproximation(faster)
        field = get_array_association(self, scalars_, preference=preference)
        # args: (idx, port, connection, field, name)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars_)
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Computing Derivative')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['shrink_factor'])
    def shrink(  # type: ignore[misc]
        self: _DataSetType,
        shrink_factor: float = 1.0,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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
        shrink_factor = _validation.validate_number(
            shrink_factor,
            must_have_dtype=float,
            must_be_in_range=[0.0, 1.0],
        )
        alg = _vtk.vtkShrinkFilter()
        alg.SetInputData(self)
        alg.SetShrinkFactor(shrink_factor)
        _update_alg(alg, progress_bar=progress_bar, message='Shrinking Mesh')
        output = _get_output(alg)
        if isinstance(self, _vtk.vtkPolyData):
            return output.extract_surface()  # type: ignore[unreachable]
        return output

    @_deprecate_positional_args
    def tessellate(  # type: ignore[misc]
        self: _DataSetType,
        max_n_subdivide: int = 3,
        merge_points: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Tessellate a mesh.

        This filter approximates nonlinear FEM-like elements with linear
        simplices. The output mesh will have geometry and any fields specified
        as attributes in the input mesh's point data. The attribute's copy
        flags are honored, except for normals.

        For more details see :vtk:`vtkTessellatorFilter`.

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
            msg = 'Tessellate filter is not supported for PolyData objects.'  # type: ignore[unreachable]
            raise TypeError(msg)
        alg = _vtk.vtkTessellatorFilter()
        alg.SetInputData(self)
        alg.SetMergePoints(merge_points)
        alg.SetMaximumNumberOfSubdivisions(max_n_subdivide)
        _update_alg(alg, progress_bar=progress_bar, message='Tessellating Mesh')
        return _get_output(alg)

    @_deprecate_positional_args
    def integrate_data(  # type: ignore[misc]
        self: _DataSetType,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Integrate point and cell data.

        Area or volume is also provided in point data.

        This filter uses :vtk:`vtkIntegrateAttributes`.

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
        >>> sphere.point_data['data'] = 2 * np.ones(sphere.n_points)
        >>> integrated = sphere.integrate_data()

        There is only 1 point and cell, so access the only value.

        >>> integrated['Area'][0]
        np.float64(3.14)
        >>> integrated['data'][0]
        np.float64(6.28)

        See the :ref:`integrate_data_example` for more examples using this filter.

        """
        alg = _vtk.vtkIntegrateAttributes()
        alg.SetInputData(self)
        alg.SetDivideAllCellDataByVolume(False)
        _update_alg(alg, progress_bar=progress_bar, message='Integrating Variables')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['n_partitions'])
    def partition(  # type: ignore[misc]
        self: _DataSetType,
        n_partitions: int,
        generate_global_id: bool = False,  # noqa: FBT001, FBT002
        as_composite: bool = True,  # noqa: FBT001, FBT002
    ):
        """Break down input dataset into a requested number of partitions.

        Cells on boundaries are uniquely assigned to each partition without duplication.

        It uses a kdtree implementation that builds balances the cell
        centers among a requested number of partitions. The current implementation
        only supports power-of-2 target partition. If a non-power of two value
        is specified for ``n_partitions``, then the load balancing simply
        uses the power-of-two greater than the requested value

        For more details, see :vtk:`vtkRedistributeDataSetFilter`.

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

        as_composite : bool, default: True
            Return the partitioned dataset as a :class:`pyvista.MultiBlock`.

        See Also
        --------
        split_bodies, extract_values

        Returns
        -------
        output : pyvista.MultiBlock | pyvista.UnstructuredGrid
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
        if not hasattr(_vtk, 'vtkRedistributeDataSetFilter'):  # pragma: no cover
            msg = (
                '`partition` requires vtkRedistributeDataSetFilter, but it '
                f'was not found in VTK {pv.vtk_version_info}'
            )
            raise VTKVersionError(msg)

        alg = _vtk.vtkRedistributeDataSetFilter()
        alg.SetInputData(self)
        alg.SetNumberOfPartitions(n_partitions)
        alg.SetPreservePartitionsInOutput(True)
        alg.SetGenerateGlobalCellIds(generate_global_id)
        alg.Update()

        # pyvista does not yet support vtkPartitionedDataSet
        part = alg.GetOutput()
        datasets = [part.GetPartition(ii) for ii in range(part.GetNumberOfPartitions())]
        output = pv.MultiBlock(datasets)
        if not as_composite:
            # note, SetPreservePartitionsInOutput does not work correctly in
            # vtk 9.2.0, so instead we set it to True always and simply merge
            # the result. See:
            # https://gitlab.kitware.com/vtk/vtk/-/issues/18632
            return pv.merge(list(output), merge_points=False)
        return output

    def oriented_bounding_box(  # type: ignore[misc]
        self: _DataSetType,
        box_style: Literal['frame', 'outline', 'face'] = 'face',
        *,
        axis_0_direction: VectorLike[float] | str | None = None,
        axis_1_direction: VectorLike[float] | str | None = None,
        axis_2_direction: VectorLike[float] | str | None = None,
        frame_width: float = 0.1,
        return_meta: bool = False,
        as_composite: bool = True,
    ):
        """Return an oriented bounding box (OBB) for this dataset.

        By default, the bounding box is a :class:`~pyvista.MultiBlock` with six
        :class:`PolyData` comprising the faces of a cube. The blocks are named and
        ordered as ``('+X','-X','+Y','-Y','+Z','-Z')``.

        The box can optionally be styled as an outline or frame.

        .. note::

            The names of the blocks of the returned :class:`~pyvista.MultiBlock`
            correspond to the oriented box's local axes, not the global x-y-z axes.
            E.g. the normal of the ``'+X'`` face of the returned box has the same
            direction as the box's primary axis, and is not necessarily pointing in
            the +x direction ``(1, 0, 0)``.

        .. versionadded:: 0.45

        Parameters
        ----------
        box_style : 'frame' | 'outline' | 'face', default: 'face'
            Choose the style of the box. If ``'face'`` (default), each face of the box
            is a single quad cell. If ``'outline'``, the edges of each face are returned
            as line cells. If ``'frame'``, the center portion of each face is removed to
            create a picture-frame style border with each face having four quads (one
            for each side of the frame). Use ``frame_width`` to control the size of the
            frame.

        axis_0_direction : VectorLike[float] | str, optional
            Approximate direction vector of this mesh's primary axis. If set, the first
            axis in the returned ``axes`` metadata is flipped such that it best aligns
            with the specified vector. Can be a vector or string specifying the axis by
            name (e.g. ``'x'`` or ``'-x'``, etc.).

        axis_1_direction : VectorLike[float] | str, optional
            Approximate direction vector of this mesh's secondary axis. If set, the second
            axis in the returned ``axes`` metadata is flipped such that it best aligns
            with the specified vector. Can be a vector or string specifying the axis by
            name (e.g. ``'x'`` or ``'-x'``, etc.).

        axis_2_direction : VectorLike[float] | str, optional
            Approximate direction vector of this mesh's third axis. If set, the third
            axis in the returned ``axes`` metadata is flipped such that it best aligns
            with the specified vector. Can be a vector or string specifying the axis by
            name (e.g. ``'x'`` or ``'-x'``, etc.).

        frame_width : float, optional
            Set the width of the frame. Only has an effect if ``box_style`` is
            ``'frame'``. Values must be between ``0.0`` (minimal frame) and ``1.0``
            (large frame). The frame is scaled to ensure it has a constant width.

        return_meta : bool, default: False
            If ``True``, also returns the corner point and the three axes vectors
            defining the orientation of the box. The sign of the axes vectors can be
            controlled using the ``axis_#_direction`` arguments.

        as_composite : bool, default: True
            Return the box as a :class:`pyvista.MultiBlock` with six blocks: one for
            each face. Set this ``False`` to merge the output and return
            :class:`~pyvista.PolyData`.

        See Also
        --------
        bounding_box
            Similar filter for an axis-aligned bounding box (AABB).

        align_xyz
            Align a mesh to the world x-y-z axes. Used internally by this filter.

        pyvista.Plotter.add_bounding_box
            Add a bounding box to a scene.

        pyvista.CubeFacesSource
            Generate the faces of a cube. Used internally by this filter.

        Returns
        -------
        output : pyvista.MultiBlock | pyvista.PolyData
            MultiBlock with six named cube faces when ``as_composite=True`` and
            PolyData otherwise.

        numpy.ndarray
            The box's corner point corresponding to the origin of its axes if
            ``return_meta=True``.

        numpy.ndarray
            The box's orthonormal axes vectors if ``return_meta=True``.

        Examples
        --------
        Create a bounding box for a dataset.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_oblique_cone()
        >>> box = mesh.oriented_bounding_box()

        Plot the mesh and its bounding box.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='red')
        >>> _ = pl.add_mesh(box, opacity=0.5)
        >>> pl.show()

        Return the metadata for the box.

        >>> box, point, axes = mesh.oriented_bounding_box('outline', return_meta=True)

        Use the metadata to plot the box's axes using :class:`~pyvista.AxesAssembly`.
        The assembly is aligned with the x-y-z axes and positioned at the origin by
        default. Create a transformation to scale, then rotate, then translate the
        assembly to the corner point of the box. The transpose of the axes is used
        as an inverted rotation matrix.

        >>> scale = box.length / 4
        >>> transform = pv.Transform().scale(scale).rotate(axes.T).translate(point)
        >>> axes_assembly = pv.AxesAssembly(user_matrix=transform.matrix)

        Plot the box and the axes.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(box, color='black', line_width=5)
        >>> _ = pl.add_actor(axes_assembly)
        >>> pl.show()

        Note how the box's z-axis is pointing from the cone's tip to its base. If we
        want to flip this axis, we can "seed" its direction as the ``'-z'`` direction.

        >>> box, _, axes = mesh.oriented_bounding_box(
        ...     'outline', axis_2_direction='-z', return_meta=True
        ... )
        >>>

        Plot the box and axes again. This time, use :class:`~pyvista.AxesAssemblySymmetric`
        and position the axes in the center of the box.

        >>> center = pv.merge(box).points.mean(axis=0)
        >>> scale = box.length / 2
        >>> transform = pv.Transform().scale(scale).rotate(axes.T).translate(center)
        >>> axes_assembly = pv.AxesAssemblySymmetric(user_matrix=transform.matrix)

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(box, color='black', line_width=5)
        >>> _ = pl.add_actor(axes_assembly)
        >>> pl.show()

        """
        alg_input, matrix = self.align_xyz(
            axis_0_direction=axis_0_direction,
            axis_1_direction=axis_1_direction,
            axis_2_direction=axis_2_direction,
            return_matrix=True,
        )
        oriented = True
        inverse_matrix = Transform(matrix).inverse_matrix

        return alg_input._bounding_box(
            matrix=matrix,
            inverse_matrix=inverse_matrix,
            box_style=box_style,
            oriented=oriented,
            frame_width=frame_width,
            return_meta=return_meta,
            as_composite=as_composite,
        )

    def bounding_box(  # type: ignore[misc]
        self: _DataSetType,
        box_style: Literal['frame', 'outline', 'face'] = 'face',
        *,
        oriented: bool = False,
        frame_width: float = 0.1,
        return_meta: bool = False,
        as_composite: bool = True,
    ):
        """Return a bounding box for this dataset.

        By default, the box is an axis-aligned bounding box (AABB) returned as a
        :class:`~pyvista.MultiBlock` with six :class:`PolyData` comprising the faces of
        the box. The blocks are named and ordered as ``('+X','-X','+Y','-Y','+Z','-Z')``.

        The box can optionally be styled as an outline or frame. It may also be
        oriented to generate an oriented bounding box (OBB).

        .. versionadded:: 0.45

        Parameters
        ----------
        box_style : 'frame' | 'outline' | 'face', default: 'face'
            Choose the style of the box. If ``'face'`` (default), each face of the box
            is a single quad cell. If ``'outline'``, the edges of each face are returned
            as line cells. If ``'frame'``, the center portion of each face is removed to
            create a picture-frame style border with each face having four quads (one
            for each side of the frame). Use ``frame_width`` to control the size of the
            frame.

        oriented : bool, default: False
            Orient the box using this dataset's :func:`~pyvista.principal_axes`. This
            will generate a box that best fits this dataset's points. See
            :meth:`oriented_bounding_box` for more details.

        frame_width : float, optional
            Set the width of the frame. Only has an effect if ``box_style`` is
            ``'frame'``. Values must be between ``0.0`` (minimal frame) and ``1.0``
            (large frame). The frame is scaled to ensure it has a constant width.

        return_meta : bool, default: False
            If ``True``, also returns the corner point and the three axes vectors
            defining the orientation of the box.

        as_composite : bool, default: True
            Return the box as a :class:`pyvista.MultiBlock` with six blocks: one for
            each face. Set this ``False`` to merge the output and return
            :class:`~pyvista.PolyData` with six cells instead. The faces in both
            outputs are separate, i.e. there are duplicate points at the corners.

        See Also
        --------
        outline
            Lightweight version of this filter with fewer options.

        oriented_bounding_box
            Similar filter with ``oriented=True`` by default and more options.

        pyvista.Plotter.add_bounding_box
            Add a bounding box to a scene.

        pyvista.CubeFacesSource
            Generate the faces of a cube. Used internally by this filter.

        Returns
        -------
        output : pyvista.MultiBlock | pyvista.PolyData
            MultiBlock with six named cube faces when ``as_composite=True`` and
            PolyData otherwise.

        numpy.ndarray
            The box's corner point corresponding to the origin of its axes if
            ``return_meta=True``.

        numpy.ndarray
            The box's orthonormal axes vectors if ``return_meta=True``.

        Examples
        --------
        Create a bounding box for a dataset.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_oblique_cone()
        >>> box = mesh.bounding_box()

        Plot the mesh and its bounding box.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='red')
        >>> _ = pl.add_mesh(box, opacity=0.5)
        >>> pl.show()

        Create a frame instead.

        >>> frame = mesh.bounding_box('frame')

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='red')
        >>> _ = pl.add_mesh(frame, show_edges=True)
        >>> pl.show()

        Create an oriented bounding box (OBB) and compare it to the non-oriented one.
        Use the outline style for both.

        >>> box = mesh.bounding_box('outline')
        >>> obb = mesh.bounding_box('outline', oriented=True)

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(box, color='red', line_width=5)
        >>> _ = pl.add_mesh(obb, color='blue', line_width=5)
        >>> pl.show()

        Return the metadata for the box.

        >>> box, point, axes = mesh.bounding_box('outline', return_meta=True)

        Use the metadata to plot the box's axes using :class:`~pyvista.AxesAssembly`.
        Create the assembly and position it at the box's corner. Scale it to a fraction
        of the box's length.

        >>> scale = box.length / 4
        >>> axes_assembly = pv.AxesAssembly(position=point, scale=scale)

        Plot the box and the axes.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(box, color='black', line_width=5)
        >>> _ = pl.add_actor(axes_assembly)
        >>> _ = pl.view_yz()
        >>> pl.show()

        """
        if oriented:
            return self.oriented_bounding_box(
                box_style=box_style,
                frame_width=frame_width,
                return_meta=return_meta,
                as_composite=as_composite,
            )
        else:
            alg_input = self
            matrix = None
            inverse_matrix = None

            return alg_input._bounding_box(
                matrix=matrix,
                inverse_matrix=inverse_matrix,
                box_style=box_style,
                oriented=oriented,
                frame_width=frame_width,
                return_meta=return_meta,
                as_composite=as_composite,
            )

    def _bounding_box(  # type: ignore[misc]
        self: _DataSetType,
        *,
        matrix: NumpyArray[float] | None,
        inverse_matrix: NumpyArray[float] | None,
        box_style: Literal['frame', 'outline', 'face'],
        oriented: bool,
        frame_width: float,
        return_meta: bool,
        as_composite: bool,
    ):
        def _multiblock_to_polydata(multiblock):
            return multiblock.combine(merge_points=False).extract_geometry()

        # Validate style
        _validation.check_contains(['frame', 'outline', 'face'], must_contain=box_style)

        # Create box
        source = pv.CubeFacesSource(bounds=self.bounds)
        if box_style == 'frame':
            source.frame_width = frame_width
        box = source.output

        # Modify box
        for face in box:
            face = cast('pv.PolyData', face)
            if box_style == 'outline':
                face.copy_from(pv.lines_from_points(face.points))
            if oriented:
                face.transform(inverse_matrix, inplace=True)

        # Get output
        alg_output = box if as_composite else _multiblock_to_polydata(box)
        if return_meta:
            if not oriented:
                axes = np.eye(3)
                point = np.reshape(alg_output.bounds, (3, 2))[:, 0]  # point at min bounds
            else:
                matrix = cast('NumpyArray[float]', matrix)
                inverse_matrix = cast('NumpyArray[float]', inverse_matrix)
                axes = matrix[:3, :3]  # type: ignore[assignment]
                # We need to figure out which corner of the box to position the axes
                # To do this we compare output axes to expected axes for all 8 corners
                # of the box
                diagonals = [
                    [1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                ]
                # Choose the best-aligned axes (whichever has the largest combined dot product)
                dots = [np.dot(axes, diag) for diag in diagonals]
                match = diagonals[np.argmax(np.sum(dots, axis=1))]
                # Choose min bound for positive direction, max bound for negative
                bnds = self.bounds
                point = np.ones(3)
                point[0] = bnds.x_min if match[0] == 1 else bnds.x_max
                point[1] = bnds.y_min if match[1] == 1 else bnds.y_max
                point[2] = bnds.z_min if match[2] == 1 else bnds.z_max

                # Transform point
                point = (inverse_matrix @ [*point, 1])[:3]
                # Make sure the point we return is one of the box's points
                box_poly = (
                    _multiblock_to_polydata(alg_output)
                    if isinstance(alg_output, pv.MultiBlock)
                    else alg_output
                )
                point_id = box_poly.find_closest_point(point)
                point = box_poly.points[point_id]

            return alg_output, point, axes
        return alg_output

    def explode(  # type: ignore[misc]
        self: _DataSetType, factor: float = 0.1
    ):
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
        if not isinstance(split, pv.UnstructuredGrid):
            split = split.cast_to_unstructured_grid()

        vec = (split.cell_centers().points - split.center) * factor
        split.points += np.repeat(vec, np.diff(split.offset), axis=0)
        return split

    def separate_cells(  # type: ignore[misc]
        self: _DataSetType,
    ):
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

    @_deprecate_positional_args(allowed=['cell_types'])
    def extract_cells_by_type(  # type: ignore[misc]
        self: _DataSetType,
        cell_types: int | VectorLike[int],
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Extract cells of a specified type.

        Given an input dataset and a list of cell types, produce an output
        dataset containing only cells of the specified type(s). Note that if
        the input dataset is homogeneous (e.g., all cells are of the same type)
        and the cell type is one of the cells specified, then the input dataset
        is shallow copied to the output.

        The type of output dataset is always the same as the input type. Since
        structured types of data (i.e., :class:`pyvista.ImageData`,
        :class:`pyvista.StructuredGrid`, :class:`pyvista.RectilinearGrid`)
        are all composed of a cell of the same
        type, the output is either empty, or a shallow copy of the input.
        Unstructured data (:class:`pyvista.UnstructuredGrid`,
        :class:`pyvista.PolyData`) input may produce a subset of the input data
        (depending on the selected cell types).

        Parameters
        ----------
        cell_types :  int | VectorLike[int]
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
        valid_cell_types = _validation.validate_arrayN(
            cell_types,
            must_be_integer=True,
            name='cell_types',
        )
        for cell_type in valid_cell_types:
            alg.AddCellType(int(cell_type))
        _update_alg(alg, progress_bar=progress_bar, message='Extracting cell types')
        return _get_output(alg)

    @_deprecate_positional_args(allowed=['scalars'])
    def sort_labels(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        scalars: str | None = None,
        preference: Literal['point', 'cell'] = 'point',
        output_scalars: str | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
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
        pyvista.DataSet
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
        ...     sorted_labels['packed_labels'], return_counts=True
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

    @_deprecate_positional_args
    def pack_labels(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetType,
        sort: bool = False,  # noqa: FBT001, FBT002
        scalars: str | None = None,
        preference: Literal['point', 'cell'] = 'point',
        output_scalars: str | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
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
        This filter uses :vtk:`vtkPackLabels` as the underlying method which
        requires VTK version 9.3 or higher. If :vtk:`vtkPackLabels` is not
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
        pyvista.DataSet
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
        scalars = set_default_active_scalars(self).name if scalars is None else scalars
        field = get_array_association(self, scalars, preference=preference)

        # Determine output scalars
        default_output_scalars = 'packed_labels'
        if output_scalars is None:
            output_scalars = default_output_scalars
        if not isinstance(output_scalars, str):
            msg = f'Output scalars must be a string, got {type(output_scalars)} instead.'  # type: ignore[unreachable]
            raise TypeError(msg)

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
            _update_alg(alg, progress_bar=progress_bar, message='Packing labels')
            result = _get_output(alg)

            if output_scalars is not scalars:
                # vtkPackLabels does not pass un-packed labels through to the
                # output, so add it back here
                if field == FieldAssociation.POINT:
                    result.point_data[scalars] = self.point_data[scalars]
                else:
                    result.cell_data[scalars] = self.cell_data[scalars]
            result.rename_array('PackedLabels', output_scalars)

            if inplace:
                self.copy_from(result, deep=False)
                return self
            return result

        else:  # Use numpy
            # Get mapping from input ID to output ID
            arr = cast(
                'pv.pyvista_ndarray',
                get_array(self, scalars, preference=preference, err=True),
            )
            label_numbers_in, label_sizes = np.unique(arr, return_counts=True)
            if sort:
                label_numbers_in = label_numbers_in[np.argsort(label_sizes)[::-1]]
            label_range_in = np.arange(0, np.max(label_numbers_in))
            label_numbers_out = label_range_in[: len(label_numbers_in)]

            # Pack/sort array
            packed_array = np.zeros_like(arr)
            for num_in, num_out in zip(label_numbers_in, label_numbers_out, strict=False):
                packed_array[arr == num_in] = num_out

            result = self if inplace else self.copy(deep=True)

            # Add output to mesh
            if field == FieldAssociation.POINT:
                result.point_data[output_scalars] = packed_array
            else:
                result.cell_data[output_scalars] = packed_array

            # vtkPackLabels sets active scalars by default, so do the same here
            result.set_active_scalars(output_scalars, preference=field)  # type: ignore[arg-type]

            return result

    def color_labels(  # type: ignore[misc]
        self: DataSet,
        colors: str
        | ColorLike
        | Sequence[ColorLike]
        | dict[float, ColorLike] = 'glasbey_category10',
        *,
        coloring_mode: Literal['index', 'cycle'] | None = None,
        color_type: Literal['int_rgb', 'float_rgb', 'int_rgba', 'float_rgba'] = 'int_rgb',
        negative_indexing: bool = False,
        scalars: str | None = None,
        preference: Literal['point', 'cell'] = 'cell',
        output_scalars: str | None = None,
        return_dict: bool = False,
        inplace: bool = False,
    ):
        """Add RGB(A) scalars to labeled data.

        This filter adds a color array to map label values to specific colors.
        The mapping can be specified explicitly with a dictionary or implicitly
        with a colormap or sequence of colors. The implicit mapping is controlled
        with two coloring modes:

        -   ``'index'`` : The input scalar values (label ids) are used as index values for
            indexing the specified ``colors``. This creates a direct relationship
            between labels and colors such that a given label will always have the same
            color, regardless of the number of labels present in the dataset.

            This option is used by default for unsigned 8-bit integer inputs, i.e.
            scalars with whole numbers and a maximum range of ``[0, 255]``.

        -   ``'cycle'`` : The specified ``colors`` are cycled through sequentially,
            and each unique value in the input scalars is assigned a color in increasing
            order. Unlike with ``'index'`` mode, the colors are not directly mapped to
            the labels, but instead depends on the number of labels at the input.

            This option is used by default for floating-point inputs or for inputs
            with values out of the range ``[0, 255]``.

        By default, a new ``'int_rgb'`` array is added with the same name as the
        specified ``scalars`` but with ``_rgb`` appended.

        .. note::
            The package ``colorcet`` is required to use the default colors from the
            ``'glasbey_category10'`` colormap. For a similar, but very limited,
            alternative that does not require ``colorcet``, set ``colors='tab10'``
            and consider setting the coloring mode explicitly.

        .. versionadded:: 0.45

        See Also
        --------
        pyvista.DataSetFilters.connectivity
            Label data based on its connectivity.

        pyvista.ImageDataFilters.contour_labels
            Generate contours from labeled images. The contours may be colored with this filter.

        pack_labels
            Make labeled data contiguous. May be used as a pre-processing step before
            coloring.

        :ref:`anatomical_groups_example`
            Additional examples using this filter.

        Parameters
        ----------
        colors : str | ColorLike | Sequence[ColorLike] | dict[float, ColorLike],
            Color(s) to use. Specify a dictionary to explicitly control the mapping
            from label values to colors. Alternatively, specify colors only using a
            colormap or a sequence of colors and use ``coloring_mode`` to implicitly
            control the mapping. A single color is also supported to color the entire
            mesh with one color.

            By default,the ``'glasbey_category10'`` categorical colormap is used
            where the first 10 colors are the same default colors used by ``matplotlib``.
            See `colorcet categorical colormaps <https://colorcet.holoviz.org/user_guide/Categorical.html#>`_
            for more information.

            .. note::
                When a dictionary is specified, any scalar values for which a key is
                not provided is assigned default RGB(A) values of ``nan`` for float colors
                or ``0``  for integer colors (see ``color_type``). To ensure the color
                array has no default values, be sure to provide a mapping for any and
                all possible input label values.

        coloring_mode : 'index' | 'cycle', optional
            Control how colors are mapped to label values. Has no effect if ``colors``
            is a dictionary. Specify one of:

            - ``'index'``: The input scalar values (label ids) are used as index
              values for indexing the specified ``colors``.
            - ``'cycle'``: The specified ``colors`` are cycled through sequentially,
              and each unique value in the input scalars is assigned a color in increasing
              order. Colors are repeated if there are fewer colors than unique values
              in the input ``scalars``.

            By default, ``'index'`` mode is used if the values can be used to index
            the input ``colors``, and ``'cycle'`` mode is used otherwise.

        color_type : 'int_rgb' | 'float_rgb' | 'int_rgba' | 'float_rgba', default: 'int_rgb'
            Type of the color array to store. By default, the colors are stored as
            RGB integers to reduce memory usage.

            .. note::
                The color type affects the default value for unspecified colors when
                a dictionary is used. See ``colors`` for details.

        negative_indexing : bool, default: False
            Allow indexing ``colors`` with negative values. Only valid when
            ``coloring_mode`` is ``'index'``. This option is useful for coloring data
            with two independent categories since positive values will be colored
            differently than negative values.

        scalars : str, optional
            Name of scalars with labels. Defaults to currently active scalars.

        preference : str, default: "cell"
            When ``scalars`` is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        output_scalars : str, optional
            Name of the color scalars array. By default, the output array
            is the same as ``scalars`` with ``_rgb`` or ``_rgba`` appended
            depending on ``color_type``.

        return_dict : bool, default: False
            Return a dictionary with a mapping from label values to the
            colors applied by the filter. The colors have the same type
            specified by ``color_type``.

            .. versionadded:: 0.46

        inplace : bool, default: False
            If ``True``, the mesh is updated in-place.

        Returns
        -------
        pyvista.DataSet
            Dataset with RGB(A) scalars. Output type matches input type.

        Examples
        --------
        Load labeled data and crop it with :meth:`~pyvista.ImageDataFilters.extract_subset`
        to simplify the data.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> import numpy as np
        >>> image_labels = examples.load_channels()
        >>> image_labels = image_labels.extract_subset(voi=(75, 109, 75, 109, 85, 100))

        Plot the dataset with default coloring using a categorical color map. The
        plotter by default uniformly samples from all 256 colors in the color map based
        on the data's range.

        >>> image_labels.plot(cmap='glasbey_category10')

        Show label ids of the dataset.

        >>> label_ids = np.unique(image_labels.active_scalars)
        >>> label_ids
        pyvista_ndarray([0, 1, 2, 3, 4])

        Color the labels with the filter. Note that the
        ``'glasbey_category10'`` color map is used by default.

        >>> colored_labels, color_dict = image_labels.color_labels(return_dict=True)

        Plot the labels. We define a helper function to include a legend with the plot.

        >>> def labels_plotter(mesh, color_dict):
        ...     pl = pv.Plotter()
        ...     pl.add_mesh(mesh)
        ...     pl.add_legend(color_dict)
        ...     return pl

        >>> labels_plotter(colored_labels, color_dict).show()

        Since the labels are unsigned integers, the ``'index'`` coloring mode is used
        by default. Unlike the uniform sampling used by the plotter in the previous
        plot, the colormap is instead indexed using the label values. This ensures
        that labels have a consistent coloring regardless of the input. For example,
        we can crop the dataset further.

        >>> subset_labels = image_labels.extract_subset(voi=(15, 34, 28, 34, 12, 15))

        And show that only three labels remain.

        >>> label_ids = np.unique(subset_labels.active_scalars)
        >>> label_ids
        pyvista_ndarray([1, 2, 3])

        Despite the changes to the dataset, the regions have the same coloring
        as before.

        >>> colored_labels, color_dict = subset_labels.color_labels(return_dict=True)
        >>> labels_plotter(colored_labels, color_dict).show()

        Use the ``'cycle'`` coloring mode instead to map label values to colors
        sequentially.

        >>> colored_labels, color_dict = subset_labels.color_labels(
        ...     coloring_mode='cycle', return_dict=True
        ... )
        >>> labels_plotter(colored_labels, color_dict).show()

        Map the colors explicitly using a dictionary.

        >>> color_dict = {0: 'black', 1: 'red', 2: 'lime', 3: 'blue', 4: 'gold'}
        >>> colored_labels = image_labels.color_labels(color_dict)
        >>> labels_plotter(colored_labels, color_dict).show()

        Omit the background value from the mapping and specify float colors. When
        floats are specified, values without a mapping are assigned ``nan`` values
        and are not plotted by default.

        >>> color_dict.pop(0)
        'black'
        >>> colored_labels = image_labels.color_labels(
        ...     color_dict, color_type='float_rgba'
        ... )
        >>> labels_plotter(colored_labels, color_dict).show()

        Modify the scalars and make two of the labels negative.

        >>> scalars = image_labels.active_scalars
        >>> scalars[scalars > 2] *= -1
        >>> np.unique(scalars)
        pyvista_ndarray([-4, -3,  0,  1,  2])

        Color the mesh and enable ``negative_indexing``. With this option enabled,
        the ``'index'`` coloring mode is used by default, and therefore the positive
        values ``0``, ``1``, and ``2`` are colored with the first, second, and third
        color in the colormap, respectively. Negative values ``-3`` and ``-4`` are
        colored with the third-last and fourth-last color in the colormap, respectively.

        >>> colored_labels, color_dict = image_labels.color_labels(
        ...     negative_indexing=True, return_dict=True
        ... )
        >>> labels_plotter(colored_labels, color_dict).show()

        If ``negative_indexing`` is disabled, the coloring defaults to the
        ``'cycle'`` coloring mode instead.

        >>> colored_labels, color_dict = image_labels.color_labels(
        ...     negative_indexing=False, return_dict=True
        ... )
        >>> labels_plotter(colored_labels, color_dict).show()

        Load the :func:`~pyvista.examples.downloads.download_foot_bones` dataset.

        >>> dataset = examples.download_foot_bones()

        Label the bones using :meth:`~pyvista.DataSetFilters.connectivity` and show
        the label values.

        >>> labeled_data = dataset.connectivity()
        >>> np.unique(labeled_data.active_scalars)
        pyvista_ndarray([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

        Color the dataset with default arguments. Despite having 26 separately colored
        regions, the colors from the default glasbey-style colormap are all relatively
        distinct.

        >>> colored_labels = labeled_data.color_labels()
        >>> colored_labels.plot()

        Color the mesh with fewer colors than there are label values. In this case
        the ``'cycle'`` mode is used by default and the colors are reused.

        >>> colored_labels = labeled_data.color_labels(['red', 'lime', 'blue'])
        >>> colored_labels.plot()

        Color all labels with a single color.

        >>> colored_labels = labeled_data.color_labels('red')
        >>> colored_labels.plot()


        """
        # Lazy import since these are from plotting module
        import matplotlib.colors  # noqa: PLC0415

        from pyvista.core._validation.validate import _validate_color_sequence  # noqa: PLC0415
        from pyvista.plotting.colors import get_cmap_safe  # noqa: PLC0415

        def _local_validate_color_sequence(
            seq: ColorLike | Sequence[ColorLike],
        ) -> Sequence[Color]:
            try:
                return _validate_color_sequence(seq)
            except ValueError:
                msg = (
                    'Invalid colors. Colors must be one of:\n'
                    '  - sequence of color-like values,\n'
                    '  - dict with color-like values,\n'
                    '  - named colormap string.\n'
                    f'Got: {seq}'
                )
                raise ValueError(msg)

        def _is_index_like(array_, max_value):
            min_value = -max_value if negative_indexing else 0
            return (array_ == np.floor(array_)) & (array_ >= min_value) & (array_ <= max_value)

        _validation.check_contains(
            ['int_rgb', 'float_rgb', 'int_rgba', 'float_rgba'],
            must_contain=color_type,
            name='color_type',
        )

        if 'rgba' in color_type:
            num_components = 4
            scalars_suffix = '_rgba'
        else:
            num_components = 3
            scalars_suffix = '_rgb'
        if 'float' in color_type:
            default_channel_value = np.nan
            color_dtype = 'float'
        else:
            default_channel_value = 0
            color_dtype = 'uint8'

        if scalars is None:
            field, name = set_default_active_scalars(self)
        else:
            name = scalars
            field = get_array_association(self, name, preference=preference, err=True)
        output_mesh = self if inplace else self.copy()
        data = output_mesh.point_data if field == FieldAssociation.POINT else output_mesh.cell_data
        array = data[name]

        if isinstance(colors, dict):
            if coloring_mode is not None:
                msg = 'Coloring mode cannot be set when a color dictionary is specified.'
                raise TypeError(msg)
            colors_ = _local_validate_color_sequence(
                cast('list[ColorLike]', list(colors.values()))
            )
            color_rgb_sequence = [getattr(c, color_type) for c in colors_]
            items = zip(colors.keys(), color_rgb_sequence, strict=True)

        else:
            if array.ndim > 1:
                msg = (
                    f'Multi-component scalars are not supported for coloring. '
                    f'Scalar array {scalars} must be one-dimensional.'
                )
                raise ValueError(msg)
            _is_rgb_sequence = False
            if isinstance(colors, str):
                try:
                    cmap = get_cmap_safe(cast('ColormapOptions', colors))
                except ValueError:
                    pass
                else:
                    if not isinstance(cmap, matplotlib.colors.ListedColormap):
                        msg = (
                            f"Colormap '{colors}' must be a ListedColormap, "
                            f'got {cmap.__class__.__name__} instead.'
                        )
                        raise TypeError(msg)
                    # Avoid unnecessary conversion and set color sequence directly in float cases
                    cmap_colors = cast('list[list[float]]', cmap.colors)
                    if color_type == 'float_rgb':
                        color_rgb_sequence = cmap_colors
                        _is_rgb_sequence = True
                    elif color_type == 'float_rgba':
                        color_rgb_sequence = [(*c, 1.0) for c in cmap_colors]
                        _is_rgb_sequence = True
                    else:
                        colors = cmap_colors

            if not _is_rgb_sequence:
                color_rgb_sequence = [
                    getattr(c, color_type) for c in _local_validate_color_sequence(colors)
                ]
                if len(color_rgb_sequence) == 1:
                    color_rgb_sequence = color_rgb_sequence * len(array)

            n_colors = len(color_rgb_sequence)
            if coloring_mode is None:
                coloring_mode = (
                    'index' if np.all(_is_index_like(array, max_value=n_colors)) else 'cycle'
                )

            _validation.check_contains(
                ['index', 'cycle'], must_contain=coloring_mode, name='coloring_mode'
            )
            if coloring_mode == 'index':
                if not np.all(_is_index_like(array, max_value=n_colors)):
                    msg = (
                        f"Index coloring mode cannot be used with scalars '{name}'. "
                        f'Scalars must be positive integers \n'
                        f'and the max value ({self.get_data_range(name)[1]}) must be less '
                        f'than the number of colors ({n_colors}).'
                    )
                    raise ValueError(msg)
                keys: Iterable[float]
                values: Iterable[Any]

                keys_ = np.arange(n_colors)
                values_ = color_rgb_sequence
                if negative_indexing:
                    keys_ = np.append(keys_, keys_[::-1] - len(keys_))
                    values_.extend(values_[::-1])
                keys = keys_
                values = values_
            elif coloring_mode == 'cycle':
                if negative_indexing:
                    msg = "Negative indexing is not supported with 'cycle' mode enabled."
                    raise ValueError(msg)
                keys = np.unique(array)
                values = itertools.cycle(color_rgb_sequence)

            items = zip(keys, values, strict=False)

        colors_out = np.full(
            (len(array), num_components), default_channel_value, dtype=color_dtype
        )
        mapping = {}
        for label, color in items:
            mask = array == label
            if np.any(mask):
                colors_out[mask, :] = color
                if return_dict:
                    mapping[label] = color

        colors_name = name + scalars_suffix if output_scalars is None else output_scalars
        data[colors_name] = colors_out
        output_mesh.set_active_scalars(colors_name)

        if return_dict:
            return output_mesh, mapping
        return output_mesh

    def voxelize_binary_mask(  # type: ignore[misc]
        self: DataSet,
        *,
        background_value: int | float = 0,  # noqa: PYI041
        foreground_value: int | float = 1,  # noqa: PYI041
        reference_volume: ImageData | None = None,
        dimensions: VectorLike[int] | None = None,
        spacing: float | VectorLike[float] | None = None,
        rounding_func: Callable[[VectorLike[float]], VectorLike[int]] | None = None,
        cell_length_percentile: float | None = None,
        cell_length_sample_size: int | None = None,
        progress_bar: bool = False,
    ):
        """Voxelize mesh as a binary :class:`~pyvista.ImageData` mask.

        The binary mask is a point data array where points inside and outside of the
        input surface are labelled with ``foreground_value`` and ``background_value``,
        respectively.

        This filter implements :vtk:`vtkPolyDataToImageStencil`. This
        algorithm operates as follows:

        * The algorithm iterates through the z-slice of the ``reference_volume``.
        * For each slice, it cuts the input :class:`~pyvista.PolyData` surface to create
          2D polylines at that z position. It attempts to close any open polylines.
        * For each x position along the polylines, the corresponding y positions are
          determined.
        * For each slice, the grid points are labelled as foreground or background based
          on their xy coordinates.

        The voxelization can be controlled in several ways:

        #. Specify the output geometry using a ``reference_volume``.

        #. Specify the ``spacing`` explicitly.

        #. Specify the ``dimensions`` explicitly.

        #. Specify the ``cell_length_percentile``. The spacing is estimated from the
           surface's cells using the specified percentile.

        Use ``reference_volume`` for full control of the output mask's geometry. For
        all other options, the geometry is implicitly defined such that the generated
        mask fits the bounds of the input surface.

        If no inputs are provided, ``cell_length_percentile=0.1`` (10th percentile) is
        used by default to estimate the spacing. On systems with VTK < 9.2, the default
        spacing is set to ``1/100`` of the input mesh's length.

        .. versionadded:: 0.45.0

        .. note::
            For best results, ensure the input surface is a closed surface. The
            surface is considered closed if it has zero :attr:`~pyvista.PolyData.n_open_edges`.

        .. note::
            This filter returns voxels represented as point data, not
            :attr:`~pyvista.CellType.VOXEL` cells.
            This differs from :func:`~pyvista.voxelize` and :func:`~pyvista.voxelize_volume`
            which return meshes with voxel cells. See :ref:`image_representations_example`
            for examples demonstrating the difference.

        .. note::
            This filter does not discard internal surfaces, due, for instance, to
            intersecting meshes. Instead, the intersection will be considered as
            background which may produce unexpected results. See `Examples`.

        Parameters
        ----------
        background_value : int, default: 0
            Background value of the generated mask.

        foreground_value : int, default: 1
            Foreground value of the generated mask.

        reference_volume : ImageData, optional
            Volume to use as a reference. The output will have the same ``dimensions``,
            ``origin``, ``spacing``, and ``direction_matrix`` as the reference.

        dimensions : VectorLike[int], optional
            Dimensions of the generated mask image. Set this value to control the
            dimensions explicitly. If unset, the dimensions are defined implicitly
            through other parameter. See summary and examples for details.

        spacing : float | VectorLike[float], optional
            Approximate spacing to use for the generated mask image. Set this value
            to control the spacing explicitly. If unset, the spacing is defined
            implicitly through other parameters. See summary and examples for details.

        rounding_func : Callable[VectorLike[float], VectorLike[int]], optional
            Control how the dimensions are rounded to integers based on the provided or
            calculated ``spacing``. Should accept a length-3 vector containing the
            dimension values along the three directions and return a length-3 vector.
            :func:`numpy.round` is used by default.

            Rounding the dimensions implies rounding the actual spacing.

            Has no effect if ``reference_volume`` or ``dimensions`` are specified.

        cell_length_percentile : float, optional
            Cell length percentage ``p`` to use for computing the default ``spacing``.
            Default is ``0.1`` (10th percentile) and must be between ``0`` and ``1``.
            The ``p``-th percentile is computed from the cumulative distribution function
            (CDF) of lengths which are representative of the cell length scales present
            in the input. The CDF is computed by:

            #. Triangulating the input cells.
            #. Sampling a subset of up to ``cell_length_sample_size`` cells.
            #. Computing the distance between two random points in each cell.
            #. Inserting the distance into an ordered set to create the CDF.

            Has no effect if ``dimensions`` or ``reference_volume`` are specified.

            .. note::
                This option is only available for VTK 9.2 or greater.

        cell_length_sample_size : int, optional
            Number of samples to use for the cumulative distribution function (CDF)
            when using the ``cell_length_percentile`` option. ``100 000`` samples are
            used by default.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        ImageData
            Generated binary mask with a ``'mask'``  point data array. The data array
            has dtype :class:`numpy.uint8` if the foreground and background values are
            unsigned and less than 256.

        See Also
        --------
        voxelize
            Similar function that returns a :class:`~pyvista.UnstructuredGrid` of
            :attr:`~pyvista.CellType.VOXEL` cells.

        voxelize_rectilinear
            Similar function that returns a :class:`~pyvista.RectilinearGrid` with cell data.

        pyvista.ImageDataFilters.contour_labels
            Filter that generates surface contours from labeled image data. Can be
            loosely considered as an inverse of this filter.

        pyvista.ImageDataFilters.points_to_cells
            Convert voxels represented as points to :attr:`~pyvista.CellType.VOXEL`
            cells.

        ImageData
            Class used to build custom ``reference_volume``.

        Examples
        --------
        Generate a binary mask from a coarse mesh.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> poly = examples.download_bunny_coarse()
        >>> mask = poly.voxelize_binary_mask()

        The mask is stored as :class:`~pyvista.ImageData` with point data scalars
        (zeros for background, ones for foreground).

        >>> mask
        ImageData (...)
          N Cells:      7056
          N Points:     8228
          X Bounds:     -1.245e-01, 1.731e-01
          Y Bounds:     -1.135e-01, 1.807e-01
          Z Bounds:     -1.359e-01, 9.140e-02
          Dimensions:   22, 22, 17
          Spacing:      1.417e-02, 1.401e-02, 1.421e-02
          N Arrays:     1

        >>> np.unique(mask.point_data['mask'])
        pyvista_ndarray([0, 1], dtype=uint8)

        To visualize it as voxel cells, use :meth:`~pyvista.ImageDataFilters.points_to_cells`,
        then use :meth:`~pyvista.DataSetFilters.threshold` to extract the foreground.

        We also plot the voxel cells in blue and the input poly data in green for
        comparison.

        >>> def mask_and_polydata_plotter(mask, poly):
        ...     voxel_cells = mask.points_to_cells().threshold(0.5)
        ...
        ...     plot = pv.Plotter()
        ...     _ = plot.add_mesh(voxel_cells, color='blue')
        ...     _ = plot.add_mesh(poly, color='lime')
        ...     plot.camera_position = 'xy'
        ...     return plot

        >>> plot = mask_and_polydata_plotter(mask, poly)
        >>> plot.show()

        The spacing of the mask image is automatically adjusted to match the
        density of the input.

        Repeat the previous example with a finer mesh.

        >>> poly = examples.download_bunny()
        >>> mask = poly.voxelize_binary_mask()
        >>> plot = mask_and_polydata_plotter(mask, poly)
        >>> plot.show()

        Control the spacing manually instead. Here, a very coarse spacing is used.

        >>> mask = poly.voxelize_binary_mask(spacing=(0.01, 0.04, 0.02))
        >>> plot = mask_and_polydata_plotter(mask, poly)
        >>> plot.show()

        Note that the spacing is only approximate. Check the mask's actual spacing.

        >>> mask.spacing
        (0.009731187485158443, 0.03858340159058571, 0.020112216472625732)

        The actual values may be greater or less than the specified values. Use
        ``rounding_func=np.floor`` to force all values to be greater.

        >>> mask = poly.voxelize_binary_mask(
        ...     spacing=(0.01, 0.04, 0.02), rounding_func=np.floor
        ... )
        >>> mask.spacing
        (0.01037993331750234, 0.05144453545411428, 0.020112216472625732)

        Set the dimensions instead of the spacing.

        >>> mask = poly.voxelize_binary_mask(dimensions=(10, 20, 30))
        >>> plot = mask_and_polydata_plotter(mask, poly)
        >>> plot.show()

        >>> mask.dimensions
        (10, 20, 30)

        Create a mask using a reference volume. First generate polydata from
        an existing mask.

        >>> volume = examples.load_frog_tissues()
        >>> poly = volume.contour_labels()

        Now create the mask from the polydata using the volume as a reference.

        >>> mask = poly.voxelize_binary_mask(reference_volume=volume)
        >>> plot = mask_and_polydata_plotter(mask, poly)
        >>> plot.show()

        Visualize the effect of internal surfaces.

        >>> mesh = pv.Cylinder() + pv.Cylinder(center=(0, 0.75, 0))
        >>> binary_mask = mesh.voxelize_binary_mask(
        ...     dimensions=(1, 100, 50)
        ... ).points_to_cells()
        >>> plot = pv.Plotter()
        >>> _ = plot.add_mesh(binary_mask)
        >>> _ = plot.add_mesh(mesh.slice(), color='red')
        >>> plot.show(cpos='yz')

        Note how the intersection is excluded from the mask.
        To include the voxels delimited by internal surfaces in the foreground, the internal
        surfaces should be removed, for instance by applying a boolean union. Note that
        this operation in unreliable in VTK but may be performed with external tools such
        as `vtkbool <https://github.com/zippy84/vtkbool>`_.

        Alternatively, the intersecting parts of the mesh can be processed sequentially.

        >>> cylinder_1 = pv.Cylinder()
        >>> cylinder_2 = pv.Cylinder(center=(0, 0.75, 0))

        >>> reference_volume = pv.ImageData(
        ...     dimensions=(1, 100, 50),
        ...     spacing=(1, 0.0175, 0.02),
        ...     origin=(0, -0.5 + 0.0175 / 2, -0.5 + 0.02 / 2),
        ... )

        >>> binary_mask_1 = cylinder_1.voxelize_binary_mask(
        ...     reference_volume=reference_volume
        ... ).points_to_cells()
        >>> binary_mask_2 = cylinder_2.voxelize_binary_mask(
        ...     reference_volume=reference_volume
        ... ).points_to_cells()

        >>> binary_mask_1['mask'] = binary_mask_1['mask'] | binary_mask_2['mask']

        >>> plot = pv.Plotter()
        >>> _ = plot.add_mesh(binary_mask_1)
        >>> _ = plot.add_mesh(cylinder_1.slice(), color='red')
        >>> _ = plot.add_mesh(cylinder_2.slice(), color='red')
        >>> plot.show(cpos='yz')

        When multiple internal surfaces are nested, they are successively treated as
        interfaces between background and foreground.

        >>> mesh = pv.Tube(radius=2) + pv.Tube(radius=3) + pv.Tube(radius=4)
        >>> binary_mask = mesh.voxelize_binary_mask(
        ...     dimensions=(1, 50, 50)
        ... ).points_to_cells()
        >>> plot = pv.Plotter()
        >>> _ = plot.add_mesh(binary_mask)
        >>> _ = plot.add_mesh(mesh.slice(), color='red')
        >>> plot.show(cpos='yz')

        """
        surface = wrap(self).extract_geometry()
        if not (surface.faces.size or surface.strips.size):
            # we have a point cloud or an empty mesh
            msg = 'Input mesh must have faces for voxelization.'
            raise ValueError(msg)

        def _preprocess_polydata(poly_in):
            return poly_in.compute_normals().triangulate()

        if reference_volume is not None:
            if (
                dimensions is not None
                or spacing is not None
                or rounding_func is not None
                or cell_length_percentile is not None
                or cell_length_sample_size is not None
            ):
                msg = (
                    'Cannot specify a reference volume with other geometry parameters. '
                    '`reference_volume` must define the geometry exclusively.'
                )
                raise TypeError(msg)
            _validation.check_instance(reference_volume, pv.ImageData, name='reference volume')
            # The image stencil filters do not support orientation, so we apply the
            # inverse direction matrix to "remove" orientation from the polydata
            poly_ijk = surface.transform(reference_volume.direction_matrix.T, inplace=False)
            poly_ijk = _preprocess_polydata(poly_ijk)
        else:
            # Compute reference volume geometry
            if spacing is not None and dimensions is not None:
                msg = 'Spacing and dimensions cannot both be set. Set one or the other.'
                raise TypeError(msg)

            # Need to preprocess so that we have a triangle mesh for computing
            # cell length percentile
            poly_ijk = _preprocess_polydata(surface)

            if spacing is None:
                # Estimate spacing from cell length percentile
                cell_length_percentile = (
                    0.1 if cell_length_percentile is None else cell_length_percentile
                )
                cell_length_sample_size = (
                    100_000 if cell_length_sample_size is None else cell_length_sample_size
                )
                spacing = _length_distribution_percentile(
                    poly_ijk,
                    cell_length_percentile,
                    cell_length_sample_size,
                    progress_bar=progress_bar,
                )
            # Spacing is specified directly. Make sure other params are not set.
            elif cell_length_percentile is not None or cell_length_sample_size is not None:
                msg = 'Spacing and cell length options cannot both be set. Set one or the other.'
                raise TypeError(msg)

            # Get initial spacing (will be adjusted later)
            initial_spacing = _validation.validate_array3(spacing, broadcast=True)

            # Get size of poly data for computing dimensions
            size = np.array(surface.bounds_size)

            if dimensions is None:
                rounding_func = np.round if rounding_func is None else rounding_func
                initial_dimensions = size / initial_spacing
                # Make sure we don't round dimensions to zero, make it one instead
                initial_dimensions[initial_dimensions < 1] = 1
                dimensions = np.array(rounding_func(initial_dimensions), dtype=int)
            elif rounding_func is not None:
                msg = (
                    'Rounding func cannot be set when dimensions is specified. '
                    'Set one or the other.'
                )
                raise TypeError(msg)

            reference_volume = pv.ImageData()
            reference_volume.dimensions = dimensions
            # Dimensions are now fixed, now adjust spacing to match poly data bounds
            # Since we are dealing with voxels as points, we want the bounds of the
            # points to be 1/2 spacing width smaller than the polydata bounds
            final_spacing = size / np.array(reference_volume.dimensions)
            reference_volume.spacing = final_spacing
            reference_volume.origin = np.array(surface.bounds[::2]) + final_spacing / 2

        # Init output structure. The image stencil filters do not support
        # orientation, so we do not set the direction matrix
        binary_mask = pv.ImageData()
        binary_mask.dimensions = reference_volume.dimensions
        binary_mask.spacing = reference_volume.spacing
        binary_mask.origin = reference_volume.origin

        # Init output scalars. Use uint8 dtype if possible.
        scalars_shape = (binary_mask.n_points,)
        scalars_dtype: type[np.uint8 | float | int]
        if all(
            isinstance(val, int) and val < 256 and val >= 0
            for val in (background_value, foreground_value)
        ):
            scalars_dtype = np.uint8
        elif all(round(val) == val for val in (background_value, foreground_value)):
            scalars_dtype = np.int_
        else:
            scalars_dtype = np.float64
        scalars = (  # Init with background value
            np.zeros(scalars_shape, dtype=scalars_dtype)
            if background_value == 0
            else np.ones(scalars_shape, dtype=scalars_dtype) * background_value
        )
        binary_mask['mask'] = scalars  # type: ignore[assignment]
        # Make sure that we have a clean triangle-strip polydata
        # Note: Poly was partially pre-processed earlier
        poly_ijk = poly_ijk.strip()

        # Convert polydata to stencil
        poly_to_stencil = _vtk.vtkPolyDataToImageStencil()
        poly_to_stencil.SetInputData(poly_ijk)
        poly_to_stencil.SetOutputSpacing(*reference_volume.spacing)
        poly_to_stencil.SetOutputOrigin(*reference_volume.origin)  # type: ignore[call-overload]
        poly_to_stencil.SetOutputWholeExtent(*reference_volume.extent)
        _update_alg(poly_to_stencil, progress_bar=progress_bar, message='Converting polydata')

        # Convert stencil to image
        stencil = _vtk.vtkImageStencil()
        stencil.SetInputData(binary_mask)
        stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
        stencil.ReverseStencilOn()
        stencil.SetBackgroundValue(foreground_value)
        _update_alg(stencil, progress_bar=progress_bar, message='Generating binary mask')
        output_volume = _get_output(stencil)

        # Set the orientation of the output
        output_volume.direction_matrix = reference_volume.direction_matrix

        return output_volume

    def _voxelize_binary_mask_cells(  # type: ignore[misc]
        self: DataSet,
        *,
        background_value: float = 0.0,
        foreground_value: float = 1.0,
        reference_volume: ImageData | None,
        dimensions: VectorLike[int] | None,
        spacing: float | VectorLike[float] | None,
        rounding_func: Callable[[VectorLike[float]], VectorLike[int]] | None,
        cell_length_percentile: float | None,
        cell_length_sample_size: int | None,
        progress_bar: bool,
    ):
        if dimensions is not None:
            dimensions_ = _validation.validate_array3(
                dimensions, must_be_integer=True, dtype_out=int, name='dimensions'
            )
            dimensions = cast('NumpyArray[int]', dimensions_) - 1

        binary_mask = self.voxelize_binary_mask(
            background_value=background_value,
            foreground_value=foreground_value,
            reference_volume=reference_volume,
            dimensions=dimensions,
            spacing=spacing,
            rounding_func=rounding_func,
            cell_length_percentile=cell_length_percentile,
            cell_length_sample_size=cell_length_sample_size,
            progress_bar=progress_bar,
        )
        return binary_mask.points_to_cells(dimensionality='3D', copy=False)

    def voxelize_rectilinear(  # type: ignore[misc]
        self: DataSet,
        *,
        background_value: int | float = 0,  # noqa: PYI041
        foreground_value: int | float = 1,  # noqa: PYI041
        reference_volume: ImageData | None = None,
        dimensions: VectorLike[int] | None = None,
        spacing: float | VectorLike[float] | None = None,
        rounding_func: Callable[[VectorLike[float]], VectorLike[int]] | None = None,
        cell_length_percentile: float | None = None,
        cell_length_sample_size: int | None = None,
        progress_bar: bool = False,
    ) -> RectilinearGrid:
        """Voxelize mesh to create a RectilinearGrid voxel volume.

        The voxelization can be controlled in several ways:

        #. Specify the output geometry using a ``reference_volume``.

        #. Specify the ``spacing`` explicitly.

        #. Specify the ``dimensions`` explicitly.

        #. Specify the ``cell_length_percentile``. The spacing is estimated from the
           surface's cells using the specified percentile.

        Use ``reference_volume`` for full control of the output grid's geometry. For
        all other options, the geometry is implicitly defined such that the generated
        grid fits the bounds of the input mesh.

        If no inputs are provided, ``cell_length_percentile=0.1`` (10th percentile) is
        used by default to estimate the spacing. On systems with VTK < 9.2, the default
        spacing is set to ``1/100`` of the input mesh's length.

        A point data array ``mask`` is included where points inside and outside of the
        input surface are labelled with ``foreground_value`` and ``background_value``,
        respectively.

        .. versionadded:: 0.46

        .. note::

            This method is a wrapper around :meth:`voxelize_binary_mask`. See that
            method for additional information.

        Parameters
        ----------
        background_value : int, default: 0
            Background value of the generated grid.

        foreground_value : int, default: 1
            Foreground value of the generated grid.

        reference_volume : ImageData, optional
            Volume to use as a reference. The output will have the same ``dimensions``,
            ``origin``, and ``spacing`` as the reference.

        dimensions : VectorLike[int], optional
            Dimensions of the generated rectilinear grid. Set this value to control the
            dimensions explicitly. If unset, the dimensions are defined implicitly
            through other parameter. See summary and examples for details.

            .. note::

                Dimensions is the number of points along each axis, not cells. Set
                dimensions to ``N+1`` instead for ``N`` cells along each axis.

        spacing : float | VectorLike[float], optional
            Approximate spacing to use for the generated grid. Set this value
            to control the spacing explicitly. If unset, the spacing is defined
            implicitly through other parameters. See summary and examples for details.

        rounding_func : Callable[VectorLike[float], VectorLike[int]], optional
            Control how the dimensions are rounded to integers based on the provided or
            calculated ``spacing``. Should accept a length-3 vector containing the
            dimension values along the three directions and return a length-3 vector.
            :func:`numpy.round` is used by default.

            Rounding the dimensions implies rounding the actual spacing.

            Has no effect if ``reference_volume`` or ``dimensions`` are specified.

        cell_length_percentile : float, optional
            Cell length percentage ``p`` to use for computing the default ``spacing``.
            Default is ``0.1`` (10th percentile) and must be between ``0`` and ``1``.
            The ``p``-th percentile is computed from the cumulative distribution function
            (CDF) of lengths which are representative of the cell length scales present
            in the input. The CDF is computed by:

            #. Triangulating the input cells.
            #. Sampling a subset of up to ``cell_length_sample_size`` cells.
            #. Computing the distance between two random points in each cell.
            #. Inserting the distance into an ordered set to create the CDF.

            Has no effect if ``dimensions`` or ``reference_volume`` are specified.

            .. note::
                This option is only available for VTK 9.2 or greater.

        cell_length_sample_size : int, optional
            Number of samples to use for the cumulative distribution function (CDF)
            when using the ``cell_length_percentile`` option. ``100 000`` samples are
            used by default.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        RectilinearGrid
            RectilinearGrid as voxelized volume with discretized cells.

        See Also
        --------
        voxelize
            Similar function that returns a :class:`~pyvista.UnstructuredGrid` of
            :attr:`~pyvista.CellType.VOXEL` cells.

        voxelize_binary_mask
            Similar function that returns a :class:`~pyvista.ImageData` with point data.

        Examples
        --------
        Create a voxel volume of a nut. By default, the spacing is automatically
        estimated.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = pv.examples.load_nut()
        >>> vox = mesh.voxelize_rectilinear()

        Plot the mesh together with its volume.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh=vox, show_edges=True)
        >>> _ = pl.add_mesh(mesh=mesh, show_edges=True, opacity=1)
        >>> pl.show()

        Load a mesh of a cow.

        >>> mesh = examples.download_cow()

        Create an equal density voxel volume and plot the result.

        >>> vox = mesh.voxelize_rectilinear(spacing=0.15)
        >>> cpos = pv.CameraPosition(
        ...     position=(15, 3, 15), focal_point=(0, 0, 0), viewup=(0, 0, 0)
        ... )
        >>> vox.plot(scalars='mask', show_edges=True, cpos=cpos)

        Slice the voxel volume to view the ``mask`` scalars.

        >>> slices = vox.slice_orthogonal()
        >>> slices.plot(scalars='mask', show_edges=True)

        Create a voxel volume from unequal density dimensions and plot result.

        >>> vox = mesh.voxelize_rectilinear(spacing=(0.15, 0.15, 0.5))
        >>> vox.plot(scalars='mask', show_edges=True, cpos=cpos)

        Slice the unequal density voxel volume to view the ``mask`` scalars.

        >>> slices = vox.slice_orthogonal()
        >>> slices.plot(scalars='mask', show_edges=True, cpos=cpos)

        """
        voxel_cells = self._voxelize_binary_mask_cells(
            background_value=background_value,
            foreground_value=foreground_value,
            reference_volume=reference_volume,
            dimensions=dimensions,
            spacing=spacing,
            rounding_func=rounding_func,
            cell_length_percentile=cell_length_percentile,
            cell_length_sample_size=cell_length_sample_size,
            progress_bar=progress_bar,
        )
        return voxel_cells.cast_to_rectilinear_grid()

    def voxelize(  # type: ignore[misc]
        self: DataSet,
        *,
        reference_volume: ImageData | None = None,
        dimensions: VectorLike[int] | None = None,
        spacing: float | VectorLike[float] | None = None,
        rounding_func: Callable[[VectorLike[float]], VectorLike[int]] | None = None,
        cell_length_percentile: float | None = None,
        cell_length_sample_size: int | None = None,
        progress_bar: bool = False,
    ) -> UnstructuredGrid:
        """Voxelize mesh to UnstructuredGrid.

        The voxelization can be controlled in several ways:

        #. Specify the output geometry using a ``reference_volume``.

        #. Specify the ``spacing`` explicitly.

        #. Specify the ``dimensions`` explicitly.

        #. Specify the ``cell_length_percentile``. The spacing is estimated from the
           surface's cells using the specified percentile.

        Use ``reference_volume`` for full control of the output geometry. For
        all other options, the geometry is implicitly defined such that the generated
        mesh fits the bounds of the input mesh.

        If no inputs are provided, ``cell_length_percentile=0.1`` (10th percentile) is
        used by default to estimate the spacing.

        .. versionadded:: 0.46

        .. note::

            This method is a wrapper around :meth:`voxelize_binary_mask`. See that
            method for additional information.

        Parameters
        ----------
        reference_volume : ImageData, optional
            Volume to use as a reference. The output will have the same ``dimensions``,
            and ``spacing`` as the reference.

        dimensions : VectorLike[int], optional
            Dimensions of the voxelized mesh. Set this value to control the
            dimensions explicitly. If unset, the dimensions are defined implicitly
            through other parameter. See summary and examples for details.

            .. note::

                Dimensions is the number of points along each axis, not cells. Set
                dimensions to ``N+1`` instead for ``N`` cells along each axis.

        spacing : float | VectorLike[float], optional
            Approximate spacing to use for the generated mesh. Set this value
            to control the spacing explicitly. If unset, the spacing is defined
            implicitly through other parameters. See summary and examples for details.

        rounding_func : Callable[VectorLike[float], VectorLike[int]], optional
            Control how the dimensions are rounded to integers based on the provided or
            calculated ``spacing``. Should accept a length-3 vector containing the
            dimension values along the three directions and return a length-3 vector.
            :func:`numpy.round` is used by default.

            Rounding the dimensions implies rounding the actual spacing.

            Has no effect if ``dimensions`` is specified.

        cell_length_percentile : float, optional
            Cell length percentage ``p`` to use for computing the default ``spacing``.
            Default is ``0.1`` (10th percentile) and must be between ``0`` and ``1``.
            The ``p``-th percentile is computed from the cumulative distribution function
            (CDF) of lengths which are representative of the cell length scales present
            in the input. The CDF is computed by:

            #. Triangulating the input cells.
            #. Sampling a subset of up to ``cell_length_sample_size`` cells.
            #. Computing the distance between two random points in each cell.
            #. Inserting the distance into an ordered set to create the CDF.

            Has no effect if ``dimensions`` is specified.

        cell_length_sample_size : int, optional
            Number of samples to use for the cumulative distribution function (CDF)
            when using the ``cell_length_percentile`` option. ``100 000`` samples are
            used by default.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        UnstructuredGrid
            Voxelized unstructured grid of the original mesh.

        See Also
        --------
        voxelize_rectilinear
            Similar function that returns a :class:`~pyvista.RectilinearGrid` with cell data.

        voxelize_binary_mask
            Similar function that returns a :class:`~pyvista.ImageData` with point data.

        Examples
        --------
        Create a voxelized mesh with uniform spacing.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_bunny_coarse()
        >>> vox = mesh.voxelize(spacing=0.01)
        >>> vox.plot(show_edges=True)

        Create a voxelized mesh using non-uniform spacing.

        >>> vox = mesh.voxelize(spacing=(0.01, 0.005, 0.002))
        >>> vox.plot(show_edges=True)

        The bounds of the voxelized mesh always match the bounds of the input.

        >>> mesh.bounds
        BoundsTuple(x_min = -0.13155962526798248,
                    x_max =  0.18016336858272552,
                    y_min = -0.12048563361167908,
                    y_max =  0.18769524991512299,
                    z_min = -0.14300920069217682,
                    z_max =  0.09850578755140305)

        >>> vox.bounds
        BoundsTuple(x_min = -0.13155962526798248,
                    x_max =  0.18016336858272552,
                    y_min = -0.12048563361167908,
                    y_max =  0.18769524991512299,
                    z_min = -0.14300920069217682,
                    z_max =  0.09650979936122894)

        Create a voxelized mesh with ``3 x 4 x 5`` cells. Since ``dimensions`` is the
        number of points, not cells, we need to add ``1`` to get the number of desired cells.

        >>> mesh = pv.Box()
        >>> cell_dimensions = np.array((3, 4, 5))
        >>> vox = mesh.voxelize(dimensions=cell_dimensions + 1)
        >>> vox.plot(show_edges=True)

        """
        voxel_cells = self._voxelize_binary_mask_cells(
            reference_volume=reference_volume,
            dimensions=dimensions,
            spacing=spacing,
            rounding_func=rounding_func,
            cell_length_percentile=cell_length_percentile,
            cell_length_sample_size=cell_length_sample_size,
            progress_bar=progress_bar,
        )
        ugrid = voxel_cells.threshold(0.5)
        del ugrid.cell_data['mask']
        return ugrid


def _length_distribution_percentile(poly, percentile, cell_length_sample_size, *, progress_bar):
    percentile = _validation.validate_number(
        percentile, must_be_in_range=[0.0, 1.0], name='percentile'
    )
    distribution = _vtk.vtkLengthDistribution()
    distribution.SetInputData(poly)
    distribution.SetSampleSize(cell_length_sample_size)
    _update_alg(
        distribution, progress_bar=progress_bar, message='Computing cell length distribution'
    )
    return distribution.GetLengthQuantile(percentile)


def _set_threshold_limit(alg, *, value, method, invert):
    """Set vtkThreshold limits and function.

    Addresses VTK API deprecations and previous PyVista inconsistencies with ParaView. Reference:

    * https://github.com/pyvista/pyvista/issues/2850
    * https://github.com/pyvista/pyvista/issues/3610
    * https://discourse.vtk.org/t/unnecessary-vtk-api-change/9929

    """
    # Check value
    if isinstance(value, (np.ndarray, Sequence)):
        if len(value) != 2:
            msg = (
                f'Value range must be length one for a float value '
                f'or two for min/max; not ({value}).'
            )
            raise ValueError(msg)
        # Check range
        if value[0] > value[1]:
            msg = (
                'Value sequence is invalid, please use (min, max). '
                'The provided first value is greater than the second.'
            )
            raise ValueError(msg)
    elif isinstance(value, Iterable):
        msg = 'Value must either be a single scalar or a sequence.'
        raise TypeError(msg)
    alg.SetInvert(invert)
    # Set values and function
    if isinstance(value, (np.ndarray, Sequence)):
        alg.SetThresholdFunction(_vtk.vtkThreshold.THRESHOLD_BETWEEN)
        alg.SetLowerThreshold(value[0])
        alg.SetUpperThreshold(value[1])
    # Single value
    elif method.lower() == 'lower':
        alg.SetLowerThreshold(value)
        alg.SetThresholdFunction(_vtk.vtkThreshold.THRESHOLD_LOWER)
    elif method.lower() == 'upper':
        alg.SetUpperThreshold(value)
        alg.SetThresholdFunction(_vtk.vtkThreshold.THRESHOLD_UPPER)
    else:
        msg = 'Invalid method choice. Either `lower` or `upper`'
        raise ValueError(msg)


def _swap_axes(vectors, values):
    """Swap axes vectors based on their respective values.

    The vector with the larger component along its projected axis is selected to precede
    the vector with the smaller component. E.g. a symmetric point cloud with equal
    std in any direction could have its principal axes computed such that the first
    axis is +Y, second is +X, and third is +Z. This function will swap the first two
    axes so that the order is XYZ instead of YXZ.

    This function is intended to be used by `align_xyz` and is only exposed as a
    module-level function for testing purposes.
    """

    def _swap(axis_a, axis_b) -> None:
        axis_order = np.argmax(np.abs(vectors), axis=1)
        if axis_order[axis_a] > axis_order[axis_b]:
            vectors[[axis_a, axis_b]] = vectors[[axis_b, axis_a]]

    if np.isclose(values[0], values[1]) and np.isclose(values[1], values[2]):
        # Sort all axes by largest 'x' component
        vectors = vectors[np.argsort(np.abs(vectors)[:, 0])[::-1]]
        _swap(1, 2)
    elif np.isclose(values[0], values[1]):
        _swap(0, 1)
    elif np.isclose(values[1], values[2]):
        _swap(1, 2)
    return vectors
