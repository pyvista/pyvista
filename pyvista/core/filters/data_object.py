"""Filters module with a class of common filters that can be applied to any vtkDataSet."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import numpy as np

import pyvista
from pyvista.core import _validation
import pyvista.core._vtk_core as _vtk
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.utilities.geometric_objects import NORMALS
from pyvista.core.utilities.geometric_objects import NormalsLiteral
from pyvista.core.utilities.helpers import generate_plane
from pyvista.core.utilities.helpers import wrap

if TYPE_CHECKING:
    from pyvista import DataSet
    from pyvista import MultiBlock
    from pyvista.core._typing_core import ConcreteDataSetType
    from pyvista.core._typing_core import VectorLike


class DataObjectFilters:
    """A set of common filters that can be applied to any DataSet or MultiBlock."""

    def _clip_with_function(
        self: ConcreteDataSetType | MultiBlock,
        function: _vtk.vtkImplicitFunction,
        invert: bool = True,
        value: float = 0.0,
        return_clipped: bool = False,
        progress_bar: bool = False,
        crinkle: bool = False,
    ):
        """Clip using an implicit function (internal helper)."""
        if crinkle:
            # Add Cell IDs
            self.cell_data['cell_ids'] = np.arange(self.n_cells)

        if isinstance(self, _vtk.vtkPolyData):
            alg: _vtk.vtkClipPolyData | _vtk.vtkTableBasedClipDataSet = _vtk.vtkClipPolyData()
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
                a = self.extract_cells(list(set_a))  # type: ignore[union-attr]
                b = self.extract_cells(list(set_b))  # type: ignore[union-attr]
            return a, b
        clipped = _get_output(alg)
        if crinkle:
            clipped = self.extract_cells(np.unique(clipped.cell_data['cell_ids']))  # type: ignore[union-attr]
        return clipped

    def clip(
        self: ConcreteDataSetType | MultiBlock,
        normal: VectorLike[float] | NormalsLiteral = 'x',
        origin: VectorLike[float] | None = None,
        invert: bool = True,
        value: float = 0.0,
        inplace: bool = False,
        return_clipped: bool = False,
        progress_bar: bool = False,
        crinkle: bool = False,
    ):
        """Clip a dataset by a plane by specifying the origin and normal.

        If no parameters are given the clip will occur in the center
        of that dataset.

        Parameters
        ----------
        normal : tuple(float) | str, default: 'x'
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
        pyvista.PolyData | tuple[pyvista.PolyData]
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
        normal_: VectorLike[float] = NORMALS[normal.lower()] if isinstance(normal, str) else normal
        # find center of data if origin not specified
        origin_ = self.center if origin is None else origin
        # create the plane for clipping
        function = generate_plane(normal_, origin_)
        # run the clip
        result = self._clip_with_function(
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
        self: ConcreteDataSetType | MultiBlock,
        bounds: float | VectorLike[float] | pyvista.PolyData | None = None,
        invert: bool = True,
        factor: float = 0.35,
        progress_bar: bool = False,
        merge_points: bool = True,
        crinkle: bool = False,
    ):
        """Clip a dataset by a bounding box defined by the bounds.

        If no bounds are given, a corner of the dataset bounds will be removed.

        Parameters
        ----------
        bounds : sequence[float], optional
            Length 6 sequence of floats: ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
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
                raise ValueError('The bounds mesh must have only 6 faces.')
            bounds = []
            poly.compute_normals(inplace=True)
            for cid in range(6):
                cell = poly.extract_cells(cid)
                normal = cell['Normals'][0]
                bounds.append(normal)
                bounds.append(cell.center)
        bounds_ = _validation.validate_array(
            bounds, dtype_out=float, must_have_length=[3, 6, 12], name='bounds'
        )
        if len(bounds_) == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            bounds_ = np.array(
                (
                    xmin,
                    xmin + bounds_[0],
                    ymin,
                    ymin + bounds_[1],
                    zmin,
                    zmin + bounds_[2],
                )
            )
        if crinkle:
            self.cell_data['cell_ids'] = np.arange(self.n_cells)
        alg = _vtk.vtkBoxClipDataSet()
        if not merge_points:
            # vtkBoxClipDataSet uses vtkMergePoints by default
            alg.SetLocator(_vtk.vtkNonMergingPointLocator())
        alg.SetInputDataObject(self)
        alg.SetBoxClip(*bounds_)
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

    def slice_implicit(
        self: ConcreteDataSetType | MultiBlock,
        implicit_function: _vtk.vtkImplicitFunction,
        generate_triangles: bool = False,
        contour: bool = False,
        progress_bar: bool = False,
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
        self: ConcreteDataSetType | MultiBlock,
        normal: VectorLike[float] | NormalsLiteral = 'x',
        origin: VectorLike[float] | None = None,
        generate_triangles: bool = False,
        contour: bool = False,
        progress_bar: bool = False,
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
        normal_: VectorLike[float] = NORMALS[normal.lower()] if isinstance(normal, str) else normal
        # find center of data if origin not specified
        origin_ = self.center if origin is None else origin

        # create the plane for clipping
        plane = generate_plane(normal_, origin_)
        return self.slice_implicit(
            plane,
            generate_triangles=generate_triangles,
            contour=contour,
            progress_bar=progress_bar,
        )

    def slice_orthogonal(
        self: ConcreteDataSetType | MultiBlock,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        generate_triangles: bool = False,
        contour: bool = False,
        progress_bar: bool = False,
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
                data = self[i]
                output.append(
                    data.slice_orthogonal(
                        x=x,
                        y=y,
                        z=z,
                        generate_triangles=generate_triangles,
                        contour=contour,
                    )
                    if data is not None
                    else data
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
        self: ConcreteDataSetType | MultiBlock,
        n: int = 5,
        axis: Literal['x', 'y', 'z', 0, 1, 2] = 'x',
        tolerance: float | None = None,
        generate_triangles: bool = False,
        contour: bool = False,
        bounds=None,
        center=None,
        progress_bar: bool = False,
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
        XYZLiteral = Literal['x', 'y', 'z']
        labels: list[XYZLiteral] = ['x', 'y', 'z']
        label_to_index: dict[Literal['x', 'y', 'z'], Literal[0, 1, 2]] = {'x': 0, 'y': 1, 'z': 2}
        if isinstance(axis, int):
            ax_index = axis
            ax_label = labels[ax_index]
        elif isinstance(axis, str):
            ax_str = axis.lower()
            if ax_str in labels:
                ax_label = cast(XYZLiteral, ax_str)
                ax_index = label_to_index[ax_label]
            else:
                raise ValueError(
                    f'Axis ({axis!r}) not understood. Choose one of {labels}.',
                ) from None
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
                data = self[i]
                output.append(
                    data.slice_along_axis(
                        n=n,
                        axis=ax_label,
                        tolerance=tolerance,
                        generate_triangles=generate_triangles,
                        contour=contour,
                        bounds=bounds,
                        center=center,
                    )
                    if data is not None
                    else data
                )
            return output
        for i in range(n):
            center[ax_index] = rng[i]
            slc = self.slice(
                normal=ax_label,
                origin=center,
                generate_triangles=generate_triangles,
                contour=contour,
                progress_bar=progress_bar,
            )
            output.append(slc, f'slice{i}')
        return output

    def slice_along_line(
        self: ConcreteDataSetType | MultiBlock,
        line: pyvista.PolyData,
        generate_triangles: bool = False,
        contour: bool = False,
        progress_bar: bool = False,
    ):
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

    def extract_all_edges(
        self: ConcreteDataSetType | MultiBlock,
        use_all_points: bool = False,
        clear_data: bool = False,
        progress_bar: bool = False,
    ):
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
        self: ConcreteDataSetType | MultiBlock,
        low_point: VectorLike[float] | None = None,
        high_point: VectorLike[float] | None = None,
        scalar_range: str | VectorLike[float] | None = None,
        preference: Literal['point', 'cell'] = 'point',
        set_active: bool = True,
        progress_bar: bool = False,
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
            low_point_ = list(self.center)
            low_point_[2] = self.bounds.z_min
        else:
            low_point_ = _validation.validate_array3(low_point)
        if high_point is None:
            high_point_ = list(self.center)
            high_point_[2] = self.bounds.z_max
        else:
            high_point_ = _validation.validate_array3(high_point)
        # Fix scalar_range:
        if scalar_range is None:
            scalar_range_ = (low_point_[2], high_point_[2])
        elif isinstance(scalar_range, str):
            scalar_range_ = self.get_data_range(scalar_range, preference=preference)
        else:
            scalar_range_ = _validation.validate_data_range(scalar_range)

        # Construct the filter
        alg = _vtk.vtkElevationFilter()
        alg.SetInputDataObject(self)
        # Set the parameters
        alg.SetScalarRange(scalar_range_)
        alg.SetLowPoint(low_point_)
        alg.SetHighPoint(high_point_)
        _update_alg(alg, progress_bar, 'Computing Elevation')
        # Decide on updating active scalars array
        output = _get_output(alg)
        if not set_active:
            # 'Elevation' is automatically made active by the VTK filter
            output.point_data.active_scalars_name = self.point_data.active_scalars_name
        return output

    def compute_cell_sizes(
        self: ConcreteDataSetType | MultiBlock,
        length: bool = True,
        area: bool = True,
        volume: bool = True,
        progress_bar: bool = False,
        vertex_count: bool = False,
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

    def cell_centers(
        self: ConcreteDataSetType | MultiBlock, vertex: bool = True, progress_bar: bool = False
    ):
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
        input_mesh = self.cast_to_poly_points() if isinstance(self, pyvista.PointSet) else self
        alg = _vtk.vtkCellCenters()
        alg.SetInputDataObject(input_mesh)
        alg.SetVertexCells(vertex)
        _update_alg(alg, progress_bar, 'Generating Points at the Center of the Cells')
        return _get_output(alg)

    def cell_data_to_point_data(
        self: ConcreteDataSetType | MultiBlock,
        pass_cell_data: bool = False,
        progress_bar: bool = False,
    ):
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

    def ctp(
        self: ConcreteDataSetType | MultiBlock,
        pass_cell_data: bool = False,
        progress_bar: bool = False,
        **kwargs,
    ):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        This method is an alias for :func:`cell_data_to_point_data`.

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
        return self.cell_data_to_point_data(
            pass_cell_data=pass_cell_data,
            progress_bar=progress_bar,
            **kwargs,
        )

    def point_data_to_cell_data(
        self: ConcreteDataSetType | MultiBlock,
        pass_point_data: bool = False,
        categorical: bool = False,
        progress_bar: bool = False,
    ):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified within cells.
        Optionally, the input point data can be passed through to the output.

        Parameters
        ----------
        pass_point_data : bool, default: False
            If enabled, pass the input point data through to the output.

        categorical : bool, default: False
            Control whether the source point data is to be treated as
            categorical. If ``True``,  histograming is used to assign the
            cell data. Specifically, a histogram is populated for each cell
            from the scalar values at each point, and the bin with the most
            elements is selected. In case of a tie, the smaller value is selected.

            .. note::

                If the point data is continuous, values that are almost equal (within
                ``1e-6``) are merged into a single bin. Otherwise, for discrete data
                the number of bins equals the number of unique values.

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
        alg.SetCategoricalData(categorical)
        _update_alg(alg, progress_bar, 'Transforming point data into cell data')
        active_scalars = None
        if not isinstance(self, pyvista.MultiBlock):
            active_scalars = self.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    def ptc(
        self: ConcreteDataSetType | MultiBlock,
        pass_point_data: bool = False,
        progress_bar: bool = False,
        **kwargs,
    ):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        This method is an alias for :func:`point_data_to_cell_data`.

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
        return self.point_data_to_cell_data(
            pass_point_data=pass_point_data,
            progress_bar=progress_bar,
            **kwargs,
        )

    def triangulate(
        self: ConcreteDataSetType | MultiBlock, inplace: bool = False, progress_bar: bool = False
    ):
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

    def sample(
        self: ConcreteDataSetType | MultiBlock,
        target: DataSet | _vtk.vtkDataSet,
        tolerance: float | None = None,
        pass_cell_data: bool = True,
        pass_point_data: bool = True,
        categorical: bool = False,
        progress_bar: bool = False,
        locator: Literal['cell', 'cell_tree', 'obb_tree', 'static_cell']
        | _vtk.vtkAbstractCellLocator
        | None = 'static_cell',
        pass_field_data: bool = True,
        mark_blank: bool = True,
        snap_to_closest_point: bool = False,
    ):
        """Resample array data from a passed mesh onto this mesh.

        For `mesh1.sample(mesh2)`, the arrays from `mesh2` are sampled onto
        the points of `mesh1`.  This function interpolates within an
        enclosing cell.  This contrasts with
        :func:`pyvista.DataSetFilters.interpolate` that uses a distance
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

        locator : vtkAbstractCellLocator or str or None, default: 'static_cell'
            Prototype cell locator to perform the ``FindCell()``
            operation.  If ``None``, uses the DataSet ``FindCell`` method.
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
            Interpolate values from one mesh onto another.

        pyvista.ImageDataFilters.resample
            Resample image data to modify its dimensions and spacing.

        Examples
        --------
        Resample data from another dataset onto a sphere.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
        >>> data_to_probe = examples.load_uniform()
        >>> result = mesh.sample(data_to_probe)
        >>> result.plot(scalars='Spatial Point Data')

        If sampling from a set of points represented by a ``(n, 3)``
        shaped ``numpy.ndarray``, they need to be converted to a
        PyVista DataSet, e.g. :class:`pyvista.PolyData`, first.

        >>> import numpy as np
        >>> points = np.array([[1.5, 5.0, 6.2], [6.7, 4.2, 8.0]])
        >>> mesh = pv.PolyData(points)
        >>> result = mesh.sample(data_to_probe)
        >>> result['Spatial Point Data']
        pyvista_ndarray([ 46.5 , 225.12])

        See :ref:`resampling_example` for more examples using this filter.

        """
        alg = _vtk.vtkResampleWithDataSet()  # Construct the ResampleWithDataSet object
        alg.SetInputData(self)  # Set the Input data (actually the source i.e. where to sample from)
        # Set the Source data (actually the target, i.e. where to sample to)
        alg.SetSourceData(wrap(target))
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
                    'cell': _vtk.vtkCellLocator(),
                    'cell_tree': _vtk.vtkCellTreeLocator(),
                    'obb_tree': _vtk.vtkOBBTree(),
                    'static_cell': _vtk.vtkStaticCellLocator(),
                }
                try:
                    locator = locator_map[locator]
                except KeyError as err:
                    raise ValueError(
                        f'locator must be a string from {locator_map.keys()}, got {locator}',
                    ) from err
            alg.SetCellLocatorPrototype(locator)

        if snap_to_closest_point:
            try:
                alg.SnapToCellWithClosestPointOn()
            except AttributeError:  # pragma: no cover
                raise VTKVersionError('`snap_to_closest_point=True` requires vtk 9.3.0 or newer')
        _update_alg(alg, progress_bar, 'Resampling array Data from a Passed Mesh onto Mesh')
        return _get_output(alg)


# clip = DataSetFilters.clip
#
# clip_box = DataSetFilters.clip_box
#
# slice = DataSetFilters.slice
#
# slice_orthogonal = DataSetFilters.slice_orthogonal
#
# slice_along_axis = DataSetFilters.slice_along_axis
#
# slice_along_line = DataSetFilters.slice_along_line
#
# extract_all_edges = DataSetFilters.extract_all_edges
#
# elevation = DataSetFilters.elevation
#
# compute_cell_sizes = DataSetFilters.compute_cell_sizes
#
# cell_centers = DataSetFilters.cell_centers
#
# cell_data_to_point_data = DataObjectFilters.cell_data_to_point_data
#
# point_data_to_cell_data = DataObjectFilters.cell_data_to_point_data
#
# sample = DataSetFilters.sample
#
# triangulate = DataSetFilters.triangulate
