"""Filters module with a class of common filters that can be applied to any vtkDataSet."""
import collections.abc
from typing import Union

import numpy as np

import pyvista
from pyvista import _vtk, FieldAssociation
from pyvista.utilities import (
    NORMALS, assert_empty_kwargs, generate_plane, get_array, wrap, abstract_class
)
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output, _update_alg
from pyvista.utilities import transformations
from pyvista.utilities.cells import numpy_to_idarr


@abstract_class
class DataSetFilters:
    """A set of common filters that can be applied to any vtkDataSet."""

    def _clip_with_function(dataset, function, invert=True, value=0.0, return_clipped=False):
        """Clip using an implicit function (internal helper)."""
        if isinstance(dataset, _vtk.vtkPolyData):
            alg = _vtk.vtkClipPolyData()
        # elif isinstance(dataset, vtk.vtkImageData):
        #     alg = vtk.vtkClipVolume()
        #     alg.SetMixed3DCellGeneration(True)
        else:
            alg = _vtk.vtkTableBasedClipDataSet()
        alg.SetInputDataObject(dataset)  # Use the grid as the data we desire to cut
        alg.SetValue(value)
        alg.SetClipFunction(function)  # the implicit function
        alg.SetInsideOut(invert)  # invert the clip if needed
        if return_clipped:
            alg.GenerateClippedOutputOn()
        alg.Update()  # Perform the Cut

        if return_clipped:
            a = _get_output(alg, oport=0)
            b = _get_output(alg, oport=1)
            return a, b
        else:
            return _get_output(alg)

    def clip(dataset, normal='x', origin=None, invert=True, value=0.0, inplace=False,
             return_clipped=False):
        """Clip a dataset by a plane by specifying the origin and normal.

        If no parameters are given the clip will occur in the center of that dataset.

        Parameters
        ----------
        normal : tuple(float) or str
            Length 3 tuple for the normal vector direction. Can also be
            specified as a string conventional direction such as ``'x'`` for
            ``(1,0,0)`` or ``'-x'`` for ``(-1,0,0)``, etc.

        origin : tuple(float), optional
            The center ``(x,y,z)`` coordinate of the plane on which the clip
            occurs. The default is the center of the dataset.

        invert : bool, optional
            Flag on whether to flip/invert the clip.

        value : float, optional
            Set the clipping value along the normal direction.
            The default value is 0.0.

        inplace : bool, optional
            Updates mesh in-place.

        return_clipped : bool, optional
            Return both unclipped and clipped parts of the dataset.

        Returns
        -------
        mesh : pyvista.PolyData or tuple(pyvista.PolyData)
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

        Clip a cube in the +Z direction.  This leaves half a cube
        below the XY plane.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip('z')

        """
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = dataset.center
        # create the plane for clipping
        function = generate_plane(normal, origin)
        # run the clip
        result = DataSetFilters._clip_with_function(dataset, function,
                                                    invert=invert, value=value,
                                                    return_clipped=return_clipped)
        if inplace:
            if return_clipped:
                dataset.overwrite(result[0])
                return dataset, result[1]
            else:
                dataset.overwrite(result)
                return dataset
        else:
            return result

    def clip_box(dataset, bounds=None, invert=True, factor=0.35):
        """Clip a dataset by a bounding box defined by the bounds.

        If no bounds are given, a corner of the dataset bounds will be removed.

        Parameters
        ----------
        bounds : tuple(float)
            Length 6 sequence of floats: (xmin, xmax, ymin, ymax, zmin, zmax).
            Length 3 sequence of floats: distances from the min coordinate of
            of the input mesh. Single float value: uniform distance from the
            min coordinate. Length 12 sequence of length 3 sequence of floats:
            a plane collection (normal, center, ...).
            :class:`pyvista.PolyData`: if a poly mesh is passed that represents
            a box with 6 faces that all form a standard box, then planes will
            be extracted from the box to define the clipping region.

        invert : bool
            Flag on whether to flip/invert the clip

        factor : float, optional
            If bounds are not given this is the factor along each axis to
            extract the default box.

        Examples
        --------
        Clip a corner of a cube.  The bounds of a cube are normally
        ``[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]``, and this removes 1/8 of
        the cube's surface.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip_box([0, 1, 0, 1, 0, 1])

        """
        if bounds is None:
            def _get_quarter(dmin, dmax):
                """Get a section of the given range (internal helper)."""
                return dmax - ((dmax - dmin) * factor)
            xmin, xmax, ymin, ymax, zmin, zmax = dataset.bounds
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
            poly.compute_normals()
            for cid in range(6):
                cell = poly.extract_cells(cid)
                normal = cell["Normals"][0]
                bounds.append(normal)
                bounds.append(cell.center)
        if not isinstance(bounds, (np.ndarray, collections.abc.Sequence)):
            raise TypeError('Bounds must be a sequence of floats with length 3, 6 or 12.')
        if len(bounds) not in [3, 6, 12]:
            raise ValueError('Bounds must be a sequence of floats with length 3, 6 or 12.')
        if len(bounds) == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = dataset.bounds
            bounds = (xmin, xmin+bounds[0], ymin, ymin+bounds[1], zmin, zmin+bounds[2])
        alg = _vtk.vtkBoxClipDataSet()
        alg.SetInputDataObject(dataset)
        alg.SetBoxClip(*bounds)
        port = 0
        if invert:
            # invert the clip if needed
            port = 1
            alg.GenerateClippedOutputOn()
        alg.Update()
        return _get_output(alg, oport=port)

    def compute_implicit_distance(dataset, surface, inplace=False):
        """Compute the implicit distance from the points to a surface.

        This filter will compute the implicit distance from all of the nodes of
        this mesh to a given surface. This distance will be added as a point
        array called ``'implicit_distance'``.

        Parameters
        ----------
        surface : pyvista.DataSet
            The surface used to compute the distance

        inplace : bool
            If True, a new scalar array will be added to the ``point_arrays``
            of this mesh. Otherwise a copy of this mesh is returned with that
            scalar field.

        Examples
        --------
        Compute the distance between all the points on a sphere and a
        plane.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> plane = pv.Plane()
        >>> _ = sphere.compute_implicit_distance(plane, inplace=True)
        >>> dist = sphere['implicit_distance']
        >>> print(type(dist))
        <class 'numpy.ndarray'>

        Plot these distances as a heatmap

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere, scalars='implicit_distance', cmap='bwr')
        >>> _ = pl.add_mesh(plane, color='w', style='wireframe')
        >>> cpos = pl.show()

        """
        function = _vtk.vtkImplicitPolyDataDistance()
        function.SetInput(surface)
        points = pyvista.convert_array(dataset.points)
        dists = _vtk.vtkDoubleArray()
        function.FunctionValue(points, dists)
        if inplace:
            dataset.point_arrays['implicit_distance'] = pyvista.convert_array(dists)
            return dataset
        result = dataset.copy()
        result.point_arrays['implicit_distance'] = pyvista.convert_array(dists)
        return result

    def clip_scalar(dataset, scalars=None, invert=True, value=0.0, inplace=False):
        """Clip a dataset by a scalar.

        Parameters
        ----------
        scalars : str, optional
            Name of scalars to clip on.  Defaults to currently active scalars.

        invert : bool, optional
            Flag on whether to flip/invert the clip.  When ``True``,
            only the mesh below ``value`` will be kept.  When
            ``False``, only values above ``value`` will be kept.

        value : float, optional
            Set the clipping value.  The default value is 0.0.

        inplace : bool, optional
            Update mesh in-place.

        Returns
        -------
        pdata : pyvista.PolyData
            Clipped dataset.

        Examples
        --------
        Remove the part of the mesh with "sample_point_scalars" above 100.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.load_hexbeam()
        >>> clipped = dataset.clip_scalar(scalars="sample_point_scalars", value=100)

        Remove the part of the mesh with "sample_point_scalars" below
        100.  Since these scalars are already active, there's no need
        to specify ``scalars=``

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.load_hexbeam()
        >>> clipped = dataset.clip_scalar(value=100, invert=False)

        """
        if isinstance(dataset, _vtk.vtkPolyData):
            alg = _vtk.vtkClipPolyData()
        else:
            alg = _vtk.vtkTableBasedClipDataSet()

        alg.SetInputDataObject(dataset)
        alg.SetValue(value)
        if scalars is None:
            field, scalars = dataset.active_scalars_info
        _, field = get_array(dataset, scalars, preference='point', info=True)

        # SetInputArrayToProcess(idx, port, connection, field, name)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
        alg.SetInsideOut(invert)  # invert the clip if needed
        alg.Update()  # Perform the Cut
        result = _get_output(alg)

        if inplace:
            dataset.overwrite(result)
            return dataset
        else:
            return result

    def clip_surface(dataset, surface, invert=True, value=0.0,
                     compute_distance=False):
        """Clip any mesh type using a :class:`pyvista.PolyData` surface mesh.

        This will return a :class:`pyvista.UnstructuredGrid` of the clipped
        mesh. Geometry of the input dataset will be preserved where possible -
        geometries near the clip intersection will be triangulated/tessellated.

        Parameters
        ----------
        surface : pyvista.PolyData
            The PolyData surface mesh to use as a clipping function. If this
            mesh is not PolyData, the external surface will be extracted.

        invert : bool
            Flag on whether to flip/invert the clip

        value : float:
            Set the clipping value of the implicit function (if clipping with
            implicit function) or scalar value (if clipping with scalars).
            The default value is 0.0.

        compute_distance : bool, optional
            Compute the implicit distance from the mesh onto the input dataset.
            A new array called ``'implicit_distance'`` will be added to the
            output clipped mesh.

        """
        if not isinstance(surface, _vtk.vtkPolyData):
            surface = DataSetFilters.extract_geometry(surface)
        function = _vtk.vtkImplicitPolyDataDistance()
        function.SetInput(surface)
        if compute_distance:
            points = pyvista.convert_array(dataset.points)
            dists = _vtk.vtkDoubleArray()
            function.FunctionValue(points, dists)
            dataset['implicit_distance'] = pyvista.convert_array(dists)
        # run the clip
        result = DataSetFilters._clip_with_function(dataset, function,
                                                    invert=invert, value=value)
        return result

    def slice(dataset, normal='x', origin=None, generate_triangles=False,
              contour=False):
        """Slice a dataset by a plane at the specified origin and normal vector orientation.

        If no origin is specified, the center of the input dataset will be used.

        Parameters
        ----------
        normal : tuple(float) or str
            Length 3 tuple for the normal vector direction. Can also be
            specified as a string conventional direction such as ``'x'`` for
            ``(1,0,0)`` or ``'-x'`` for ``(-1,0,0)```, etc.

        origin : tuple(float)
            The center (x,y,z) coordinate of the plane on which the slice occurs

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

        """
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = dataset.center
        # create the plane for clipping
        plane = generate_plane(normal, origin)
        # create slice
        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(dataset)  # Use the grid as the data we desire to cut
        alg.SetCutFunction(plane)  # the cutter to use the plane we made
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        alg.Update()  # Perform the Cut
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output

    def slice_orthogonal(dataset, x=None, y=None, z=None,
                         generate_triangles=False, contour=False):
        """Create three orthogonal slices through the dataset on the three cartesian planes.

        Yields a MutliBlock dataset of the three slices.

        Parameters
        ----------
        x : float
            The X location of the YZ slice

        y : float
            The Y location of the XZ slice

        z : float
            The Z location of the XY slice

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

        """
        # Create the three slices
        if x is None:
            x = dataset.center[0]
        if y is None:
            y = dataset.center[1]
        if z is None:
            z = dataset.center[2]
        output = pyvista.MultiBlock()
        if isinstance(dataset, pyvista.MultiBlock):
            for i in range(dataset.n_blocks):
                output[i] = dataset[i].slice_orthogonal(x=x, y=y, z=z,
                    generate_triangles=generate_triangles,
                    contour=contour)
            return output
        output[0, 'YZ'] = dataset.slice(normal='x', origin=[x,y,z], generate_triangles=generate_triangles)
        output[1, 'XZ'] = dataset.slice(normal='y', origin=[x,y,z], generate_triangles=generate_triangles)
        output[2, 'XY'] = dataset.slice(normal='z', origin=[x,y,z], generate_triangles=generate_triangles)
        return output

    def slice_along_axis(dataset, n=5, axis='x', tolerance=None,
                         generate_triangles=False, contour=False,
                         bounds=None, center=None):
        """Create many slices of the input dataset along a specified axis.

        Parameters
        ----------
        n : int
            The number of slices to create

        axis : str or int
            The axis to generate the slices along. Perpendicular to the slices.
            Can be string name (``'x'``, ``'y'``, or ``'z'``) or axis index
            (``0``, ``1``, or ``2``).

        tolerance : float, optional
            The tolerance to the edge of the dataset bounds to create the slices

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

        """
        axes = {'x':0, 'y':1, 'z':2}
        if isinstance(axis, int):
            ax = axis
            axis = list(axes.keys())[list(axes.values()).index(ax)]
        elif isinstance(axis, str):
            try:
                ax = axes[axis]
            except KeyError:
                raise ValueError(f'Axis ({axis}) not understood')
        # get the locations along that axis
        if bounds is None:
            bounds = dataset.bounds
        if center is None:
            center = dataset.center
        if tolerance is None:
            tolerance = (bounds[ax*2+1] - bounds[ax*2]) * 0.01
        rng = np.linspace(bounds[ax*2]+tolerance, bounds[ax*2+1]-tolerance, n)
        center = list(center)
        # Make each of the slices
        output = pyvista.MultiBlock()
        if isinstance(dataset, pyvista.MultiBlock):
            for i in range(dataset.n_blocks):
                output[i] = dataset[i].slice_along_axis(n=n, axis=axis,
                    tolerance=tolerance, generate_triangles=generate_triangles,
                    contour=contour, bounds=bounds, center=center)
            return output
        for i in range(n):
            center[ax] = rng[i]
            slc = DataSetFilters.slice(dataset, normal=axis, origin=center,
                                       generate_triangles=generate_triangles,
                                       contour=contour)
            output[i, f'slice{i}'] = slc
        return output

    def slice_along_line(dataset, line, generate_triangles=False,
                         contour=False):
        """Slice a dataset using a polyline/spline as the path.

        This also works for lines generated with :func:`pyvista.Line`

        Parameters
        ----------
        line : pyvista.PolyData
            A PolyData object containing one single PolyLine cell.

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

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
        alg.SetInputDataObject(dataset)  # Use the grid as the data we desire to cut
        alg.SetCutFunction(polyplane)  # the cutter to use the poly planes
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        alg.Update()  # Perform the Cut
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output

    def threshold(dataset, value=None, scalars=None, invert=False, continuous=False,
                  preference='cell', all_scalars=False):
        """Apply a ``vtkThreshold`` filter to the input dataset.

        This filter will apply a ``vtkThreshold`` filter to the input dataset
        and return the resulting object. This extracts cells where the scalar
        value in each cell satisfies threshold criterion.  If scalars is None,
        the inputs active scalars is used.

        Parameters
        ----------
        value : float or sequence, optional
            Single value or (min, max) to be used for the data threshold.  If
            a sequence, then length must be 2. If no value is specified, the
            non-NaN data range will be used to remove any NaN values.

        scalars : str, optional
            Name of scalars to threshold on. Defaults to currently active scalars.

        invert : bool, optional
            If value is a single value, when invert is True cells are kept when
            their values are below parameter "value".  When invert is False
            cells are kept when their value is above the threshold "value".
            Default is False: yielding above the threshold "value".

        continuous : bool, optional
            When True, the continuous interval [minimum cell scalar,
            maximum cell scalar] will be used to intersect the threshold bound,
            rather than the set of discrete scalar values from the vertices.

        preference : str, optional
            When scalars is specified, this is the preferred array type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        all_scalars : bool, optional
            If using scalars from point data, all scalars for all
            points in a cell must satisfy the threshold when this
            value is ``True``.  When ``False``, any point of the cell
            with a scalar value satisfying the threshold criterion
            will extract the cell.

        Examples
        --------
        >>> import pyvista
        >>> import numpy as np
        >>> volume = np.zeros([10, 10, 10])
        >>> volume[:3] = 1
        >>> v = pyvista.wrap(volume)
        >>> threshed = v.threshold(0.1)

        """
        # set the scalaras to threshold on
        if scalars is None:
            field, scalars = dataset.active_scalars_info
        arr, field = get_array(dataset, scalars, preference=preference, info=True)

        if all_scalars and scalars is not None:
            raise ValueError('Setting `all_scalars=True` and designating `scalars` '
                             'is incompatible.  Set one or the other but not both')

        if arr is None:
            raise ValueError('No arrays present to threshold.')

        # If using an inverted range, merge the result of two filters:
        if isinstance(value, (np.ndarray, collections.abc.Sequence)) and invert:
            valid_range = [np.nanmin(arr), np.nanmax(arr)]
            # Create two thresholds
            t1 = dataset.threshold([valid_range[0], value[0]], scalars=scalars,
                    continuous=continuous, preference=preference, invert=False)
            t2 = dataset.threshold([value[1], valid_range[1]], scalars=scalars,
                    continuous=continuous, preference=preference, invert=False)
            # Use an AppendFilter to merge the two results
            appender = _vtk.vtkAppendFilter()
            appender.AddInputData(t1)
            appender.AddInputData(t2)
            appender.Update()
            return _get_output(appender)

        # Run a standard threshold algorithm
        alg = _vtk.vtkThreshold()
        alg.SetAllScalars(all_scalars)
        alg.SetInputDataObject(dataset)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars) # args: (idx, port, connection, field, name)
        # set thresholding parameters
        alg.SetUseContinuousCellRange(continuous)
        # use valid range if no value given
        if value is None:
            value = dataset.get_data_range(scalars)
        # check if value is a sequence (if so threshold by min max range like ParaView)
        if isinstance(value, (np.ndarray, collections.abc.Sequence)):
            if len(value) != 2:
                raise ValueError(f'Value range must be length one for a float value or two for min/max; not ({value}).')
            alg.ThresholdBetween(value[0], value[1])
        elif isinstance(value, collections.abc.Iterable):
            raise TypeError('Value must either be a single scalar or a sequence.')
        else:
            # just a single value
            if invert:
                alg.ThresholdByLower(value)
            else:
                alg.ThresholdByUpper(value)
        # Run the threshold
        alg.Update()
        return _get_output(alg)

    def threshold_percent(dataset, percent=0.50, scalars=None, invert=False,
                          continuous=False, preference='cell'):
        """Threshold the dataset by a percentage of its range on the active scalars array or as specified.

        Parameters
        ----------
        percent : float or tuple(float), optional
            The percentage (0,1) to threshold. If value is out of 0 to 1 range,
            then it will be divided by 100 and checked to be in that range.

        scalars : str, optional
            Name of scalars to threshold on. Defaults to currently active scalars.

        invert : bool, optional
            When invert is True cells are kept when their values are below the
            percentage of the range.  When invert is False, cells are kept when
            their value is above the percentage of the range.
            Default is False: yielding above the threshold "value".

        continuous : bool, optional
            When True, the continuous interval [minimum cell scalar,
            maximum cell scalar] will be used to intersect the threshold bound,
            rather than the set of discrete scalar values from the vertices.

        preference : str, optional
            When scalars is specified, this is the preferred array type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        """
        if scalars is None:
            _, tscalars = dataset.active_scalars_info
        else:
            tscalars = scalars
        dmin, dmax = dataset.get_data_range(arr_var=tscalars, preference=preference)

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
        if isinstance(percent, (np.ndarray, collections.abc.Sequence)):
            # Get two values
            value = [_get_val(percent[0], dmin, dmax), _get_val(percent[1], dmin, dmax)]
        elif isinstance(percent, collections.abc.Iterable):
            raise TypeError('Percent must either be a single scalar or a sequence.')
        else:
            # Compute one value to threshold
            value = _get_val(percent, dmin, dmax)
        # Use the normal thresholding function on these values
        return DataSetFilters.threshold(dataset, value=value, scalars=scalars,
                                        invert=invert, continuous=continuous,
                                        preference=preference)

    def outline(dataset, generate_faces=False):
        """Produce an outline of the full extent for the input dataset.

        Parameters
        ----------
        generate_faces : bool, optional
            Generate solid faces for the box. This is off by default

        """
        alg = _vtk.vtkOutlineFilter()
        alg.SetInputDataObject(dataset)
        alg.SetGenerateFaces(generate_faces)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))

    def outline_corners(dataset, factor=0.2):
        """Produce an outline of the corners for the input dataset.

        Parameters
        ----------
        factor : float, optional
            controls the relative size of the corners to the length of the
            corresponding bounds

        """
        alg = _vtk.vtkOutlineCornerFilter()
        alg.SetInputDataObject(dataset)
        alg.SetCornerFactor(factor)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))

    def extract_geometry(dataset):
        """Extract the outer surface of a volume or structured grid dataset as PolyData.

        This will extract all 0D, 1D, and 2D cells producing the
        boundary faces of the dataset.

        """
        alg = _vtk.vtkGeometryFilter()
        alg.SetInputDataObject(dataset)
        alg.Update()
        return _get_output(alg)

    def extract_all_edges(dataset, progress_bar=False):
        """Extract all the internal/external edges of the dataset as PolyData.

        This produces a full wireframe representation of the input dataset.

        Parameters
        ----------
        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        """
        alg = _vtk.vtkExtractEdges()
        alg.SetInputDataObject(dataset)
        _update_alg(alg, progress_bar, 'Extracting All Edges')
        return _get_output(alg)

    def elevation(dataset, low_point=None, high_point=None, scalar_range=None,
                  preference='point', set_active=True, progress_bar=False):
        """Generate scalar values on a dataset.

        The scalar values lie within a user specified range, and are
        generated by computing a projection of each dataset point onto
        a line.  The line can be oriented arbitrarily.  A typical
        example is to generate scalars based on elevation or height
        above a plane.

        Parameters
        ----------
        low_point : tuple(float), optional
            The low point of the projection line in 3D space. Default is bottom
            center of the dataset. Otherwise pass a length 3 ``tuple(float)``.

        high_point : tuple(float), optional
            The high point of the projection line in 3D space. Default is top
            center of the dataset. Otherwise pass a length 3 ``tuple(float)``.

        scalar_range : str or tuple(float), optional
            The scalar range to project to the low and high points on the line
            that will be mapped to the dataset. If None given, the values will
            be computed from the elevation (Z component) range between the
            high and low points. Min and max of a range can be given as a length
            2 tuple(float). If ``str`` name of scalara array present in the
            dataset given, the valid range of that array will be used.

        preference : str, optional
            When an array name is specified for ``scalar_range``, this is the
            preferred array type to search for in the dataset.
            Must be either 'point' or 'cell'.

        set_active : bool, optional
            A boolean flag on whether or not to set the new `Elevation` scalar
            as the active scalars array on the output dataset.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Warning
        -------
        This will create a scalars array named `Elevation` on the point data of
        the input dataset and overasdf write an array named `Elevation` if present.

        """
        # Fix the projection line:
        if low_point is None:
            low_point = list(dataset.center)
            low_point[2] = dataset.bounds[4]
        if high_point is None:
            high_point = list(dataset.center)
            high_point[2] = dataset.bounds[5]
        # Fix scalar_range:
        if scalar_range is None:
            scalar_range = (low_point[2], high_point[2])
        elif isinstance(scalar_range, str):
            scalar_range = dataset.get_data_range(arr_var=scalar_range, preference=preference)
        elif isinstance(scalar_range, (np.ndarray, collections.abc.Sequence)):
            if len(scalar_range) != 2:
                raise ValueError('scalar_range must have a length of two defining the min and max')
        else:
            raise TypeError(f'scalar_range argument ({scalar_range}) not understood.')
        # Construct the filter
        alg = _vtk.vtkElevationFilter()
        alg.SetInputDataObject(dataset)
        # Set the parameters
        alg.SetScalarRange(scalar_range)
        alg.SetLowPoint(low_point)
        alg.SetHighPoint(high_point)
        _update_alg(alg, progress_bar, 'Computing Elevation')
        # Decide on updating active scalars array
        name = 'Elevation' # Note that this is added to the PointData
        if not set_active:
            name = None
        return _get_output(alg, active_scalars=name, active_scalars_field='point')

    def contour(dataset, isosurfaces=10, scalars=None, compute_normals=False,
                compute_gradients=False, compute_scalars=True, rng=None,
                preference='point', method='contour', progress_bar=False):
        """Contour an input dataset by an array.

        ``isosurfaces`` can be an integer specifying the number of isosurfaces in
        the data range or a sequence of values for explicitly setting the isosurfaces.

        Parameters
        ----------
        isosurfaces : int or sequence
            Number of isosurfaces to compute across valid data range or a
            sequence of float values to explicitly use as the isosurfaces.

        scalars : str, optional
            Name of scalars to threshold on. Defaults to currently active scalars.

        compute_normals : bool, optional

        compute_gradients : bool, optional
            Desc

        compute_scalars : bool, optional
            Preserves the scalar values that are being contoured

        rng : tuple(float), optional
            If an integer number of isosurfaces is specified, this is the range
            over which to generate contours. Default is the scalars arrays' full
            data range.

        preference : str, optional
            When scalars is specified, this is the preferred array type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        method : str, optional
            Specify to choose which vtk filter is used to create the contour.
            Must be one of ``'contour'``, ``'marching_cubes'`` and
            ``'flying_edges'``. Defaults to ``'contour'``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        """
        if method is None or method == 'contour':
            alg = _vtk.vtkContourFilter()
        elif method == 'marching_cubes':
            alg = _vtk.vtkMarchingCubes()
        elif method == 'flying_edges':
            alg = _vtk.vtkFlyingEdges3D()
        else:
            raise ValueError(f"Method '{method}' is not supported")
        # Make sure the input has scalars to contour on
        if dataset.n_arrays < 1:
            raise ValueError('Input dataset for the contour filter must have scalar data.')
        alg.SetInputDataObject(dataset)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        # set the array to contour on
        if scalars is None:
            field, scalars = dataset.active_scalars_info
        else:
            _, field = get_array(dataset, scalars, preference=preference, info=True)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        if field != FieldAssociation.POINT:
            raise TypeError(f'Contour filter only works on Point data. Array ({scalars}) is in the Cell data.')
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars) # args: (idx, port, connection, field, name)
        # set the isosurfaces
        if isinstance(isosurfaces, int):
            # generate values
            if rng is None:
                rng = dataset.get_data_range(scalars)
            alg.GenerateValues(isosurfaces, rng)
        elif isinstance(isosurfaces, (np.ndarray, collections.abc.Sequence)):
            alg.SetNumberOfContours(len(isosurfaces))
            for i, val in enumerate(isosurfaces):
                alg.SetValue(i, val)
        else:
            raise TypeError('isosurfaces not understood.')
        _update_alg(alg, progress_bar, 'Computing Contour')
        return _get_output(alg)

    def texture_map_to_plane(dataset, origin=None, point_u=None, point_v=None,
                             inplace=False, name='Texture Coordinates',
                             use_bounds=False):
        """Texture map this dataset to a user defined plane.

        This is often used to define a plane to texture map an image to this dataset.
        The plane defines the spatial reference and extent of that image.

        Parameters
        ----------
        origin : tuple(float)
            Length 3 iterable of floats defining the XYZ coordinates of the
            BOTTOM LEFT CORNER of the plane

        point_u : tuple(float)
            Length 3 iterable of floats defining the XYZ coordinates of the
            BOTTOM RIGHT CORNER of the plane

        point_v : tuple(float)
            Length 3 iterable of floats defining the XYZ coordinates of the
            TOP LEFT CORNER of the plane

        inplace : bool, optional
            If True, the new texture coordinates will be added to this
            dataset. If False (default), a new dataset is returned
            with the textures coordinates

        name : str, optional
            The string name to give the new texture coordinates if applying
            the filter inplace.

        use_bounds : bool, optional
            Use the bounds to set the mapping plane by default (bottom plane
            of the bounding box).

        """
        if use_bounds:
            if isinstance(use_bounds, (int, bool)):
                b = dataset.GetBounds()
            origin = [b[0], b[2], b[4]]   # BOTTOM LEFT CORNER
            point_u = [b[1], b[2], b[4]]  # BOTTOM RIGHT CORNER
            point_v = [b[0], b[3], b[4]]  # TOP LEFT CORNER
        alg = _vtk.vtkTextureMapToPlane()
        if origin is None or point_u is None or point_v is None:
            alg.SetAutomaticPlaneGeneration(True)
        else:
            alg.SetOrigin(origin)  # BOTTOM LEFT CORNER
            alg.SetPoint1(point_u) # BOTTOM RIGHT CORNER
            alg.SetPoint2(point_v) # TOP LEFT CORNER
        alg.SetInputDataObject(dataset)
        alg.Update()
        output = _get_output(alg)
        if not inplace:
            return output
        t_coords = output.GetPointData().GetTCoords()
        t_coords.SetName(name)
        otc = dataset.GetPointData().GetTCoords()
        dataset.GetPointData().SetTCoords(t_coords)
        dataset.GetPointData().AddArray(t_coords)
        # CRITICAL:
        dataset.GetPointData().AddArray(otc) # Add old ones back at the end
        return dataset

    def texture_map_to_sphere(dataset, center=None, prevent_seam=True,
                              inplace=False, name='Texture Coordinates'):
        """Texture map this dataset to a user defined sphere.

        This is often used to define a sphere to texture map an image to this
        dataset. The sphere defines the spatial reference and extent of that image.

        Parameters
        ----------
        center : tuple(float)
            Length 3 iterable of floats defining the XYZ coordinates of the
            center of the sphere. If ``None``, this will be automatically
            calculated.

        prevent_seam : bool
            Default true. Control how the texture coordinates are generated.
            If set, the s-coordinate ranges from 0->1 and 1->0 corresponding
            to the theta angle variation between 0->180 and 180->0 degrees.
            Otherwise, the s-coordinate ranges from 0->1 between 0->360
            degrees.

        inplace : bool, optional
            If True, the new texture coordinates will be added to the dataset
            inplace. If False (default), a new dataset is returned with the
            textures coordinates

        name : str, optional
            The string name to give the new texture coordinates if applying
            the filter inplace.

        Examples
        --------
        Map a puppy texture to a sphere

        >>> import pyvista
        >>> from pyvista import examples
        >>> sphere = pyvista.Sphere()
        >>> sphere = sphere.texture_map_to_sphere()
        >>> tex = examples.download_puppy_texture()  # doctest:+SKIP
        >>> cpos = sphere.plot(texture=tex)  # doctest:+SKIP

        """
        alg = _vtk.vtkTextureMapToSphere()
        if center is None:
            alg.SetAutomaticSphereGeneration(True)
        else:
            alg.SetAutomaticSphereGeneration(False)
            alg.SetCenter(center)
        alg.SetPreventSeam(prevent_seam)
        alg.SetInputDataObject(dataset)
        alg.Update()
        output = _get_output(alg)
        if not inplace:
            return output
        t_coords = output.GetPointData().GetTCoords()
        t_coords.SetName(name)
        otc = dataset.GetPointData().GetTCoords()
        dataset.GetPointData().SetTCoords(t_coords)
        dataset.GetPointData().AddArray(t_coords)
        # CRITICAL:
        dataset.GetPointData().AddArray(otc)  # Add old ones back at the end
        return dataset

    def compute_cell_sizes(dataset, length=True, area=True, volume=True,
                           progress_bar=False):
        """Compute sizes for 1D (length), 2D (area) and 3D (volume) cells.

        Parameters
        ----------
        length : bool
            Specify whether or not to compute the length of 1D cells.

        area : bool
            Specify whether or not to compute the area of 2D cells.

        volume : bool
            Specify whether or not to compute the volume of 3D cells.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        """
        alg = _vtk.vtkCellSizeFilter()
        alg.SetInputDataObject(dataset)
        alg.SetComputeArea(area)
        alg.SetComputeVolume(volume)
        alg.SetComputeLength(length)
        alg.SetComputeVertexCount(False)
        _update_alg(alg, progress_bar, 'Computing Cell Sizes')
        return _get_output(alg)

    def cell_centers(dataset, vertex=True):
        """Generate points at the center of the cells in this dataset.

        These points can be used for placing glyphs / vectors.

        Parameters
        ----------
        vertex : bool
            Enable/disable the generation of vertex cells.

        """
        alg = _vtk.vtkCellCenters()
        alg.SetInputDataObject(dataset)
        alg.SetVertexCells(vertex)
        alg.Update()
        output = _get_output(alg)
        return output

    def glyph(dataset, orient=True, scale=True, factor=1.0, geom=None,
              indices=None, tolerance=None, absolute=False, clamping=False,
              rng=None, progress_bar=False):
        """Copy a geometric representation (called a glyph) to the input dataset.

        The glyph may be oriented along the input vectors, and it may
        be scaled according to scalar data or vector
        magnitude. Passing a table of glyphs to choose from based on
        scalars or vector magnitudes is also supported.  The arrays
        used for ``orient`` and ``scale`` must be either both point data
        or both cell data.

        Parameters
        ----------
        orient : bool or str, optional
            If ``True``, use the active vectors array to orient the glyphs.
            If string, the vector array to use to orient the glyphs.

        scale : bool or str, optional
            If ``True``, use the active scalars to scale the glyphs.
            If string, the scalar array to use to scale the glyphs.

        factor : float, optional
            Scale factor applied to scaling array.

        geom : vtk.vtkDataSet or tuple(vtk.vtkDataSet), optional
            The geometry to use for the glyph. If missing, an arrow glyph
            is used. If a sequence, the datasets inside define a table of
            geometries to choose from based on scalars or vectors. In this
            case a sequence of numbers of the same length must be passed as
            ``indices``. The values of the range (see ``rng``) affect lookup
            in the table.

        indices : tuple(float), optional
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

        absolute : bool, optional
            Control if ``tolerance`` is an absolute distance or a fraction.

        clamping: bool, optional
            Turn on/off clamping of "scalar" values to range. Default ``False``.

        rng: tuple(float), optional
            Set the range of values to be considered by the filter when scalars
            values are provided.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        glyphs : pyvista.PolyData
            Glyphs at either the cell centers or points.

        Examples
        --------
        Create arrow glyphs oriented by vectors and scaled by scalars.
        Factor parameter is used to reduce the size of the arrows.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_carotid().threshold(145, scalars="scalars")  # doctest:+SKIP
        >>> glyph = mesh.glyph(orient="vectors", scale="scalars", factor=0.01)  # doctest:+SKIP

        """
        # Clean the points before glyphing
        if tolerance is not None:
            small = pyvista.PolyData(dataset.points)
            small.point_arrays.update(dataset.point_arrays)
            dataset = small.clean(point_merging=True, merge_tol=tolerance,
                                  lines_to_points=False, polys_to_lines=False,
                                  strips_to_polys=False, inplace=False,
                                  absolute=absolute, progress_bar=progress_bar)
        # Make glyphing geometry if necessary
        if geom is None:
            arrow = _vtk.vtkArrowSource()
            arrow.Update()
            geom = arrow.GetOutput()
        # Check if a table of geometries was passed
        if isinstance(geom, (np.ndarray, collections.abc.Sequence)):
            if indices is None:
                # use default "categorical" indices
                indices = np.arange(len(geom))
            if not isinstance(indices, (np.ndarray, collections.abc.Sequence)):
                raise TypeError('If "geom" is a sequence then "indices" must '
                                'also be a sequence of the same length.')
            if len(indices) != len(geom) and len(geom) != 1:
                raise ValueError('The sequence "indices" must be the same length '
                                 'as "geom".')
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
            dataset.active_scalars_name = scale
            scale = True
        if scale:
            if dataset.active_scalars is not None:
                if dataset.active_scalars.ndim > 1:
                    alg.SetScaleModeToScaleByVector()
                else:
                    alg.SetScaleModeToScaleByScalar()
        else:
            alg.SetScaleModeToDataScalingOff()
        if isinstance(orient, str):
            dataset.active_vectors_name = orient
            orient = True
        if scale and orient:
            if (dataset.active_vectors_info.association == FieldAssociation.CELL
                and dataset.active_scalars_info.association == FieldAssociation.CELL
            ):
                source_data = dataset.cell_centers()
            elif(dataset.active_vectors_info.association == FieldAssociation.POINT
                and dataset.active_scalars_info.association == FieldAssociation.POINT
            ):
                source_data = dataset
            else:
                raise ValueError("Both ``scale`` and ``orient`` must use "
                                 "point data or cell data.")
        else:
            source_data = dataset
        if rng is not None:
            alg.SetRange(rng)
        alg.SetOrient(orient)
        alg.SetInputData(source_data)
        alg.SetVectorModeToUseVector()
        alg.SetScaleFactor(factor)
        alg.SetClamping(clamping)
        _update_alg(alg, progress_bar, 'Computing Glyphs')
        return _get_output(alg)

    def connectivity(dataset, largest=False):
        """Find and label connected bodies/volumes.

        This adds an ID array to the point and cell data to distinguish separate
        connected bodies. This applies a ``vtkConnectivityFilter`` filter which
        extracts cells that share common points and/or meet other connectivity
        criterion.
        (Cells that share vertices and meet other connectivity criterion such
        as scalar range are known as a region.)

        Parameters
        ----------
        largest : bool
            Extract the largest connected part of the mesh.

        """
        alg = _vtk.vtkConnectivityFilter()
        alg.SetInputData(dataset)
        if largest:
            alg.SetExtractionModeToLargestRegion()
        else:
            alg.SetExtractionModeToAllRegions()
        alg.SetColorRegions(True)
        alg.Update()
        return _get_output(alg)

    def extract_largest(dataset, inplace=False):
        """
        Extract largest connected set in mesh.

        Can be used to reduce residues obtained when generating an isosurface.
        Works only if residues are not connected (share at least one point with)
        the main component of the image.

        Parameters
        ----------
        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        mesh : pyvista.PolyData
            Largest connected set in mesh

        """
        mesh = DataSetFilters.connectivity(dataset, largest=True)
        if inplace:
            dataset.overwrite(mesh)
            return dataset
        else:
            return mesh

    def split_bodies(dataset, label=False):
        """Find, label, and split connected bodies/volumes.

        This splits different connected bodies into blocks in a MultiBlock dataset.

        Parameters
        ----------
        label : bool
            A flag on whether to keep the ID arrays given by the
            ``connectivity`` filter.

        """
        # Get the connectivity and label different bodies
        labeled = DataSetFilters.connectivity(dataset)
        classifier = labeled.cell_arrays['RegionId']
        bodies = pyvista.MultiBlock()
        for vid in np.unique(classifier):
            # Now extract it:
            b = labeled.threshold([vid-0.5, vid+0.5], scalars='RegionId')
            if not label:
                # strange behavior:
                # must use this method rather than deleting from the point_arrays
                # or else object is collected.
                b.cell_arrays.remove('RegionId')
                b.point_arrays.remove('RegionId')
            bodies.append(b)

        return bodies

    def warp_by_scalar(dataset, scalars=None, factor=1.0, normal=None,
                       inplace=False, **kwargs):
        """Warp the dataset's points by a point data scalars array's values.

        This modifies point coordinates by moving points along point normals by
        the scalar amount times the scale factor.

        Parameters
        ----------
        scalars : str, optional
            Name of scalars to warp by. Defaults to currently active scalars.

        factor : float, optional
            A scaling factor to increase the scaling effect. Alias
            ``scale_factor`` also accepted - if present, overrides ``factor``.

        normal : np.array, list, tuple of length 3
            User specified normal. If given, data normals will be ignored and
            the given normal will be used to project the warp.

        inplace : bool
            If True, the points of the given dataset will be updated.

        """
        factor = kwargs.pop('scale_factor', factor)
        assert_empty_kwargs(**kwargs)
        if scalars is None:
            field, scalars = dataset.active_scalars_info
        arr, field = get_array(dataset, scalars, preference='point', info=True)
        if field != FieldAssociation.POINT:
            raise TypeError('Dataset can only by warped by a point data array.')
        # Run the algorithm
        alg = _vtk.vtkWarpScalar()
        alg.SetInputDataObject(dataset)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars) # args: (idx, port, connection, field, name)
        alg.SetScaleFactor(factor)
        if normal is not None:
            alg.SetNormal(normal)
            alg.SetUseNormal(True)
        alg.Update()
        output = _get_output(alg)
        if inplace:
            if isinstance(dataset, (_vtk.vtkImageData, _vtk.vtkRectilinearGrid)):
                raise TypeError("This filter cannot be applied inplace for this mesh type.")
            dataset.overwrite(output)
            return dataset
        else:
            return output

    def warp_by_vector(dataset, vectors=None, factor=1.0, inplace=False):
        """Warp the dataset's points by a point data vectors array's values.

        This modifies point coordinates by moving points along point vectors by
        the local vector times the scale factor.

        A classical application of this transform is to visualize eigenmodes in
        mechanics.

        Parameters
        ----------
        vectors : str, optional
            Name of vector to warp by. Defaults to currently active vector.

        factor : float, optional
            A scaling factor that multiplies the vectors to warp by. Can
            be used to enhance the warping effect.

        inplace : bool, optional
            If True, the function will update the mesh in-place.

        Returns
        -------
        warped_mesh : mesh
            The warped mesh resulting from the operation.

        """
        if vectors is None:
            field, vectors = dataset.active_vectors_info
        arr, field = get_array(dataset, vectors, preference='point', info=True)
        if arr is None:
            raise TypeError('No active vectors')

        # check that this is indeed a vector field
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                'Dataset can only by warped by a 3D vector point data array.' + \
                'The values you provided do not satisfy this requirement')
        alg = _vtk.vtkWarpVector()
        alg.SetInputDataObject(dataset)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, vectors)
        alg.SetScaleFactor(factor)
        alg.Update()
        warped_mesh = _get_output(alg)
        if inplace:
            dataset.overwrite(warped_mesh)
            return dataset
        else:
            return warped_mesh

    def cell_data_to_point_data(dataset, pass_cell_data=False):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified within cells.
        Optionally, the input point data can be passed through to the output.

        The method of transformation is based on averaging the data values of
        all cells using a particular point. Optionally, the input cell data can
        be passed through to the output as well.

        See also: :func:`pyvista.DataSetFilters.point_data_to_cell_data`

        Parameters
        ----------
        pass_cell_data : bool
            If enabled, pass the input cell data through to the output

        """
        alg = _vtk.vtkCellDataToPointData()
        alg.SetInputDataObject(dataset)
        alg.SetPassCellData(pass_cell_data)
        alg.Update()
        active_scalars = None
        if not isinstance(dataset, pyvista.MultiBlock):
            active_scalars = dataset.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    def ctp(dataset, pass_cell_data=False):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified within cells.
        Optionally, the input point data can be passed through to the output.

        An alias/shortcut for ``cell_data_to_point_data``.

        """
        return DataSetFilters.cell_data_to_point_data(dataset, pass_cell_data=pass_cell_data)

    def point_data_to_cell_data(dataset, pass_point_data=False):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified within cells.
        Optionally, the input point data can be passed through to the output.

        See also: :func:`pyvista.DataSetFilters.cell_data_to_point_data`

        Parameters
        ----------
        pass_point_data : bool
            If enabled, pass the input point data through to the output

        """
        alg = _vtk.vtkPointDataToCellData()
        alg.SetInputDataObject(dataset)
        alg.SetPassPointData(pass_point_data)
        alg.Update()
        active_scalars = None
        if not isinstance(dataset, pyvista.MultiBlock):
            active_scalars = dataset.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    def ptc(dataset, pass_point_data=False):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified within cells.
        Optionally, the input point data can be passed through to the output.

        An alias/shortcut for ``point_data_to_cell_data``.

        """
        return DataSetFilters.point_data_to_cell_data(dataset, pass_point_data=pass_point_data)

    def triangulate(dataset, inplace=False):
        """Return an all triangle mesh.

        More complex polygons will be broken down into triangles.

        Parameters
        ----------
        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            Mesh containing only triangles.

        """
        alg = _vtk.vtkDataSetTriangleFilter()
        alg.SetInputData(dataset)
        alg.Update()

        mesh = _get_output(alg)
        if inplace:
            dataset.overwrite(mesh)
            return dataset
        else:
            return mesh

    def delaunay_3d(dataset, alpha=0, tol=0.001, offset=2.5, progress_bar=False):
        """Construct a 3D Delaunay triangulation of the mesh.

        This helps smooth out a rugged mesh.

        Parameters
        ----------
        alpha : float, optional
            Distance value to control output of this filter. For a non-zero
            alpha value, only verts, edges, faces, or tetra contained within
            the circumsphere (of radius alpha) will be output. Otherwise, only
            tetrahedra will be output.

        tol : float, optional
            tolerance to control discarding of closely spaced points.
            This tolerance is specified as a fraction of the diagonal length
            of the bounding box of the points.

        offset : float, optional
            multiplier to control the size of the initial, bounding Delaunay
            triangulation.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        """
        alg = _vtk.vtkDelaunay3D()
        alg.SetInputData(dataset)
        alg.SetAlpha(alpha)
        alg.SetTolerance(tol)
        alg.SetOffset(offset)
        _update_alg(alg, progress_bar, 'Computing 3D Triangulation')
        return _get_output(alg)

    def select_enclosed_points(dataset, surface, tolerance=0.001,
                               inside_out=False, check_surface=True):
        """Mark points as to whether they are inside a closed surface.

        This evaluates all the input points to determine whether they are in an
        enclosed surface. The filter produces a (0,1) mask
        (in the form of a vtkDataArray) that indicates whether points are
        outside (mask value=0) or inside (mask value=1) a provided surface.
        (The name of the output vtkDataArray is "SelectedPoints".)

        This filter produces and output data array, but does not modify the
        input dataset. If you wish to extract cells or poinrs, various
        threshold filters are available (i.e., threshold the output array).

        Warning
        -------
        The filter assumes that the surface is closed and manifold. A boolean
        flag can be set to force the filter to first check whether this is
        true. If false, all points will be marked outside. Note that if this
        check is not performed and the surface is not closed, the results are
        undefined.

        Parameters
        ----------
        surface : pyvista.PolyData
            Set the surface to be used to test for containment. This must be a
            :class:`pyvista.PolyData` object.

        tolerance : float
            The tolerance on the intersection. The tolerance is expressed as a
            fraction of the bounding box of the enclosing surface.

        inside_out : bool
            By default, points inside the surface are marked inside or sent
            to the output. If ``inside_out`` is ``True``, then the points
            outside the surface are marked inside.

        check_surface : bool
            Specify whether to check the surface for closure. If on, then the
            algorithm first checks to see if the surface is closed and
            manifold. If the surface is not closed and manifold, a runtime
            error is raised.

        """
        if not isinstance(surface, pyvista.PolyData):
            raise TypeError("`surface` must be `pyvista.PolyData`")
        if check_surface and surface.n_open_edges > 0:
            raise RuntimeError("Surface is not closed. Please read the warning in the "
                               "documentation for this function and either pass "
                               "`check_surface=False` or repair the surface.")
        alg = _vtk.vtkSelectEnclosedPoints()
        alg.SetInputData(dataset)
        alg.SetSurfaceData(surface)
        alg.SetTolerance(tolerance)
        alg.SetInsideOut(inside_out)
        alg.Update()
        result = _get_output(alg)
        out = dataset.copy()
        bools = result['SelectedPoints'].astype(np.uint8)
        if len(bools) < 1:
            bools = np.zeros(out.n_points, dtype=np.uint8)
        out['SelectedPoints'] = bools
        return out

    def probe(dataset, points, tolerance=None, pass_cell_arrays=True,
              pass_point_arrays=True, categorical=False):
        """Sample data values at specified point locations.

        This uses :class:`vtk.vtkProbeFilter`.

        Parameters
        ----------
        dataset: pyvista.DataSet
            The mesh to probe from - point and cell arrays from
            this object are probed onto the nodes of the ``points`` mesh

        points: pyvista.DataSet
            The points to probe values on to. This should be a PyVista mesh
            or something :func:`pyvista.wrap` can handle.

        tolerance: float, optional
            Tolerance used to compute whether a point in the source is in a
            cell of the input.  If not given, tolerance is automatically generated.

        pass_cell_arrays: bool, optional
            Preserve source mesh's original cell data arrays

        pass_point_arrays: bool, optional
            Preserve source mesh's original point data arrays

        categorical : bool, optional
            Control whether the source point data is to be treated as
            categorical. If the data is categorical, then the resultant data
            will be determined by a nearest neighbor interpolation scheme.

        Examples
        --------
        Probe the active scalars in ``grid`` at the points in ``mesh``

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
        >>> grid = examples.load_uniform()
        >>> result = grid.probe(mesh)
        >>> 'Spatial Point Data' in result.point_arrays
        True

        """
        if not pyvista.is_pyvista_dataset(points):
            points = pyvista.wrap(points)
        alg = _vtk.vtkProbeFilter()
        alg.SetInputData(points)
        alg.SetSourceData(dataset)
        alg.SetPassCellArrays(pass_cell_arrays)
        alg.SetPassPointArrays(pass_point_arrays)
        alg.SetCategoricalData(categorical)
        if tolerance is not None:
            alg.SetComputeTolerance(False)
            alg.SetTolerance(tolerance)
        alg.Update()  # Perform the resampling
        return _get_output(alg)

    def sample(dataset, target, tolerance=None, pass_cell_arrays=True,
               pass_point_arrays=True, categorical=False):
        """Resample array data from a passed mesh onto this mesh.

        This uses :class:`vtk.vtkResampleWithDataSet`.

        Parameters
        ----------
        dataset: pyvista.DataSet
            The source vtk data object as the mesh to sample values on to

        target: pyvista.DataSet
            The vtk data object to sample from - point and cell arrays from
            this object are sampled onto the nodes of the ``dataset`` mesh

        tolerance: float, optional
            Tolerance used to compute whether a point in the source is in a
            cell of the input.  If not given, tolerance is automatically generated.

        pass_cell_arrays: bool, optional
            Preserve source mesh's original cell data arrays

        pass_point_arrays: bool, optional
            Preserve source mesh's original point data arrays

        categorical : bool, optional
            Control whether the source point data is to be treated as
            categorical. If the data is categorical, then the resultant data
            will be determined by a nearest neighbor interpolation scheme.

        """
        if not pyvista.is_pyvista_dataset(target):
            raise TypeError('`target` must be a PyVista mesh type.')
        alg = _vtk.vtkResampleWithDataSet() # Construct the ResampleWithDataSet object
        alg.SetInputData(dataset)  # Set the Input data (actually the source i.e. where to sample from)
        alg.SetSourceData(target) # Set the Source data (actually the target, i.e. where to sample to)
        alg.SetPassCellArrays(pass_cell_arrays)
        alg.SetPassPointArrays(pass_point_arrays)
        alg.SetCategoricalData(categorical)
        if tolerance is not None:
            alg.SetComputeTolerance(False)
            alg.SetTolerance(tolerance)
        alg.Update() # Perform the resampling
        return _get_output(alg)

    def interpolate(dataset, target, sharpness=2, radius=1.0,
                    strategy='null_value', null_value=0.0, n_points=None,
                    pass_cell_arrays=True, pass_point_arrays=True,
                    progress_bar=False, ):
        """Interpolate values onto this mesh from a given dataset.

        The input dataset is typically a point cloud.

        This uses a gaussian interpolation kernel. Use the ``sharpness`` and
        ``radius`` parameters to adjust this kernel. You can also switch this
        kernel to use an N closest points approach.

        Parameters
        ----------
        target: pyvista.DataSet
            The vtk data object to sample from - point and cell arrays from
            this object are interpolated onto this mesh.

        sharpness : float
            Set / Get the sharpness (i.e., falloff) of the Gaussian. By
            default Sharpness=2. As the sharpness increases the effects of
            distant points are reduced.

        radius : float
            Specify the radius within which the basis points must lie.

        n_points : int, optional
            If given, specifies the number of the closest points used to form
            the interpolation basis. This will invalidate the radius argument
            in favor of an N closest points approach. This typically has poorer
            results.

        strategy : str, optional
            Specify a strategy to use when encountering a "null" point during
            the interpolation process. Null points occur when the local
            neighborhood (of nearby points to interpolate from) is empty. If
            the strategy is set to ``'mask_points'``, then an output array is
            created that marks points as being valid (=1) or null (invalid
            =0) (and the NullValue is set as well). If the strategy is set to
            ``'null_value'`` (this is the default), then the output data
            value(s) are set to the ``null_value`` (specified in the output
            point data). Finally, the strategy ``'closest_point'`` is to simply
            use the closest point to perform the interpolation.

        null_value : float, optional
            Specify the null point value. When a null point is encountered
            then all components of each null tuple are set to this value. By
            default the null value is set to zero.

        pass_cell_arrays: bool, optional
            Preserve input mesh's original cell data arrays

        pass_point_arrays: bool, optional
            Preserve input mesh's original point data arrays

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        """
        if not pyvista.is_pyvista_dataset(target):
            raise TypeError('`target` must be a PyVista mesh type.')

        # Must cast to UnstructuredGrid in some cases (e.g. vtkImageData/vtkRectilinearGrid)
        # I believe the locator and the interpolator call `GetPoints` and not all mesh types have that method
        if isinstance(target, (pyvista.UniformGrid, pyvista.RectilinearGrid)):
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
        interpolator.SetInputData(dataset)
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
        interpolator.SetPassPointArrays(pass_point_arrays)
        interpolator.SetPassCellArrays(pass_cell_arrays)
        _update_alg(interpolator, progress_bar, 'Interpolating')
        return _get_output(interpolator)

    def streamlines(dataset, vectors=None, source_center=None,
                    source_radius=None, n_points=100,
                    start_position=None,
                    return_source=False, pointa=None, pointb=None,
                    **kwargs):
        """Integrate a vector field to generate streamlines.

        The default behavior uses a Sphere as the source - set it's location and
        radius via the ``source_center`` and ``source_radius`` keyword arguments.
        ``n_points`` defines the number of starting points on the sphere surface.
        Alternatively, a Line source can be used by specifying ``pointa`` and ``pointb``.
        ``n_points`` again defines the number of points on the line.

        You can retrieve the source by specifying ``return_source=True``.

        Optional keyword parameters from :func:`pyvista.DataSetFilters.streamlines_from_source`
        can be used here to control the generation of streamlines.

        Parameters
        ----------
        vectors : str, optional
            The string name of the active vector field to integrate across.

        source_center : tuple(float), optional
            Length 3 tuple of floats defining the center of the source
            particles. Defaults to the center of the dataset.

        source_radius : float, optional
            Float radius of the source particle cloud. Defaults to one-tenth of
            the diagonal of the dataset's spatial extent.

        n_points : int, optional
            Number of particles present in source sphere or line.

        start_position : tuple(float), optional
            A single point.  This will override the sphere point source.

        return_source : bool, optional
            Return the source particles as :class:`pyvista.PolyData` as well as the
            streamlines. This will be the second value returned if ``True``.

        pointa, pointb : tuple(float), optional
            The coordinates of a start and end point for a line source. This
            will override the sphere and start_position point source.

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

        """
        if source_center is None:
            source_center = dataset.center
        if source_radius is None:
            source_radius = dataset.length / 10.0

        # A single point at start_position
        if start_position is not None:
            source_center = start_position
            source_radius = 0.
            n_points = 1

        if (
            (pointa is not None and pointb is None) or
            (pointa is None and pointb is not None)
        ):
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
        input_source = pyvista.wrap(source.GetOutput())
        output = dataset.streamlines_from_source(input_source, vectors, **kwargs)
        if return_source:
            return output, input_source
        return output


    def streamlines_from_source(dataset, source, vectors=None,
                    integrator_type=45, integration_direction='both',
                    surface_streamlines=False, initial_step_length=0.5,
                    step_unit='cl', min_step_length=0.01, max_step_length=1.0,
                    max_steps=2000, terminal_speed=1e-12, max_error=1e-6,
                    max_time=None, compute_vorticity=True, rotation_scale=1.0,
                    interpolator_type='point'):
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

        integrator_type : int, optional
            The integrator type to be used for streamline generation.
            The default is Runge-Kutta45. The recognized solvers are:
            RUNGE_KUTTA2 (``2``),  RUNGE_KUTTA4 (``4``), and RUNGE_KUTTA45
            (``45``). Options are ``2``, ``4``, or ``45``. Default is ``45``.

        integration_direction : str, optional
            Specify whether the streamline is integrated in the upstream or
            downstream directions (or both). Options are ``'both'``,
            ``'backward'``, or ``'forward'``.

        surface_streamlines : bool, optional
            Compute streamlines on a surface. Default ``False``.

        initial_step_length : float, optional
            Initial step size used for line integration, expressed ib length
            unitsL or cell length units (see ``step_unit`` parameter).
            either the starting size for an adaptive integrator, e.g., RK45, or
            the constant / fixed size for non-adaptive ones, i.e., RK2 and RK4).

        step_unit : str, optional
            Uniform integration step unit. The valid unit is now limited to
            only LENGTH_UNIT (``'l'``) and CELL_LENGTH_UNIT (``'cl'``).
            Default is CELL_LENGTH_UNIT: ``'cl'``.

        min_step_length : float, optional
            Minimum step size used for line integration, expressed in length or
            cell length units. Only valid for an adaptive integrator, e.g., RK45.

        max_step_length : float, optional
            Maximum step size used for line integration, expressed in length or
            cell length units. Only valid for an adaptive integrator, e.g., RK45.

        max_steps : int, optional
            Maximum number of steps for integrating a streamline.
            Defaults to ``2000``

        terminal_speed : float, optional
            Terminal speed value, below which integration is terminated.

        max_error : float, optional
            Maximum error tolerated throughout streamline integration.

        max_time : float, optional
            Specify the maximum length of a streamline expressed in LENGTH_UNIT.

        compute_vorticity : bool, optional
            Vorticity computation at streamline points (necessary for generating
            proper stream-ribbons using the ``vtkRibbonFilter``.

        interpolator_type : str, optional
            Set the type of the velocity field interpolator to locate cells
            during streamline integration either by points or cells.
            The cell locator is more robust then the point locator. Options
            are ``'point'`` or ``'cell'`` (abbreviations of ``'p'`` and ``'c'``
            are also supported).

        rotation_scale : float, optional
            This can be used to scale the rate with which the streamribbons
            twist. The default is 1.

        Returns
        -------
        streamlines : pyvista.PolyData
            This produces polylines as the output, with each cell
            (i.e., polyline) representing a streamline. The attribute values
            associated with each streamline are stored in the cell data, whereas
            those associated with streamline-points are stored in the point data.

        """
        integration_direction = str(integration_direction).strip().lower()
        if integration_direction not in ['both', 'back', 'backward', 'forward']:
            raise ValueError("Integration direction must be one of:\n 'backward', "
                             f"'forward', or 'both' - not '{integration_direction}'.")
        if integrator_type not in [2, 4, 45]:
            raise ValueError('Integrator type must be one of `2`, `4`, or `45`.')
        if interpolator_type not in ['c', 'cell', 'p', 'point']:
            raise ValueError("Interpolator type must be either 'cell' or 'point'")
        if step_unit not in ['l', 'cl']:
            raise ValueError("Step unit must be either 'l' or 'cl'")
        step_unit = {'cl': _vtk.vtkStreamTracer.CELL_LENGTH_UNIT,
                     'l': _vtk.vtkStreamTracer.LENGTH_UNIT}[step_unit]
        if isinstance(vectors, str):
            dataset.set_active_scalars(vectors)
            dataset.set_active_vectors(vectors)
        if max_time is None:
            max_velocity = dataset.get_data_range()[-1]
            max_time = 4.0 * dataset.GetLength() / max_velocity
        if not isinstance(source, pyvista.DataSet):
            raise TypeError("source must be a pyvista.DataSet")

        # vtk throws error with two Structured Grids
        # See: https://github.com/pyvista/pyvista/issues/1373
        if isinstance(dataset, pyvista.StructuredGrid) and isinstance(source, pyvista.StructuredGrid):
            source = source.cast_to_unstructured_grid()

        # Build the algorithm
        alg = _vtk.vtkStreamTracer()
        # Inputs
        alg.SetInputDataObject(dataset)
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
        alg.Update()
        return _get_output(alg)

    def decimate_boundary(dataset, target_reduction=0.5):
        """Return a decimated version of a triangulation of the boundary.

        Only the outer surface of the input dataset will be considered.

        Parameters
        ----------
        target_reduction : float
            Fraction of the original mesh to remove. Default is ``0.5``
            TargetReduction is set to ``0.9``, this filter will try to reduce
            the data set to 10% of its original size and will remove 90%
            of the input triangles.

        """
        return dataset.extract_geometry().triangulate().decimate(target_reduction)

    def sample_over_line(dataset, pointa, pointb, resolution=None, tolerance=None):
        """Sample a dataset onto a line.

        Parameters
        ----------
        pointa : np.ndarray or list
            Location in [x, y, z].

        pointb : np.ndarray or list
            Location in [x, y, z].

        resolution : int
            Number of pieces to divide line into. Defaults to number of cells
            in the input mesh. Must be a positive integer.

        tolerance: float, optional
            Tolerance used to compute whether a point in the source is in a
            cell of the input.  If not given, tolerance is automatically generated.

        Returns
        -------
        sampled_line : pv.PolyData
            Line object with sampled data from dataset.

        """
        if resolution is None:
            resolution = int(dataset.n_cells)
        # Make a line and sample the dataset
        line = pyvista.Line(pointa, pointb, resolution=resolution)

        sampled_line = line.sample(dataset, tolerance=tolerance)
        return sampled_line

    def plot_over_line(dataset, pointa, pointb, resolution=None, scalars=None,
                       title=None, ylabel=None, figsize=None, figure=True,
                       show=True, tolerance=None, fname=None):
        """Sample a dataset along a high resolution line and plot.

        Plot the variables of interest in 2D where the X-axis is distance from
        Point A and the Y-axis is the variable of interest. Note that this filter
        returns None.

        Parameters
        ----------
        pointa : np.ndarray or list
            Location in [x, y, z].

        pointb : np.ndarray or list
            Location in [x, y, z].

        resolution : int
            number of pieces to divide line into. Defaults to number of cells
            in the input mesh. Must be a positive integer.

        scalars : str
            The string name of the variable in the input dataset to probe. The
            active scalar is used by default.

        title : str
            The string title of the `matplotlib` figure

        ylabel : str
            The string label of the Y-axis. Defaults to variable name

        figsize : tuple(int)
            the size of the new figure

        figure : bool
            flag on whether or not to create a new figure

        show : bool
            Shows the matplotlib figure

        tolerance: float, optional
            Tolerance used to compute whether a point in the source is in a
            cell of the input.  If not given, tolerance is automatically generated.

        fname : str, optional
            Save the figure this file name when set.

        """
        # Ensure matplotlib is available
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError('matplotlib must be available to use this filter.')

        # Sample on line
        sampled = DataSetFilters.sample_over_line(dataset, pointa, pointb, resolution, tolerance)

        # Get variable of interest
        if scalars is None:
            field, scalars = dataset.active_scalars_info
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
            return plt.show()

    def sample_over_circular_arc(dataset, pointa, pointb, center,
                                 resolution=None, tolerance=None):
        """Sample a dataset over a circular arc.

        Parameters
        ----------
        pointa : np.ndarray or list
            Location in ``[x, y, z]``.

        pointb : np.ndarray or list
            Location in ``[x, y, z]``.

        center : np.ndarray or list
            Location in ``[x, y, z]``.

        resolution : int, optional
            Number of pieces to divide circular arc into. Defaults to
            number of cells in the input mesh. Must be a positive
            integer.

        tolerance: float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        Examples
        --------
        Sample a dataset over a circular arc.

        >>> from pyvista import examples
        >>> uniform = examples.load_uniform()
        >>> uniform["height"] = uniform.points[:, 2]
        >>> pointa = [uniform.bounds[0], uniform.bounds[2], uniform.bounds[5]]
        >>> pointb = [uniform.bounds[1], uniform.bounds[2], uniform.bounds[4]]
        >>> center = [uniform.bounds[0], uniform.bounds[2], uniform.bounds[4]]
        >>> sampled_arc = uniform.sample_over_circular_arc(pointa, pointb, center)

        """
        if resolution is None:
            resolution = int(dataset.n_cells)
        # Make a circular arc and sample the dataset
        circular_arc = pyvista.CircularArc(pointa, pointb, center, resolution=resolution)

        sampled_circular_arc = circular_arc.sample(dataset, tolerance=tolerance)
        return sampled_circular_arc

    def sample_over_circular_arc_normal(dataset, center, resolution=None, normal=None,
                                        polar=None, angle=None, tolerance=None):
        """Sample a dataset over a circular arc defined by a normal and polar vector and plot it.

        The number of segments composing the polyline is controlled by
        setting the object resolution.

        Parameters
        ----------
        center : np.ndarray or list
            Location in ``[x, y, z]``.

        resolution : int, optional
            Number of pieces to divide circular arc into. Defaults to
            number of cells in the input mesh. Must be a positive
            integer.

        normal : np.ndarray or list, optional
            The normal vector to the plane of the arc.  By default it
            points in the positive Z direction.

        polar : np.ndarray or list, optional
            Starting point of the arc in polar coordinates.  By
            default it is the unit vector in the positive x direction.

        angle : float, optional
            Arc length (in degrees), beginning at the polar vector.  The
            direction is counterclockwise.  By default it is 360.

        tolerance: float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        Examples
        --------
        Sample a dataset over a circular arc.

        >>> from pyvista import examples
        >>> uniform = examples.load_uniform()
        >>> uniform["height"] = uniform.points[:, 2]
        >>> normal = [0, 0, 1]
        >>> polar = [uniform.bounds[0], uniform.bounds[2], uniform.bounds[5]]
        >>> center = [uniform.bounds[0], uniform.bounds[2], uniform.bounds[4]]
        >>> sampled_arc = uniform.sample_over_circular_arc_normal(center, normal=normal, polar=polar)

        """
        if resolution is None:
            resolution = int(dataset.n_cells)
        # Make a circular arc and sample the dataset
        circular_arc = pyvista.CircularArcFromNormal(center,
                                                     resolution=resolution,
                                                     normal=normal,
                                                     polar=polar,
                                                     angle=angle)

        sampled_circular_arc = circular_arc.sample(dataset, tolerance=tolerance)
        return sampled_circular_arc

    def plot_over_circular_arc(dataset, pointa, pointb, center,
                               resolution=None, scalars=None,
                               title=None, ylabel=None, figsize=None,
                               figure=True, show=True, tolerance=None, fname=None):
        """Sample a dataset along a circular arc and plot it.

        Plot the variables of interest in 2D where the X-axis is
        distance from Point A and the Y-axis is the variable of
        interest. Note that this filter returns ``None``.

        Parameters
        ----------
        pointa : np.ndarray or list
            Location in ``[x, y, z]``.

        pointb : np.ndarray or list
            Location in ``[x, y, z]``.

        center : np.ndarray or list
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

        figure : bool, optional
            Flag on whether or not to create a new figure.

        show : bool, optional
            Shows the ``matplotlib`` figure when ``True``.

        tolerance: float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        fname : str, optional
            Save the figure this file name when set.

        Examples
        --------
        Sample a dataset along a high resolution circular arc and plot.

        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> a = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[5]]
        >>> b = [mesh.bounds[1], mesh.bounds[2], mesh.bounds[4]]
        >>> center = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
        >>> mesh.plot_over_circular_arc(a, b, center, resolution=1000, show=False)

        """
        # Ensure matplotlib is available
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError('matplotlib must be installed to use this filter.')

        # Sample on circular arc
        sampled = DataSetFilters.sample_over_circular_arc(dataset,
                                                          pointa,
                                                          pointb,
                                                          center,
                                                          resolution,
                                                          tolerance)

        # Get variable of interest
        if scalars is None:
            field, scalars = dataset.active_scalars_info
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
            return plt.show()

    def plot_over_circular_arc_normal(dataset, center, resolution=None, normal=None,
                                      polar=None, angle=None, scalars=None,
                                      title=None, ylabel=None, figsize=None,
                                      figure=True, show=True, tolerance=None, fname=None):
        """Sample a dataset along a resolution circular arc defined by a normal and polar vector and plot it.

        Plot the variables of interest in 2D where the X-axis is
        distance from Point A and the Y-axis is the variable of
        interest. Note that this filter returns ``None``.

        Parameters
        ----------
        center : np.ndarray or list
            Location in ``[x, y, z]``.

        resolution : int, optional
            number of pieces to divide circular arc into. Defaults to
            number of cells in the input mesh. Must be a positive
            integer.

        normal : np.ndarray or list, optional
            The normal vector to the plane of the arc.  By default it
            points in the positive Z direction.

        polar : np.ndarray or list, optional
            (starting point of the arc).  By default it is the unit vector
            in the positive x direction.

        angle : float, optional
            Arc length (in degrees), beginning at the polar vector.  The
            direction is counterclockwise.  By default it is 360.

        scalars : str, optional
            The string name of the variable in the input dataset to
            probe. The active scalar is used by default.

        title : str, optional
            The string title of the `matplotlib` figure

        ylabel : str, optional
            The string label of the Y-axis. Defaults to variable name

        figsize : tuple(int), optional
            the size of the new figure

        figure : bool, optional
            flag on whether or not to create a new figure

        show : bool, optional
            Shows the matplotlib figure

        tolerance: float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        fname : str, optional
            Save the figure this file name when set.

        Examples
        --------
        Sample a dataset along a high resolution circular arc and plot.

        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> normal = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[5]]
        >>> polar = [mesh.bounds[0], mesh.bounds[3], mesh.bounds[4]]
        >>> angle = 90
        >>> center = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
        >>> mesh.plot_over_circular_arc_normal(center, polar=polar, angle=angle)  # doctest:+SKIP

        """
        # Ensure matplotlib is available
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError('matplotlib must be installed to use this filter.')

        # Sample on circular arc
        sampled = DataSetFilters.sample_over_circular_arc_normal(dataset,
                                                                 center,
                                                                 resolution,
                                                                 normal,
                                                                 polar,
                                                                 angle,
                                                                 tolerance)

        # Get variable of interest
        if scalars is None:
            field, scalars = dataset.active_scalars_info
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
            return plt.show()

    def extract_cells(dataset, ind):
        """Return a subset of the grid.

        Parameters
        ----------
        ind : np.ndarray
            Numpy array of cell indices to be extracted.

        Returns
        -------
        subgrid : pyvista.UnstructuredGrid
            Subselected grid

        """
        # Create selection objects
        selectionNode = _vtk.vtkSelectionNode()
        selectionNode.SetFieldType(_vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(_vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(numpy_to_idarr(ind))

        selection = _vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = _vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, dataset)
        extract_sel.SetInputData(1, selection)
        extract_sel.Update()
        subgrid = _get_output(extract_sel)

        # extracts only in float32
        if subgrid.n_points:
            if dataset.points.dtype is not np.dtype('float32'):
                ind = subgrid.point_arrays['vtkOriginalPointIds']
                subgrid.points = dataset.points[ind]

        return subgrid

    def extract_points(dataset, ind, adjacent_cells=True, include_cells=True):
        """Return a subset of the grid (with cells) that contains any of the given point indices.

        Parameters
        ----------
        ind : np.ndarray, list, or sequence
            Numpy array of point indices to be extracted.
        adjacent_cells : bool, optional
            If True, extract the cells that contain at least one of the
            extracted points. If False, extract the cells that contain
            exclusively points from the extracted points list. The default is
            True.
        include_cells : bool, optional
            Specifies if the cells shall be returned or not. The default is
            True.

        Returns
        -------
        subgrid : pyvista.UnstructuredGrid
            Subselected grid.

        """
        # Create selection objects
        selectionNode = _vtk.vtkSelectionNode()
        selectionNode.SetFieldType(_vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(_vtk.vtkSelectionNode.INDICES)
        if not include_cells:
            adjacent_cells = True
        if not adjacent_cells:
            # Build array of point indices to be removed.
            ind_rem = np.ones(dataset.n_points, dtype='bool')
            ind_rem[ind] = False
            ind = np.arange(dataset.n_points)[ind_rem]
            # Invert selection
            selectionNode.GetProperties().Set(_vtk.vtkSelectionNode.INVERSE(), 1)
        selectionNode.SetSelectionList(numpy_to_idarr(ind))
        if include_cells:
            selectionNode.GetProperties().Set(_vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)

        selection = _vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = _vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, dataset)
        extract_sel.SetInputData(1, selection)
        extract_sel.Update()
        return _get_output(extract_sel)

    def extract_surface(dataset, pass_pointid=True, pass_cellid=True,
                        nonlinear_subdivision=1):
        """Extract surface mesh of the grid.

        Parameters
        ----------
        pass_pointid : bool, optional
            Adds a point array "vtkOriginalPointIds" that idenfities which
            original points these surface points correspond to

        pass_cellid : bool, optional
            Adds a cell array "vtkOriginalPointIds" that idenfities which
            original cells these surface cells correspond to

        nonlinear_subdivision : int, optional
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

        Returns
        -------
        pyvista.PolyData
            Surface mesh of the grid.

        Examples
        --------
        Extract the surface of an UnstructuredGrid.

        >>> import pyvista
        >>> from pyvista import examples
        >>> grid = examples.load_hexbeam()
        >>> surf = grid.extract_surface()
        >>> type(surf)
        <class 'pyvista.core.pointset.PolyData'>

        """
        surf_filter = _vtk.vtkDataSetSurfaceFilter()
        surf_filter.SetInputData(dataset)
        if pass_pointid:
            surf_filter.PassThroughCellIdsOn()
        if pass_cellid:
            surf_filter.PassThroughPointIdsOn()

        if nonlinear_subdivision != 1:
            surf_filter.SetNonlinearSubdivisionLevel(nonlinear_subdivision)

        # available in 9.0.2
        # surf_filter.SetDelegation(delegation)

        surf_filter.Update()
        return _get_output(surf_filter)

    def surface_indices(dataset):
        """Return the surface indices of a grid.

        Returns
        -------
        surf_ind : np.ndarray
            Indices of the surface points.

        """
        surf = DataSetFilters.extract_surface(dataset, pass_cellid=True)
        return surf.point_arrays['vtkOriginalPointIds']

    def extract_feature_edges(dataset, feature_angle=30, boundary_edges=True,
                              non_manifold_edges=True, feature_edges=True,
                              manifold_edges=True, inplace=False):
        """Extract edges from the surface of the mesh.

        If the given mesh is not PolyData, the external surface of the given
        mesh is extracted and used.
        From vtk documentation, the edges are one of the following

            1) boundary (used by one polygon) or a line cell
            2) non-manifold (used by three or more polygons)
            3) feature edges (edges used by two triangles and whose
               dihedral angle > feature_angle)
            4) manifold edges (edges used by exactly two polygons).

        Parameters
        ----------
        feature_angle : float, optional
            Defaults to 30 degrees.

        boundary_edges : bool, optional
            Defaults to True

        non_manifold_edges : bool, optional
            Defaults to True

        feature_edges : bool, optional
            Defaults to True

        manifold_edges : bool, optional
            Defaults to True

        inplace : bool, optional
            Updates existing dataset with the extracted features.

        Returns
        -------
        edges : pyvista.vtkPolyData
            Extracted edges.

        """
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
        featureEdges.Update()

        mesh = _get_output(featureEdges)
        if inplace:
            dataset.overwrite(mesh)
            return dataset
        else:
            return mesh

    def merge(dataset, grid=None, merge_points=True, inplace=False,
              main_has_priority=True):
        """Join one or many other grids to this grid.

        Grid is updated in-place by default.

        Can be used to merge points of adjacent cells when no grids
        are input.

        Parameters
        ----------
        grid : vtk.UnstructuredGrid or list of vtk.UnstructuredGrids
            Grids to merge to this grid.

        merge_points : bool, optional
            Points in exactly the same location will be merged between
            the two meshes. Warning: this can leave degenerate point data.

        inplace : bool, optional
            Updates grid inplace when True if the input type is an
            :class:`pyvista.UnstructuredGrid`.

        main_has_priority : bool, optional
            When this parameter is true and merge_points is true,
            the arrays of the merging grids will be overwritten
            by the original main mesh.

        Returns
        -------
        merged_grid : vtk.UnstructuredGrid
            Merged grid.

        Notes
        -----
        When two or more grids are joined, the type and name of each
        array must match or the arrays will be ignored and not
        included in the final merged mesh.

        """
        append_filter = _vtk.vtkAppendFilter()
        append_filter.SetMergePoints(merge_points)

        if not main_has_priority:
            append_filter.AddInputData(dataset)

        if isinstance(grid, pyvista.DataSet):
            append_filter.AddInputData(grid)
        elif isinstance(grid, (list, tuple, pyvista.MultiBlock)):
            grids = grid
            for grid in grids:
                append_filter.AddInputData(grid)

        if main_has_priority:
            append_filter.AddInputData(dataset)

        append_filter.Update()
        merged = _get_output(append_filter)
        if inplace:
            if type(dataset) == type(merged):
                dataset.deep_copy(merged)
                return dataset
            else:
                raise TypeError(f"Mesh type {type(dataset)} cannot be overridden by output.")
        else:
            return merged

    def __add__(dataset, grid):
        """Combine this mesh with another into an :class:`pyvista.UnstructuredGrid`."""
        return DataSetFilters.merge(dataset, grid)

    def compute_cell_quality(dataset, quality_measure='scaled_jacobian', null_value=-1.0):
        """Compute a function of (geometric) quality for each cell of a mesh.

        The per-cell quality is added to the mesh's cell data, in an array
        named "CellQuality". Cell types not supported by this filter or
        undefined quality of supported cell types will have an entry of -1.

        Defaults to computing the scaled jacobian.

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

        Parameters
        ----------
        quality_measure : str
            The cell quality measure to use

        null_value : float
            Float value for undefined quality. Undefined quality are qualities
            that could be addressed by this filter but is not well defined for
            the particular geometry of cell in question, e.g. a volume query
            for a triangle. Undefined quality will always be undefined.
            The default value is -1.

        """
        alg = _vtk.vtkCellQuality()
        measure_setters = {
            'area': alg.SetQualityMeasureToArea,
            'aspect_beta': alg.SetQualityMeasureToAspectBeta,
            'aspect_frobenius': alg.SetQualityMeasureToAspectFrobenius,
            'aspect_gamma': alg.SetQualityMeasureToAspectGamma,
            'aspect_ratio': alg.SetQualityMeasureToAspectRatio,
            'collapse_ratio': alg.SetQualityMeasureToCollapseRatio,
            'condition': alg.SetQualityMeasureToCondition,
            'diagonal': alg.SetQualityMeasureToDiagonal,
            'dimension': alg.SetQualityMeasureToDimension,
            'distortion': alg.SetQualityMeasureToDistortion,
            'jacobian': alg.SetQualityMeasureToJacobian,
            'max_angle': alg.SetQualityMeasureToMaxAngle,
            'max_aspect_frobenius': alg.SetQualityMeasureToMaxAspectFrobenius,
            'max_edge_ratio': alg.SetQualityMeasureToMaxEdgeRatio,
            'med_aspect_frobenius': alg.SetQualityMeasureToMedAspectFrobenius,
            'min_angle': alg.SetQualityMeasureToMinAngle,
            'oddy': alg.SetQualityMeasureToOddy,
            'radius_ratio': alg.SetQualityMeasureToRadiusRatio,
            'relative_size_squared': alg.SetQualityMeasureToRelativeSizeSquared,
            'scaled_jacobian': alg.SetQualityMeasureToScaledJacobian,
            'shape': alg.SetQualityMeasureToShape,
            'shape_and_size': alg.SetQualityMeasureToShapeAndSize,
            'shear': alg.SetQualityMeasureToShear,
            'shear_and_size': alg.SetQualityMeasureToShearAndSize,
            'skew': alg.SetQualityMeasureToSkew,
            'stretch': alg.SetQualityMeasureToStretch,
            'taper': alg.SetQualityMeasureToTaper,
            'volume': alg.SetQualityMeasureToVolume,
            'warpage': alg.SetQualityMeasureToWarpage
        }
        try:
            # Set user specified quality measure
            measure_setters[quality_measure]()
        except (KeyError, IndexError):
            options = ', '.join([f"'{s}'" for s in list(measure_setters.keys())])
            raise KeyError(f'Cell quality type ({quality_measure}) not available. Options are: {options}')
        alg.SetInputData(dataset)
        alg.SetUndefinedQuality(null_value)
        alg.Update()
        return _get_output(alg)

    def compute_derivative(dataset, scalars=None, gradient=True,
                           divergence=None, vorticity=None, qcriterion=None,
                           faster=False, preference='point'):
        """Compute derivative-based quantities of point/cell scalar field.

        Utilize ``vtkGradientFilter`` to compute derivative-based quantities,
        such as gradient, divergence, vorticity, and Q-criterion, of the
        selected point or cell scalar field.

        Parameters
        ----------
        scalars : str, optional
            String name of the scalars array to use when computing the
            derivative quantities.

        gradient: bool, str, optional
            Calculate gradient. If a string is passed, the string will be used
            for the resulting array name. Otherwise, array name will be
            'gradient'. Default: True

        divergence: bool, str, optional
            Calculate divergence. If a string is passed, the string will be
            used for the resulting array name. Otherwise, array name will be
            'divergence'. Default: None

        vorticity: bool, str, optional
            Calculate vorticity. If a string is passed, the string will be used
            for the resulting array name. Otherwise, array name will be
            'vorticity'. Default: None

        qcriterion: bool, str, optional
            Calculate qcriterion. If a string is passed, the string will be
            used for the resulting array name. Otherwise, array name will be
            'qcriterion'. Default: None

        faster: bool, optional
            Use faster algorithm for computing derivative quantities. Result is
            less accurate and performs fewer derivative calculations,
            increasing computation speed. The error will feature smoothing of
            the output and possibly errors at boundaries. Option has no effect
            if DataSet is not UnstructuredGrid. Default: False

        preference: str, optional
            Data type preference. Either 'point' or 'cell'.

        """
        alg = _vtk.vtkGradientFilter()
        # Check if scalars array given
        if scalars is None:
            field, scalars = dataset.active_scalars_info
            if scalars is None:
                raise TypeError('No active scalars.  Must input scalars array name')
        if not isinstance(scalars, str):
            raise TypeError('scalars array must be given as a string name')
        if not any((gradient, divergence, vorticity, qcriterion)):
            raise ValueError('must set at least one of gradient, divergence, vorticity, or qcriterion')

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
        _, field = dataset.get_array(scalars, preference=preference, info=True)
        # args: (idx, port, connection, field, name)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
        alg.SetInputData(dataset)
        alg.Update()
        return _get_output(alg)

    def shrink(dataset, shrink_factor=1.0, progress_bar=False):
        """Shrink the individual faces of a mesh.

        This filter shrinks the individual faces of a mesh rather than scaling
        the entire mesh.

        Parameters
        ----------
        shrink_factor : float, optional
            fraction of shrink for each cell.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Examples
        --------
        Extrude shrink mesh

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> shrunk_mesh = mesh.shrink(shrink_factor=0.8)

        """
        if not (0.0 <= shrink_factor <= 1.0):
            raise ValueError('`shrink_factor` should be between 0.0 and 1.0')
        alg = _vtk.vtkShrinkFilter()
        alg.SetInputData(dataset)
        alg.SetShrinkFactor(shrink_factor)
        _update_alg(alg, progress_bar, 'Shrinking Mesh')
        output = pyvista.wrap(alg.GetOutput())
        if isinstance(dataset, _vtk.vtkPolyData):
            return output.extract_surface()

    def transform(dataset: _vtk.vtkDataSet,
                  trans: Union[_vtk.vtkMatrix4x4, _vtk.vtkTransform, np.ndarray],
                  transform_all_input_vectors=False, inplace=True):
        """Transform this mesh with a 4x4 transform.

        Parameters
        ----------
        trans : vtk.vtkMatrix4x4, vtk.vtkTransform, or np.ndarray
            Accepts a vtk transformation object or a 4x4
            transformation matrix.

        transform_all_input_vectors: bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            When ``True``, modifies the dataset inplace.

        Examples
        --------
        Translate a mesh by ``(50, 100, 200)``.

        >>> import numpy as np
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()

        Here a 4x4 ``numpy`` array is used, but
        ``vtk.vtkMatrix4x4`` and ``vtk.vtkTransform`` are also
        accepted.

        >>> transform_matrix = np.array([[1, 0, 0, 50],
        ...                              [0, 1, 0, 100],
        ...                              [0, 0, 1, 200],
        ...                              [0, 0, 0, 1]])
        >>> transformed = mesh.transform(transform_matrix)
        >>> cpos = transformed.plot(show_edges=True)

        """
        if isinstance(trans, _vtk.vtkMatrix4x4):
            m = trans
            t = _vtk.vtkTransform()
            t.SetMatrix(m)
        elif isinstance(trans, _vtk.vtkTransform):
            t = trans
            m = trans.GetMatrix()
        elif isinstance(trans, np.ndarray):
            if trans.ndim != 2:
                raise ValueError('Transformation array must be 4x4')
            elif trans.shape[0] != 4 or trans.shape[1] != 4:
                raise ValueError('Transformation array must be 4x4')
            m = pyvista.vtkmatrix_from_array(trans)
            t = _vtk.vtkTransform()
            t.SetMatrix(m)
        else:
            raise TypeError('Input transform must be either:\n'
                            '\tvtk.vtkMatrix4x4\n'
                            '\tvtk.vtkTransform\n'
                            '\t4x4 np.ndarray\n')

        if m.GetElement(3, 3) == 0:
            raise ValueError(
                "Transform element (3,3), the inverse scale term, is zero")

        # vtkTransformFilter sometimes doesn't transform all vector arrays
        # when there are active point/cell scalars. Use this workaround
        active_scalars_name = dataset.active_scalars_name
        dataset.set_active_scalars(None)

        f = _vtk.vtkTransformFilter()
        f.SetInputDataObject(dataset)
        f.SetTransform(t)

        if hasattr(f, 'SetTransformAllInputVectors'):
            f.SetTransformAllInputVectors(transform_all_input_vectors)
        else:
            # In VTK 8.1.2 and earlier, vtkTransformFilter does not support the transformation of all input vectors.
            # Raise an error if the user requested for input vectors to be transformed and it is not supported
            if transform_all_input_vectors:
                raise VTKVersionError('The installed version of VTK does not support '
                                      'transformation of all input vectors.')

        f.Update()
        res = pyvista.core.filters._get_output(f)

        # make the previously active scalars active again
        dataset.set_active_scalars(active_scalars_name)
        res.set_active_scalars(active_scalars_name)

        if inplace:
            if not isinstance(dataset, type(res)):
                raise ValueError('Unable to perform in-place transformation. '
                                 f'Input was `{dataset.GetClassName()}` '
                                 f'but output is `{res.GetClassName()}`.')
            dataset.overwrite(res)
            return dataset
        else:
            return res

    def reflect(dataset, normal, point=None, inplace=False,
                transform_all_input_vectors=False):
        """Reflect a dataset across a plane.

        Parameters
        ----------
        normal : tuple(float)
            Normal direction for reflection.

        point : tuple(float), optional
            Point which, along with `normal`, defines the reflection plane. If not
            specified, this is the origin.

        inplace : bool, optional
            When ``True``, modifies the dataset inplace.

        transform_all_input_vectors: bool, optional
            When ``True``, all input vectors are transformed. Otherwise, only the
            points, normals and active vectors are transformed.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh = mesh.reflect((0, 0, 1), point=(0, 0, -100))
        >>> cpos = mesh.plot(show_edges=True)

        """
        t = transformations.reflection(normal, point=point)
        return dataset.transform(t, transform_all_input_vectors=transform_all_input_vectors,
                                 inplace=inplace)
