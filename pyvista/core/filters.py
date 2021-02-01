"""These classes hold methods to apply general filters to any data type.

By inheriting these classes into the wrapped VTK data structures, a user
can easily apply common filters in an intuitive manner.

Example
-------
>>> import pyvista
>>> from pyvista import examples
>>> dataset = examples.load_uniform()

>>> # Threshold
>>> thresh = dataset.threshold([100, 500])

>>> # Slice
>>> slc = dataset.slice()

>>> # Clip
>>> clp = dataset.clip(invert=True)

>>> # Contour
>>> iso = dataset.contour()

"""
import collections.abc
import logging
from functools import wraps

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from pyvista.utilities import (FieldAssociation, NORMALS, assert_empty_kwargs,
                               generate_plane, get_array, vtk_id_list_to_array,
                               wrap, ProgressMonitor, abstract_class)
from pyvista.utilities.cells import numpy_to_idarr
from pyvista.core.errors import NotAllTrianglesError
from pyvista.utilities import transformations


def _update_alg(alg, progress_bar=False, message=''):
    """Update an algorithm with or without a progress bar."""
    if progress_bar:
        with ProgressMonitor(alg, message=message):
            alg.Update()
    else:
        alg.Update()


def _get_output(algorithm, iport=0, iconnection=0, oport=0, active_scalars=None,
                active_scalars_field='point'):
    """Get the algorithm's output and copy input's pyvista meta info."""
    ido = algorithm.GetInputDataObject(iport, iconnection)
    data = wrap(algorithm.GetOutputDataObject(oport))
    if not isinstance(data, pyvista.MultiBlock):
        data.copy_meta_from(ido)
        if not data.field_arrays and ido.field_arrays:
            data.field_arrays.update(ido.field_arrays)
        if active_scalars is not None:
            data.set_active_scalars(active_scalars, preference=active_scalars_field)
    return data


@abstract_class
class DataSetFilters:
    """A set of common filters that can be applied to any vtkDataSet."""

    def _clip_with_function(dataset, function, invert=True, value=0.0, return_clipped=False):
        """Clip using an implicit function (internal helper)."""
        if isinstance(dataset, vtk.vtkPolyData):
            alg = vtk.vtkClipPolyData()
        # elif isinstance(dataset, vtk.vtkImageData):
        #     alg = vtk.vtkClipVolume()
        #     alg.SetMixed3DCellGeneration(True)
        else:
            alg = vtk.vtkTableBasedClipDataSet()
        alg.SetInputDataObject(dataset) # Use the grid as the data we desire to cut
        alg.SetValue(value)
        alg.SetClipFunction(function) # the implicit function
        alg.SetInsideOut(invert) # invert the clip if needed
        if return_clipped:
            alg.GenerateClippedOutputOn()
        alg.Update() # Perform the Cut

        if return_clipped:
            a = _get_output(alg, oport=0)
            b = _get_output(alg, oport=1)
            return a, b
        else:
            return _get_output(alg)

    def clip(dataset, normal='x', origin=None, invert=True, value=0.0, inplace=False, return_clipped=False):
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
            Updates mesh in-place while returning nothing.

        return_clipped : bool, optional
            Return both unclipped and clipped parts of the dataset.

        Returns
        -------
        mesh : pyvista.PolyData or tuple(pyvista.PolyData)
            Clipped mesh when ``inplace=False``.  When
            ``inplace=True``, ``None``. When ``return_clipped=True``,
            a tuple containing the unclipped and clipped datasets,
            regardless of the setting of ``inplace``.

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
            overwrite_with = result[0] if return_clipped else result
            dataset.overwrite(overwrite_with)
            if return_clipped:
                # normally if inplace=True, filters return None. But if
                # return_clipped=True, the user still wants the clipped data,
                # so return both the unclipped and clipped data as a tuple
                return result
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
            bounds = (xmin,xmin+bounds[0], ymin,ymin+bounds[1], zmin,zmin+bounds[2])
        alg = vtk.vtkBoxClipDataSet()
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

        This filter will comput the implicit distance from all of the nodes of
        this mesh to a given surface. This distance will be added as a point
        array called ``'implicit_distance'``.

        Parameters
        ----------
        surface : pyvista.Common
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
        >>> sphere.compute_implicit_distance(plane, inplace=True)
        >>> dist = sphere['implicit_distance']
        >>> print(type(dist))
        <class 'numpy.ndarray'>

        Plot these distances as a heatmap

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere, scalars='implicit_distance', cmap='bwr')
        >>> _ = pl.add_mesh(plane, color='w', style='wireframe')
        >>> pl.show()  # doctest:+SKIP

        """
        function = vtk.vtkImplicitPolyDataDistance()
        function.SetInput(surface)
        points = pyvista.convert_array(dataset.points)
        dists = vtk.vtkDoubleArray()
        function.FunctionValue(points, dists)
        if inplace:
            dataset.point_arrays['implicit_distance'] = pyvista.convert_array(dists)
            return
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
            Updates mesh in-place while returning nothing.

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
        if isinstance(dataset, vtk.vtkPolyData):
            alg = vtk.vtkClipPolyData()
        else:
            alg = vtk.vtkTableBasedClipDataSet()

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
        if not isinstance(surface, vtk.vtkPolyData):
            surface = DataSetFilters.extract_geometry(surface)
        function = vtk.vtkImplicitPolyDataDistance()
        function.SetInput(surface)
        if compute_distance:
            points = pyvista.convert_array(dataset.points)
            dists = vtk.vtkDoubleArray()
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
        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(dataset) # Use the grid as the data we desire to cut
        alg.SetCutFunction(plane) # the cutter to use the plane we made
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        alg.Update() # Perform the Cut
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
        if not isinstance(polyline, vtk.vtkPolyLine):
            raise TypeError(f'Input line must have a PolyLine cell, not ({type(polyline)})')
        # Generate PolyPlane
        polyplane = vtk.vtkPolyPlane()
        polyplane.SetPolyLine(polyline)
        # Create slice
        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(dataset) # Use the grid as the data we desire to cut
        alg.SetCutFunction(polyplane) # the cutter to use the poly planes
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        alg.Update() # Perform the Cut
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
            appender = vtk.vtkAppendFilter()
            appender.AddInputData(t1)
            appender.AddInputData(t2)
            appender.Update()
            return _get_output(appender)

        # Run a standard threshold algorithm
        alg = vtk.vtkThreshold()
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
        alg = vtk.vtkOutlineFilter()
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
        alg = vtk.vtkOutlineCornerFilter()
        alg.SetInputDataObject(dataset)
        alg.SetCornerFactor(factor)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))

    def extract_geometry(dataset):
        """Extract the outer surface of a volume or structured grid dataset as PolyData.

        This will extract all 0D, 1D, and 2D cells producing the
        boundary faces of the dataset.

        """
        alg = vtk.vtkGeometryFilter()
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
        alg = vtk.vtkExtractEdges()
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
        alg = vtk.vtkElevationFilter()
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
            alg = vtk.vtkContourFilter()
        elif method == 'marching_cubes':
            alg = vtk.vtkMarchingCubes()
        elif method == 'flying_edges':
            alg = vtk.vtkFlyingEdges3D()
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
            If True, the new texture coordinates will be added to the dataset
            inplace. If False (default), a new dataset is returned with the
            textures coordinates

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
            point_v = [b[0], b[3], b[4]] # TOP LEFT CORNER
        alg = vtk.vtkTextureMapToPlane()
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
        return # No return type because it is inplace

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
        >>> sphere = pyvista.Sphere()
        >>> sphere.texture_map_to_sphere(inplace=True)
        >>> tex = examples.download_puppy_texture()  # doctest:+SKIP
        >>> sphere.plot(texture=tex)  # doctest:+SKIP
        """
        alg = vtk.vtkTextureMapToSphere()
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
        dataset.GetPointData().AddArray(otc) # Add old ones back at the end
        return # No return type because it is inplace

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
        alg = vtk.vtkCellSizeFilter()
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
        alg = vtk.vtkCellCenters()
        alg.SetInputDataObject(dataset)
        alg.SetVertexCells(vertex)
        alg.Update()
        output = _get_output(alg)
        return output

    def glyph(dataset, orient=True, scale=True, factor=1.0, geom=None,
              indices=None, tolerance=None, absolute=False, clamping=False,
              rng=None, progress_bar=False):
        """Copy a geometric representation (called a glyph) to every point in the input dataset.

        The glyph may be oriented along the input vectors, and it may be scaled according to scalar
        data or vector magnitude. Passing a table of glyphs to choose from based on scalars or
        vector magnitudes is also supported.

        Parameters
        ----------
        orient : bool
            Use the active vectors array to orient the glyphs

        scale : bool
            Use the active scalars to scale the glyphs

        factor : float
            Scale factor applied to scaling array

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
            If None, points merging as a preprocessing step is disabled.

        absolute : bool, optional
            Control if ``tolerance`` is an absolute distance or a fraction.

        clamping: bool
            Turn on/off clamping of "scalar" values to range.

        rng: tuple(float), optional
            Set the range of values to be considered by the filter when scalars
            values are provided.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

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
            arrow = vtk.vtkArrowSource()
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
        if any(not isinstance(subgeom, vtk.vtkPolyData) for subgeom in geom):
            raise TypeError('Only PolyData objects can be used as glyphs.')
        # Run the algorithm
        alg = vtk.vtkGlyph3D()
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
        if rng is not None:
            alg.SetRange(rng)
        alg.SetOrient(orient)
        alg.SetInputData(dataset)
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
        alg = vtk.vtkConnectivityFilter()
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
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Largest connected set in mesh

        """
        mesh = DataSetFilters.connectivity(dataset, largest=True)
        if inplace:
            dataset.overwrite(mesh)
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
            If True, the points of the give dataset will be updated.

        """
        factor = kwargs.pop('scale_factor', factor)
        assert_empty_kwargs(**kwargs)
        if scalars is None:
            field, scalars = dataset.active_scalars_info
        arr, field = get_array(dataset, scalars, preference='point', info=True)
        if field != FieldAssociation.POINT:
            raise TypeError('Dataset can only by warped by a point data array.')
        # Run the algorithm
        alg = vtk.vtkWarpScalar()
        alg.SetInputDataObject(dataset)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars) # args: (idx, port, connection, field, name)
        alg.SetScaleFactor(factor)
        if normal is not None:
            alg.SetNormal(normal)
            alg.SetUseNormal(True)
        alg.Update()
        output = _get_output(alg)
        if inplace:
            if isinstance(dataset, (vtk.vtkImageData, vtk.vtkRectilinearGrid)):
                raise TypeError("This filter cannot be applied inplace for this mesh type.")
            dataset.overwrite(output)
            return
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
            If True, the function will update the mesh in-place and
            return ``None``.

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
        alg = vtk.vtkWarpVector()
        alg.SetInputDataObject(dataset)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, vectors)
        alg.SetScaleFactor(factor)
        alg.Update()
        warped_mesh = _get_output(alg)
        if inplace:
            dataset.overwrite(warped_mesh)
            return
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
        alg = vtk.vtkCellDataToPointData()
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
        alg = vtk.vtkPointDataToCellData()
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
            Updates mesh in-place while returning ``None``.

        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            Mesh containing only triangles. ``None`` when ``inplace=True``

        """
        alg = vtk.vtkDataSetTriangleFilter()
        alg.SetInputData(dataset)
        alg.Update()

        mesh = _get_output(alg)
        if inplace:
            dataset.overwrite(mesh)
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
        alg = vtk.vtkDelaunay3D()
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
            raise RuntimeError("Surface is not closed. Please read the warning in the documentation for this function and either pass `check_surface=False` or repair the surface.")
        alg = vtk.vtkSelectEnclosedPoints()
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
        dataset: pyvista.Common
            The mesh to probe from - point and cell arrays from
            this object are probed onto the nodes of the ``points`` mesh

        points: pyvista.Common
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
        >>> mesh = pyvista.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
        >>> grid = examples.load_uniform()
        >>> result = grid.probe(mesh)
        >>> 'Spatial Point Data' in result.point_arrays
        True

        """
        if not pyvista.is_pyvista_dataset(points):
            points = pyvista.wrap(points)
        alg = vtk.vtkProbeFilter()
        alg.SetInputData(points)
        alg.SetSourceData(dataset)
        alg.SetPassCellArrays(pass_cell_arrays)
        alg.SetPassPointArrays(pass_point_arrays)
        alg.SetCategoricalData(categorical)
        if tolerance is not None:
            alg.SetComputeTolerance(False)
            alg.SetTolerance(tolerance)
        alg.Update() # Perform the resampling
        return _get_output(alg)

    def sample(dataset, target, tolerance=None, pass_cell_arrays=True,
               pass_point_arrays=True, categorical=False):
        """Resample array data from a passed mesh onto this mesh.

        This uses :class:`vtk.vtkResampleWithDataSet`.

        Parameters
        ----------
        dataset: pyvista.Common
            The source vtk data object as the mesh to sample values on to

        target: pyvista.Common
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
        alg = vtk.vtkResampleWithDataSet() # Construct the ResampleWithDataSet object
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
        target: pyvista.Common
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
            the interpolation basis. This will invalidate the radius and
            sharpness arguments in favor of an N closest points approach. This
            typically has poorer results.

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

        gaussian_kernel = vtk.vtkGaussianKernel()
        gaussian_kernel.SetSharpness(sharpness)
        gaussian_kernel.SetRadius(radius)
        gaussian_kernel.SetKernelFootprintToRadius()
        if n_points:
            gaussian_kernel.SetNumberOfPoints(n_points)
            gaussian_kernel.SetKernelFootprintToNClosest()

        locator = vtk.vtkStaticPointLocator()
        locator.SetDataSet(target)
        locator.BuildLocator()

        interpolator = vtk.vtkPointInterpolator()
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
                    integrator_type=45, integration_direction='both',
                    surface_streamlines=False, initial_step_length=0.5,
                    step_unit='cl', min_step_length=0.01, max_step_length=1.0,
                    max_steps=2000, terminal_speed=1e-12, max_error=1e-6,
                    max_time=None, compute_vorticity=True, rotation_scale=1.0,
                    interpolator_type='point', start_position=(0.0, 0.0, 0.0),
                    return_source=False, pointa=None, pointb=None):
        """Integrate a vector field to generate streamlines.

        The integration is performed using a specified integrator, by default
        Runge-Kutta2. This supports integration through any type of dataset.
        Thus if the dataset contains 2D cells like polygons or triangles, the
        integration is constrained to lie on the surface defined by 2D cells.

        This produces polylines as the output, with each cell
        (i.e., polyline) representing a streamline. The attribute values
        associated with each streamline are stored in the cell data, whereas
        those associated with streamline-points are stored in the point data.

        This uses a Sphere as the source - set it's location and radius via
        the ``source_center`` and ``source_radius`` keyword arguments.
        You can retrieve the source as :class:`pyvista.PolyData` by specifying
        ``return_source=True``.

        Parameters
        ----------
        vectors : str
            The string name of the active vector field to integrate across

        source_center : tuple(float)
            Length 3 tuple of floats defining the center of the source
            particles. Defaults to the center of the dataset

        source_radius : float
            Float radius of the source particle cloud. Defaults to one-tenth of
            the diagonal of the dataset's spatial extent

        n_points : int
            Number of particles present in source sphere

        integrator_type : int
            The integrator type to be used for streamline generation.
            The default is Runge-Kutta45. The recognized solvers are:
            RUNGE_KUTTA2 (``2``),  RUNGE_KUTTA4 (``4``), and RUNGE_KUTTA45
            (``45``). Options are ``2``, ``4``, or ``45``. Default is ``45``.

        integration_direction : str
            Specify whether the streamline is integrated in the upstream or
            downstream directions (or both). Options are ``'both'``,
            ``'backward'``, or ``'forward'``.

        surface_streamlines : bool
            Compute streamlines on a surface. Default ``False``

        initial_step_length : float
            Initial step size used for line integration, expressed ib length
            unitsL or cell length units (see ``step_unit`` parameter).
            either the starting size for an adaptive integrator, e.g., RK45, or
            the constant / fixed size for non-adaptive ones, i.e., RK2 and RK4)

        step_unit : str
            Uniform integration step unit. The valid unit is now limited to
            only LENGTH_UNIT (``'l'``) and CELL_LENGTH_UNIT (``'cl'``).
            Default is CELL_LENGTH_UNIT: ``'cl'``.

        min_step_length : float
            Minimum step size used for line integration, expressed in length or
            cell length units. Only valid for an adaptive integrator, e.g., RK45

        max_step_length : float
            Maximum step size used for line integration, expressed in length or
            cell length units. Only valid for an adaptive integrator, e.g., RK45

        max_steps : int
            Maximum number of steps for integrating a streamline.
            Defaults to ``2000``

        terminal_speed : float
            Terminal speed value, below which integration is terminated.

        max_error : float
            Maximum error tolerated throughout streamline integration.

        max_time : float
            Specify the maximum length of a streamline expressed in LENGTH_UNIT.

        compute_vorticity : bool
            Vorticity computation at streamline points (necessary for generating
            proper stream-ribbons using the ``vtkRibbonFilter``.

        interpolator_type : str
            Set the type of the velocity field interpolator to locate cells
            during streamline integration either by points or cells.
            The cell locator is more robust then the point locator. Options
            are ``'point'`` or ``'cell'`` (abbreviations of ``'p'`` and ``'c'``
            are also supported).

        rotation_scale : float
            This can be used to scale the rate with which the streamribbons
            twist. The default is 1.

        start_position : tuple(float)
            Set the start position. Default is ``(0.0, 0.0, 0.0)``

        return_source : bool
            Return the source particles as :class:`pyvista.PolyData` as well as the
            streamlines. This will be the second value returned if ``True``.

        pointa, pointb : tuple(float)
            The coordinates of a start and end point for a line source. This
            will override the sphere point source.

        """
        integration_direction = str(integration_direction).strip().lower()
        if integration_direction not in ['both', 'back', 'backward', 'forward']:
            raise ValueError(f"integration direction must be one of: 'backward', 'forward', or 'both' - not '{integration_direction}'.")
        if integrator_type not in [2, 4, 45]:
            raise ValueError('integrator type must be one of `2`, `4`, or `45`.')
        if interpolator_type not in ['c', 'cell', 'p', 'point']:
            raise ValueError("interpolator type must be either 'cell' or 'point'")
        if step_unit not in ['l', 'cl']:
            raise ValueError("step unit must be either 'l' or 'cl'")
        step_unit = {'cl': vtk.vtkStreamTracer.CELL_LENGTH_UNIT,
                     'l': vtk.vtkStreamTracer.LENGTH_UNIT}[step_unit]
        if isinstance(vectors, str):
            dataset.set_active_scalars(vectors)
            dataset.set_active_vectors(vectors)
        if max_time is None:
            max_velocity = dataset.get_data_range()[-1]
            max_time = 4.0 * dataset.GetLength() / max_velocity
        # Generate the source
        if source_center is None:
            source_center = dataset.center
        if source_radius is None:
            source_radius = dataset.length / 10.0
        if pointa is not None and pointb is not None:
            source = vtk.vtkLineSource()
            source.SetPoint1(pointa)
            source.SetPoint2(pointb)
            source.SetResolution(n_points)
        else:
            source = vtk.vtkPointSource()
            source.SetCenter(source_center)
            source.SetRadius(source_radius)
            source.SetNumberOfPoints(n_points)
        # Build the algorithm
        alg = vtk.vtkStreamTracer()
        # Inputs
        alg.SetInputDataObject(dataset)
        # NOTE: not sure why we can't pass a PolyData object
        #       setting the connection is the only I could get it to work
        alg.SetSourceConnection(source.GetOutputPort())
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
        alg.SetStartPosition(start_position)
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
        output = _get_output(alg)
        if return_source:
            source.Update()
            src = pyvista.wrap(source.GetOutput())
            return output, src
        return output

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
                       show=True, tolerance=None):
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
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(numpy_to_idarr(ind))

        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = vtk.vtkExtractSelection()
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
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        if not include_cells:
            adjacent_cells = True        
        if not adjacent_cells:
            # Build array of point indices to be removed.
            ind_rem = np.ones(dataset.n_points, dtype='bool')
            ind_rem[ind] = False
            ind = np.arange(dataset.n_points)[ind_rem]
            # Invert selection
            selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(), 1)
        selectionNode.SetSelectionList(numpy_to_idarr(ind))
        if include_cells:
            selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)
        
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, dataset)
        extract_sel.SetInputData(1, selection)
        extract_sel.Update()
        return _get_output(extract_sel)

    def extract_surface(dataset, pass_pointid=True, pass_cellid=True, inplace=False):
        """Extract surface mesh of the grid.

        Parameters
        ----------
        pass_pointid : bool, optional
            Adds a point array "vtkOriginalPointIds" that idenfities which
            original points these surface points correspond to

        pass_cellid : bool, optional
            Adds a cell array "vtkOriginalPointIds" that idenfities which
            original cells these surface cells correspond to

        Returns
        -------
        extsurf : pyvista.PolyData
            Surface mesh of the grid

        """
        surf_filter = vtk.vtkDataSetSurfaceFilter()
        surf_filter.SetInputData(dataset)
        if pass_pointid:
            surf_filter.PassThroughCellIdsOn()
        if pass_cellid:
            surf_filter.PassThroughPointIdsOn()
        surf_filter.Update()

        # need to add
        # surf_filter.SetNonlinearSubdivisionLevel(subdivision)

        mesh = _get_output(surf_filter)
        return mesh

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
            Return new mesh or overwrite input.

        Returns
        -------
        edges : pyvista.vtkPolyData
            Extracted edges. None if inplace=True.

        """
        if not isinstance(dataset, vtk.vtkPolyData):
            dataset = DataSetFilters.extract_surface(dataset)
        featureEdges = vtk.vtkFeatureEdges()
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
            Merged grid.  Returned when inplace is False.

        Notes
        -----
        When two or more grids are joined, the type and name of each
        array must match or the arrays will be ignored and not
        included in the final merged mesh.

        """
        append_filter = vtk.vtkAppendFilter()
        append_filter.SetMergePoints(merge_points)

        if not main_has_priority:
            append_filter.AddInputData(dataset)

        if isinstance(grid, pyvista.Common):
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
        alg = vtk.vtkCellQuality()
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
        alg = vtk.vtkGradientFilter()
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
        >>> shrunk_mesh = mesh.shrink(shrink_factor=0.8)  # doctest:+SKIP
        """
        if not (0.0 <= shrink_factor <= 1.0):
            raise ValueError('`shrink_factor` should be between 0.0 and 1.0')
        alg = vtk.vtkShrinkFilter()
        alg.SetInputData(dataset)
        alg.SetShrinkFactor(shrink_factor)
        _update_alg(alg, progress_bar, 'Shrinking Mesh')
        output = pyvista.wrap(alg.GetOutput())
        if isinstance(dataset, vtk.vtkPolyData):
            return output.extract_surface()

    def reflect(dataset, normal, point=None, inplace=False):
        """Reflect a dataset across a plane.

        Parameters
        ----------
        normal : tuple(float)
            Normal direction for reflection.

        point : tuple(float), optional
            Point which, along with `normal`, defines the reflection plane. If not
            specified, this is the origin.

        inplace : bool, optional
            When ``True``, modifies the dataset and returns nothing.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh = mesh.reflect((0, 0, 1), point=(0, 0, -100))
        >>> mesh.plot(show_edges=True)  # doctest:+SKIP

        """
        t = transformations.reflection(normal, point=point)
        if inplace:
            dataset.transform(t)
        else:
            mirror = dataset.copy()
            mirror.transform(t)
            return mirror


@abstract_class
class CompositeFilters:
    """An internal class to manage filters/algorithms for composite datasets."""

    def extract_geometry(composite):
        """Combine the geomertry of all blocks into a single ``PolyData`` object.

        Place this filter at the end of a pipeline before a polydata
        consumer such as a polydata mapper to extract geometry from all blocks
        and append them to one polydata object.

        """
        gf = vtk.vtkCompositeDataGeometryFilter()
        gf.SetInputData(composite)
        gf.Update()
        return wrap(gf.GetOutputDataObject(0))

    def combine(composite, merge_points=False):
        """Append all blocks into a single unstructured grid.

        Parameters
        ----------
        merge_points : bool, optional
            Merge coincidental points.

        """
        alg = vtk.vtkAppendFilter()
        for block in composite:
            if isinstance(block, vtk.vtkMultiBlockDataSet):
                block = CompositeFilters.combine(block, merge_points=merge_points)
            alg.AddInputData(block)
        alg.SetMergePoints(merge_points)
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

    def outline(composite, generate_faces=False, nested=False):
        """Produce an outline of the full extent for the all blocks in this composite dataset.

        Parameters
        ----------
        generate_faces : bool, optional
            Generate solid faces for the box. This is off by default

        nested : bool, optional
            If True, these creates individual outlines for each nested dataset

        """
        if nested:
            return DataSetFilters.outline(composite, generate_faces=generate_faces)
        box = pyvista.Box(bounds=composite.bounds)
        return box.outline(generate_faces=generate_faces)

    def outline_corners(composite, factor=0.2, nested=False):
        """Produce an outline of the corners for the all blocks in this composite dataset.

        Parameters
        ----------
        factor : float, optional
            controls the relative size of the corners to the length of the
            corresponding bounds

        nested : bool, optional
            If True, these creates individual outlines for each nested dataset

        """
        if nested:
            return DataSetFilters.outline_corners(composite, factor=factor)
        box = pyvista.Box(bounds=composite.bounds)
        return box.outline_corners(factor=factor)


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
        featureEdges = vtk.vtkFeatureEdges()
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
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            The cut mesh when inplace=False

        """
        if not isinstance(cut, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")
        if not poly_data.is_all_triangles() or not cut.is_all_triangles():
            raise NotAllTrianglesError("Make sure both the input and output are triangulated.")

        bfilter = vtk.vtkBooleanOperationPolyDataFilter()
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
            Updates mesh in-place while returning nothing.

        Returns
        -------
        joinedmesh : pyvista.PolyData
            Initial mesh and the new mesh when inplace=False.

        """
        if not isinstance(mesh, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")

        vtkappend = vtk.vtkAppendPolyData()
        vtkappend.AddInputData(poly_data)
        vtkappend.AddInputData(mesh)
        vtkappend.Update()

        mesh = _get_output(vtkappend)
        if inplace:
            poly_data.overwrite(mesh)
        else:
            return mesh

    def __add__(poly_data, mesh):
        """Merge these two meshes."""
        if not isinstance(mesh, vtk.vtkPolyData):
            return DataSetFilters.__add__(poly_data, mesh)
        return PolyDataFilters.boolean_add(poly_data, mesh)

    def boolean_union(poly_data, mesh, inplace=False):
        """Combine two meshes and attempts to create a manifold mesh.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to perform a union against.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        union : pyvista.PolyData
            The union mesh when inplace=False.

        """
        if not isinstance(mesh, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")

        bfilter = vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToUnion()
        bfilter.SetInputData(1, mesh)
        bfilter.SetInputData(0, poly_data)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.Update()

        mesh = _get_output(bfilter)
        if inplace:
            poly_data.overwrite(mesh)
        else:
            return mesh

    def boolean_difference(poly_data, mesh, inplace=False):
        """Combine two meshes and retains only the volume in common between the meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to perform a union against.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        union : pyvista.PolyData
            The union mesh when inplace=False.

        """
        if not isinstance(mesh, pyvista.PolyData):
            raise TypeError("Input mesh must be PolyData.")

        bfilter = vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToDifference()
        bfilter.SetInputData(1, mesh)
        bfilter.SetInputData(0, poly_data)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.Update()

        mesh = _get_output(bfilter)
        if inplace:
            poly_data.overwrite(mesh)
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
        intfilter = vtk.vtkIntersectionPolyDataFilter()
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
        curvefilter = vtk.vtkCurvatures()
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
        return vtk_to_numpy(curv.GetPointData().GetScalars())

    def plot_curvature(poly_data, curv_type='mean', **kwargs):
        """Plot the curvature.

        Parameters
        ----------
        curvtype : str, optional
            One of the following strings indicating curvature type

            - Mean
            - Gaussian
            - Maximum
            - Minimum

        **kwargs : optional
            See :func:`pyvista.plot`

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up

        """
        return poly_data.plot(scalars=poly_data.curvature(curv_type),
                              stitle=f'{curv_type}\nCurvature', **kwargs)

    def triangulate(poly_data, inplace=False):
        """Return an all triangle mesh.

        More complex polygons will be broken down into tetrahedrals.

        Parameters
        ----------
        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh containing only triangles.  None when inplace=True

        """
        trifilter = vtk.vtkTriangleFilter()
        trifilter.SetInputData(poly_data)
        trifilter.PassVertsOff()
        trifilter.PassLinesOff()
        trifilter.Update()

        mesh = _get_output(trifilter)
        if inplace:
            poly_data.overwrite(mesh)
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
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Smoothed mesh. None when inplace=True.

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
        alg = vtk.vtkSmoothPolyDataFilter()
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
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Decimated mesh. None when inplace=True.

        """
        alg = vtk.vtkDecimatePro()
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
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Tube-filtered mesh. None when inplace=True.

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
        tube = vtk.vtkTubeFilter()
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
        else:
            return mesh

    def subdivide(poly_data, nsub, subfilter='linear', inplace=False):
        """Increase the number of triangles in a single, connected triangular mesh.

        Uses one of the following vtk subdivision filters to subdivide a mesh.
        vtkButterflySubdivisionFilter
        vtkLoopSubdivisionFilter
        vtkLinearSubdivisionFilter

        Linear subdivision results in the fastest mesh subdivision, but it
        does not smooth mesh edges, but rather splits each triangle into 4
        smaller triangles.

        Butterfly and loop subdivision perform smoothing when dividing, and may
        introduce artifacts into the mesh when dividing.

        Subdivision filter appears to fail for multiple part meshes.  Should
        be one single mesh.

        Parameters
        ----------
        nsub : int
            Number of subdivisions.  Each subdivision creates 4 new triangles,
            so the number of resulting triangles is nface*4**nsub where nface
            is the current number of faces.

        subfilter : string, optional
            Can be one of the following: 'butterfly', 'loop', 'linear'

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : Polydata object
            pyvista polydata object.  None when inplace=True

        Examples
        --------
        >>> from pyvista import examples
        >>> import pyvista
        >>> mesh = pyvista.PolyData(examples.planefile)
        >>> submesh = mesh.subdivide(1, 'loop') # doctest:+SKIP

        Alternatively, update the mesh in-place

        >>> mesh.subdivide(1, 'loop', inplace=True) # doctest:+SKIP

        """
        subfilter = subfilter.lower()
        if subfilter == 'linear':
            sfilter = vtk.vtkLinearSubdivisionFilter()
        elif subfilter == 'butterfly':
            sfilter = vtk.vtkButterflySubdivisionFilter()
        elif subfilter == 'loop':
            sfilter = vtk.vtkLoopSubdivisionFilter()
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
            Updates mesh in-place while returning nothing.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        outmesh : pyvista.PolyData
            Decimated mesh.  None when inplace=True.

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
        alg = vtk.vtkQuadricDecimation()  # vtkDecimatePro as well

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
        else:
            return mesh

    def compute_normals(poly_data, cell_normals=True, point_normals=True,
                        split_vertices=False, flip_normals=False,
                        consistent_normals=True,
                        auto_orient_normals=False,
                        non_manifold_traversal=True,
                        feature_angle=30.0, inplace=False):
        """Compute point and/or cell normals for a mesh.

        The filter can reorder polygons to insure consistent orientation across
        polygon neighbors. Sharp edges can be split and points duplicated
        with separate normals to give crisp (rendered) surface definition. It is
        also possible to globally flip the normal orientation.

        The algorithm works by determining normals for each polygon and then
        averaging them at shared points. When sharp edges are present, the edges
        are split and new points generated to prevent blurry edges (due to
        Gouraud shading).

        Parameters
        ----------
        cell_normals : bool, optional
            Calculation of cell normals. Defaults to True.

        point_normals : bool, optional
            Calculation of point normals. Defaults to True.

        split_vertices : bool, optional
            Splitting of sharp edges. Defaults to False.

        flip_normals : bool, optional
            Set global flipping of normal orientation. Flipping modifies both
            the normal direction and the order of a cell's points. Defaults to
            False.

        consistent_normals : bool, optional
            Enforcement of consistent polygon ordering. Defaults to True.

        auto_orient_normals : bool, optional
            Turn on/off the automatic determination of correct normal
            orientation. NOTE: This assumes a completely closed surface (i.e. no
            boundary edges) and no non-manifold edges. If these constraints do
            not hold, all bets are off. This option adds some computational
            complexity, and is useful if you don't want to have to inspect the
            rendered image to determine whether to turn on the FlipNormals flag.
            However, this flag can work with the FlipNormals flag, and if both
            are set, all the normals in the output will point "inward". Defaults
            to False.

        non_manifold_traversal : bool, optional
            Turn on/off traversal across non-manifold edges. Changing this may
            prevent problems where the consistency of polygonal ordering is
            corrupted due to topological loops. Defaults to True.

        feature_angle : float, optional
            The angle that defines a sharp edge. If the difference in angle
            across neighboring polygons is greater than this value, the shared
            edge is considered "sharp". Defaults to 30.0.

        inplace : bool, optional
            Updates mesh in-place while returning nothing. Defaults to False.

        Returns
        -------
        mesh : pyvista.PolyData
            Updated mesh with cell and point normals if inplace=False

        Examples
        --------
        Compute the point normals of the surface of a sphere

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.compute_normals(cell_normals=False, inplace=True)
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

        Normals are computed only for polygons and triangle strips. Normals are
        not computed for lines or vertices.

        Triangle strips are broken up into triangle polygons. You may want to
        restrip the triangles.

        May be easier to run mesh.point_normals or mesh.cell_normals

        """
        normal = vtk.vtkPolyDataNormals()
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
        else:
            return mesh

    def clip_closed_surface(poly_data, normal='x', origin=None,
                            tolerance=1e-06, inplace=False):
        """Clip a closed polydata surface with a plane.

        This currently only supports one plane but could be implemented to
        handle a plane collection.

        It will produce a new closed surface by creating new polygonal faces
        where the input data was clipped.

        Non-manifold surfaces should not be used as input for this filter.
        The input surface should have no open edges, and must not have any
        edges that are shared by more than two faces. In addition, the input
        surface should not self-intersect, meaning that the faces of the
        surface should only touch at their edges.

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
            to ``[0, 0, 0]```

        tolerance : float, optional
            The tolerance for creating new points while clipping.  If
            the tolerance is too small, then degenerate triangles
            might be produced.

        inplace : bool, optional
            Updates mesh in-place while returning nothing. Defaults to False.

        Returns
        -------
        clipped_mesh : pyvista.PolyData
            The clipped mesh resulting from this operation when
            ``inplace==False``.  Otherwise, ``None``.

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
        collection = vtk.vtkPlaneCollection()
        collection.AddItem(plane)

        alg = vtk.vtkClipClosedSurface()
        alg.SetGenerateFaces(True)
        alg.SetInputDataObject(poly_data)
        alg.SetTolerance(tolerance)
        alg.SetClippingPlanes(collection)
        alg.Update() # Perform the Cut
        result = _get_output(alg)

        if inplace:
            poly_data.overwrite(result)
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
            Mesh with holes filled.  None when inplace=True

        Examples
        --------
        Create a partial sphere with a hole and then fill it

        >>> import pyvista as pv
        >>> sphere_with_hole = pv.Sphere(end_theta=330)
        >>> sphere_with_hole.fill_holes(1000, inplace=True)
        >>> edges = sphere_with_hole.extract_feature_edges(feature_edges=False, manifold_edges=False)
        >>> assert edges.n_cells == 0

        """
        logging.warning('pyvista.PolyData.fill_holes is known to segfault. '
                        'Use at your own risk')
        alg = vtk.vtkFillHolesFilter()
        alg.SetHoleSize(hole_size)
        alg.SetInputData(poly_data)
        _update_alg(alg, progress_bar, 'Filling Holes')

        mesh = _get_output(alg)
        if inplace:
            poly_data.overwrite(mesh)
        else:
            return mesh

    def clean(poly_data, point_merging=True, tolerance=None, lines_to_points=True,
              polys_to_lines=True, strips_to_polys=True, inplace=False,
              absolute=True, progress_bar=False, **kwargs):
        """Clean the mesh.

        This merges duplicate points, removes unused points, and/or removes
        degenerate cells.

        Parameters
        ----------
        point_merging : bool, optional
            Enables point merging.  On by default.

        tolerance : float, optional
            Set merging tolerance.  When enabled merging is set to
            absolute distance. If ``absolute`` is False, then the merging
            tolerance is a fraction of the bounding box length. The alias
            ``merge_tol`` is also excepted.

        lines_to_points : bool, optional
            Turn on/off conversion of degenerate lines to points.  Enabled by
            default.

        polys_to_lines : bool, optional
            Turn on/off conversion of degenerate polys to lines.  Enabled by
            default.

        strips_to_polys : bool, optional
            Turn on/off conversion of degenerate strips to polys.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.  Default True.

        absolute : bool, optional
            Control if ``tolerance`` is an absolute distance or a fraction.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        mesh : pyvista.PolyData
            Cleaned mesh.  None when inplace=True

        Examples
        --------
        Create a mesh with a degenerate face and then clean it,
        removing the degenerate face

        >>> import pyvista as pv
        >>> points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        >>> faces = np.array([3, 0, 1, 2, 3, 0, 3, 3])
        >>> mesh = pv.PolyData(points, faces)
        >>> mout = mesh.clean()
        >>> print(mout.faces)
        [3 0 1 2]

        """
        if tolerance is None:
            tolerance = kwargs.pop('merge_tol', None)
        assert_empty_kwargs(**kwargs)
        alg = vtk.vtkCleanPolyData()
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
        else:
            return output

    def geodesic(poly_data, start_vertex, end_vertex, inplace=False):
        """Calculate the geodesic path between two vertices using Dijkstra's algorithm.

        This will add an array titled `vtkOriginalPointIds` of the input
        mesh's point ids to the output mesh.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        Returns
        -------
        output : pyvista.PolyData
            PolyData object consisting of the line segment between the
            two given vertices.

        Examples
        --------
        Plot the path between two points on a sphere

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> path = sphere.geodesic(0, 100)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere)
        >>> _ = pl.add_mesh(path, line_width=5, color='k')
        >>> pl.show()  # doctest:+SKIP

        """
        if start_vertex < 0 or end_vertex > poly_data.n_points - 1:
            raise IndexError('Invalid indices.')
        if not poly_data.is_all_triangles():
            raise NotAllTrianglesError("Input mesh for geodesic path must be all triangles.")

        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(poly_data)
        dijkstra.SetStartVertex(start_vertex)
        dijkstra.SetEndVertex(end_vertex)
        dijkstra.Update()
        original_ids = vtk_id_list_to_array(dijkstra.GetIdList())

        output = _get_output(dijkstra)
        output["vtkOriginalPointIds"] = original_ids

        # Do not copy textures from input
        output.clear_textures()

        if inplace:
            poly_data.overwrite(output)
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
        points = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()
        poly_data.obbTree.IntersectWithLine(np.array(origin),
                                            np.array(end_point),
                                            points, cell_ids)

        intersection_points = vtk_to_numpy(points.GetData())
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
        Compute the intersection between rays from the origin in directions
        [1, 0, 0], [0, 1, 0] and [0, 0, 1], and a sphere with radius 0.5 centered at the origin

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

        faces_as_array = poly_data.faces.reshape((poly_data.number_of_faces, 4))[:, 1:]
        tmesh = trimesh.Trimesh(poly_data.points, faces_as_array)
        locations, index_ray, index_tri = tmesh.ray.intersects_location(
            origins, directions, multiple_hits=not first_point
        )
        if retry:
            ray_tuples = [(id_r, l, id_t) for id_r, l, id_t in zip(index_ray, locations, index_tri)]
            for id_r in range(len(origins)):
                if id_r not in index_ray:
                    origin = np.array(origins[id_r])
                    vector = np.array(directions[id_r])
                    unit_vector = vector / np.sqrt(np.sum(np.power(vector, 2)))
                    second_point = origin + (unit_vector * poly_data.length)
                    locs, indexes = poly_data.ray_trace(origin, second_point, first_point=first_point)
                    if locs.any():
                        if first_point:
                            locs = locs.reshape([1, 3])
                        for loc, id_t in zip(locs, indexes):
                            ray_tuples.append((id_r, loc, id_t))
            sorted_results = sorted(ray_tuples, key=lambda x: x[0])
            locations = np.array([loc for id_r, loc, id_t in sorted_results])
            index_ray = np.array([id_r for id_r, loc, id_t in sorted_results])
            index_tri = np.array([id_t for id_r, loc, id_t in sorted_results])
        return locations, index_ray, index_tri

    def plot_boundaries(poly_data, edge_color="red", **kwargs):
        """Plot boundaries of a mesh.

        Parameters
        ----------
        edge_color : str, etc.
            The color of the edges when they are added to the plotter.

        kwargs : optional
            All additional keyword arguments will be passed to
            :func:`pyvista.BasePlotter.add_mesh`

        """
        edges = DataSetFilters.extract_feature_edges(poly_data)

        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', False),
                                  notebook=kwargs.pop('notebook', None))
        plotter.add_mesh(edges, color=edge_color, style='wireframe', label='Edges')
        plotter.add_mesh(poly_data, label='Mesh', **kwargs)
        plotter.add_legend()
        return plotter.show()

    def plot_normals(poly_data, show_mesh=True, mag=1.0, flip=False,
                     use_every=1, **kwargs):
        """Plot the point normals of a mesh."""
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', False),
                                  notebook=kwargs.pop('notebook', None))
        if show_mesh:
            plotter.add_mesh(poly_data, **kwargs)

        normals = poly_data.point_normals
        if flip:
            normals *= -1
        plotter.add_arrows(poly_data.points[::use_every],
                           normals[::use_every], mag=mag)
        return plotter.show()

    def remove_points(poly_data, remove, mode='any', keep_scalars=True, inplace=False):
        """Rebuild a mesh by removing points.

        Only valid for all-triangle meshes.

        Parameters
        ----------
        remove : np.ndarray
            If remove is a bool array, points that are True will be
            removed.  Otherwise, it is treated as a list of indices.

        mode : str, optional
            When 'all', only faces containing all points flagged for
            removal will be removed.  Default 'all'

        keep_scalars : bool, optional
            When True, point and cell scalars will be passed on to the
            new mesh.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh without the points flagged for removal.  Not returned
            when inplace=False.

        ridx : np.ndarray
            Indices of new points relative to the original mesh.  Not
            returned when inplace=False.

        Examples
        --------
        Remove the first 100 points from a sphere

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> reduced_sphere = sphere.remove_points(range(100))

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
        >>> sphere.plot_normals(mag=0.1)  # doctest:+SKIP
        >>> sphere.flip_normals()
        >>> sphere.plot_normals(mag=0.1)  # doctest:+SKIP

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
        tol : float
            Specify a tolerance to control discarding of closely spaced
            points. This tolerance is specified as a fraction of the diagonal
            length of the bounding box of the points.

        alpha : float
            Specify alpha (or distance) value to control output of this
            filter. For a non-zero alpha value, only edges or triangles
            contained within a sphere centered at mesh vertices will be
            output. Otherwise, only triangles will be output.

        offset : float
            Specify a multiplier to control the size of the initial, bounding
            Delaunay triangulation.

        bound : bool
            Boolean controls whether bounding triangulation points (and
            associated triangles) are included in the output. (These are
            introduced as an initial triangulation to begin the triangulation
            process. This feature is nice for debugging output.)

        inplace : bool
            If True, overwrite this mesh with the triangulated mesh.

        edge_source : pyvista.PolyData, optional
            Specify the source object used to specify constrained edges and
            loops. (This is optional.) If set, and lines/polygons are
            defined, a constrained triangulation is created. The
            lines/polygons are assumed to reference points in the input point
            set (i.e. point ids are identical in the input and source). Note
            that this method does not connect the pipeline. See
            SetSourceConnection for connecting the pipeline.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

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
        alg = vtk.vtkDelaunay2D()
        alg.SetProjectionPlaneMode(vtk.VTK_BEST_FITTING_PLANE)
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
        else:
            return mesh

    def compute_arc_length(poly_data):
        """Compute the arc length over the length of the probed line.

        It adds a new point-data array named "arc_length" with the
        computed arc length for each of the polylines in the
        input. For all other cell types, the arc length is set to 0.

        Returns
        -------
        arc_length : float
            Arc length of the length of the probed line

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> path = sphere.geodesic(0, 100)
        >>> length = path.compute_arc_length()['arc_length'][-1]
        >>> print(f'Length is {length:.3f}')
        Length is 0.812

        This is identical to the geodesic_distance

        >>> length = sphere.geodesic_distance(0, 100)
        >>> print(f'Length is {length:.3f}')
        Length is 0.812

        You can also plot the arc_length

        >>> arc = path.compute_arc_length()
        >>> arc.plot(scalars="arc_length")  # doctest:+SKIP

        """
        alg = vtk.vtkAppendArcLength()
        alg.SetInputData(poly_data)
        alg.Update()
        return _get_output(alg)


    def project_points_to_plane(poly_data, origin=None, normal=(0,0,1), inplace=False):
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
        if not inplace:
            return mesh
        return

    def ribbon(poly_data, width=None, scalars=None, angle=0.0, factor=2.0,
               normal=None, tcoords=False, preference='points'):
        """Create a ribbon of the lines in this dataset.

        Note
        ----
        If there are no lines in the input dataset, then the output will be
        an empty PolyData mesh.

        Parameters
        ----------
        width : float
            Set the "half" width of the ribbon. If the width is allowed to
            vary, this is the minimum width. The default is 10% the length

        scalars : str, optional
            String name of the scalars array to use to vary the ribbon width.
            This is only used if a scalars array is specified.

        angle : float
            Set the offset angle of the ribbon from the line normal. (The
            angle is expressed in degrees.) The default is 0.0

        factor : float
            Set the maximum ribbon width in terms of a multiple of the
            minimum width. The default is 2.0

        normal : tuple(float), optional
            Normal to use as default

        tcoords : bool, str, optional
            If True, generate texture coordinates along the ribbon. This can
            also be specified to generate the texture coordinates in the
            following ways: ``'length'``, ``'normalized'``,

        Examples
        --------
        Convert a line to a ribbon and plot it.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> path = sphere.geodesic(0, 100)
        >>> ribbon = path.ribbon()
        >>> pv.plot([sphere, ribbon])  # doctest:+SKIP

        """
        if scalars is not None:
            arr, field = get_array(poly_data, scalars, preference=preference, info=True)
        if width is None:
            width = poly_data.length * 0.1
        alg = vtk.vtkRibbonFilter()
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
            Overwrites the original mesh inplace.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Examples
        --------
        Extrude a half arc circle

        >>> import pyvista
        >>> arc = pyvista.CircularArc([-1, 0, 0], [1, 0, 0], [0, 0, 0])
        >>> mesh = arc.extrude([0, 0, 1])
        >>> mesh.plot()  # doctest:+SKIP
        """
        alg = vtk.vtkLinearExtrusionFilter()
        alg.SetExtrusionTypeToVectorExtrusion()
        alg.SetVector(*vector)
        alg.SetInputData(poly_data)
        _update_alg(alg, progress_bar, 'Extruding')
        output = pyvista.wrap(alg.GetOutput())
        if not inplace:
            return output
        poly_data.overwrite(output)

    def extrude_rotate(poly_data, resolution=30, inplace=False, progress_bar=False):
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
        resolution : int
            Number of pieces to divide line into.

        inplace : bool, optional
            Overwrites the original mesh inplace.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Examples
        --------
        >>> import pyvista
        >>> line = pyvista.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))
        >>> mesh = line.extrude_rotate(resolution = 4)
        >>> mesh.show() # doctest:+SKIP
        """
        if resolution <= 0:
            raise ValueError('`resolution` should be positive')
        alg = vtk.vtkRotationalExtrusionFilter()
        alg.SetInputData(poly_data)
        alg.SetResolution(resolution)
        _update_alg(alg, progress_bar, 'Extruding')
        output = pyvista.wrap(alg.GetOutput())
        if not inplace:
            return output
        poly_data.overwrite(output)

    def strip(poly_data, join=False, max_length=1000, pass_cell_data=False,
              pass_cell_ids=False, pass_point_ids=False):
        """Strip poly data cells.

        Generates triangle strips and/or poly-lines from input polygons,
        triangle strips, and lines.

        Polygons are assembled into triangle strips only if they are
        triangles; other types of polygons are passed through to the output
        and not stripped. (Use  ``triangulate`` filter to triangulate
        non-triangular polygons prior to running this filter if you need to
        strip all the data.) The filter will pass through (to the output)
        vertices if they are present in the input polydata. Also note that if
        triangle strips or polylines are defined in the input they are passed
        through and not joined nor extended. (If you wish to strip these use
        ``triangulate`` filter to fragment the input into triangles and lines
        prior to running this filter.)

        Parameters
        ----------
        join : bool
            If on, the output polygonal segments will be joined if they are
            contiguous. This is useful after slicing a surface. The default
            is off.

        max_length : int
            Specify the maximum number of triangles in a triangle strip,
            and/or the maximum number of lines in a poly-line.

        pass_cell_data : bool
            Enable/Disable passing of the CellData in the input to the output
            as FieldData. Note the field data is transformed.

        pass_cell_ids : bool
            If on, the output polygonal dataset will have a celldata array
            that holds the cell index of the original 3D cell that produced
            each output cell. This is useful for picking. The default is off
            to conserve memory.

        pass_point_ids : bool
            If on, the output polygonal dataset will have a pointdata array
            that holds the point index of the original vertex that produced
            each output vertex. This is useful for picking. The default is
            off to conserve memory.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> slc = mesh.slice(normal='z', origin=(0,0,-10))
        >>> stripped = slc.strip()
        >>> stripped.n_cells
        1
        """
        alg = vtk.vtkStripper()
        alg.SetInputDataObject(poly_data)
        alg.SetJoinContiguousSegments(join)
        alg.SetMaximumLength(max_length)
        alg.SetPassCellDataAsFieldData(pass_cell_data)
        alg.SetPassThroughCellIds(pass_cell_ids)
        alg.SetPassThroughPointIds(pass_point_ids)
        alg.Update()
        return _get_output(alg)

@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

    def delaunay_2d(ugrid, tol=1e-05, alpha=0.0, offset=1.0, bound=False,
                    progress_bar=False):
        """Apply a delaunay 2D filter along the best fitting plane.

        This extracts the grid's points and performs the triangulation on those alone.

        Parameters
        ----------
        progress_bar : bool, optional
            Display a progress bar to indicate progress.
        """
        return pyvista.PolyData(ugrid.points).delaunay_2d(tol=tol, alpha=alpha,
                                                          offset=offset,
                                                          bound=bound,
                                                          progress_bar=progress_bar)


@abstract_class
class StructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for structured grid datasets."""

    def extract_subset(dataset, voi, rate=(1, 1, 1), boundary=False):
        """Select piece (e.g., volume of interest).

        To use this filter set the VOI ivar which are i-j-k min/max indices
        that specify a rectangular region in the data. (Note that these are
        0-offset.) You can also specify a sampling rate to subsample the
        data.

        Typical applications of this filter are to extract a slice from a
        volume for image processing, subsampling large volumes to reduce data
        size, or extracting regions of a volume with interesting data.

        Parameters
        ----------
        voi : tuple(int)
            Length 6 iterable of ints: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
            These bounds specify the volume of interest in i-j-k min/max
            indices.

        rate : tuple(int)
            Length 3 iterable of ints: ``(xrate, yrate, zrate)``.
            Default: ``(1, 1, 1)``

        boundary : bool
            Control whether to enforce that the "boundary" of the grid is
            output in the subsampling process. (This only has effect
            when the rate in any direction is not equal to 1). When
            this is on, the subsampling will always include the boundary of
            the grid even though the sample rate is not an even multiple of
            the grid dimensions.  By default this is ``False``.

        Examples
        --------
        Split a grid in half.

        >>> import numpy as np
        >>> import pyvista
        >>> from pyvista import examples
        >>> grid = examples.load_structured()
        >>> voi_1 = grid.extract_subset([0, 80, 0, 40, 0, 1], boundary=True)
        >>> voi_2 = grid.extract_subset([0, 80, 40, 80, 0, 1], boundary=True)

        For fun, add the two grids back together and show they are
        identical to the original grid.

        >>> joined = voi_1.concatenate(voi_2, axis=1)
        >>> assert np.allclose(grid.points, joined.points)
        """
        alg = vtk.vtkExtractGrid()
        alg.SetVOI(voi)
        alg.SetInputDataObject(dataset)
        alg.SetSampleRate(rate)
        alg.SetIncludeBoundary(boundary)
        alg.Update()
        return _get_output(alg)

    def concatenate(dataset, other, axis, tolerance=0.0):
        """Concatenate a structured grids to this grid.

        Joins structured grids into a single structured grid.
        Grids must be of compatible dimension, and must be coincident
        along the seam. Grids must have the same point and cell data.
        Field data is ignored.

        Parameters
        ----------
        other : pyvista.StructuredGrid
            Structured grid to concatenate.

        axis : int
            Axis along which to concatenate.

        tolerance : float
            Tolerance for point coincidence along joining seam.

        Returns
        --------
        pyvista.StructuredGrid
            Concatenated grid.

        Examples
        --------
        Split a grid in half and join them.

        >>> import numpy as np
        >>> import pyvista
        >>> from pyvista import examples
        >>> grid = examples.load_structured()
        >>> voi_1 = grid.extract_subset([0, 80, 0, 40, 0, 1], boundary=True)
        >>> voi_2 = grid.extract_subset([0, 80, 40, 80, 0, 1], boundary=True)
        >>> joined = voi_1.concatenate(voi_2, axis=1)
        >>> print(grid.dimensions, 'same as', joined.dimensions)
        [80, 80, 1] same as [80, 80, 1]
        """
        if axis > 2:
            raise RuntimeError('Concatenation axis must be <= 2.')

        # check dimensions are compatible
        for i, (dim1, dim2) in enumerate(zip(dataset.dimensions,
                                             other.dimensions)):
            if i == axis:
                continue
            if dim1 != dim2:
                raise RuntimeError('StructuredGrids with dimensions %s and %s '
                                   'are not compatible.'
                                   % (dataset.dimensions, other.dimensions))

        # check point/cell variables are the same
        if not set(dataset.point_arrays.keys()) == \
               set(other.point_arrays.keys()):
            raise RuntimeError('Grid to concatenate has different point array names.')
        if not set(dataset.cell_arrays.keys()) == \
               set(other.cell_arrays.keys()):
            raise RuntimeError('Grid to concatenate has different cell array names.')

        # check that points are coincident (within tolerance) along seam
        if not np.allclose(np.take(dataset.points_matrix, indices=-1, axis=axis),
                           np.take(other.points_matrix, indices=0, axis=axis),
                           atol=tolerance):
            raise RuntimeError('Grids cannot be joined along axis %d, as points '
                               'are not coincident within tolerance of %f.'
                               % (axis, tolerance))

        # slice to cut off the repeated grid face
        slice_spec = [slice(None, None, None)] * 3
        slice_spec[axis] = slice(0, -1, None)

        # concatenate points, cutting off duplicate
        new_points = np.concatenate((dataset.points_matrix[slice_spec],
                                     other.points_matrix), axis=axis)

        # concatenate point arrays, cutting off duplicate
        new_point_data = {}
        for name, point_array in dataset.point_arrays.items():
            arr_1 = dataset._reshape_point_array(point_array)
            arr_2 = other._reshape_point_array(other.point_arrays[name])
            if not np.array_equal(np.take(arr_1, indices=-1, axis=axis),
                                  np.take(arr_2, indices=0, axis=axis)):
                raise RuntimeError('Grids cannot be joined along axis %d, as field '
                                   '`%s` is not identical along the seam.'
                                   % (axis, name))
            new_point_data[name] = np.concatenate((arr_1[slice_spec], arr_2),
                                                  axis=axis).ravel(order='F')

        new_dims = np.array(dataset.dimensions)
        new_dims[axis] += other.dimensions[axis] - 1

        # concatenate cell arrays
        new_cell_data = {}
        for name, cell_array in dataset.cell_arrays.items():
            arr_1 = dataset._reshape_cell_array(cell_array)
            arr_2 = other._reshape_cell_array(other.cell_arrays[name])
            new_cell_data[name] = np.concatenate((arr_1, arr_2),
                                                 axis=axis).ravel(order='F')

        # assemble output
        joined = pyvista.StructuredGrid()
        joined.dimensions = list(new_dims)
        joined.points = new_points.reshape((-1, 3), order='F')
        joined.point_arrays.update(new_point_data)
        joined.cell_arrays.update(new_cell_data)

        return joined


@abstract_class
class UniformGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for uniform grid datasets."""

    def gaussian_smooth(dataset, radius_factor=1.5, std_dev=2.,
                        scalars=None, preference='points', progress_bar=False):
        """Smooth the data with a Gaussian kernel.

        Parameters
        ----------
        radius_factor : float or iterable, optional
            Unitless factor to limit the extent of the kernel.

        std_dev : float or iterable, optional
            Standard deviation of the kernel in pixel units.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, optional
            When scalars is specified, this is the preferred array type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        progress_bar : bool, optional
            Display a progress bar to indicate progress.
        """
        alg = vtk.vtkImageGaussianSmooth()
        alg.SetInputDataObject(dataset)
        if scalars is None:
            field, scalars = dataset.active_scalars_info
        else:
            _, field = dataset.get_array(scalars, preference=preference, info=True)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars) # args: (idx, port, connection, field, name)
        if isinstance(radius_factor, collections.abc.Iterable):
            alg.SetRadiusFactors(radius_factor)
        else:
            alg.SetRadiusFactors(radius_factor, radius_factor, radius_factor)
        if isinstance(std_dev, collections.abc.Iterable):
            alg.SetStandardDeviations(std_dev)
        else:
            alg.SetStandardDeviations(std_dev, std_dev, std_dev)
        _update_alg(alg, progress_bar, 'Performing Gaussian Smoothing')
        return _get_output(alg)

    def extract_subset(dataset, voi, rate=(1, 1, 1), boundary=False):
        """Select piece (e.g., volume of interest).

        To use this filter set the VOI ivar which are i-j-k min/max indices
        that specify a rectangular region in the data. (Note that these are
        0-offset.) You can also specify a sampling rate to subsample the
        data.

        Typical applications of this filter are to extract a slice from a
        volume for image processing, subsampling large volumes to reduce data
        size, or extracting regions of a volume with interesting data.

        Parameters
        ----------
        voi : tuple(int)
            Length 6 iterable of ints: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
            These bounds specify the volume of interest in i-j-k min/max
            indices.

        rate : tuple(int)
            Length 3 iterable of ints: ``(xrate, yrate, zrate)``.
            Default: ``(1, 1, 1)``

        boundary : bool
            Control whether to enforce that the "boundary" of the grid is
            output in the subsampling process. (This only has effect
            when the rate in any direction is not equal to 1). When
            this is on, the subsampling will always include the boundary of
            the grid even though the sample rate is not an even multiple of
            the grid dimensions. (By default this is off.)
        """
        alg = vtk.vtkExtractVOI()
        alg.SetVOI(voi)
        alg.SetInputDataObject(dataset)
        alg.SetSampleRate(rate)
        alg.SetIncludeBoundary(boundary)
        alg.Update()
        result = _get_output(alg)
        # Adjust for the confusing issue with the extents
        #   see https://gitlab.kitware.com/vtk/vtk/-/issues/17938
        fixed = pyvista.UniformGrid()
        fixed.origin = result.bounds[::2]
        fixed.spacing = result.spacing
        fixed.dimensions = result.dimensions
        fixed.point_arrays.update(result.point_arrays)
        fixed.cell_arrays.update(result.cell_arrays)
        fixed.field_arrays.update(result.field_arrays)
        fixed.copy_meta_from(result)
        return fixed
