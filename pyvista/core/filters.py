"""
These classes hold methods to apply general filters to any data type.
By inherritting these classes into the wrapped VTK data structures, a user
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
import collections
import logging

import numpy as np
import vtk
from vtk.util.numpy_support import (numpy_to_vtkIdTypeArray, vtk_to_numpy)

import pyvista
from pyvista.utilities import (CELL_DATA_FIELD, POINT_DATA_FIELD, NORMALS,
                               generate_plane, get_array, wrap)


def _get_output(algorithm, iport=0, iconnection=0, oport=0, active_scalar=None,
                active_scalar_field='point'):
    """A helper to get the algorithm's output and copy input's pyvista meta info"""
    ido = algorithm.GetInputDataObject(iport, iconnection)
    data = wrap(algorithm.GetOutputDataObject(oport))
    if not isinstance(data, pyvista.MultiBlock):
        data.copy_meta_from(ido)
        if active_scalar is not None:
            data.set_active_scalar(active_scalar, preference=active_scalar_field)
    return data



class DataSetFilters(object):
    """A set of common filters that can be applied to any vtkDataSet"""

    def __new__(cls, *args, **kwargs):
        if cls is DataSetFilters:
            raise TypeError("pyvista.DataSetFilters is an abstract class and may not be instantiated.")
        return object.__new__(cls)


    def _clip_with_function(dataset, function, invert=True, value=0.0):
        """Internal helper to clip using an implicit function"""
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
        alg.Update() # Perfrom the Cut
        return _get_output(alg)


    def clip(dataset, normal='x', origin=None, invert=True, value=0.0, inplace=False):
        """
        Clip a dataset by a plane by specifying the origin and normal. If no
        parameters are given the clip will occur in the center of that dataset

        Parameters
        ----------
        normal : tuple(float) or str
            Length 3 tuple for the normal vector direction. Can also be
            specified as a string conventional direction such as ``'x'`` for
            ``(1,0,0)`` or ``'-x'`` for ``(-1,0,0)``, etc.

        origin : tuple(float)
            The center ``(x,y,z)`` coordinate of the plane on which the clip
            occurs

        invert : bool
            Flag on whether to flip/invert the clip

        value : float:
            Set the clipping value of the implicit function (if clipping with
            implicit function) or scalar value (if clipping with scalars).
            The default value is 0.0.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.
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
                                                    invert=invert, value=value)
        if inplace:
            dataset.overwrite(result)
        else:
            return result


    def clip_box(dataset, bounds=None, invert=True, factor=0.35):
        """Clips a dataset by a bounding box defined by the bounds. If no bounds
        are given, a corner of the dataset bounds will be removed.

        Parameters
        ----------
        bounds : tuple(float)
            Length 6 iterable of floats: (xmin, xmax, ymin, ymax, zmin, zmax)

        invert : bool
            Flag on whether to flip/invert the clip

        factor : float, optional
            If bounds are not given this is the factor along each axis to
            extract the default box.

        """
        if bounds is None:
            def _get_quarter(dmin, dmax):
                """internal helper to get a section of the given range"""
                return dmax - ((dmax - dmin) * factor)
            xmin, xmax, ymin, ymax, zmin, zmax = dataset.bounds
            xmin = _get_quarter(xmin, xmax)
            ymin = _get_quarter(ymin, ymax)
            zmin = _get_quarter(zmin, zmax)
            bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
        if isinstance(bounds, (float, int)):
            bounds = [bounds, bounds, bounds]
        if len(bounds) == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = dataset.bounds
            bounds = (xmin,xmin+bounds[0], ymin,ymin+bounds[1], zmin,zmin+bounds[2])
        if not isinstance(bounds, collections.Iterable) or not (len(bounds) == 6 or len(bounds) == 12):
            raise AssertionError('Bounds must be a length 6 iterable of floats.')
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


    def clip_surface(dataset, surface, invert=True, value=0.0,
                     compute_distance=False):
        """Clip any mesh type using a :class:`pyvista.PolyData` surface mesh.
        This will return a :class:`pyvista.UnstructuredGrid` of the clipped
        mesh. Geometry of the input dataset will be preserved where possible -
        geometries near the clip intersection will be triangulated/tesselated.

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
        """Slice a dataset by a plane at the specified origin and normal vector
        orientation. If no origin is specified, the center of the input dataset will
        be used.

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
        alg.SetCutFunction(plane) # the the cutter to use the plane we made
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        alg.Update() # Perfrom the Cut
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output


    def slice_orthogonal(dataset, x=None, y=None, z=None,
                         generate_triangles=False, contour=False):
        """Creates three orthogonal slices through the dataset on the three
        caresian planes. Yields a MutliBlock dataset of the three slices.

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
            The toleranceerance to the edge of the dataset bounds to create the slices

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
                raise RuntimeError('Axis ({}) not understood'.format(axis))
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
            output[i, 'slice%.2d' % i] = slc
        return output


    def slice_along_line(dataset, line, generate_triangles=False,
                         contour=False):
        """Slices a dataset using a polyline/spline as the path. This also works
        for lines generated with :func:`pyvista.Line`

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
            raise AssertionError('Input line must have only one cell.')
        polyline = line.GetCell(0)
        if not isinstance(polyline, vtk.vtkPolyLine):
            raise TypeError('Input line must have a PolyLine cell, not ({})'.format(type(polyline)))
        # Generate PolyPlane
        polyplane = vtk.vtkPolyPlane()
        polyplane.SetPolyLine(polyline)
        # Create slice
        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(dataset) # Use the grid as the data we desire to cut
        alg.SetCutFunction(polyplane) # the the cutter to use the poly planes
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        alg.Update() # Perfrom the Cut
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output


    def threshold(dataset, value=None, scalars=None, invert=False, continuous=False,
                  preference='cell'):
        """
        This filter will apply a ``vtkThreshold`` filter to the input dataset and
        return the resulting object. This extracts cells where scalar value in each
        cell satisfies threshold criterion.  If scalars is None, the inputs
        active_scalar is used.

        Parameters
        ----------
        value : float or iterable, optional
            Single value or (min, max) to be used for the data threshold.  If
            iterable, then length must be 2. If no value is specified, the
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
            maxmimum cell scalar] will be used to intersect the threshold bound,
            rather than the set of discrete scalar values from the vertices.

        preference : str, optional
            When scalars is specified, this is the preferred scalar type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        """
        # set the scalaras to threshold on
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        arr, field = get_array(dataset, scalars, preference=preference, info=True)

        if arr is None:
            raise AssertionError('No arrays present to threshold.')

        # If using an inverted range, merge the result of two fitlers:
        if isinstance(value, collections.Iterable) and invert:
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
        alg.SetInputDataObject(dataset)
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars) # args: (idx, port, connection, field, name)
        # set thresholding parameters
        alg.SetUseContinuousCellRange(continuous)
        # use valid range if no value given
        if value is None:
            value = dataset.get_data_range(scalars)
        # check if value is iterable (if so threshold by min max range like ParaView)
        if isinstance(value, collections.Iterable):
            if len(value) != 2:
                raise AssertionError('Value range must be length one for a float value or two for min/max; not ({}).'.format(value))
            alg.ThresholdBetween(value[0], value[1])
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
        """Thresholds the dataset by a percentage of its range on the active
        scalar array or as specified

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
            maxmimum cell scalar] will be used to intersect the threshold bound,
            rather than the set of discrete scalar values from the vertices.

        preference : str, optional
            When scalars is specified, this is the preferred scalar type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        """
        if scalars is None:
            _, tscalars = dataset.active_scalar_info
        else:
            tscalars = scalars
        dmin, dmax = dataset.get_data_range(arr=tscalars, preference=preference)

        def _check_percent(percent):
            """Make sure percent is between 0 and 1 or fix if between 0 and 100."""
            if percent >= 1:
                percent = float(percent) / 100.0
                if percent > 1:
                    raise RuntimeError('Percentage ({}) is out of range (0, 1).'.format(percent))
            if percent < 1e-10:
                raise RuntimeError('Percentage ({}) is too close to zero or negative.'.format(percent))
            return percent

        def _get_val(percent, dmin, dmax):
            """Gets the value from a percentage of a range"""
            percent = _check_percent(percent)
            return dmin + float(percent) * (dmax - dmin)

        # Compute the values
        if isinstance(percent, collections.Iterable):
            # Get two values
            value = [_get_val(percent[0], dmin, dmax), _get_val(percent[1], dmin, dmax)]
        else:
            # Compute one value to threshold
            value = _get_val(percent, dmin, dmax)
        # Use the normal thresholding function on these values
        return DataSetFilters.threshold(dataset, value=value, scalars=scalars,
                                        invert=invert, continuous=continuous,
                                        preference=preference)


    def outline(dataset, generate_faces=False):
        """Produces an outline of the full extent for the input dataset.

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
        """Produces an outline of the corners for the input dataset.

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
        """Extract the outer surface of a volume or structured grid dataset as
        PolyData. This will extract all 0D, 1D, and 2D cells producing the
        boundary faces of the dataset.
        """
        alg = vtk.vtkGeometryFilter()
        alg.SetInputDataObject(dataset)
        alg.Update()
        return _get_output(alg)

    def wireframe(dataset):
        """Extract all the internal/external edges of the dataset as PolyData.
        This produces a full wireframe representation of the input dataset.
        """
        alg = vtk.vtkExtractEdges()
        alg.SetInputDataObject(dataset)
        alg.Update()
        return _get_output(alg)

    def elevation(dataset, low_point=None, high_point=None, scalar_range=None,
                  preference='point', set_active=True):
        """Generate scalar values on a dataset.  The scalar values lie within a
        user specified range, and are generated by computing a projection of
        each dataset point onto a line.
        The line can be oriented arbitrarily.
        A typical example is to generate scalars based on elevation or height
        above a plane.

        Parameters
        ----------
        low_point : tuple(float), optional
            The low point of the projection line in 3D space. Default is bottom
            center of the dataset. Otherwise pass a length 3 tuple(float).

        high_point : tuple(float), optional
            The high point of the projection line in 3D space. Default is top
            center of the dataset. Otherwise pass a length 3 tuple(float).

        scalar_range : str or tuple(float), optional
            The scalar range to project to the low and high points on the line
            that will be mapped to the dataset. If None given, the values will
            be computed from the elevation (Z component) range between the
            high and low points. Min and max of a range can be given as a length
            2 tuple(float). If ``str`` name of scalara array present in the
            dataset given, the valid range of that array will be used.

        preference : str, optional
            When a scalar name is specified for ``scalar_range``, this is the
            preferred scalar type to search for in the dataset.
            Must be either 'point' or 'cell'.

        set_active : bool, optional
            A boolean flag on whethter or not to set the new `Elevation` scalar
            as the active scalar array on the output dataset.

        Warning
        -------
        This will create a scalar array named `Elevation` on the point data of
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
            scalar_range = dataset.get_data_range(arr=scalar_range, preference=preference)
        elif isinstance(scalar_range, collections.Iterable):
            if len(scalar_range) != 2:
                raise AssertionError('scalar_range must have a length of two defining the min and max')
        else:
            raise RuntimeError('scalar_range argument ({}) not understood.'.format(type(scalar_range)))
        # Construct the filter
        alg = vtk.vtkElevationFilter()
        alg.SetInputDataObject(dataset)
        # Set the parameters
        alg.SetScalarRange(scalar_range)
        alg.SetLowPoint(low_point)
        alg.SetHighPoint(high_point)
        alg.Update()
        # Decide on updating active scalar array
        name = 'Elevation' # Note that this is added to the PointData
        if not set_active:
            name = None
        return _get_output(alg, active_scalar=name, active_scalar_field='point')


    def contour(dataset, isosurfaces=10, scalars=None, compute_normals=False,
                compute_gradients=False, compute_scalars=True, rng=None,
                preference='point', method='contour'):
        """Contours an input dataset by an array. ``isosurfaces`` can be an integer
        specifying the number of isosurfaces in the data range or an iterable set of
        values for explicitly setting the isosurfaces.

        Parameters
        ----------
        isosurfaces : int or iterable
            Number of isosurfaces to compute across valid data range or an
            iterable of float values to explicitly use as the isosurfaces.

        scalars : str, optional
            Name of scalars to threshold on. Defaults to currently active scalars.

        compute_normals : bool, optional

        compute_gradients : bool, optional
            Desc

        compute_scalars : bool, optional
            Preserves the scalar values that are being contoured

        rng : tuple(float), optional
            If an integer number of isosurfaces is specified, this is the range
            over which to generate contours. Default is the scalar arrays's full
            data range.

        preference : str, optional
            When scalars is specified, this is the preferred scalar type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        method : str, optional
            Specify to choose which vtk filter is used to create the contour.
            Must be one of ``'contour'``, ``'marching_cubes'`` and
            ``'flying_edges'``. Defaults to ``'contour'``.

        """
        if method is None or method == 'contour':
            alg = vtk.vtkContourFilter()
        elif method == 'marching_cubes':
            alg = vtk.vtkMarchingCubes()
        elif method == 'flying_edges':
            alg = vtk.vtkFlyingEdges3D()
        else:
            raise RuntimeError("Method '{}' is not supported".format(method))
        # Make sure the input has scalars to contour on
        if dataset.n_arrays < 1:
            raise AssertionError('Input dataset for the contour filter must have scalar data.')
        alg.SetInputDataObject(dataset)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        # set the array to contour on
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        else:
            _, field = get_array(dataset, scalars, preference=preference, info=True)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        if field != pyvista.POINT_DATA_FIELD:
            raise AssertionError('Contour filter only works on Point data. Array ({}) is in the Cell data.'.format(scalars))
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars) # args: (idx, port, connection, field, name)
        # set the isosurfaces
        if isinstance(isosurfaces, int):
            # generate values
            if rng is None:
                rng = dataset.get_data_range(scalars)
            alg.GenerateValues(isosurfaces, rng)
        elif isinstance(isosurfaces, collections.Iterable):
            alg.SetNumberOfContours(len(isosurfaces))
            for i, val in enumerate(isosurfaces):
                alg.SetValue(i, val)
        else:
            raise RuntimeError('isosurfaces not understood.')
        alg.Update()
        return _get_output(alg)


    def texture_map_to_plane(dataset, origin=None, point_u=None, point_v=None,
                             inplace=False, name='Texture Coordinates',
                             use_bounds=False):
        """Texture map this dataset to a user defined plane. This is often used
        to define a plane to texture map an image to this dataset. The plane
        defines the spatial reference and extent of that image.

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

        use_bounds : bool
            Use the bounds to set the mapping plane by default (bottom plane
            of the bounding box).
        """
        if use_bounds:
            if isinstance(use_bounds, (int, bool)):
                b = dataset.GetBounds()
            else:
                b = use_bounds
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

    def compute_cell_sizes(dataset, length=True, area=True, volume=True):
        """This filter computes sizes for 1D (length), 2D (area) and 3D (volume)
        cells.

        Parameters
        ----------
        length : bool
            Specify whether or not to compute the length of 1D cells.

        area : bool
            Specify whether or not to compute the area of 2D cells.

        volume : bool
            Specify whether or not to compute the volume of 3D cells.

        """
        alg = vtk.vtkCellSizeFilter()
        alg.SetInputDataObject(dataset)
        alg.SetComputeArea(area)
        alg.SetComputeVolume(volume)
        alg.SetComputeLength(length)
        alg.SetComputeVertexCount(False)
        alg.Update()
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
              tolerance=0.0, absolute=False):
        """
        Copies a geometric representation (called a glyph) to every
        point in the input dataset.  The glyph may be oriented along
        the input vectors, and it may be scaled according to scalar
        data or vector magnitude.

        Parameters
        ----------
        orient : bool
            Use the active vectors array to orient the the glyphs

        scale : bool
            Use the active scalars to scale the glyphs

        factor : float
            Scale factor applied to sclaing array

        geom : vtk.vtkDataSet
            The geometry to use for the glyph

        tolerance : float, optional
            Specify tolerance in terms of fraction of bounding box length.
            Float value is between 0 and 1. Default is 0.0. If ``absolute``
            is ``True`` then the tolerance can be an absolute distance.

        absolute : bool, optional
            Control if ``tolerance`` is an absolute distance or a fraction.
        """
        # Clean the points before glyphing
        small = pyvista.PolyData(dataset.points)
        small.point_arrays.update(dataset.point_arrays)
        dataset = small.clean(point_merging=True, merge_tol=tolerance,
                              lines_to_points=False, polys_to_lines=False,
                              strips_to_polys=False, inplace=False,
                              absolute=absolute)
        # Make glyphing geometry
        if geom is None:
            arrow = vtk.vtkArrowSource()
            arrow.Update()
            geom = arrow.GetOutput()
        # Run the algorithm
        alg = vtk.vtkGlyph3D()
        alg.SetSourceData(geom)
        if isinstance(scale, str):
            dataset.active_scalar_name = scale
            scale = True
        if scale:
            if dataset.active_scalar is not None:
                if dataset.active_scalar.ndim > 1:
                    alg.SetScaleModeToScaleByVector()
                else:
                    alg.SetScaleModeToScaleByScalar()
        else:
            alg.SetScaleModeToDataScalingOff()
        if isinstance(orient, str):
            dataset.active_vectors_name = orient
            orient = True
        alg.SetOrient(orient)
        alg.SetInputData(dataset)
        alg.SetVectorModeToUseVector()
        alg.SetScaleFactor(factor)
        alg.Update()
        return _get_output(alg)


    def connectivity(dataset, largest=False):
        """Find and label connected bodies/volumes. This adds an ID array to
        the point and cell data to distinguish seperate connected bodies.
        This applies a ``vtkConnectivityFilter`` filter which extracts cells
        that share common points and/or meet other connectivity criterion.
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
        """Find, label, and split connected bodies/volumes. This splits
        different connected bodies into blocks in a MultiBlock dataset.

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
                b._remove_array(CELL_DATA_FIELD, 'RegionId')
                b._remove_array(POINT_DATA_FIELD, 'RegionId')
            bodies.append(b)

        return bodies


    def warp_by_scalar(dataset, scalars=None, factor=1.0, normal=None,
                       inplace=False, **kwargs):
        """
        Warp the dataset's points by a point data scalar array's values.
        This modifies point coordinates by moving points along point normals by
        the scalar amount times the scale factor.

        Parameters
        ----------
        scalars : str, optional
            Name of scalars to warb by. Defaults to currently active scalars.

        factor : float, optional
            A scalaing factor to increase the scaling effect. Alias
            ``scale_factor`` also accepted - if present, overrides ``factor``.

        normal : np.array, list, tuple of length 3
            User specified normal. If given, data normals will be ignored and
            the given normal will be used to project the warp.

        inplace : bool
            If True, the points of the give dataset will be updated.
        """
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        arr, field = get_array(dataset, scalars, preference='point', info=True)
        if field != pyvista.POINT_DATA_FIELD:
            raise AssertionError('Dataset can only by warped by a point data array.')
        scale_factor = kwargs.get('scale_factor', None)
        if scale_factor is not None:
            factor = scale_factor
        # Run the algorithm
        alg = vtk.vtkWarpScalar()
        alg.SetInputDataObject(dataset)
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars) # args: (idx, port, connection, field, name)
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


    def cell_data_to_point_data(dataset, pass_cell_data=False):
        """Transforms cell data (i.e., data specified per cell) into point data
        (i.e., data specified at cell points).
        The method of transformation is based on averaging the data values of
        all cells using a particular point. Optionally, the input cell data can
        be passed through to the output as well.

        See aslo: :func:`pyvista.DataSetFilters.point_data_to_cell_data`

        Parameters
        ----------
        pass_cell_data : bool
            If enabled, pass the input cell data through to the output
        """
        alg = vtk.vtkCellDataToPointData()
        alg.SetInputDataObject(dataset)
        alg.SetPassCellData(pass_cell_data)
        alg.Update()
        active_scalar = None
        if not isinstance(dataset, pyvista.MultiBlock):
            active_scalar = dataset.active_scalar_name
        return _get_output(alg, active_scalar=active_scalar)


    def ctp(dataset, pass_cell_data=False):
        """An alias/shortcut for ``cell_data_to_point_data``"""
        return DataSetFilters.cell_data_to_point_data(dataset, pass_cell_data=pass_cell_data)


    def point_data_to_cell_data(dataset, pass_point_data=False):
        """Transforms point data (i.e., data specified per node) into cell data
        (i.e., data specified within cells).
        Optionally, the input point data can be passed through to the output.

        See aslo: :func:`pyvista.DataSetFilters.cell_data_to_point_data`

        Parameters
        ----------
        pass_point_data : bool
            If enabled, pass the input point data through to the output
        """
        alg = vtk.vtkPointDataToCellData()
        alg.SetInputDataObject(dataset)
        alg.SetPassPointData(pass_point_data)
        alg.Update()
        active_scalar = None
        if not isinstance(dataset, pyvista.MultiBlock):
            active_scalar = dataset.active_scalar_name
        return _get_output(alg, active_scalar=active_scalar)


    def ptc(dataset, pass_point_data=False):
        """An alias/shortcut for ``point_data_to_cell_data``"""
        return DataSetFilters.point_data_to_cell_data(dataset, pass_point_data=pass_point_data)


    def triangulate(dataset, inplace=False):
        """
        Returns an all triangle mesh.  More complex polygons will be broken
        down into triangles.

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


    def delaunay_3d(dataset, alpha=0, tol=0.001, offset=2.5):
        """Constructs a 3D Delaunay triangulation of the mesh.
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
        """
        alg = vtk.vtkDelaunay3D()
        alg.SetInputData(dataset)
        alg.SetAlpha(alpha)
        alg.SetTolerance(tol)
        alg.SetOffset(offset)
        alg.Update()
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


    def sample(dataset, target, tolerance=None, pass_cell_arrays=True,
               pass_point_arrays=True):
        """Resample scalar data from a passed mesh onto this mesh using
        :class:`vtk.vtkResampleWithDataSet`.

        Parameters
        ----------
        dataset: pyvista.Common
            The source vtk data object as the mesh to sample values on to

        target: pyvista.Common
            The vtk data object to sample from - point and cell arrays from
            this object are sampled onto the nodes of the ``dataset`` mesh

        tolerance: flaot, optional
            tolerance used to compute whether a point in the source is in a
            cell of the input.  If not given, tolerance automatically generated.

        pass_cell_arrays: bool, optional
            Preserve source mesh's original cell data arrays

        pass_point_arrays: bool, optional
            Preserve source mesh's original point data arrays
        """
        alg = vtk.vtkResampleWithDataSet() # Construct the ResampleWithDataSet object
        alg.SetInputData(dataset)  # Set the Input data (actually the source i.e. where to sample from)
        alg.SetSourceData(target) # Set the Source data (actually the target, i.e. where to sample to)
        alg.SetPassCellArrays(pass_cell_arrays)
        alg.SetPassPointArrays(pass_point_arrays)
        if tolerance is not None:
            alg.SetComputeTolerance(False)
            alg.SetTolerance(tolerance)
        alg.Update() # Perfrom the resampling
        return _get_output(alg)


    def interpolate(dataset, points, sharpness=2, radius=1.0,
                    dimensions=(101, 101, 101), pass_cell_arrays=True,
                    pass_point_arrays=True, null_value=0.0):
        """Interpolate values onto this mesh from the point data of a given
        :class:`pyvista.PolyData` object (typically a point cloud).

        This uses a guassian interpolation kernel. Use the ``sharpness`` and
        ``radius`` parameters to adjust this kernel.

        Please note that the source dataset is first interpolated onto a fine
        UniformGrid which is then sampled to this mesh. The interpolation grid's
        dimensions will likely need to be tweaked for each individual use case.

        Parameters
        ----------
        points : pyvista.PolyData
            The points whose values will be interpolated onto this mesh.

        sharpness : float
            Set / Get the sharpness (i.e., falloff) of the Gaussian. By
            default Sharpness=2. As the sharpness increases the effects of
            distant points are reduced.

        radius : float
            Specify the radius within which the basis points must lie.

        dimensions : tuple(int)
            When interpolating the points, they are first interpolating on to a
            :class:`pyvista.UniformGrid` with the same spatial extent -
            ``dimensions`` is number of points along each axis for that grid.

        pass_cell_arrays: bool, optional
            Preserve source mesh's original cell data arrays

        pass_point_arrays: bool, optional
            Preserve source mesh's original point data arrays

        null_value : float, optional
            Specify the null point value. When a null point is encountered
            then all components of each null tuple are set to this value. By
            default the null value is set to zero.
        """
        box = pyvista.create_grid(dataset, dimensions=dimensions)

        gaussian_kernel = vtk.vtkGaussianKernel()
        gaussian_kernel.SetSharpness(sharpness)
        gaussian_kernel.SetRadius(radius)

        interpolator = vtk.vtkPointInterpolator()
        interpolator.SetInputData(box)
        interpolator.SetSourceData(points)
        interpolator.SetKernel(gaussian_kernel)
        interpolator.SetNullValue(null_value)
        interpolator.Update()

        return dataset.sample(interpolator.GetOutput(),
                              pass_cell_arrays=pass_cell_arrays,
                              pass_point_arrays=pass_point_arrays)

    def streamlines(dataset, vectors=None, source_center=None,
                    source_radius=None, n_points=100,
                    integrator_type=45, integration_direction='both',
                    surface_streamlines=False, initial_step_length=0.5,
                    step_unit='cl', min_step_length=0.01, max_step_length=1.0,
                    max_steps=2000, terminal_speed=1e-12, max_error=1e-6,
                    max_time=None, compute_vorticity=True, rotation_scale=1.0,
                    interpolator_type='point', start_position=(0.0, 0.0, 0.0),
                    return_source=False, pointa=None, pointb=None):
        """Integrate a vector field to generate streamlines. The integration is
        performed using a specified integrator, by default Runge-Kutta2.
        This supports integration through any type of dataset.
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
            Maxmimum step size used for line integration, expressed in length or
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
            are ``'point'`` or ``'cell'`` (abreviations of ``'p'`` and ``'c'``
            are also supported).

        rotation_scale : float
            This can be used to scale the rate with which the streamribbons
            twist. The default is 1.

        start_position : tuple(float)
            Set the start position. Default is ``(0.0, 0.0, 0.0)``

        return_source : bool
            Return the source particles as :class:`pyvista.PolyData` as well as the
            streamlines. This will be the second value returned if ``True``.

        pointa, pointb : tuple(flaot)
            The coordinates of a start and end point for a line source. This
            will override the sphere point source.
        """
        integration_direction = str(integration_direction).strip().lower()
        if integration_direction not in ['both', 'back', 'backward', 'forward']:
            raise RuntimeError("integration direction must be one of: 'backward', 'forward', or 'both' - not '{}'.".format(integration_direction))
        if integrator_type not in [2, 4, 45]:
            raise RuntimeError('integrator type must be one of `2`, `4`, or `45`.')
        if interpolator_type not in ['c', 'cell', 'p', 'point']:
            raise RuntimeError("interpolator type must be either 'cell' or 'point'")
        if step_unit not in ['l', 'cl']:
            raise RuntimeError("step unit must be either 'l' or 'cl'")
        step_unit = {'cl':vtk.vtkStreamTracer.CELL_LENGTH_UNIT,
                     'l':vtk.vtkStreamTracer.LENGTH_UNIT}[step_unit]
        if isinstance(vectors, str):
            dataset.set_active_scalar(vectors)
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
        """Return a decimated version of a triangulation of the boundary of
        this mesh's outer surface

        Parameters
        ----------
        target_reduction : float
            Fraction of the original mesh to remove. Default is ``0.5``
            TargetReduction is set to ``0.9``, this filter will try to reduce
            the data set to 10% of its original size and will remove 90%
            of the input triangles.
        """
        return dataset.extract_geometry().triangulate().decimate(target_reduction)


    def plot_over_line(dataset, pointa, pointb, resolution=None, scalars=None,
                       title=None, ylabel=None, figsize=None, figure=True,
                       show=True):
        """Sample a dataset along a high resolution line and plot the variables
        of interest in 2D where the X-axis is distance from Point A and the
        Y-axis is the varaible of interest. Note that this filter returns None.

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
        """
        # Ensure matplotlib is available
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('matplotlib must be available to use this filter.')

        if resolution is None:
            resolution = int(dataset.n_cells)
        if not isinstance(resolution, int) or resolution < 0:
            raise RuntimeError('`resolution` must be a positive integer, not {}'.format(type(resolution)))
        # Make a line and probe the dataset
        line = pyvista.Line(pointa, pointb, resolution=resolution)
        sampled = line.sample(dataset)

        # Get variable of interest
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        values = sampled.get_array(scalars)
        distance = sampled['Distance']

        # Remainder of the is plotting
        if figure:
            plt.figure(figsize=figsize)
        # Plot it in 2D
        if values.ndim > 1:
            for i in range(values.shape[1]):
                plt.plot(distance, values[:, i], label='Component {}'.format(i))
            plt.legend()
        else:
            plt.plot(distance, values)
        plt.xlabel('Distance')
        if ylabel is None:
            plt.ylabel(scalars)
        else:
            plt.ylabel(ylabel)
        if title is None:
            plt.title('{} Profile'.format(scalars))
        else:
            plt.title(title)
        if show:
            return plt.show()


    def extract_cells(dataset, ind):
        """
        Returns a subset of the grid

        Parameters
        ----------
        ind : np.ndarray
            Numpy array of cell indices to be extracted.

        Returns
        -------
        subgrid : pyvista.UnstructuredGrid
            Subselected grid

        """
        if not isinstance(ind, np.ndarray):
            ind = np.array(ind, np.ndarray)

        if ind.dtype == np.bool:
            ind = ind.nonzero()[0].astype(pyvista.ID_TYPE)

        if ind.dtype != pyvista.ID_TYPE:
            ind = ind.astype(pyvista.ID_TYPE)

        if not ind.flags.c_contiguous:
            ind = np.ascontiguousarray(ind)

        vtk_ind = numpy_to_vtkIdTypeArray(ind, deep=False)

        # Create selection objects
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(vtk_ind)

        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, dataset)
        extract_sel.SetInputData(1, selection)
        extract_sel.Update()
        subgrid = _get_output(extract_sel)

        # extracts only in float32
        if dataset.points.dtype is not np.dtype('float32'):
            ind = subgrid.point_arrays['vtkOriginalPointIds']
            subgrid.points = dataset.points[ind]

        return subgrid


    def extract_points(dataset, ind):
        """Returns a subset of the grid (with cells) that contains the points
        that contain any of the given point indices.

        Parameters
        ----------
        ind : np.ndarray, list, or iterable
            Numpy array of point indices to be extracted.

        Returns
        -------
        subgrid : pyvista.UnstructuredGrid
            Subselected grid.
        """
        try:
            ind = np.array(ind)
        except:
            raise Exception('indices must be either a mask, array, list, or iterable')

        # Convert to vtk indices
        if ind.dtype == np.bool:
            ind = ind.nonzero()[0]

        if ind.dtype != np.int64:
            ind = ind.astype(np.int64)
        vtk_ind = numpy_to_vtkIdTypeArray(ind, deep=True)

        # Create selection objects
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(vtk_ind)
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)

        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, dataset)
        extract_sel.SetInputData(1, selection)
        extract_sel.Update()
        return _get_output(extract_sel)


    def extract_selection_points(dataset, ind):
        logging.warning("DEPRECATED: use ``extract_points`` instead.")
        return DataSetFilters.extract_points(dataset, ind)


    def extract_surface(dataset, pass_pointid=True, pass_cellid=True, inplace=False):
        """
        Extract surface mesh of the grid

        Parameters
        ----------
        pass_pointid : bool, optional
            Adds a point scalar "vtkOriginalPointIds" that idenfities which
            original points these surface points correspond to

        pass_cellid : bool, optional
            Adds a cell scalar "vtkOriginalPointIds" that idenfities which
            original cells these surface cells correspond to

        inplace : bool, optional
            Return new mesh or overwrite input.

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

        mesh = _get_output(surf_filter)
        if inplace:
            dataset.overwrite(mesh)
        else:
            return mesh


    def surface_indices(dataset):
        """
        The surface indices of a grid.

        Returns
        -------
        surf_ind : np.ndarray
            Indices of the surface points.

        """
        surf = DataSetFilters.extract_surface(dataset, pass_cellid=True)
        return surf.point_arrays['vtkOriginalPointIds']


    def extract_edges(dataset, feature_angle=30, boundary_edges=True,
                      non_manifold_edges=True, feature_edges=True,
                      manifold_edges=True, inplace=False):
        """
        Extracts edges from the surface of the mesh. If the given mesh is not
        PolyData, the external surface of the given mesh is extracted and used.
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
        """
        Join one or many other grids to this grid.  Grid is updated
        in-place by default.

        Can be used to merge points of adjcent cells when no grids
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
            the scalar arrays of the merging grids will be overwritten
            by the original main mesh.

        Returns
        -------
        merged_grid : vtk.UnstructuredGrid
            Merged grid.  Returned when inplace is False.

        Notes
        -----
        When two or more grids are joined, the type and name of each
        scalar array must match or the arrays will be ignored and not
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
                raise TypeError("Mesh tpye {} not able to be overridden by output.".format(type(dataset)))
        else:
            return merged


    def __add__(dataset, grid):
        """Combine this mesh with another into an
        :class:`pyvista.UnstructuredGrid`"""
        return DataSetFilters.merge(dataset, grid)


    def compute_cell_quality(dataset, quality_measure='scaled_jacobian', null_value=-1.0):
        """compute a function of (geometric) quality for each cell of a mesh.
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
            options = ', '.join(["'{}'".format(s) for s in list(measure_setters.keys())])
            raise KeyError('Cell quality type ({}) not available. Options are: {}'.format(quality_measure, options))
        alg.SetInputData(dataset)
        alg.SetUndefinedQuality(null_value)
        alg.Update()
        return _get_output(alg)


    def compute_gradient(dataset, scalars=None, gradient_name='gradient',
                         preference='point'):
        """Computes per cell gradient of point scalar field or per point
        gradient of cell scalar field.

        Parameters
        ----------
        scalars : str
            String name of the scalars array to use when computing gradient.

        gradient_name : str, optional
            The name of the output array of the computed gradient.
        """
        alg = vtk.vtkGradientFilter()
        # Check if scalar array given
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        if not isinstance(scalars, str):
            raise TypeError('Scalar array must be given as a string name')
        _, field = dataset.get_array(scalars, preference=preference, info=True)
        # args: (idx, port, connection, field, name)
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars)
        alg.SetInputData(dataset)
        alg.SetResultArrayName(gradient_name)
        alg.Update()
        return _get_output(alg)


class CompositeFilters(object):
    """An internal class to manage filtes/algorithms for composite datasets.
    """
    def __new__(cls, *args, **kwargs):
        if cls is CompositeFilters:
            raise TypeError("pyvista.CompositeFilters is an abstract class and may not be instantiated.")
        return object.__new__(cls)


    def extract_geometry(composite):
        """Combines the geomertry of all blocks into a single ``PolyData``
        object. Place this filter at the end of a pipeline before a polydata
        consumer such as a polydata mapper to extract geometry from all blocks
        and append them to one polydata object.
        """
        gf = vtk.vtkCompositeDataGeometryFilter()
        gf.SetInputData(composite)
        gf.Update()
        return wrap(gf.GetOutputDataObject(0))


    def combine(composite, merge_points=False):
        """Appends all blocks into a single unstructured grid.

        Parameters
        ----------
        merge_points : bool, optional
            Merge coincidental points.

        """
        alg = vtk.vtkAppendFilter()
        for block in composite:
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


    wireframe = DataSetFilters.wireframe


    elevation = DataSetFilters.elevation


    compute_cell_sizes = DataSetFilters.compute_cell_sizes


    cell_centers = DataSetFilters.cell_centers


    cell_data_to_point_data = DataSetFilters.cell_data_to_point_data


    point_data_to_cell_data = DataSetFilters.point_data_to_cell_data


    triangulate = DataSetFilters.triangulate


    def outline(composite, generate_faces=False, nested=False):
        """Produces an outline of the full extent for the all blocks in this
        composite dataset.

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
        """Produces an outline of the corners for the all blocks in this
        composite dataset.

        Parameters
        ----------
        factor : float, optional
            controls the relative size of the corners to the length of the
            corresponding bounds

        ested : bool, optional
            If True, these creates individual outlines for each nested dataset
        """
        if nested:
            return DataSetFilters.outline_corners(composite, factor=factor)
        box = pyvista.Box(bounds=composite.bounds)
        return box.outline_corners(factor=factor)



class PolyDataFilters(DataSetFilters):

    def __new__(cls, *args, **kwargs):
        if cls is PolyDataFilters:
            raise TypeError("pyvista.PolyDataFilters is an abstract class and may not be instantiated.")
        return object.__new__(cls)

    def edge_mask(poly_data, angle):
        """
        Returns a mask of the points of a surface mesh that have a surface
        angle greater than angle

        Parameters
        ----------
        angle : float
            Angle to consider an edge.

        """
        if not isinstance(poly_data, pyvista.PolyData):
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
        orig_id = pyvista.point_scalar(edges, 'point_ind')

        return np.in1d(poly_data.point_arrays['point_ind'], orig_id,
                       assume_unique=True)


    def boolean_cut(poly_data, cut, tolerance=1E-5, inplace=False):
        """
        Performs a Boolean cut using another mesh.

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
        """
        Add a mesh to the current mesh.  Does not attempt to "join"
        the meshes.

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
        """Merge these two meshes"""
        if not isinstance(mesh, vtk.vtkPolyData):
            return DataSetFilters.__add__(poly_data, mesh)
        return PolyDataFilters.boolean_add(poly_data, mesh)


    def boolean_union(poly_data, mesh, inplace=False):
        """
        Combines two meshes and attempts to create a manifold mesh.

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
        """
        Combines two meshes and retains only the volume in common
        between the meshes.

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


    def curvature(poly_data, curv_type='mean'):
        """
        Returns the pointwise curvature of a mesh

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
            raise Exception('Curv_Type must be either "Mean", '
                            '"Gaussian", "Maximum", or "Minimum"')
        curvefilter.Update()

        # Compute and return curvature
        curv = _get_output(curvefilter)
        return vtk_to_numpy(curv.GetPointData().GetScalars())


    def plot_curvature(poly_data, curv_type='mean', **kwargs):
        """
        Plots curvature

        Parameters
        ----------
        curvtype : str, optional
            One of the following strings indicating curvature type

            - Mean
            - Gaussian
            - Maximum
            - Minimum

        **kwargs : optional
            See help(pyvista.plot)

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up
        """
        return poly_data.plot(scalars=poly_data.curvature(curv_type),
                              stitle='%s\nCurvature' % curv_type, **kwargs)


    def triangulate(poly_data, inplace=False):
        """
        Returns an all triangle mesh.  More complex polygons will be broken
        down into triangles.

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


    def tri_filter(poly_data, inplace=False):
        """DEPRECATED: use ``.triangulate`` instead"""
        logging.warning("DEPRECATED: ``.tri_filter`` is deprecated. Use ``.triangulate`` instead.")
        return PolyDataFilters.triangulate(poly_data, inplace=inplace)


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
            Decimated mesh. None when inplace=True.

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
        """Reduce the number of triangles in a triangular mesh, forming a good
        approximation to the original geometry. Based on the algorithm originally
        described in "Decimation of Triangle Meshes", Proc Siggraph 92.

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
        """Generate a tube around each input line. The radius of the tube can be
        set to linearly vary with a scalar value.

        Parameters
        ----------
        radius : float
            Minimum tube radius (minimum because the tube radius may vary).

        scalars : str, optional
            Scalar array by which the radius varies

        capping : bool
            Turn on/off whether to cap the ends with polygons. Default True.

        n_sides : int
            Set the number of sides for the tube. Minimum of 3.

        radius_factor : float
            Maximum tube radius in terms of a multiple of the minimum radius.

        preference : str
            The field preference when searching for the scalar array by name

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Tube-filtered mesh. None when inplace=True.

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
        # Check if scalar array given
        if scalars is not None:
            if not isinstance(scalars, str):
                raise TypeError('Scalar array must be given as a string name')
            _, field = poly_data.get_array(scalars, preference=preference, info=True)
            # args: (idx, port, connection, field, name)
            tube.SetInputArrayToProcess(0, 0, 0, field, scalars)
            tube.SetVaryRadiusToVaryRadiusByScalar()
        # Apply the filter
        tube.Update()

        mesh = _get_output(tube)
        if inplace:
            poly_data.overwrite(mesh)
        else:
            return mesh


    def subdivide(poly_data, nsub, subfilter='linear', inplace=False):
        """
        Increase the number of triangles in a single, connected triangular
        mesh.

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

        alternatively, update mesh in-place

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
            raise Exception("Subdivision filter must be one of the following: "
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
                 tensors_weight=0.1, inplace=False):
        """
        Reduces the number of triangles in a triangular mesh using
        vtkQuadricDecimation.

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

        Returns
        -------
        outmesh : pyvista.PolyData
            Decimated mesh.  None when inplace=True.

        """
        # create decimation filter
        decimate = vtk.vtkQuadricDecimation()  # vtkDecimatePro as well

        decimate.SetVolumePreservation(volume_preservation)
        decimate.SetAttributeErrorMetric(attribute_error)
        decimate.SetScalarsAttribute(scalars)
        decimate.SetVectorsAttribute(vectors)
        decimate.SetNormalsAttribute(normals)
        decimate.SetTCoordsAttribute(tcoords)
        decimate.SetTensorsAttribute(tensors)
        decimate.SetScalarsWeight(scalars_weight)
        decimate.SetVectorsWeight(vectors_weight)
        decimate.SetNormalsWeight(normals_weight)
        decimate.SetTCoordsWeight(tcoords_weight)
        decimate.SetTensorsWeight(tensors_weight)
        decimate.SetTargetReduction(target_reduction)

        decimate.SetInputData(poly_data)
        decimate.Update()

        mesh = _get_output(decimate)
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
        """
        Compute point and/or cell normals for a mesh.

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


    def clip_with_plane(poly_data, origin, normal, value=0, invert=False, inplace=False):
        """DEPRECATED: Use ``.clip`` instead."""
        logging.warning('DEPRECATED: ``clip_with_plane`` is deprecated. Use ``.clip`` instead.')
        return DataSetFilters.clip(poly_data, normal=normal, origin=origin, value=value, invert=invert, inplace=inplace)


    def fill_holes(poly_data, hole_size, inplace=False):  # pragma: no cover
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

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh with holes filled.  None when inplace=True

        """
        logging.warning('pyvista.PolyData.fill_holes is known to segfault. '
                        'Use at your own risk')
        fill = vtk.vtkFillHolesFilter()
        fill.SetHoleSize(hole_size)
        fill.SetInputData(poly_data)
        fill.Update()

        mesh = _get_output(fill)
        if inplace:
            poly_data.overwrite(mesh)
        else:
            return mesh

    def clean(poly_data, point_merging=True, tolerance=None, lines_to_points=True,
              polys_to_lines=True, strips_to_polys=True, inplace=False,
              absolute=True, **kwargs):
        """
        Cleans mesh by merging duplicate points, remove unused
        points, and/or remove degenerate cells.

        Parameters
        ----------
        point_merging : bool, optional
            Enables point merging.  On by default.

        tolerance : float, optional
            Set merging tolerance.  When enabled merging is set to
            absolute distance. If ``absolute`` is False, then the merging
            tolerance is a fraction of the bounding box legnth. The alias
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

        Returns
        -------
        mesh : pyvista.PolyData
            Cleaned mesh.  None when inplace=True
        """
        if tolerance is None:
            tolerance = kwargs.pop('merge_tol', None)
        clean = vtk.vtkCleanPolyData()
        clean.SetPointMerging(point_merging)
        clean.SetConvertLinesToPoints(lines_to_points)
        clean.SetConvertPolysToLines(polys_to_lines)
        clean.SetConvertStripsToPolys(strips_to_polys)
        if isinstance(tolerance, (int, float)):
            if absolute:
                clean.ToleranceIsAbsoluteOn()
                clean.SetAbsoluteTolerance(tolerance)
            else:
                clean.SetTolerance(tolerance)
        clean.SetInputData(poly_data)
        clean.Update()

        output = _get_output(clean)

        # Check output so no segfaults occur
        if output.n_points < 1:
            raise AssertionError('Clean tolerance is too high. Empty mesh returned.')

        if inplace:
            poly_data.overwrite(output)
        else:
            return output


    def geodesic(poly_data, start_vertex, end_vertex, inplace=False):
        """
        Calculates the geodesic path betweeen two vertices using Dijkstra's
        algorithm.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        Returns
        -------
        output : pyvista.PolyData
            PolyData object consisting of the line segment between the two given
            vertices.

        """
        if start_vertex < 0 or end_vertex > poly_data.n_points - 1:
            raise IndexError('Invalid indices.')

        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(poly_data)
        dijkstra.SetStartVertex(start_vertex)
        dijkstra.SetEndVertex(end_vertex)
        dijkstra.Update()

        output = _get_output(dijkstra)

        # Do not copy textures from input
        output.clear_textures()

        if inplace:
            poly_data.overwrite(output)
        else:
            return output


    def geodesic_distance(poly_data, start_vertex, end_vertex):
        """
        Calculates the geodesic distance betweeen two vertices using Dijkstra's
        algorithm.

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

        """
        path = poly_data.geodesic(start_vertex, end_vertex)
        sizes = path.compute_cell_sizes(length=True, area=False, volume=False)
        distance = np.sum(sizes['Length'])
        del path
        del sizes
        return distance

    def ray_trace(poly_data, origin, end_point, first_point=False, plot=False,
                  off_screen=False):
        """
        Performs a single ray trace calculation given a mesh and a line segment
        defined by an origin and end_point.

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
            Plots off screen.  Used for unit testing.

        Returns
        -------
        intersection_points : np.ndarray
            Location of the intersection points.  Empty array if no
            intersections.

        intersection_cells : np.ndarray
            Indices of the intersection cells.  Empty array if no
            intersections.

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


    def plot_boundaries(poly_data, **kwargs):
        """ Plots boundaries of a mesh """
        edges = DataSetFilters.extract_edges(poly_data)

        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', False),
                                  notebook=kwargs.pop('notebook', None))
        plotter.add_mesh(edges, 'r', style='wireframe', legend='Edges')
        plotter.add_mesh(poly_data, legend='Mesh', **kwargs)
        return plotter.show()


    def plot_normals(poly_data, show_mesh=True, mag=1.0, flip=False,
                     use_every=1, **kwargs):
        """
        Plot the point normals of a mesh.
        """
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
        """
        Rebuild a mesh by removing points.  Only valid for
        all-triangle meshes.

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

        """
        if isinstance(remove, list):
            remove = np.asarray(remove)

        if remove.dtype == np.bool:
            if remove.size != poly_data.n_points:
                raise AssertionError('Mask different size than n_points')
            remove_mask = remove
        else:
            remove_mask = np.zeros(poly_data.n_points, np.bool)
            remove_mask[remove] = True

        try:
            f = poly_data.faces.reshape(-1, 4)[:, 1:]
        except:
            raise Exception('Mesh must consist of only triangles')

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
                    logging.warning('Unable to pass cell key %s onto reduced mesh' % key)

        # Return vtk surface and reverse indexing array
        if inplace:
            poly_data.overwrite(newmesh)
        else:
            return newmesh, ridx


    def flip_normals(poly_data):
        """
        Flip normals of a triangular mesh by reversing the point ordering.

        """
        if poly_data.faces.size % 4:
            raise Exception('Can only flip normals on an all triangular mesh')

        f = poly_data.faces.reshape((-1, 4))
        f[:, 1:] = f[:, 1:][:, ::-1]


    def delaunay_2d(poly_data, tol=1e-05, alpha=0.0, offset=1.0, bound=False, inplace=False):
        """Apply a delaunay 2D filter along the best fitting plane"""
        alg = vtk.vtkDelaunay2D()
        alg.SetProjectionPlaneMode(vtk.VTK_BEST_FITTING_PLANE)
        alg.SetInputDataObject(poly_data)
        alg.SetTolerance(tol)
        alg.SetAlpha(alpha)
        alg.SetOffset(offset)
        alg.SetBoundingTriangulation(bound)
        alg.Update()

        mesh = _get_output(alg)
        if inplace:
            poly_data.overwrite(mesh)
        else:
            return mesh


    def delauney_2d(poly_data):
        """DEPRECATED. Please see :func:`pyvista.PolyData.delaunay_2d`"""
        raise AttributeError('`delauney_2d` is deprecated because we made a '
                             'spelling mistake. Please use `delaunay_2d`.')


    def compute_arc_length(poly_data):
        """Computes the arc length over the length of the probed line.
        It adds a new point-data array named "arc_length" with the computed arc
        length for each of the polylines in the input. For all other cell types,
        the arc length is set to 0.
        """
        alg = vtk.vtkAppendArcLength()
        alg.SetInputData(poly_data)
        alg.Update()
        return _get_output(alg)


    def project_points_to_plane(poly_data, origin=None, normal=(0,0,1), inplace=False):
        """Project points of this mesh to a plane"""
        if not isinstance(normal, collections.Iterable) or len(normal) != 3:
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
            alg.SetInputArrayToProcess(0, 0, 0, field, scalars) # args: (idx, port, connection, field, name)
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


class UnstructuredGridFilters(DataSetFilters):

    def __new__(cls, *args, **kwargs):
        if cls is UnstructuredGridFilters:
            raise TypeError("pyvista.UnstructuredGridFilters is an abstract class and may not be instantiated.")
        return object.__new__(cls)

    def delaunay_2d(ugrid, tol=1e-05, alpha=0.0, offset=1.0, bound=False):
        """Apply a delaunay 2D filter along the best fitting plane. This
        extracts the grid's points and perfoms the triangulation on those alone.
        """
        return pyvista.PolyData(ugrid.points).delaunay_2d(tol=tol, alpha=alpha,
                                                          offset=offset,
                                                          bound=bound)


class UniformGridFilters(DataSetFilters):

    def __new__(cls, *args, **kwargs):
        if cls is UniformGridFilters:
            raise TypeError("pyvista.UniformGridFilters is an abstract class and may not be instantiated.")
        return object.__new__(cls)

    def gaussian_smooth(dataset, radius_factor=1.5, std_dev=2.,
                        scalars=None, preference='points'):
        """Smooths the data with a Gaussian kernel

        Parameters
        ----------
        radius_factor : float or iterable, optional
            Unitless factor to limit the extent of the kernel.

        std_dev : float or iterable, optional
            Standard deviation of the kernel in pixel units.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, optional
            When scalars is specified, this is the preferred scalar type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``
        """
        alg = vtk.vtkImageGaussianSmooth()
        alg.SetInputDataObject(dataset)
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        else:
            _, field = dataset.get_array(scalars, preference=preference, info=True)
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars) # args: (idx, port, connection, field, name)
        if isinstance(radius_factor, collections.Iterable):
            alg.SetRadiusFactors(radius_factor)
        else:
            alg.SetRadiusFactors(radius_factor, radius_factor, radius_factor)
        if isinstance(std_dev, collections.Iterable):
            alg.SetStandardDeviations(std_dev)
        else:
            alg.SetStandardDeviations(std_dev, std_dev, std_dev)
        alg.Update()
        return _get_output(alg)
