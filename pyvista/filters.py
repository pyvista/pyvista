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

import numpy as np
import vtk

import pyvista
from pyvista.utilities import get_scalar, is_inside_bounds, wrap

NORMALS = {
    'x': [1, 0, 0],
    'y': [0, 1, 0],
    'z': [0, 0, 1],
    '-x': [-1, 0, 0],
    '-y': [0, -1, 0],
    '-z': [0, 0, -1],
}


def _get_output(algorithm, iport=0, iconnection=0, oport=0, active_scalar=None,
                active_scalar_field='point'):
    """A helper to get the algorithm's output and copy input's pyvista meta info"""
    ido = algorithm.GetInputDataObject(iport, iconnection)
    data = wrap(algorithm.GetOutputDataObject(oport))
    data.copy_meta_from(ido)
    if active_scalar is not None:
        data.set_active_scalar(active_scalar, preference=active_scalar_field)
    return data


def _generate_plane(normal, origin):
    """ Returns a vtk.vtkPlane """
    plane = vtk.vtkPlane()
    plane.SetNormal(normal[0], normal[1], normal[2])
    plane.SetOrigin(origin[0], origin[1], origin[2])
    return plane



class DataSetFilters(object):
    """A set of common filters that can be applied to any vtkDataSet"""

    def __new__(cls, *args, **kwargs):
        if cls is DataSetFilters:
            raise TypeError("pyvista.DataSetFilters is an abstract class and may not be instantiated.")
        return object.__new__(cls)


    def clip(dataset, normal='x', origin=None, invert=True):
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

        """
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = dataset.center
        # create the plane for clipping
        plane = _generate_plane(normal, origin)
        # run the clip
        alg = vtk.vtkClipDataSet()
        alg.SetInputDataObject(dataset) # Use the grid as the data we desire to cut
        alg.SetClipFunction(plane) # the the cutter to use the plane we made
        alg.SetInsideOut(invert) # invert the clip if needed
        alg.Update() # Perfrom the Cut
        return _get_output(alg)

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
        if not isinstance(bounds, collections.Iterable) or len(bounds) != 6:
            raise AssertionError('Bounds must be a length 6 iterable of floats')
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        alg = vtk.vtkBoxClipDataSet()
        alg.SetInputDataObject(dataset)
        alg.SetBoxClip(xmin, xmax, ymin, ymax, zmin, zmax)
        port = 0
        if invert:
            # invert the clip if needed
            port = 1
            alg.GenerateClippedOutputOn()
        alg.Update()
        return _get_output(alg, oport=port)

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
        if not is_inside_bounds(origin, dataset.bounds):
            raise AssertionError('Slice is outside data bounds.')
        # create the plane for clipping
        plane = _generate_plane(normal, origin)
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
        caresian planes. Yields a MutliBlock dataset of the three slices

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
        output = pyvista.MultiBlock()
        # Create the three slices
        if x is None:
            x = dataset.center[0]
        if y is None:
            y = dataset.center[1]
        if z is None:
            z = dataset.center[2]
        output[0, 'YZ'] = dataset.slice(normal='x', origin=[x,y,z], generate_triangles=generate_triangles)
        output[1, 'XZ'] = dataset.slice(normal='y', origin=[x,y,z], generate_triangles=generate_triangles)
        output[2, 'XY'] = dataset.slice(normal='z', origin=[x,y,z], generate_triangles=generate_triangles)
        return output


    def slice_along_axis(dataset, n=5, axis='x', tolerance=None,
                         generate_triangles=False, contour=False):
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
        output = pyvista.MultiBlock()
        if isinstance(axis, int):
            ax = axis
            axis = list(axes.keys())[list(axes.values()).index(ax)]
        elif isinstance(axis, str):
            try:
                ax = axes[axis]
            except KeyError:
                raise RuntimeError('Axis ({}) not understood'.format(axis))
        # get the locations along that axis
        if tolerance is None:
            tolerance = (dataset.bounds[ax*2+1] - dataset.bounds[ax*2]) * 0.01
        rng = np.linspace(dataset.bounds[ax*2]+tolerance, dataset.bounds[ax*2+1]-tolerance, n)
        center = list(dataset.center)
        # Make each of the slices
        for i in range(n):
            center[ax] = rng[i]
            slc = DataSetFilters.slice(dataset, normal=axis, origin=center,
                    generate_triangles=generate_triangles, contour=contour)
            output[i, 'slice%.2d'%i] = slc
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
            When scalars is specified, this is the perfered scalar type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        """
        # set the scalaras to threshold on
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        arr, field = get_scalar(dataset, scalars, preference=preference, info=True)

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
            When scalars is specified, this is the perfered scalar type to
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
                    invert=invert, continuous=continuous, preference=preference)


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
            perfered scalar type to search for in the dataset.
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
                compute_gradients=False, compute_scalars=True,  rng=None,
                preference='point'):
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
            When scalars is specified, this is the perfered scalar type to
            search for in the dataset.  Must be either ``'point'`` or ``'cell'``

        """
        # Make sure the input has scalars to contour on
        if dataset.n_scalars < 1:
            raise AssertionError('Input dataset for the contour filter must have scalar data.')
        alg = vtk.vtkContourFilter()
        alg.SetInputDataObject(dataset)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        # set the array to contour on
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        else:
            _, field = get_scalar(dataset, scalars, preference=preference, info=True)
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
                             inplace=False, name='Texture Coordinates'):
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

        """
        alg = vtk.vtkTextureMapToPlane()
        if origin is None or point_u is None or point_v is None:
            alg.SetAutomaticPlaneGeneration(True)
        else:
            alg.SetOrigin(origin) # BOTTOM LEFT CORNER
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

    def compute_cell_sizes(dataset, length=False, area=True, volume=True):
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


    def glyph(dataset, orient=True, scale=True, factor=1.0, geom=None):
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
        """
        if geom is None:
            arrow = vtk.vtkArrowSource()
            arrow.Update()
            geom = arrow.GetOutput()
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
        labeled = dataset.connectivity()
        classifier = labeled.cell_arrays['RegionId']
        bodies = pyvista.MultiBlock()
        for vid in np.unique(classifier):
            # Now extract it:
            b = labeled.threshold([vid-0.5, vid+0.5], scalars='RegionId')
            if not label:
                # strange behavior:
                # must use this method rather than deleting from the point_arrays
                # or else object is collected.
                b._remove_cell_scalar('RegionId')
                b._remove_point_scalar('RegionId')
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
        arr, field = get_scalar(dataset, scalars, preference='point', info=True)
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
            dataset.points = output.points
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
        return _get_output(alg, active_scalar=dataset.active_scalar_name)


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
        return _get_output(alg, active_scalar=dataset.active_scalar_name)


    def triangulate(dataset):
        """
        Returns an all triangle mesh.  More complex polygons will be broken
        down into triangles.

        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            Mesh containing only triangles.

        """
        alg = vtk.vtkDataSetTriangleFilter()
        alg.SetInputData(dataset)
        alg.Update()
        return _get_output(alg)


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
        (The name of the output vtkDataArray is "SelectedPointsArray".)

        The filter assumes that the surface is closed and manifold. A boolean
        flag can be set to force the filter to first check whether this is
        true. If false, all points will be marked outside. Note that if this
        check is not performed and the surface is not closed, the results are
        undefined.

        This filter produces and output data array, but does not modify the
        input dataset. If you wish to extract cells or poinrs, various
        threshold filters are available (i.e., threshold the output array).

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
            manifold.
        """
        alg = vtk.vtkSelectEnclosedPoints()
        alg.SetInputData(dataset)
        alg.SetSurfaceData(surface)
        alg.SetTolerance(tolerance)
        alg.SetCheckSurface(check_surface)
        alg.Update()
        return _get_output(alg)



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
            dimensions=(101, 101, 101), pass_cell_arrays=True, pass_point_arrays=True):
        """Interpolate values onto this mesh from the point data of a given
        :class:`pyvista.PolyData` object (typically a point cloud).

        This uses a guassian interpolation kernel. Use the ``sharpness`` and
        ``radius`` parameters to adjust this kernel.

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
        """
        box = pyvista.create_grid(dataset, dimensions=dimensions)

        gaussian_kernel = vtk.vtkGaussianKernel()
        gaussian_kernel.SetSharpness(sharpness)
        gaussian_kernel.SetRadius(radius)

        interpolator = vtk.vtkPointInterpolator()
        interpolator.SetInputData(box)
        interpolator.SetSourceData(points)
        interpolator.SetKernel(gaussian_kernel)
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
                    return_source=False):
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
        source = vtk.vtkPointSource()
        source.SetNumberOfPoints(n_points);
        source.SetCenter(source_center);
        source.SetRadius(source_radius);
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
        return dataset.extract_geometry().tri_filter().decimate(target_reduction)
