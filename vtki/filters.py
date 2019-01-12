"""
These classes hold methods to apply general filters to any data type.
By inherritting these classes into the wrapped VTK data structures, a user
can easily apply common filters in an intuitive manner.

Example:

    >>> import vtki
    >>> from vtki import examples
    >>> dataset = examples.load_uniform()
    >>> dataset.set_active_scalar('Spatial Point Data') # Array the filters will use

    >>> # Threshold
    >>> thresh = dataset.threshold([100, 500])
    >>> thresh.plot(scalars='Spatial Point Data')

    >>> # Slice
    >>> slc = dataset.slice()
    >>> slc.plot(scalars='Spatial Point Data')

    >>> # Clip
    >>> clp = dataset.clip(invert=True)
    >>> clp.plot(scalars='Spatial Point Data')

    >>> # Contour
    >>> iso = dataset.contour()
    >>> iso.plot(scalars='Spatial Point Data')

"""
import collections
import logging
import numpy as np
import vtk

import vtki
from vtki.utilities import get_scalar, wrap

NORMALS = {
    'x': [1, 0, 0],
    'y': [0, 1, 0],
    'z': [0, 0, 1],
    '-x': [-1, 0, 0],
    '-y': [0, -1, 0],
    '-z': [0, 0, -1],
}


def _get_output(algorithm, iport=0, iconnection=0, oport=0):
    """A helper to get the algorithm's output and copy input's vtki meta info"""
    ido = algorithm.GetInputDataObject(iport, iconnection)
    data = wrap(algorithm.GetOutputDataObject(oport))
    data.copy_meta_from(ido)
    return data


def _generate_plane(normal, origin):
    """ Returns a vtk.vtkPlane """
    plane = vtk.vtkPlane()
    plane.SetNormal(normal[0], normal[1], normal[2])
    plane.SetOrigin(origin[0], origin[1], origin[2])
    return plane


def _is_inside_bounds(point, bounds):
    """ Checks if a point is inside a set of bounds """
    if not (bounds[0] < point[0] < bounds[1]):
        return False
    if not (bounds[2] < point[1] < bounds[3]):
        return False
    if not (bounds[4] < point[2] < bounds[5]):
        return False
    return True


class DataSetFilters(object):
    """A set of common filters that can be applied to any vtkDataSet"""


    def clip(dataset, normal='x', origin=None, invert=True):
        """
        Clip a dataset by a plane by specifying the origin and normal. If no
        parameters are given the clip will occur in the center of that dataset

        Parameters
        ----------
        normal : tuple(float) or str
            Length 3 tuple for the normal vector direction. Can also be specified
            as a string conventional direction such as ``'x'`` for ``(1,0,0)``
            or ``'-x'`` for ``(-1,0,0), etc.

        origin : tuple(float)
            The center (x,y,z) coordinate of the plane on which the clip occurs

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


    def slice(dataset, normal='x', origin=None):
        """Slice a dataset by a plane at the specified origin and normal vector
        orientation. If no origin is specified, the center of the input dataset will
        be used.

        Parameters
        ----------
        normal : tuple(float) or str
            Length 3 tuple for the normal vector direction. Can also be specified
            as a string conventional direction such as ``'x'`` for ``(1,0,0)``
            or ``'-x'`` for ``(-1,0,0), etc.

        origin : tuple(float)
            The center (x,y,z) coordinate of the plane on which the slice occurs

        """
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = dataset.center
        if not _is_inside_bounds(origin, dataset.bounds):
            raise RuntimeError('Slice is outside data bounds.')
        # create the plane for clipping
        plane = _generate_plane(normal, origin)
        # create slice
        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(dataset) # Use the grid as the data we desire to cut
        alg.SetCutFunction(plane) # the the cutter to use the plane we made
        alg.Update() # Perfrom the Cut
        return _get_output(alg)


    def slice_orthogonal(dataset, x=None, y=None, z=None):
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

        """
        output = vtki.MultiBlock()
        # Create the three slices
        if x is None:
            x = dataset.center[0]
        if y is None:
            y = dataset.center[1]
        if z is None:
            z = dataset.center[2]
        output[0, 'YZ'] = dataset.slice(normal='x', origin=[x,y,z])
        output[1, 'XZ'] = dataset.slice(normal='y', origin=[x,y,z])
        output[2, 'XY'] = dataset.slice(normal='z', origin=[x,y,z])
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
            When scalars is specified, this is the perfered scalar type to search
            for in the dataset.  Must be either 'point' or 'cell'.

        """
        alg = vtk.vtkThreshold()
        alg.SetInputDataObject(dataset)
        # set the scalaras to threshold on
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        else:
            _, field = get_scalar(dataset, scalars, preference=preference, info=True)
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars) # args: (idx, port, connection, field, name)
        # set thresholding parameters
        alg.SetUseContinuousCellRange(continuous)
        # use valid range if no value given
        if value is None:
            value = dataset.get_data_range(scalars)
        # check if value is iterable (if so threshold by min max range like ParaView)
        if isinstance(value, collections.Iterable):
            if len(value) != 2:
                raise RuntimeError('Value range must be length one for a float value or two for min/max; not ({}).'.format(value))
            alg.ThresholdBetween(value[0], value[1])
            # NOTE: Invert for ThresholdBetween is coming in vtk=>8.2.x
            version = vtk.VTK_VERSION.split('.')
            if invert:
                if (int(version[0]) <= 8 or int(version[0]) < 2):
                    logging.warning(' invert option for range thresholding is not supported before VTK version 8.2.x. You are running VTK version {}.'.format(vtk.VTK_VERSION))
                else:
                    alg.SetInvert(invert)
        else:
            # just a single value
            if invert:
                alg.ThresholdByLower(value)
            else:
                alg.ThresholdByUpper(value)
        # Run the threshold
        alg.Update()
        return _get_output(alg)


    def threshold_percent(dataset, percent=50, scalars=None, invert=False,
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
            When scalars is specified, this is the perfered scalar type to search
            for in the dataset.  Must be either 'point' or 'cell'.

        """
        if scalars is None:
            field, tscalars = dataset.active_scalar_info
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


    def outline(dataset, gen_faces=False):
        """Produces an outline of the full extent for the input dataset.

        Parameters
        ----------
        gen_faces : bool, optional
            Generate solid faces for the box. This is off by default

        """
        alg = vtk.vtkOutlineFilter()
        alg.SetInputDataObject(dataset)
        alg.SetGenerateFaces(gen_faces)
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
        """Extract the geometry of the dataset as PolyData"""
        alg = vtk.vtkGeometryFilter()
        alg.SetInputDataObject(dataset)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))


class PointSetFilters(object):
    """Filters that can be applied to point set data objects"""


    def contour(dataset, isosurfaces=10, scalars=None, compute_normals=False,
                compute_gradients=False, compute_scalars=True, preference='point'):
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

        preference : str, optional
            When scalars is specified, this is the perfered scalar type to search
            for in the dataset.  Must be either 'point' or 'cell'.

        """
        alg = vtk.vtkContourFilter() #vtkMarchingCubes
        alg.SetInputDataObject(dataset)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        # set the array to contour on
        #dataset.set_active_scalar(scalars, preference=preference)
        if scalars is None:
            field, scalars = dataset.active_scalar_info
        else:
            _, field = get_scalar(dataset, scalars, preference=preference, info=True)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        # if field != 0:
        #     raise RuntimeError('Can only contour by Point data at this time.')
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars) # args: (idx, port, connection, field, name)
        # set the isosurfaces
        if isinstance(isosurfaces, int):
            # generate values
            alg.GenerateValues(isosurfaces, dataset.get_data_range(scalars))
        elif isinstance(isosurfaces, collections.Iterable):
            alg.SetNumberOfContours(len(isosurfaces))
            for i, val in enumerate(isosurfaces):
                alg.SetValue(i, val)
        else:
            raise RuntimeError('isosurfaces not understood.')
        alg.Update()
        return _get_output(alg)
