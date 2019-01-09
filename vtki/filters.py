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
import numpy as np
import vtk

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


class DataSetFilters(object):
    """A set of common filters that can be applied to any vtkDataSet"""

    def clip(dataset, normal='x', origin=None, invert=True):
        """
        Clip a dataset by a plane by specifying the origin and normal. If no
        parameters are given the clip will occur in the center of that dataset
        
        dataset : 
        
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
        """
        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]
        # find center of data if origin not specified
        if origin is None:
            origin = dataset.center
        # create the plane for clipping
        plane = _generate_plane(normal, origin)
        # create slice
        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(dataset) # Use the grid as the data we desire to cut
        alg.SetCutFunction(plane) # the the cutter to use the plane we made
        alg.Update() # Perfrom the Cut
        return _get_output(alg)


    def threshold(dataset, value, scalars=None, invert=False, continuous=False,
                  preference='cell'):
        """
        This filter will apply a ``vtkThreshold`` filter to the input dataset and
        return the resulting object. This extracts cells where scalar value in each
        cell satisfies threshold criterion.  If scalars is None, the inputs 
        active_scalar is used.

        Parameters
        ----------
        dataset : vtk.vtkDataSet object
            Input dataset.

        value : float or iterable
            Single value or (min, max) to be used for the data threshold.  If
            iterable then length must be 2.

        scalars : str
            Name of scalars.

        invert : bool, optional
            If value is a single value, when invert is True cells are kept when
            their values are below parameter "value".  When invert is False
            cells are kept when their value is above the threshold "value".
            
        continuous : bool, optional
            When True, the continuous interval [minimum cell scalar, 
            maxmimum cell scalar] will be used to intersect the threshold bound, 
            rather than the set of discrete scalar values from the vertices.
            
        preference : str, optional
            When scalars is None, this is the perfered scalar type to search for
            in the dataset.  Must be either 'point' or 'cell'.

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
        # check if value is iterable (if so threshold by min max range like ParaView)
        if isinstance(value, collections.Iterable):
            if len(value) != 2:
                raise RuntimeError('Value range must be length one for a float value or two for min/max; not ({}).'.format(value))
            alg.ThresholdBetween(value[0], value[1])
            # NOTE: Invert for ThresholdBetween is coming in vtk=>8.2.x
            #alg.SetInvert(invert)
        else:
            # just a single value
            if invert:
                alg.ThresholdByLower(value)
            else:
                alg.ThresholdByUpper(value)
        # Run the threshold
        alg.Update()
        return _get_output(alg)


    def outline(dataset, gen_faces=False):
        alg = vtk.vtkOutlineFilter()
        alg.SetInputDataObject(dataset)
        alg.SetGenerateFaces(gen_faces)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))


class PointSetFilters(object):
    """Filters that can be applied to point set data objects"""


    def contour(dataset, isosurfaces=10, scalars=None, compute_normals=False,
                compute_gradients=False, compute_scalars=True, preference='point'):
        """Contours an input dataset by an array. ``isosurfaces`` can be an integer
        specifying the number of isosurfaces in the data range or an iterable set of
        values for explicitly setting the isosurfaces.
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
        # NOTE: only point data is allowed
        if field != 0:
            raise RuntimeError('Can only contour by Point data at this time.')
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
