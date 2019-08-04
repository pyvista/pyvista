"""
This module provides a number of algorithms that can be used with
the dataset classes defined in the dataset_adapter module.
See the documentation of the dataset_adapter for some examples.
These algorithms work in serial and in parallel as long as the
data is partitioned according to VTK data parallel execution
guidelines. For details, see the documentation of individual
algorithms.
"""
from __future__ import absolute_import
import sys

try:
    import numpy
except ImportError:
    raise RuntimeError("This module depends on the numpy module. Please make\
sure that it is installed properly.")

from . import dataset_adapter as dsa
from . import internal_algorithms as algs
import itertools
try:
    from vtk.vtkParallelCore import vtkMultiProcessController
    from vtk.vtkParallelMPI4Py import vtkMPI4PyCommunicator
except ImportError:
    vtkMultiProcessController = None
    vtkMPI4PyCommunicator = None

if sys.hexversion < 0x03000000:
    izip = itertools.izip
    def next(itr):
        return itr.next()
else:
    izip = zip

def _apply_func2(func, array, args):
    """Apply a function to each member of a VTKCompositeDataArray.
    Returns a list of arrays.

    Note that this function is mainly for internal use by this module."""
    if array is dsa.NoneArray:
        return []
    res = []
    for a in array.Arrays:
        if a is dsa.NoneArray:
            res.append(dsa.NoneArray)
        else:
            res.append(func(a, *args))
    return res

def apply_ufunc(func, array, args=()):
    """Apply a function to each member of a VTKCompositeDataArray.
    VTKArray and numpy arrays are also supported."""
    if array is dsa.NoneArray:
        return dsa.NoneArray
    elif type(array) == dsa.VTKCompositeDataArray:
        return dsa.VTKCompositeDataArray(_apply_func2(func, array, args), dataset = array.DataSet)
    else:
        return func(array)

def _make_ufunc(ufunc):
    """ Given a ufunc, creates a closure that applies it to each member
    of a VTKCompositeDataArray.

    Note that this function is mainly for internal use by this module."""
    def new_ufunc(array):
        return apply_ufunc(ufunc, array, ())
    return new_ufunc

def apply_dfunc(dfunc, array1, val2):
    """Apply a two argument function to each member of a VTKCompositeDataArray
    and another argument The second argument can be a VTKCompositeDataArray, in
    which case a one-to-one match between arrays is assumed. Otherwise, the
    function is applied to the composite array with the second argument repeated.
    VTKArray and numpy arrays are also supported."""
    if type(array1) == dsa.VTKCompositeDataArray and type(val2) == dsa.VTKCompositeDataArray:
        res = []
        for a1, a2 in izip(array1.Arrays, val2.Arrays):
            if a1 is dsa.NoneArray or a2 is dsa.NoneArray:
                res.append(dsa.NoneArray)
            else:
                l = dsa.reshape_append_ones(a1, a2)
                res.append(dfunc(l[0], l[1]))
        return dsa.VTKCompositeDataArray(res, dataset = array1.DataSet)
    elif type(array1) == dsa.VTKCompositeDataArray:
        res = []
        for a in array1.Arrays :
            if a is dsa.NoneArray:
                res.append(dsa.NoneArray)
            else:
                l = dsa.reshape_append_ones(a, val2)
                res.append(dfunc(l[0], l[1]))
        return dsa.VTKCompositeDataArray(res, dataset = array1.DataSet)
    elif array1 is dsa.NoneArray:
        return dsa.NoneArray
    else:
        l = dsa.reshape_append_ones(array1, val2)
        return dfunc(l[0], l[1])

def _make_dfunc(dfunc):
    """ Given a function that requires two arguments, creates a closure that
    applies it to each member of a VTKCompositeDataArray.

    Note that this function is mainly for internal use by this module."""
    def new_dfunc(array1, val2):
        return apply_dfunc(dfunc, array1, val2)
    return new_dfunc

def _make_dsfunc(dsfunc):
    """ Given a function that requires two arguments (one array, one dataset),
    creates a closure that applies it to each member of a VTKCompositeDataArray.
    Note that this function is mainly for internal use by this module."""
    def new_dsfunc(array, ds=None):
        if type(array) == dsa.VTKCompositeDataArray:
            res = []
            for a in array.Arrays:
                if a is dsa.NoneArray:
                    res.append(dsa.NoneArray)
                else:
                    res.append(dsfunc(a, ds))
            return dsa.VTKCompositeDataArray(res, dataset = array.DataSet)
        elif array is dsa.NoneArray:
            return dsa.NoneArray
        else:
            return dsfunc(array, ds)
    return new_dsfunc

def _make_dsfunc2(dsfunc):
    """ Given a function that requires a dataset, creates a closure that
    applies it to each member of a VTKCompositeDataArray.

    Note that this function is mainly for internal use by this module."""
    def new_dsfunc2(ds):
        if type(ds) == dsa.CompositeDataSet:
            res = []
            for dataset in ds:
                res.append(dsfunc(dataset))
            return dsa.VTKCompositeDataArray(res, dataset = ds)
        else:
            return dsfunc(ds)
    return new_dsfunc2

def _lookup_mpi_type(ntype):
    from mpi4py import MPI
    if ntype == numpy.bool:
        typecode = 'b'
    else:
        typecode = numpy.dtype(ntype).char
    try:
        return MPI.__TypeDict__[typecode]
    except AttributeError:
        # handle mpi4py 2.0.0
        return MPI._typedict[typecode]

def _reduce_dims(array, comm):
    from mpi4py import MPI
    dims = numpy.array([0, 0], dtype=numpy.int32)
    if array is not dsa.NoneArray:
        shp = shape(array)
        if len(shp) == 0:
            dims = numpy.array([1, 0], dtype=numpy.int32)
        elif len(shp) == 1:
            dims = numpy.array([shp[0], 0], dtype=numpy.int32)
        else:
            dims = numpy.array(shp, dtype=numpy.int32)
    max_dims = numpy.array(dims, dtype=numpy.int32)
    mpitype = _lookup_mpi_type(numpy.int32)
    comm.Allreduce([dims, mpitype], [max_dims, mpitype], MPI.MAX)

    if max_dims[1] == 0:
        max_dims = numpy.array((max_dims[0],))
        size = max_dims[0]
    else:
        size = max_dims[0]*max_dims[1]

    if max_dims[0] == 1:
        max_dims = 1

    return (max_dims, size)

def _global_func(impl, array, axis, controller):
    if type(array) == dsa.VTKCompositeDataArray:
        if axis is None or axis == 0:
            res = impl.serial_composite(array, axis)
        else:
            res = apply_ufunc(impl.op(), array, (axis,))
    else:
        res = impl.op()(array, axis)
        if res is not dsa.NoneArray:
            res = res.astype(numpy.float64)

    if axis is None or axis == 0:
        if controller is None and vtkMultiProcessController is not None:
            controller = vtkMultiProcessController.GetGlobalController()
        if controller and controller.IsA("vtkMPIController"):
            from mpi4py import MPI
            comm = vtkMPI4PyCommunicator.ConvertToPython(controller.GetCommunicator())

            max_dims, size = _reduce_dims(res, comm)

            # All NoneArrays
            if size == 0:
                return dsa.NoneArray;

            if res is dsa.NoneArray:
                if max_dims is 1:
                    # Weird trick to make the array look like a scalar
                    max_dims = ()
                res = numpy.empty(max_dims)
                res.fill(impl.default())

            res_recv = numpy.array(res)
            mpi_type = _lookup_mpi_type(res.dtype)
            comm.Allreduce([res, mpi_type], [res_recv, mpi_type], impl.mpi_op())
            if array is dsa.NoneArray:
                return dsa.NoneArray
            res = res_recv

    return res

def bitwise_or(array1, array2):
    """Implements element by element or (bitwise, | in C/C++) operation.
    If one of the arrays is a NoneArray, this will return the array
    that is not NoneArray, treating NoneArray as 0 in the or operation."""
    if type(array1) == dsa.VTKCompositeDataArray and type(array2) == dsa.VTKCompositeDataArray:
        res = []
        for a1, a2 in izip(array1.Arrays, array2.Arrays):
            l = dsa.reshape_append_ones(a1, a2)
            res.append(bitwise_or(l[0], l[1]))
        return dsa.VTKCompositeDataArray(res, dataset = array1.DataSet)
    elif type(array1) == dsa.VTKCompositeDataArray:
        res = []
        for a in array1.Arrays :
            l = dsa.reshape_append_ones(a, array2)
            res.append(bitwise_or(l[0], l[1]))
        return dsa.VTKCompositeDataArray(res, dataset = array1.DataSet)
    elif array1 is dsa.NoneArray:
        return array2
    elif array2 is dsa.NoneArray:
        return array1
    else:
        l = dsa.reshape_append_ones(array1, array2)
        return numpy.bitwise_or(l[0], l[1])

def make_point_mask_from_NaNs(dataset, array):
    """This method will create a ghost array corresponding to an
    input with NaN values. For each NaN value, the output array will
    have a corresponding value of vtk.vtkDataSetAttributes.HIDDENPOINT.
    These values are also combined with any ghost values that the
    dataset may have."""
    from vtk import vtkDataSetAttributes
    ghosts = dataset.PointData[vtkDataSetAttributes.GhostArrayName()]
    return make_mask_from_NaNs(array, ghosts)

def make_cell_mask_from_NaNs(dataset, array):
    """This method will create a ghost array corresponding to an
    input with NaN values. For each NaN value, the output array will
    have a corresponding value of vtk.vtkDataSetAttributes.HIDDENCELL.
    These values are also combined with any ghost values that the
    dataset may have."""
    from vtk import vtkDataSetAttributes
    ghosts = dataset.CellData[vtkDataSetAttributes.GhostArrayName()]
    return make_mask_from_NaNs(array, ghosts, True)

def make_mask_from_NaNs(array, ghost_array=dsa.NoneArray, is_cell=False):
    """This method will create a ghost array corresponding to an
    input with NaN values. For each NaN value, the output array will
    have a corresponding value of vtk.vtkDataSetAttributes.HIDDENPOINT or
    HIDDENCELL is the is_cell argument is true. If an input ghost_array
    is passed, the array is bitwise_or'ed with it, simply adding
    the new ghost values to it."""
    from vtk import vtkDataSetAttributes
    if is_cell:
        mask_value = vtkDataSetAttributes.HIDDENCELL
    else:
        mask_value = vtkDataSetAttributes.HIDDENPOINT

    return bitwise_or(isnan(array).astype(numpy.uint8) * mask_value,
        ghost_array)

def sum(array, axis=None, controller=None):
    """Returns the sum of all values along a particular axis (dimension).
    Given an array of m tuples and n components:
    * Default is to return the sum of all values in an array.
    * axis=0: Sum values of all components and return a one tuple,
      n-component array.
    * axis=1: Sum values of all components of each tuple and return an
      m-tuple, 1-component array.

    When called in parallel, this function will sum across processes
    when a controller argument is passed or the global controller is
    defined. To disable parallel summing when running in parallel, pass
    a dummy controller as follows:

    sum(array, controller=vtk.vtkDummyController()).
    """
    class SumImpl:
        def op(self):
            return algs.sum

        def mpi_op(self):
            from mpi4py import MPI
            return MPI.SUM

        def serial_composite(self, array, axis):
            res = None
            arrays = array.Arrays
            for a in arrays:
                if a is not dsa.NoneArray:
                    if res is None:
                        res = algs.sum(a, axis).astype(numpy.float64)
                    else:
                        res += algs.sum(a, axis)
            return res

        def default(self):
            return numpy.float64(0)

    return _global_func(SumImpl(), array, axis, controller)

def max(array, axis=None, controller=None):
    """Returns the max of all values along a particular axis (dimension).
    Given an array of m tuples and n components:
    * Default is to return the max of all values in an array.
    * axis=0: Return the max values of all tuples and return a
      one tuple, n-component array.
    * axis=1: Return the max values of all components of each tuple
      and return an m-tuple, 1-component array.

    When called in parallel, this function will compute the max across
    processes when a controller argument is passed or the global controller
    is defined. To disable parallel summing when running in parallel, pass a
    dummy controller as follows:

    max(array, controller=vtk.vtkDummyController()).
    """
    class MaxImpl:
        def op(self):
            return algs.max

        def mpi_op(self):
            from mpi4py import MPI
            return MPI.MAX

        def serial_composite(self, array, axis):
            res = _apply_func2(algs.max, array, (axis,))
            clean_list = []
            for a in res:
                if a is not dsa.NoneArray:
                    clean_list.append(a)
            if clean_list is []:
                return None
            return algs.max(clean_list, axis=0).astype(numpy.float64)

        def default(self):
            return numpy.finfo(numpy.float64).min

    return _global_func(MaxImpl(), array, axis, controller)

def min(array, axis=None, controller=None):
    """Returns the min of all values along a particular axis (dimension).
    Given an array of m tuples and n components:
    * Default is to return the min of all values in an array.
    * axis=0: Return the min values of all tuples and return a one
      tuple, n-component array.
    * axis=1: Return the min values of all components of each tuple and
      return an m-tuple, 1-component array.

    When called in parallel, this function will compute the min across processes
    when a controller argument is passed or the global controller is defined.
    To disable parallel summing when running in parallel, pass a dummy controller as follows:

    min(array, controller=vtk.vtkDummyController()).
    """
    class MinImpl:
        def op(self):
            return algs.min

        def mpi_op(self):
            from mpi4py import MPI
            return MPI.MIN

        def serial_composite(self, array, axis):
            res = _apply_func2(algs.min, array, (axis,))
            clean_list = []
            for a in res:
                if a is not dsa.NoneArray:
                    clean_list.append(a)
            if clean_list is []:
                return None
            return algs.min(clean_list, axis=0).astype(numpy.float64)

        def default(self):
            return numpy.finfo(numpy.float64).max

    return _global_func(MinImpl(), array, axis, controller)

def _global_per_block(impl, array, axis=None, controller=None):
    if axis is not None and axis > 0:
        return impl.op()(array, axis=axis, controller=controller)

    try:
        dataset = array.DataSet
    except AttributeError:
        dataset = None

    t = type(array)
    if t == dsa.VTKArray or t == numpy.ndarray:
        from vtk.vtkCommonDataModel import vtkMultiBlockDataSet
        array = dsa.VTKCompositeDataArray([array])
        ds = vtkMultiBlockDataSet()
        ds.SetBlock(0, dataset.VTKObject)
        dataset = ds

    results = _apply_func2(impl.op2(), array, (axis,))

    if controller is None and vtkMultiProcessController is not None:
        controller = vtkMultiProcessController.GetGlobalController()
    if controller and controller.IsA("vtkMPIController"):
        from mpi4py import MPI
        comm = vtkMPI4PyCommunicator.ConvertToPython(controller.GetCommunicator())

        # First determine the number of components to use
        # for reduction
        res = dsa.NoneArray
        for res in results:
            if res is not dsa.NoneArray:
                break

        max_dims, size = _reduce_dims(res, comm)

        # All NoneArrays
        if size == 0:
            return dsa.NoneArray;

        # Next determine the max id to use for reduction
        # operations

        # Get all ids from dataset, including empty ones.
        ids = []
        lmax_id = numpy.int32(0)
        if dataset is not None:
            it = dataset.NewIterator()
            it.UnRegister(None)
            it.SetSkipEmptyNodes(False)
            while not it.IsDoneWithTraversal():
                _id = it.GetCurrentFlatIndex()
                lmax_id = numpy.max((lmax_id, _id)).astype(numpy.int32)
                if it.GetCurrentDataObject() is not None:
                    ids.append(_id)
                it.GoToNextItem()
        max_id = numpy.array(0, dtype=numpy.int32)
        mpitype = _lookup_mpi_type(numpy.int32)
        comm.Allreduce([lmax_id, mpitype], [max_id, mpitype], MPI.MAX)

        has_ids = numpy.zeros(max_id+1, dtype=numpy.int32)
        for _id in ids:
            has_ids[_id] = 1
        id_count = numpy.array(has_ids)
        comm.Allreduce([has_ids, mpitype], [id_count, mpitype], MPI.SUM)

        if numpy.all(id_count <= 1):
            return dsa.VTKCompositeDataArray(results, dataset=dataset)

        # Now that we know which blocks are shared by more than
        # 1 rank. The ones that have a count of 2 or more.
        reduce_ids = []
        for _id in range(len(id_count)):
            if id_count[_id] > 1:
                reduce_ids.append(_id)

        to_reduce = len(reduce_ids)
        # If not block is shared, short circuit. No need to
        # communicate any more.
        if to_reduce == 0:
            return dsa.VTKCompositeDataArray(results, dataset=dataset)

        # Create the local array that will be used for
        # reduction. Set it to a value that won't effect
        # the reduction.
        lresults = numpy.empty(size*to_reduce)
        lresults.fill(impl.default())

        # Just get non-empty ids. Doing this again in case
        # the traversal above results in a different order.
        # We need the same order since we'll use izip below.
        if dataset is not None:
            it = dataset.NewIterator()
            it.UnRegister(None)
            ids = []
            while not it.IsDoneWithTraversal():
                ids.append(it.GetCurrentFlatIndex())
                it.GoToNextItem()

        # Fill the local array with available values.
        for _id, _res in izip(ids, results):
            success = True
            try:
                loc = reduce_ids.index(_id)
            except ValueError:
                success = False
            if success:
                if _res is not dsa.NoneArray:
                    lresults[loc*size:(loc+1)*size] = _res.flatten()

        # Now do the MPI reduction.
        rresults = numpy.array(lresults)
        mpitype = _lookup_mpi_type(numpy.double)
        comm.Allreduce([lresults, mpitype], [rresults, mpitype], impl.mpi_op())

        if array is dsa.NoneArray:
            return dsa.NoneArray

        # Fill in the reduced values.
        for i in range(to_reduce):
            _id = reduce_ids[i]
            success = True
            try:
                loc = ids.index(_id)
            except ValueError:
                success = False
            if success:
                if size == 1:
                    results[loc] = dsa.VTKArray(rresults[i])
                else:
                    results[loc] = rresults[i*size:(i+1)*size].reshape(max_dims)

    return dsa.VTKCompositeDataArray(results, dataset=dataset)

def sum_per_block(array, axis=None, controller=None):
    """Returns the sum of all values along a particular axis (dimension) for
    each block of an VTKCompositeDataArray.

    Given an array of m tuples and n components:
    * Default is to return the sum of all values in an array.
    * axis=0: Sum values of all components and return a one tuple,
      n-component array.
    * axis=1: Sum values of all components of each tuple and return an
      m-tuple, 1-component array.

    When called in parallel, this function will sum across processes
    when a controller argument is passed or the global controller is
    defined. To disable parallel summing when running in parallel, pass
    a dummy controller as follows:

    sum_per_block(array, controller=vtk.vtkDummyController()).
    """
    class SumPerBlockImpl:
        def op(self):
            return sum

        def op2(self):
            return algs.sum

        def mpi_op(self):
            from mpi4py import MPI
            return MPI.SUM

        def default(self):
            return numpy.float64(0)

    return _global_per_block(SumPerBlockImpl(), array, axis, controller)

def count_per_block(array, axis=None, controller=None):
    """Return the number of elements of each block in a VTKCompositeDataArray
    along an axis.

    - if axis is None, the number of all elements (ntuples * ncomponents) is
    returned.
    - if axis is 0, the number of tuples is returned.
    """

    if axis is not None and axis > 0:
        raise ValueError("Only axis=None and axis=0 are supported for count")

    class CountPerBlockImpl:
        def op(self):
            return _array_count

        def op2(self):
            return _local_array_count

        def mpi_op(self):
            from mpi4py import MPI
            return MPI.SUM

        def default(self):
            return numpy.float64(0)

    return _global_per_block(CountPerBlockImpl(), array, axis, controller)

def mean_per_block(array, axis=None, controller=None):
    """Returns the mean of all values along a particular axis (dimension)
    for each block of a VTKCompositeDataArray.

    Given an array of m tuples and n components:
    * Default is to return the mean of all values in an array.
    * axis=0: Return the mean values of all components and return a one
      tuple, n-component array.
    * axis=1: Return the mean values of all components of each tuple and
      return an m-tuple, 1-component array.

    When called in parallel, this function will compute the mean across
    processes when a controller argument is passed or the global controller
    is defined. To disable parallel summing when running in parallel, pass a
    dummy controller as follows:

    mean(array, controller=vtk.vtkDummyController()).
    """
    if axis is None or axis == 0:
        return sum_per_block(array, axis, controller) / count_per_block(array, axis, controller)
    else:
        return sum(array, axis, controller)

def max_per_block(array, axis=None, controller=None):
    """Returns the max of all values along a particular axis (dimension)
    for each block of a VTKCompositeDataArray.
    Given an array of m tuples and n components:
    * Default is to return the max of all values in an array.
    * axis=0: Return the max values of all components and return a one
      tuple, n-component array.
    * axis=1: Return the max values of all components of each tuple and return
      an m-tuple, 1-component array.

    When called in parallel, this function will compute the max across
    processes when a controller argument is passed or the global controller
    is defined. To disable parallel summing when running in parallel, pass a
    dummy controller as follows:

    max_per_block(array, controller=vtk.vtkDummyController()).
    """
    class MaxPerBlockImpl:
        def op(self):
            return max

        def op2(self):
            return algs.max

        def mpi_op(self):
            from mpi4py import MPI
            return MPI.MAX

        def default(self):
            return numpy.finfo(numpy.float64).min

    return _global_per_block(MaxPerBlockImpl(), array, axis, controller)

def min_per_block(array, axis=None, controller=None):
    """Returns the min of all values along a particular axis (dimension)
    for each block of a VTKCompositeDataArray.
    Given an array of m tuples and n components:
    * Default is to return the min of all values in an array.
    * axis=0: Return the min values of all components and return a one
      tuple, n-component array.
    * axis=1: Return the min values of all components of each tuple and
      return an m-tuple, 1-component array.

    When called in parallel, this function will compute the min across
    processes when a controller argument is passed or the global controller
    is defined. To disable parallel summing when running in parallel, pass a
    dummy controller as follows:

    min_per_block(array, controller=vtk.vtkDummyController()).
    """
    class MinPerBlockImpl:
        def op(self):
            return min

        def op2(self):
            return algs.min

        def mpi_op(self):
            from mpi4py import MPI
            return MPI.MIN

        def default(self):
            return numpy.finfo(numpy.float64).max

    return _global_per_block(MinPerBlockImpl(), array, axis, controller)

def all(array, axis=None, controller=None):
    """Returns True if all values of an array evaluate to True, returns
    False otherwise.
    This is useful to check if all values of an array match a certain
    condition such as:

    algorithms.all(array > 5)
    """
    class MinImpl:
        def op(self):
            return algs.all

        def mpi_op(self):
            from mpi4py import MPI
            return MPI.LAND

        def serial_composite(self, array, axis):
            res = _apply_func2(algs.all, array, (axis,))
            clean_list = []
            for a in res:
                if a is not dsa.NoneArray:
                    clean_list.append(a)
            if clean_list is []:
                return None
            return algs.all(clean_list, axis=0)

        def default(self, max_comps):
            return numpy.ones(max_comps, dtype=numpy.bool)

    return _global_func(MinImpl(), array, axis, controller)

def _local_array_count(array, axis):

    if array is dsa.NoneArray:
        return numpy.int64(0)
    elif axis is None:
        return numpy.int64(array.size)
    else:
        return numpy.int64(shape(array)[0])

def _array_count(array, axis, controller):

    if array is dsa.NoneArray:
        size = numpy.int64(0)
    elif axis is None:
        size = numpy.int64(array.size)
    else:
        size = numpy.int64(shape(array)[0])

    if controller is None and vtkMultiProcessController is not None:
        controller = vtkMultiProcessController.GetGlobalController()

    if controller and controller.IsA("vtkMPIController"):
        from mpi4py import MPI
        comm = vtkMPI4PyCommunicator.ConvertToPython(controller.GetCommunicator())

        total_size = numpy.array(size, dtype=numpy.int64)
        mpitype = _lookup_mpi_type(numpy.int64)
        comm.Allreduce([size, mpitype], [total_size, mpitype], MPI.SUM)
        size = total_size

    return size

def mean(array, axis=None, controller=None, size=None):
    """Returns the mean of all values along a particular axis (dimension).
    Given an array of m tuples and n components:
    * Default is to return the mean of all values in an array.
    * axis=0: Return the mean values of all components and return a one
      tuple, n-component array.
    * axis=1: Return the mean values of all components of each tuple and
      return an m-tuple, 1-component array.

    When called in parallel, this function will compute the mean across
    processes when a controller argument is passed or the global controller
    is defined. To disable parallel summing when running in parallel, pass a
    dummy controller as follows:

    mean(array, controller=vtk.vtkDummyController()).
    """

    if axis is None or axis == 0:
        if size is None:
            size = _array_count(array, axis, controller)
        return sum(array, axis) / size
    else:
        if type(array) == dsa.VTKCompositeDataArray:
            return apply_ufunc(algs.mean, array, (axis,))
        else:
            return algs.mean(array, axis)

def var(array, axis=None, controller=None):
    """Returns the variance of all values along a particular axis (dimension).
    Given an array of m tuples and n components:
    * Default is to return the variance of all values in an array.
    * axis=0: Return the variance values of all components and return a one
      tuple, n-component array.
    * axis=1: Return the variance values of all components of each tuple and
      return an m-tuple, 1-component array.

    When called in parallel, this function will compute the variance across
    processes when a controller argument is passed or the global controller
    is defined. To disable parallel summing when running in parallel, pass a
    dummy controller as follows:

    var(array, controller=vtk.vtkDummyController()).
    """

    if axis is None or axis == 0:
        size = _array_count(array, axis, controller)
        tmp = array - mean(array, axis, controller, size)
        return sum(tmp*tmp, axis, controller) / size
    else:
        if type(array) == dsa.VTKCompositeDataArray:
            return apply_ufunc(algs.var, array, (axis,))
        else:
            return algs.var(array, axis)

def std(array, axis=None, controller=None):
    """Returns the standard deviation of all values along a particular
    axis (dimension).
    Given an array of m tuples and n components:
    * Default is to return the standard deviation of all values in an array.
    * axis=0: Return the standard deviation values of all components and
      return a one tuple, n-component array.
    * axis=1: Return the standard deviation values of all components of
      each tuple and return an m-tuple, 1-component array.

    When called in parallel, this function will compute the standard deviation
    across processes when a controller argument is passed or the global controller
    is defined. To disable parallel summing when running in parallel, pass a dummy
    controller as follows:

    std(array, controller=vtk.vtkDummyController()).
    """
    return sqrt(var(array, axis, controller))

def shape(array):
    "Returns the shape (dimensions) of an array."
    if type(array) == dsa.VTKCompositeDataArray:
        shp = None
        for a in array.Arrays:
            if a is not dsa.NoneArray:
                if shp is None:
                    shp = list(a.shape)
                else:
                    tmp = a.shape
                    if (len(shp) != len(tmp)):
                        raise ValueError("Expected arrays of same shape")
                    shp[0] += tmp[0]
                    for idx in range(1,len(tmp)):
                        if shp[idx] != tmp[idx]:
                            raise ValueError("Expected arrays of same shape")
        return tuple(shp)
    elif array is dsa.NoneArray:
        return ()
    else:
        return numpy.shape(array)

def make_vector(arrayx, arrayy, arrayz=None):
    """Given 2 or 3 scalar arrays, returns a vector array. If only
    2 scalars are provided, the third component will be set to 0."""
    if type(arrayx) == dsa.VTKCompositeDataArray and type(arrayy) == dsa.VTKCompositeDataArray and (type(arrayz) == dsa.VTKCompositeDataArray or arrayz is None):
        res = []
        if arrayz is None:
            for ax, ay in izip(arrayx.Arrays, arrayy.Arrays):
                if ax is not dsa.NoneArray and ay is not dsa.NoneArray:
                    res.append(algs.make_vector(ax, ay))
                else:
                    res.append(dsa.NoneArray)
        else:
            for ax, ay, az in izip(arrayx.Arrays, arrayy.Arrays, arrayz.Arrays):
                if ax is not dsa.NoneArray and ay is not dsa.NoneArray and az is not dsa.NoneArray:
                    res.append(algs.make_vector(ax, ay, az))
                else:
                    res.append(dsa.NoneArray)
        return dsa.VTKCompositeDataArray(res, dataset = arrayx.DataSet)
    else:
        return algs.make_vector(arrayx, arrayy, arrayz)

def unstructured_from_composite_arrays(points, arrays, controller=None):
    """Given a set of VTKCompositeDataArrays, creates a vtkUnstructuredGrid.
    The main goal of this function is to transform the output of XXX_per_block()
    methods to a single dataset that can be visualized and further processed.
    Here arrays is an iterable (e.g. list) of (array, name) pairs. Here is
    an example:

    centroid = mean_per_block(composite_data.Points)
    T = mean_per_block(composite_data.PointData['Temperature'])
    ug = unstructured_from_composite_arrays(centroid, (T, 'Temperature'))

    When called in parallel, this function makes sure that each array in
    the input dataset is represented only on 1 process. This is important
    because methods like mean_per_block() return the same value for blocks
    that are partitioned on all of the participating processes. If the
    same point were to be created across multiple processes in the output,
    filters like histogram would report duplicate values erroneously.
    """

    try:
        dataset = points.DataSet
    except AttributeError:
        dataset = None

    if dataset is None and points is not dsa.NoneArray:
        raise ValueError("Expecting a points arrays with an associated dataset.")

    if points is dsa.NoneArray:
        cpts = []
    else:
        cpts = points.Arrays
    ownership = numpy.zeros(len(cpts), dtype=numpy.int32)
    rank = 0

    # Let's first create a map of array index to composite ids.
    if dataset is None:
        ids = []
    else:
        it = dataset.NewIterator()
        it.UnRegister(None)
        itr = cpts.__iter__()
        ids = numpy.empty(len(cpts), dtype=numpy.int32)
        counter = 0
        while not it.IsDoneWithTraversal():
            _id = it.GetCurrentFlatIndex()
            ids[counter] = _id
            counter += 1
            it.GoToNextItem()

    if controller is None and vtkMultiProcessController is not None:
        controller = vtkMultiProcessController.GetGlobalController()
    if controller and controller.IsA("vtkMPIController"):
        from mpi4py import MPI
        comm = vtkMPI4PyCommunicator.ConvertToPython(controller.GetCommunicator())
        rank = comm.Get_rank()

        # Determine the max id to use for reduction
        # operations

        # Get all ids from dataset, including empty ones.
        lmax_id = numpy.int32(0)
        if dataset is not None:
            it = dataset.NewIterator()
            it.UnRegister(None)
            it.SetSkipEmptyNodes(False)
            while not it.IsDoneWithTraversal():
                _id = it.GetCurrentFlatIndex()
                lmax_id = numpy.max((lmax_id, _id)).astype(numpy.int32)
                it.GoToNextItem()
        max_id = numpy.array(0, dtype=numpy.int32)
        mpitype = _lookup_mpi_type(numpy.int32)
        comm.Allreduce([lmax_id, mpitype], [max_id, mpitype], MPI.MAX)

        # Now we figure out which processes have which ids
        lownership = numpy.empty(max_id, dtype = numpy.int32)
        lownership.fill(numpy.iinfo(numpy.int32).max)

        ownership = numpy.empty(max_id, dtype = numpy.int32)

        if dataset is not None:
            it = dataset.NewIterator()
            it.UnRegister(None)
            it.InitTraversal()
            itr = cpts.__iter__()
            while not it.IsDoneWithTraversal():
                _id = it.GetCurrentFlatIndex()
                if next(itr) is not dsa.NoneArray:
                    lownership[_id] = rank
                it.GoToNextItem()
        mpitype = _lookup_mpi_type(numpy.int32)
        # The process with the lowest id containing a block will
        # produce the output for that block.
        comm.Allreduce([lownership, mpitype], [ownership, mpitype], MPI.MIN)

    # Iterate over blocks to produce points and arrays
    from vtk.vtkCommonDataModel import vtkUnstructuredGrid
    from vtk.vtkCommonCore import vtkDoubleArray, vtkPoints
    ugrid = vtkUnstructuredGrid()
    da = vtkDoubleArray()
    da.SetNumberOfComponents(3)
    pts = vtkPoints()
    pts.SetData(da)
    counter = 0
    for pt in cpts:
        if ownership[ids[counter]] == rank:
            pts.InsertNextPoint(tuple(pt))
        counter += 1
    ugrid.SetPoints(pts)

    for ca, name in arrays:
        if ca is not dsa.NoneArray:
            da = vtkDoubleArray()
            ncomps = ca.Arrays[0].flatten().shape[0]
            da.SetNumberOfComponents(ncomps)
            counter = 0
            for a in ca.Arrays:
                if ownership[ids[counter]] == rank:
                    a = a.flatten()
                    for i in range(ncomps):
                        da.InsertNextValue(a[i])
                counter += 1
            if len(a) > 0:
                da.SetName(name)
                ugrid.GetPointData().AddArray(da)
    return ugrid

isnan = _make_ufunc(numpy.isnan)
isnan.__doc__ = "Returns a bool array, true if values is nan."

sqrt = _make_ufunc(numpy.sqrt)
sqrt.__doc__ = "Computes square root."

negative = _make_ufunc(numpy.negative)
negative.__doc__ = "Numerical negative, element-wise."

reciprocal = _make_ufunc(numpy.reciprocal)
reciprocal.__doc__ = "Return the reciprocal (1/x) of the argument, element-wise."

square = _make_ufunc(numpy.square)
square.__doc__ = "Return the element-wise square of the input."

exp = _make_ufunc(numpy.exp)
exp.__doc__ = "The exponential function."

floor = _make_ufunc(numpy.floor)
floor.__doc__ = "Returns the floor of floating point values."

ceil = _make_ufunc(numpy.ceil)
ceil.__doc__ = "Returns the ceiling of floating point values."

rint = _make_ufunc(numpy.rint)
rint.__doc__ = "Round elements of the array to the nearest integer."

sin = _make_ufunc(numpy.sin)
sin.__doc__ = "Computes sine of values in radians."

cos = _make_ufunc(numpy.cos)
cos.__doc__ = "Computes cosine of values in radians."

tan = _make_ufunc(numpy.tan)
tan.__doc__ = "Computes tangent of values in radians."

arcsin = _make_ufunc(numpy.arcsin)
arcsin.__doc__ = "Computes inverse sine."

arccos = _make_ufunc(numpy.arccos)
arccos.__doc__ = "Computes inverse cosine."

arctan = _make_ufunc(numpy.arctan)
arctan.__doc__ = "Computes inverse tangent."

arctan2 = _make_dfunc(numpy.arctan2)
arctan2.__doc__ = "Computes inverse tangent using two arguments."

sinh = _make_ufunc(numpy.sinh)
sinh.__doc__ = "Computes hyperbolic sine."

cosh = _make_ufunc(numpy.cosh)
cosh.__doc__ = "Computes hyperbolic cosine."

tanh = _make_ufunc(numpy.tanh)
tanh.__doc__ = "Computes hyperbolic tangent."

arcsinh = _make_ufunc(numpy.arcsinh)
arcsinh.__doc__ = "Computes inverse hyperbolic sine."

arccosh = _make_ufunc(numpy.arccosh)
arccosh.__doc__ = "Computes inverse hyperbolic cosine."

arctanh = _make_ufunc(numpy.arctanh)
arctanh.__doc__ = "Computes inverse hyperbolic tangent."

where = _make_ufunc(numpy.where)
where.__doc__ = """Returns the location (indices) of an array where the given
expression is true. For scalars, it returns a single array of indices.
For vectors and matrices, it returns two arrays: first with tuple indices,
second with component indices. The output of this method can be used to
extract the values from the array also by using it as the index of the [] operator.

For example:

>>> algs.where(algs.array([1,2,3]) == 2)
(array([1]),)

>>> algs.where(algs.array([[1,2,3], [2,1,1]]) == 2)
(array([0, 1]), array([1, 0]))

>>> a = array([[1,2,3], [2,1,1]])
>>> indices = algs.where(a > 2)
>>> a[indices]
array([3])
"""

flatnonzero = _make_ufunc(numpy.flatnonzero)
flatnonzero.__doc__ = "Return indices that are non-zero in the flattened version of the input array."

nonzero = _make_ufunc(numpy.nonzero)
nonzero.__doc__ = "Return the indices of the non-zero elements of the input array."

expand_dims = _make_dfunc(numpy.expand_dims)
expand_dims.__doc__ = """Insert a new dimension, corresponding to a given
position in the array shape. In VTK, this function's main use is to
enable an operator to work on a vector and a scalar field. For example,
say you want to divide each component of a vector by the magnitude of
that vector. You might try this:

>>> v
VTKArray([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> algs.mag(v)
VTKArray([ 1.73205081,  1.73205081,  1.73205081,  1.73205081,  1.73205081])
>>> v / algs.mag(v)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (5,3) (5)

The division operator does not know how to map a scalar to a vector
due to a mismatch in dimensions. This can be solved by making the
scalar a vector of 1 component (increasing its dimension to 2) as follows:

>>> v / algs.expand_dims(algs.mag(v), 1)
VTKArray([[ 0.57735027,  0.57735027,  0.57735027],
       [ 0.57735027,  0.57735027,  0.57735027],
       [ 0.57735027,  0.57735027,  0.57735027],
       [ 0.57735027,  0.57735027,  0.57735027],
       [ 0.57735027,  0.57735027,  0.57735027]])"""

abs = _make_ufunc(algs.abs)
abs.__doc__ = "Returns the absolute values of an array of scalars/vectors/tensors."

area = _make_dsfunc2(algs.area)
area.__doc__ = "Returns the surface area of each 2D cell in a mesh."

aspect = _make_dsfunc2(algs.aspect)
aspect.__doc__ = "Returns the aspect ratio of each cell in a mesh. See Verdict documentation for details."

aspect_gamma = _make_dsfunc2(algs.aspect_gamma)
aspect_gamma.__doc__ = "Returns the aspect gamma of each cell in a mesh. This metric compares root-mean-square edge length to volume. See Verdict documentation for details."

condition = _make_dsfunc2(algs.condition)
condition.__doc__ = "Returns the condition number of each cell in a mesh. See Verdict documentation for details."

cross = _make_dfunc(algs.cross)
cross.__doc__ = "Return the cross product of two vectors."

curl = _make_dsfunc(algs.curl)
curl.__doc__ = "Returns the curl a vector field."

divergence = _make_dsfunc(algs.divergence)
divergence.__doc__ = "Returns the divergence of a vector field."

det = _make_ufunc(algs.det)
det.__doc__ = "Returns the determinant of 2D matrices."

determinant = _make_ufunc(algs.determinant)
determinant.__doc__ = "Returns the determinant of 2D matrices."

diagonal = _make_dsfunc2(algs.diagonal)
diagonal.__doc__ = "Returns the diagonal length of each cell in a dataset. See Verdict documentation for details"

dot = _make_dfunc(algs.dot)
dot.__doc__ = "Returns the dot product of two vectors."

eigenvalue = _make_ufunc(algs.eigenvalue)
eigenvalue.__doc__ = "Returns the eigenvalues of 3x3 matrices. Currently only works with symmetric matrices."

eigenvector = _make_ufunc(algs.eigenvector)
eigenvector.__doc__ = "Returns the eigenvectors of 3x3 matrices. Currently only works with symmetric matrices."

gradient = _make_dsfunc(algs.gradient)
gradient.__doc__ = "Returns the gradient of  scalars or vectors."

inv = _make_ufunc(algs.inv)
inv.__doc__ = "Returns the inverse of 3x3 matrices."

inverse = _make_ufunc(algs.inverse)
inverse.__doc__ = "Returns the inverse of 3x3 matrices."

jacobian = _make_dsfunc2(algs.jacobian)
jacobian.__doc__ = "Returns the Jacobian of a dataset."

laplacian = _make_dsfunc(algs.laplacian)
laplacian.__doc__ = "Returns the Laplacian of a scalar field."

ln = _make_ufunc(algs.ln)
ln.__doc__ = "Returns the natural logarithm of its input."

log = _make_ufunc(algs.log)
log.__doc__ = "Returns the natural logarithm of its input."

log10 = _make_ufunc(algs.log10)
log10.__doc__ = "Returns the base 10 logarithm of its input."

max_angle = _make_dsfunc2(algs.max_angle)
max_angle.__doc__ = "Returns the maximum angle of each cell in a dataset. See Verdict documentation for details"

mag = _make_ufunc(algs.mag)
mag.__doc__ = "Returns the magnitude of vectors."

min_angle = _make_dsfunc2(algs.min_angle)
min_angle.__doc__ = "Returns the minimum angle of each cell in a dataset."

norm = _make_ufunc(algs.norm)
norm.__doc__ = "Computes the normalized values of vectors."

shear = _make_dsfunc2(algs.shear)
shear.__doc__ = "Returns the shear of each cell in a dataset. See Verdict documentation for details."

skew = _make_dsfunc2(algs.skew)
skew.__doc__ = "Returns the skew of each cell in a dataset. See Verdict documentation for details."

strain = _make_dsfunc(algs.strain)
strain.__doc__ = "Given a deformation vector, this function computes the infinitesimal (Cauchy) strain tensor. It can also be used to compute strain rate if the input is velocity."

surface_normal = _make_dsfunc2(algs.surface_normal)
surface_normal.__doc__ = "Returns the surface normal of each cell in a dataset."

trace = _make_ufunc(algs.trace)
trace.__doc__ = "Returns the trace of square matrices."

volume = _make_dsfunc2(algs.volume)
volume.__doc__ = "Returns the volume of each cell in a dataset. Use sum to calculate total volume of a dataset."

vorticity = _make_dsfunc(algs.vorticity)
vorticity.__doc__ = "Given a velocity field, calculates vorticity."

vertex_normal = _make_dsfunc2(algs.vertex_normal)
vertex_normal.__doc__ = "Returns the normal at each vertex of a dataset, which is defined as the average of the cell normals of all cells containing that vertex."

logical_not = _make_ufunc(numpy.logical_not)
logical_not.__doc__ = "Computes the truth value of NOT x element-wise."

divide = _make_dfunc(numpy.divide)
divide.__doc__ = "Element by element division. Both elements can be single values or arrays. Same as /."

multiply = _make_dfunc(numpy.multiply)
multiply.__doc__ = "Element by element multiplication. Both elements can be single values or arrays. Same as *."

add = _make_dfunc(numpy.add)
add.__doc__ = "Element by element addition. Both elements can be single values or arrays. Same as +."

subtract = _make_dfunc(numpy.subtract)
subtract.__doc__ = "Returns the difference of two values element-wise. Same as x - y."

mod = _make_dfunc(numpy.mod)
mod.__doc__ = "Computes x1 - floor(x1 / x2) * x2, the result has the same sign as the divisor x2. It is equivalent to the Python modulus operator x1 % x2. Same as remainder."

remainder = _make_dfunc(numpy.remainder)
remainder.__doc__ = "Computes x1 - floor(x1 / x2) * x2, the result has the same sign as the divisor x2. It is equivalent to the Python modulus operator x1 % x2. Same as mod."

power = _make_dfunc(numpy.power)
power.__doc__ = "First array elements raised to powers from second array, element-wise."

hypot = _make_dfunc(numpy.hypot)
hypot.__doc__ = "Given the 'legs' of a right triangle, return its hypotenuse."
