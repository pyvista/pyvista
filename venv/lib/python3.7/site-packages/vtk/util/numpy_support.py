"""This module adds support to easily import and export NumPy
(http://numpy.scipy.org) arrays into/out of VTK arrays.  The code is
loosely based on TVTK (https://svn.enthought.com/enthought/wiki/TVTK).

This code depends on an addition to the VTK data arrays made by Berk
Geveci to make it support Python's buffer protocol (on Feb. 15, 2008).

The main functionality of this module is provided by the two functions:
    numpy_to_vtk,
    vtk_to_numpy.


Caveats:
--------

 - Bit arrays in general do not have a numpy equivalent and are not
   supported.  Char arrays are also not easy to handle and might not
   work as you expect.  Patches welcome.

 - You need to make sure you hold a reference to a Numpy array you want
   to import into VTK.  If not you'll get a segfault (in the best case).
   The same holds in reverse when you convert a VTK array to a numpy
   array -- don't delete the VTK array.


Created by Prabhu Ramachandran in Feb. 2008.
"""

import vtk
import numpy

# Useful constants for VTK arrays.
VTK_ID_TYPE_SIZE = vtk.vtkIdTypeArray().GetDataTypeSize()
if VTK_ID_TYPE_SIZE == 4:
    ID_TYPE_CODE = numpy.int32
elif VTK_ID_TYPE_SIZE == 8:
    ID_TYPE_CODE = numpy.int64

VTK_LONG_TYPE_SIZE = vtk.vtkLongArray().GetDataTypeSize()
if VTK_LONG_TYPE_SIZE == 4:
    LONG_TYPE_CODE = numpy.int32
    ULONG_TYPE_CODE = numpy.uint32
elif VTK_LONG_TYPE_SIZE == 8:
    LONG_TYPE_CODE = numpy.int64
    ULONG_TYPE_CODE = numpy.uint64


def get_vtk_array_type(numpy_array_type):
    """Returns a VTK typecode given a numpy array."""
    # This is a Mapping from numpy array types to VTK array types.
    _np_vtk = {numpy.character:vtk.VTK_UNSIGNED_CHAR,
                numpy.uint8:vtk.VTK_UNSIGNED_CHAR,
                numpy.uint16:vtk.VTK_UNSIGNED_SHORT,
                numpy.uint32:vtk.VTK_UNSIGNED_INT,
                numpy.uint64:vtk.VTK_UNSIGNED_LONG_LONG,
                numpy.int8:vtk.VTK_CHAR,
                numpy.int16:vtk.VTK_SHORT,
                numpy.int32:vtk.VTK_INT,
                numpy.int64:vtk.VTK_LONG_LONG,
                numpy.float32:vtk.VTK_FLOAT,
                numpy.float64:vtk.VTK_DOUBLE,
                numpy.complex64:vtk.VTK_FLOAT,
                numpy.complex128:vtk.VTK_DOUBLE}
    for key, vtk_type in _np_vtk.items():
        if numpy_array_type == key or \
           numpy.issubdtype(numpy_array_type, key) or \
           numpy_array_type == numpy.dtype(key):
            return vtk_type
    raise TypeError(
        'Could not find a suitable VTK type for %s' % (str(numpy_array_type)))

def get_vtk_to_numpy_typemap():
    """Returns the VTK array type to numpy array type mapping."""
    _vtk_np = {vtk.VTK_BIT:numpy.bool,
                vtk.VTK_CHAR:numpy.int8,
                vtk.VTK_SIGNED_CHAR:numpy.int8,
                vtk.VTK_UNSIGNED_CHAR:numpy.uint8,
                vtk.VTK_SHORT:numpy.int16,
                vtk.VTK_UNSIGNED_SHORT:numpy.uint16,
                vtk.VTK_INT:numpy.int32,
                vtk.VTK_UNSIGNED_INT:numpy.uint32,
                vtk.VTK_LONG:LONG_TYPE_CODE,
                vtk.VTK_LONG_LONG:numpy.int64,
                vtk.VTK_UNSIGNED_LONG:ULONG_TYPE_CODE,
                vtk.VTK_UNSIGNED_LONG_LONG:numpy.uint64,
                vtk.VTK_ID_TYPE:ID_TYPE_CODE,
                vtk.VTK_FLOAT:numpy.float32,
                vtk.VTK_DOUBLE:numpy.float64}
    return _vtk_np


def get_numpy_array_type(vtk_array_type):
    """Returns a numpy array typecode given a VTK array type."""
    return get_vtk_to_numpy_typemap()[vtk_array_type]


def create_vtk_array(vtk_arr_type):
    """Internal function used to create a VTK data array from another
    VTK array given the VTK array type.
    """
    return vtk.vtkDataArray.CreateDataArray(vtk_arr_type)


def numpy_to_vtk(num_array, deep=0, array_type=None):
    """Converts a real numpy Array to a VTK array object.

    This function only works for real arrays.
    Complex arrays are NOT handled.  It also works for multi-component
    arrays.  However, only 1, and 2 dimensional arrays are supported.
    This function is very efficient, so large arrays should not be a
    problem.

    If the second argument is set to 1, the array is deep-copied from
    from numpy. This is not as efficient as the default behavior
    (shallow copy) and uses more memory but detaches the two arrays
    such that the numpy array can be released.

    WARNING: You must maintain a reference to the passed numpy array, if
    the numpy data is gc'd and VTK will point to garbage which will in
    the best case give you a segfault.

    Parameters:

    num_array
      a 1D or 2D, real numpy array.

    """

    z = numpy.asarray(num_array)
    if not z.flags.contiguous:
        z = numpy.ascontiguousarray(z)

    shape = z.shape
    assert z.flags.contiguous, 'Only contiguous arrays are supported.'
    assert len(shape) < 3, \
           "Only arrays of dimensionality 2 or lower are allowed!"
    assert not numpy.issubdtype(z.dtype, complex), \
           "Complex numpy arrays cannot be converted to vtk arrays."\
           "Use real() or imag() to get a component of the array before"\
           " passing it to vtk."

    # First create an array of the right type by using the typecode.
    if array_type:
        vtk_typecode = array_type
    else:
        vtk_typecode = get_vtk_array_type(z.dtype)
    result_array = create_vtk_array(vtk_typecode)

    # Fixup shape in case its empty or scalar.
    try:
        testVar = shape[0]
    except:
        shape = (0,)

    # Find the shape and set number of components.
    if len(shape) == 1:
        result_array.SetNumberOfComponents(1)
    else:
        result_array.SetNumberOfComponents(shape[1])

    result_array.SetNumberOfTuples(shape[0])

    # Ravel the array appropriately.
    arr_dtype = get_numpy_array_type(vtk_typecode)
    if numpy.issubdtype(z.dtype, arr_dtype) or \
       z.dtype == numpy.dtype(arr_dtype):
        z_flat = numpy.ravel(z)
    else:
        z_flat = numpy.ravel(z).astype(arr_dtype)
        # z_flat is now a standalone object with no references from the caller.
        # As such, it will drop out of this scope and cause memory issues if we
        # do not deep copy its data.
        deep = 1

    # Point the VTK array to the numpy data.  The last argument (1)
    # tells the array not to deallocate.
    result_array.SetVoidArray(z_flat, len(z_flat), 1)
    if deep:
        copy = result_array.NewInstance()
        copy.DeepCopy(result_array)
        result_array = copy
    else:
        result_array._numpy_reference = z
    return result_array

def numpy_to_vtkIdTypeArray(num_array, deep=0):
    isize = vtk.vtkIdTypeArray().GetDataTypeSize()
    dtype = num_array.dtype
    if isize == 4:
        if dtype != numpy.int32:
            raise ValueError(
             'Expecting a numpy.int32 array, got %s instead.' % (str(dtype)))
    else:
        if dtype != numpy.int64:
            raise ValueError(
             'Expecting a numpy.int64 array, got %s instead.' % (str(dtype)))

    return numpy_to_vtk(num_array, deep, vtk.VTK_ID_TYPE)

def vtk_to_numpy(vtk_array):
    """Converts a VTK data array to a numpy array.

    Given a subclass of vtkDataArray, this function returns an
    appropriate numpy array containing the same data -- it actually
    points to the same data.

    WARNING: This does not work for bit arrays.

    Parameters

    vtk_array
      The VTK data array to be converted.

    """
    typ = vtk_array.GetDataType()
    assert typ in get_vtk_to_numpy_typemap().keys(), \
           "Unsupported array type %s"%typ
    assert typ != vtk.VTK_BIT, 'Bit arrays are not supported.'

    shape = vtk_array.GetNumberOfTuples(), \
            vtk_array.GetNumberOfComponents()

    # Get the data via the buffer interface
    dtype = get_numpy_array_type(typ)
    try:
        result = numpy.frombuffer(vtk_array, dtype=dtype)
    except ValueError:
        # http://mail.scipy.org/pipermail/numpy-tickets/2011-August/005859.html
        # numpy 1.5.1 (and maybe earlier) has a bug where if frombuffer is
        # called with an empty buffer, it throws ValueError exception. This
        # handles that issue.
        if shape[0] == 0:
            # create an empty array with the given shape.
            result = numpy.empty(shape, dtype=dtype)
        else:
            raise
    if shape[1] == 1:
        shape = (shape[0], )
    try:
        result.shape = shape
    except ValueError:
        if shape[0] == 0:
           # Refer to https://github.com/numpy/numpy/issues/2536 .
           # For empty array, reshape fails. Create the empty array explicitly
           # if that happens.
           result = numpy.empty(shape, dtype=dtype)
        else: raise
    return result
