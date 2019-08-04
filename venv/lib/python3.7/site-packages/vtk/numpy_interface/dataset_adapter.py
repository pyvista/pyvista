"""This module provides classes that allow Numpy-type access
to VTK datasets and arrays. This is best described with some examples.

To normalize a VTK array:

import vtk
import vtk.numpy_interface.dataset_adapter as dsa
import vtk.numpy_interface.algorithms as algs

rt = vtk.vtkRTAnalyticSource()
rt.Update()
image = dsa.WrapDataObject(rt.GetOutput())
rtdata = image.PointData['RTData']
rtmin = algs.min(rtdata)
rtmax = algs.max(rtdata)
rtnorm = (rtdata - rtmin) / (rtmax - rtmin)
image.PointData.append(rtnorm, 'RTData - normalized')
print image.GetPointData().GetArray('RTData - normalized').GetRange()

To calculate gradient:

grad= algs.gradient(rtnorm)

To access subsets:

>>> grad[0:10]
VTKArray([[ 0.10729134,  0.03763443,  0.03136338],
       [ 0.02754352,  0.03886006,  0.032589  ],
       [ 0.02248248,  0.04127144,  0.03500038],
       [ 0.02678365,  0.04357527,  0.03730421],
       [ 0.01765099,  0.04571581,  0.03944477],
       [ 0.02344007,  0.04763837,  0.04136734],
       [ 0.01089381,  0.04929155,  0.04302051],
       [ 0.01769151,  0.05062952,  0.04435848],
       [ 0.002764  ,  0.05161414,  0.04534309],
       [ 0.01010841,  0.05221677,  0.04594573]])

>>> grad[:, 0]
VTKArray([ 0.10729134,  0.02754352,  0.02248248, ..., -0.02748174,
       -0.02410045,  0.05509736])

All of this functionality is also supported for composite datasets
even though their data arrays may be spread across multiple datasets.
We have implemented a VTKCompositeDataArray class that handles many
Numpy style operators and is supported by all algorithms in the
algorithms module.

This module also provides an API to access composite datasets.
For example:

mb = vtk.vtkMultiBlockDataSet()
mb.SetBlock(0, image.VTKObject)
mb.SetBlock(1e, image.VTKObject)
cds = dsa.WrapDataObject(mb)
for block in cds:
    print block

Note that this module implements only the wrappers for datasets
and arrays. The classes implement many useful operators. However,
to make best use of these classes, take a look at the algorithms
module.
"""
try:
    import numpy
except ImportError:
    raise RuntimeError("This module depends on the numpy module. Please make\
sure that it is installed properly.")

import itertools
import operator
import sys
from vtk import buffer_shared
from vtk.util import numpy_support
from vtk.vtkCommonDataModel import vtkDataObject
from vtk.vtkCommonCore import vtkWeakReference
import weakref

if sys.hexversion < 0x03000000:
    izip = itertools.izip
else:
    izip = zip

def reshape_append_ones (a1, a2):
    """Returns a list with the two arguments, any of them may be
    processed.  If the arguments are numpy.ndarrays, append 1s to the
    shape of the array with the smallest number of dimensions until
    the arrays have the same number of dimensions. Does nothing if the
    arguments are not ndarrays or the arrays have the same number of
    dimensions.

    """
    l = [a1, a2]
    if (isinstance(a1, numpy.ndarray) and isinstance(a2, numpy.ndarray)):
        len1 = len(a1.shape)
        len2 = len(a2.shape)
        if (len1 == len2 or len1 == 0 or len2 == 0 or
            a1.shape[0] != a2.shape[0]):
            return l;
        elif (len1 < len2):
            d = len1
            maxLength = len2
            i = 0
        else:
            d = len2
            maxLength = len1
            i = 1
        while (d < maxLength):
            l[i] = numpy.expand_dims(l[i], d)
            d = d + 1
    return l

class ArrayAssociation :
    """Easy access to vtkDataObject.AttributeTypes"""
    POINT = vtkDataObject.POINT
    CELL  = vtkDataObject.CELL
    FIELD = vtkDataObject.FIELD
    ROW = vtkDataObject.ROW

class VTKObjectWrapper(object):
    """Superclass for classes that wrap VTK objects with Python objects.
    This class holds a reference to the wrapped VTK object. It also
    forwards unresolved methods to the underlying object by overloading
    __get__attr."""
    def __init__(self, vtkobject):
        self.VTKObject = vtkobject

    def __getattr__(self, name):
        "Forwards unknown attribute requests to VTK object."
        return getattr(self.VTKObject, name)

def vtkDataArrayToVTKArray(array, dataset=None):
    "Given a vtkDataArray and a dataset owning it, returns a VTKArray."
    narray = numpy_support.vtk_to_numpy(array)

    # Make arrays of 9 components into matrices. Also transpose
    # as VTK store matrices in Fortran order
    shape = narray.shape
    if len(shape) == 2 and shape[1] == 9:
        narray = narray.reshape((shape[0], 3, 3)).transpose(0, 2, 1)

    return VTKArray(narray, array=array, dataset=dataset)

def numpyTovtkDataArray(array, name="numpy_array", array_type=None):
    """Given a numpy array or a VTKArray and a name, returns a vtkDataArray.
    The resulting vtkDataArray will store a reference to the numpy array:
    the numpy array is released only when the vtkDataArray is destroyed."""
    vtkarray = numpy_support.numpy_to_vtk(array, array_type=array_type)
    vtkarray.SetName(name)
    return vtkarray

def _make_tensor_array_contiguous(array):
    if array is None:
        return None
    if array.flags.contiguous:
        return array
    array = numpy.asarray(array)
    size = array.dtype.itemsize
    strides = array.strides
    if len(strides) == 3 and strides[1]/size == 1 and strides[2]/size == 3:
        return array.transpose(0, 2, 1)
    return array

def _metaclass(mcs):
    """For compatibility between python 2 and python 3."""
    def decorator(cls):
        body = vars(cls).copy()
        body.pop('__dict__', None)
        body.pop('__weakref__', None)
        return mcs(cls.__name__, cls.__bases__, body)
    return decorator

class VTKArrayMetaClass(type):
    def __new__(mcs, name, parent, attr):
        """We overwrite numerical/comparison operators because we might need
        to reshape one of the arrays to perform the operation without
        broadcast errors. For instace:

        An array G of shape (n,3) resulted from computing the
        gradient on a scalar array S of shape (n,) cannot be added together without
        reshaping.
        G + expand_dims(S,1) works,
        G + S gives an error:
        ValueError: operands could not be broadcast together with shapes (n,3) (n,)

        This metaclass overwrites operators such that it computes this
        reshape operation automatically by appending 1s to the
        dimensions of the array with fewer dimensions.

        """
        def add_numeric_op(attr_name):
            """Create an attribute named attr_name that calls
            _numeric_op(self, other, op)."""
            def closure(self, other):
                return VTKArray._numeric_op(self, other, attr_name)
            closure.__name__ = attr_name
            attr[attr_name] = closure

        def add_default_numeric_op(op_name):
            """Adds '__[op_name]__' attribute that uses operator.[op_name]"""
            add_numeric_op("__%s__"%op_name)

        def add_reverse_numeric_op(attr_name):
            """Create an attribute named attr_name that calls
            _reverse_numeric_op(self, other, op)."""
            def closure(self, other):
                return VTKArray._reverse_numeric_op(self, other, attr_name)
            closure.__name__ = attr_name
            attr[attr_name] = closure

        def add_default_reverse_numeric_op(op_name):
            """Adds '__r[op_name]__' attribute that uses operator.[op_name]"""
            add_reverse_numeric_op("__r%s__"%op_name)

        def add_default_numeric_ops(op_name):
            """Call both add_default_numeric_op and add_default_reverse_numeric_op."""
            add_default_numeric_op(op_name)
            add_default_reverse_numeric_op(op_name)

        add_default_numeric_ops("add")
        add_default_numeric_ops("sub")
        add_default_numeric_ops("mul")
        if sys.hexversion < 0x03000000:
            add_default_numeric_ops("div")
        add_default_numeric_ops("truediv")
        add_default_numeric_ops("floordiv")
        add_default_numeric_ops("mod")
        add_default_numeric_ops("pow")
        add_default_numeric_ops("lshift")
        add_default_numeric_ops("rshift")
        add_numeric_op("and")
        add_default_numeric_ops("xor")
        add_numeric_op("or")

        add_default_numeric_op("lt")
        add_default_numeric_op("le")
        add_default_numeric_op("eq")
        add_default_numeric_op("ne")
        add_default_numeric_op("ge")
        add_default_numeric_op("gt")
        return type.__new__(mcs, name, parent, attr)

@_metaclass(VTKArrayMetaClass)
class VTKArray(numpy.ndarray):
    """This is a sub-class of numpy ndarray that stores a
    reference to a vtk array as well as the owning dataset.
    The numpy array and vtk array should point to the same
    memory location."""

    def _numeric_op(self, other, attr_name):
        """Used to implement numpy-style numerical operations such as __add__,
        __mul__, etc."""
        l = reshape_append_ones(self, other)
        return getattr(numpy.ndarray, attr_name)(l[0], l[1])

    def _reverse_numeric_op(self, other, attr_name):
        """Used to implement numpy-style numerical operations such as __add__,
        __mul__, etc."""
        l = reshape_append_ones(self, other)
        return getattr(numpy.ndarray, attr_name)(l[0], l[1])

    def __new__(cls, input_array, array=None, dataset=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        obj.Association = ArrayAssociation.FIELD
        # add the new attributes to the created instance
        obj.VTKObject = array
        if dataset:
            obj._dataset = vtkWeakReference()
            obj._dataset.Set(dataset.VTKObject)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self,obj):
        # Copy the VTK array only if the two share data
        slf = _make_tensor_array_contiguous(self)
        obj2 = _make_tensor_array_contiguous(obj)

        self.VTKObject = None
        try:
            # This line tells us that they are referring to the same buffer.
            # Much like two pointers referring to same memory location in C/C++.
            if buffer_shared(slf, obj2):
                self.VTKObject = getattr(obj, 'VTKObject', None)
        except TypeError:
            pass

        self.Association = getattr(obj, 'Association', None)
        self.DataSet = getattr(obj, 'DataSet', None)

    def __getattr__(self, name):
        "Forwards unknown attribute requests to VTK array."
        try:
            o = self.__dict__["VTKObject"]
        except KeyError:
            o = None
        if o is None:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                 (self.__class__.__name__, name))
        return getattr(o, name)

    def __array_wrap__(self, out_arr, context=None):
        if out_arr.shape == ():
            # Convert to scalar value
            return out_arr[()]
        else:
            return numpy.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def DataSet(self):
        """
        Get the dataset this array is associated with. The reference to the
        dataset is held through a vtkWeakReference to ensure it doesn't prevent
        the dataset from being collected if necessary.
        """
        if hasattr(self, '_dataset') and self._dataset and self._dataset.Get():
            return WrapDataObject(self._dataset.Get())

        return  None

    @DataSet.setter
    def DataSet(self, dataset):
        """
        Set the dataset this array is associated with. The reference is held
        through a vtkWeakReference.
        """
        # Do we have dataset to store
        if dataset and dataset.VTKObject:
            # Do we need to create a vtkWeakReference
            if not hasattr(self, '_dataset') or self._dataset is None:
                self._dataset = vtkWeakReference()

            self._dataset.Set(dataset.VTKObject)
        else:
            self._dataset = None

class VTKNoneArrayMetaClass(type):
    def __new__(mcs, name, parent, attr):
        """Simplify the implementation of the numeric/logical sequence API."""
        def _add_op(attr_name, op):
            """Create an attribute named attr_name that calls
            _numeric_op(self, other, op)."""
            def closure(self, other):
                return VTKNoneArray._op(self, other, op)
            closure.__name__ = attr_name
            attr[attr_name] = closure

        def _add_default_reverse_op(op_name):
            """Adds '__r[op_name]__' attribute that uses operator.[op_name]"""
            _add_op("__r%s__"%op_name, getattr(operator, op_name))

        def _add_default_op(op_name):
            """Adds '__[op_name]__' attribute that uses operator.[op_name]"""
            _add_op("__%s__"%op_name, getattr(operator, op_name))

        def _add_default_ops(op_name):
            """Call both add_default_numeric_op and add_default_reverse_numeric_op."""
            _add_default_op(op_name)
            _add_default_reverse_op(op_name)

        _add_default_ops("add")
        _add_default_ops("sub")
        _add_default_ops("mul")
        if sys.hexversion < 0x03000000:
            _add_default_ops("div")
        _add_default_ops("truediv")
        _add_default_ops("floordiv")
        _add_default_ops("mod")
        _add_default_ops("pow")
        _add_default_ops("lshift")
        _add_default_ops("rshift")
        _add_op("__and__", operator.and_)
        _add_op("__rand__", operator.and_)
        _add_default_ops("xor")
        _add_op("__or__", operator.or_)
        _add_op("__ror__", operator.or_)

        _add_default_op("lt")
        _add_default_op("le")
        _add_default_op("eq")
        _add_default_op("ne")
        _add_default_op("ge")
        _add_default_op("gt")
        return type.__new__(mcs, name, parent, attr)

@_metaclass(VTKNoneArrayMetaClass)
class VTKNoneArray(object):
    """VTKNoneArray is used to represent a "void" array. An instance
    of this class (NoneArray) is returned instead of None when an
    array that doesn't exist in a DataSetAttributes is requested.
    All operations on the NoneArray return NoneArray. The main reason
    for this is to support operations in parallel where one of the
    processes may be working on an empty dataset. In such cases,
    the process is still expected to evaluate a whole expression because
    some of the functions may perform bulk MPI communication. None
    cannot be used in these instances because it cannot properly override
    operators such as __add__, __sub__ etc. This is the main raison
    d'etre for VTKNoneArray."""

    def __getitem__(self, index):
        return NoneArray

    def _op(self, other, op):
        """Used to implement numpy-style numerical operations such as __add__,
        __mul__, etc."""
        return NoneArray

    def astype(self, dtype):
        """Implements numpy array's astype method."""
        return NoneArray

NoneArray = VTKNoneArray()

class VTKCompositeDataArrayMetaClass(type):
    def __new__(mcs, name, parent, attr):
        """Simplify the implementation of the numeric/logical sequence API."""
        def add_numeric_op(attr_name, op):
            """Create an attribute named attr_name that calls
            _numeric_op(self, other, op)."""
            def closure(self, other):
                return VTKCompositeDataArray._numeric_op(self, other, op)
            closure.__name__ = attr_name
            attr[attr_name] = closure

        def add_reverse_numeric_op(attr_name, op):
            """Create an attribute named attr_name that calls
            _reverse_numeric_op(self, other, op)."""
            def closure(self, other):
                return VTKCompositeDataArray._reverse_numeric_op(self, other, op)
            closure.__name__ = attr_name
            attr[attr_name] = closure

        def add_default_reverse_numeric_op(op_name):
            """Adds '__r[op_name]__' attribute that uses operator.[op_name]"""
            add_reverse_numeric_op("__r%s__"%op_name, getattr(operator, op_name))

        def add_default_numeric_op(op_name):
            """Adds '__[op_name]__' attribute that uses operator.[op_name]"""
            add_numeric_op("__%s__"%op_name, getattr(operator, op_name))

        def add_default_numeric_ops(op_name):
            """Call both add_default_numeric_op and add_default_reverse_numeric_op."""
            add_default_numeric_op(op_name)
            add_default_reverse_numeric_op(op_name)

        add_default_numeric_ops("add")
        add_default_numeric_ops("sub")
        add_default_numeric_ops("mul")
        if sys.hexversion < 0x03000000:
            add_default_numeric_ops("div")
        add_default_numeric_ops("truediv")
        add_default_numeric_ops("floordiv")
        add_default_numeric_ops("mod")
        add_default_numeric_ops("pow")
        add_default_numeric_ops("lshift")
        add_default_numeric_ops("rshift")
        add_numeric_op("__and__", operator.and_)
        add_reverse_numeric_op("__rand__", operator.and_)
        add_default_numeric_ops("xor")
        add_numeric_op("__or__", operator.or_)
        add_reverse_numeric_op("__ror__", operator.or_)

        add_default_numeric_op("lt")
        add_default_numeric_op("le")
        add_default_numeric_op("eq")
        add_default_numeric_op("ne")
        add_default_numeric_op("ge")
        add_default_numeric_op("gt")
        return type.__new__(mcs, name, parent, attr)

@_metaclass(VTKCompositeDataArrayMetaClass)
class VTKCompositeDataArray(object):
    """This class manages a set of arrays of the same name contained
    within a composite dataset. Its main purpose is to provide a
    Numpy-type interface to composite data arrays which are naturally
    nothing but a collection of vtkDataArrays. A VTKCompositeDataArray
    makes such a collection appear as a single Numpy array and support
    all array operations that this module and the associated algorithm
    module support. Note that this is not a subclass of a Numpy array
    and as such cannot be passed to native Numpy functions. Instead
    VTK modules should be used to process composite arrays.
    """

    def __init__(self, arrays = [], dataset = None, name = None,
                 association = None):
        """Construct a composite array given a container of
        arrays, a dataset, name and association. It is sufficient
        to define a container of arrays to define a composite array.
        It is also possible to initialize an array by defining
        the dataset, name and array association. In that case,
        the underlying arrays will be created lazily when they
        are needed. It is recommended to use the latter method
        when initializing from an existing composite dataset."""
        self._Arrays = arrays
        self.DataSet = dataset
        self.Name = name
        validAssociation = True
        if association == None:
            for array in self._Arrays:
                if hasattr(array, "Association"):
                    if association == None:
                        association = array.Association
                    elif array.Association and association != array.Association:
                        validAssociation = False
                        break
        if validAssociation:
            self.Association = association
        else:
            self.Association = ArrayAssociation.FIELD
        self.Initialized = False

    def __init_from_composite(self):
        if self.Initialized:
            return

        self.Initialized = True

        if self.DataSet is None or self.Name is None:
            return

        self._Arrays = []
        for ds in self.DataSet:
            self._Arrays.append(ds.GetAttributes(self.Association)[self.Name])

    def GetSize(self):
        "Returns the number of elements in the array."
        self.__init_from_composite()
        size = numpy.int64(0)
        for a in self._Arrays:
            try:
                size += a.size
            except AttributeError:
                pass
        return size

    size = property(GetSize)

    def GetArrays(self):
        """Returns the internal container of VTKArrays. If necessary,
        this will populate the array list from a composite dataset."""
        self.__init_from_composite()
        return self._Arrays

    Arrays = property(GetArrays)

    def __getitem__(self, index):
        """Overwritten to refer indexing to underlying VTKArrays.
        For the most part, this will behave like Numpy. Note
        that indexing is done per array - arrays are never treated
        as forming a bigger array. If the index is another composite
        array, a one-to-one mapping between arrays is assumed.
        """
        self.__init_from_composite()
        res = []
        if type(index) == VTKCompositeDataArray:
            for a, idx in izip(self._Arrays, index.Arrays):
                if a is not NoneArray:
                    res.append(a.__getitem__(idx))
                else:
                    res.append(NoneArray)
        else:
            for a in self._Arrays:
                if a is not NoneArray:
                    res.append(a.__getitem__(index))
                else:
                    res.append(NoneArray)
        return VTKCompositeDataArray(res, dataset=self.DataSet)

    def _numeric_op(self, other, op):
        """Used to implement numpy-style numerical operations such as __add__,
        __mul__, etc."""
        self.__init_from_composite()
        res = []
        if type(other) == VTKCompositeDataArray:
            for a1, a2 in izip(self._Arrays, other.Arrays):
                if a1 is not NoneArray and a2 is not NoneArray:
                    l = reshape_append_ones(a1, a2)
                    res.append(op(l[0],l[1]))
                else:
                    res.append(NoneArray)
        else:
            for a in self._Arrays:
                if a is not NoneArray:
                    l = reshape_append_ones(a, other)
                    res.append(op(l[0], l[1]))
                else:
                    res.append(NoneArray)
        return VTKCompositeDataArray(
            res, dataset=self.DataSet, association=self.Association)

    def _reverse_numeric_op(self, other, op):
        """Used to implement numpy-style numerical operations such as __add__,
        __mul__, etc."""
        self.__init_from_composite()
        res = []
        if type(other) == VTKCompositeDataArray:
            for a1, a2 in izip(self._Arrays, other.Arrays):
                if a1 is not NoneArray and a2 is notNoneArray:
                    l = reshape_append_ones(a2,a1)
                    res.append(op(l[0],l[1]))
                else:
                    res.append(NoneArray)
        else:
            for a in self._Arrays:
                if a is not NoneArray:
                    l = reshape_append_ones(other, a)
                    res.append(op(l[0], l[1]))
                else:
                    res.append(NoneArray)
        return VTKCompositeDataArray(
            res, dataset=self.DataSet, association = self.Association)

    def __str__(self):
        return self.Arrays.__str__()

    def astype(self, dtype):
        """Implements numpy array's as array method."""
        res = []
        if self is not NoneArray:
            for a in self.Arrays:
                if a is NoneArray:
                    res.append(NoneArray)
                else:
                    res.append(a.astype(dtype))
        return VTKCompositeDataArray(
            res, dataset = self.DataSet, association = self.Association)


class DataSetAttributes(VTKObjectWrapper):
    """This is a python friendly wrapper of vtkDataSetAttributes. It
    returns VTKArrays. It also provides the dictionary interface."""

    def __init__(self, vtkobject, dataset, association):
        super(DataSetAttributes, self).__init__(vtkobject)
        # import weakref
        # self.DataSet = weakref.ref(dataset)
        self.DataSet = dataset
        self.Association = association

    def __getitem__(self, idx):
        """Implements the [] operator. Accepts an array name or index."""
        return self.GetArray(idx)

    def GetArray(self, idx):
        "Given an index or name, returns a VTKArray."
        if isinstance(idx, int) and idx >= self.VTKObject.GetNumberOfArrays():
            raise IndexError("array index out of range")
        vtkarray = self.VTKObject.GetArray(idx)
        if not vtkarray:
            vtkarray = self.VTKObject.GetAbstractArray(idx)
            if vtkarray:
                return vtkarray
            return NoneArray
        array = vtkDataArrayToVTKArray(vtkarray, self.DataSet)
        array.Association = self.Association
        return array

    def keys(self):
        """Returns the names of the arrays as a list."""
        kys = []
        narrays = self.VTKObject.GetNumberOfArrays()
        for i in range(narrays):
            name = self.VTKObject.GetAbstractArray(i).GetName()
            if name:
                kys.append(name)
        return kys

    def values(self):
        """Returns the arrays as a list."""
        vals = []
        narrays = self.VTKObject.GetNumberOfArrays()
        for i in range(narrays):
            a = self.VTKObject.GetAbstractArray(i)
            if a.GetName():
                vals.append(a)
        return vals

    def PassData(self, other):
        "A wrapper for vtkDataSet.PassData."
        try:
            self.VTKObject.PassData(other)
        except TypeError:
            self.VTKObject.PassData(other.VTKObject)

    def append(self, narray, name):
        """Appends a new array to the dataset attributes."""
        if narray is NoneArray:
            # if NoneArray, nothing to do.
            return

        if self.Association == ArrayAssociation.POINT:
            arrLength = self.DataSet.GetNumberOfPoints()
        elif self.Association == ArrayAssociation.CELL:
            arrLength = self.DataSet.GetNumberOfCells()
        else:
            if not isinstance(narray, numpy.ndarray):
                arrLength = 1
            else:
                arrLength = narray.shape[0]

        # Fixup input array length:
        if not isinstance(narray, numpy.ndarray) or numpy.ndim(narray) == 0: # Scalar input
            tmparray = numpy.empty(arrLength)
            tmparray.fill(narray)
            narray = tmparray
        elif narray.shape[0] != arrLength: # Vector input
            components = 1
            for l in narray.shape:
                components *= l
            tmparray = numpy.empty((arrLength, components))
            tmparray[:] = narray.flatten()
            narray = tmparray

        shape = narray.shape

        if len(shape) == 3:
            # Array of matrices. We need to make sure the order  in memory is right.
            # If column order (c order), transpose. VTK wants row order (fortran
            # order). The deep copy later will make sure that the array is contiguous.
            # If row order but not contiguous, transpose so that the deep copy below
            # does not happen.
            size = narray.dtype.itemsize
            if (narray.strides[1]/size == 3 and narray.strides[2]/size == 1) or \
                (narray.strides[1]/size == 1 and narray.strides[2]/size == 3 and \
                 not narray.flags.contiguous):
                narray  = narray.transpose(0, 2, 1)

        # If array is not contiguous, make a deep copy that is contiguous
        if not narray.flags.contiguous:
            narray = numpy.ascontiguousarray(narray)

        # Flatten array of matrices to array of vectors
        if len(shape) == 3:
            narray = narray.reshape(shape[0], shape[1]*shape[2])

        # this handle the case when an input array is directly appended on the
        # output. We want to make sure that the array added to the output is not
        # referring to the input dataset.
        copy = VTKArray(narray)
        try:
            copy.VTKObject = narray.VTKObject
        except AttributeError: pass
        arr = numpyTovtkDataArray(copy, name)
        self.VTKObject.AddArray(arr)


class CompositeDataSetAttributes():
    """This is a python friendly wrapper for vtkDataSetAttributes for composite
    datsets. Since composite datasets themselves don't have attribute data, but
    the attribute data is associated with the leaf nodes in the composite
    dataset, this class simulates a DataSetAttributes interface by taking a
    union of DataSetAttributes associated with all leaf nodes."""

    def __init__(self, dataset, association):
        # import weakref
        # self.DataSet = weakref.ref(dataset)
        self.DataSet = dataset
        self.Association = association
        self.ArrayNames = []
        self.Arrays = {}

        # build the set of arrays available in the composite dataset. Since
        # composite datasets can have partial arrays, we need to iterate over
        # all non-null blocks in the dataset.
        self.__determine_arraynames()

    def __determine_arraynames(self):
        array_set = set()
        array_list = []
        for dataset in self.DataSet:
            dsa = dataset.GetAttributes(self.Association)
            for array_name in dsa.keys():
                if array_name not in array_set:
                    array_set.add(array_name)
                    array_list.append(array_name)
        self.ArrayNames = array_list

    def keys(self):
        """Returns the names of the arrays as a list."""
        return self.ArrayNames

    def __getitem__(self, idx):
        """Implements the [] operator. Accepts an array name."""
        return self.GetArray(idx)

    def append(self, narray, name):
        """Appends a new array to the composite dataset attributes."""
        if narray is NoneArray:
            # if NoneArray, nothing to do.
            return

        added = False
        if not isinstance(narray, VTKCompositeDataArray): # Scalar input
            for ds in self.DataSet:
                ds.GetAttributes(self.Association).append(narray, name)
                added = True
            if added:
                self.ArrayNames.append(name)
                # don't add the narray since it's a scalar. GetArray() will create a
                # VTKCompositeArray on-demand.
        else:
            for ds, array in izip(self.DataSet, narray.Arrays):
                if array is not None:
                    ds.GetAttributes(self.Association).append(array, name)
                    added = True
            if added:
                self.ArrayNames.append(name)
                self.Arrays[name] = weakref.ref(narray)

    def GetArray(self, idx):
        """Given a name, returns a VTKCompositeArray."""
        arrayname = idx
        if arrayname not in self.ArrayNames:
            return NoneArray
        if arrayname not in self.Arrays or self.Arrays[arrayname]() is None:
            array = VTKCompositeDataArray(
                dataset = self.DataSet, name = arrayname, association = self.Association)
            self.Arrays[arrayname] = weakref.ref(array)
        else:
            array = self.Arrays[arrayname]()
        return array

    def PassData(self, other):
        """Emulate PassData for composite datasets."""
        for this,that in zip(self.DataSet, other.DataSet):
            for assoc in [ArrayAssociation.POINT, ArrayAssociation.CELL]:
                this.GetAttributes(assoc).PassData(that.GetAttributes(assoc))

class CompositeDataIterator(object):
    """Wrapper for a vtkCompositeDataIterator class to satisfy
       the python iterator protocol. This iterator iterates
       over non-empty leaf nodes. To iterate over empty or
       non-leaf nodes, use the vtkCompositeDataIterator directly.
       """

    def __init__(self, cds):
        self.Iterator = cds.NewIterator()
        if self.Iterator:
            self.Iterator.UnRegister(None)
            self.Iterator.GoToFirstItem()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.Iterator:
            raise StopIteration

        if self.Iterator.IsDoneWithTraversal():
            raise StopIteration
        retVal = self.Iterator.GetCurrentDataObject()
        self.Iterator.GoToNextItem()
        return WrapDataObject(retVal)

    def next(self):
        return self.__next__()

    def __getattr__(self, name):
        """Returns attributes from the vtkCompositeDataIterator."""
        return getattr(self.Iterator, name)

class MultiCompositeDataIterator(CompositeDataIterator):
    """Iterator that can be used to iterate over multiple
    composite datasets together. This iterator works only
    with arrays that were copied from an original using
    CopyStructured. The most common use case is to use
    CopyStructure, then iterate over input and output together
    while creating output datasets from corresponding input
    datasets."""
    def __init__(self, cds):
        CompositeDataIterator.__init__(self, cds[0])
        self.Datasets = cds

    def __next__(self):
        if not self.Iterator:
            raise StopIteration

        if self.Iterator.IsDoneWithTraversal():
            raise StopIteration
        retVal = []
        retVal.append(WrapDataObject(self.Iterator.GetCurrentDataObject()))
        if len(self.Datasets) > 1:
            for cd in self.Datasets[1:]:
                retVal.append(WrapDataObject(cd.GetDataSet(self.Iterator)))
        self.Iterator.GoToNextItem()
        return retVal

    def next(self):
        return self.__next__()

class DataObject(VTKObjectWrapper):
    """A wrapper for vtkDataObject that makes it easier to access FielData
    arrays as VTKArrays
    """

    def GetAttributes(self, type):
        """Returns the attributes specified by the type as a DataSetAttributes
         instance."""
        if type == ArrayAssociation.FIELD:
            return DataSetAttributes(self.VTKObject.GetFieldData(), self, type)
        return DataSetAttributes(self.VTKObject.GetAttributes(type), self, type)

    def GetFieldData(self):
        "Returns the field data as a DataSetAttributes instance."
        return DataSetAttributes(self.VTKObject.GetFieldData(), self, ArrayAssociation.FIELD)

    FieldData = property(GetFieldData, None, None, "This property returns the field data of a data object.")

class Table(DataObject):
    """A wrapper for vtkFielData that makes it easier to access RowData array as
    VTKArrays
    """
    def GetRowData(self):
        "Returns the row data as a DataSetAttributes instance."
        return self.GetAttributes(ArrayAssociation.ROW)

    RowData = property(GetRowData, None, None, "This property returns the row data of the table.")

class CompositeDataSet(DataObject):
    """A wrapper for vtkCompositeData and subclasses that makes it easier
    to access Point/Cell/Field data as VTKCompositeDataArrays. It also
    provides a Python type iterator."""

    def __init__(self, vtkobject):
        DataObject.__init__(self, vtkobject)
        self._PointData = None
        self._CellData = None
        self._FieldData = None
        self._Points = None

    def __iter__(self):
        "Creates an iterator for the contained datasets."
        return CompositeDataIterator(self)

    def GetNumberOfElements(self, assoc):
        """Returns the total number of cells or points depending
        on the value of assoc which can be ArrayAssociation.POINT or
        ArrayAssociation.CELL."""
        result = 0
        for dataset in self:
            result += dataset.GetNumberOfElements(assoc)
        return int(result)

    def GetNumberOfPoints(self):
        """Returns the total number of points of all datasets
        in the composite dataset. Note that this traverses the
        whole composite dataset every time and should not be
        called repeatedly for large composite datasets."""
        return self.GetNumberOfElements(ArrayAssociation.POINT)

    def GetNumberOfCells(self):
        """Returns the total number of cells of all datasets
        in the composite dataset. Note that this traverses the
        whole composite dataset every time and should not be
        called repeatedly for large composite datasets."""
        return self.GetNumberOfElements(ArrayAssociation.CELL)

    def GetAttributes(self, type):
        """Returns the attributes specified by the type as a
        CompositeDataSetAttributes instance."""
        return CompositeDataSetAttributes(self, type)

    def GetPointData(self):
        "Returns the point data as a DataSetAttributes instance."
        if self._PointData is None or self._PointData() is None:
            pdata = self.GetAttributes(ArrayAssociation.POINT)
            self._PointData = weakref.ref(pdata)
        return self._PointData()

    def GetCellData(self):
        "Returns the cell data as a DataSetAttributes instance."
        if self._CellData is None or self._CellData() is None:
            cdata = self.GetAttributes(ArrayAssociation.CELL)
            self._CellData = weakref.ref(cdata)
        return self._CellData()

    def GetFieldData(self):
        "Returns the field data as a DataSetAttributes instance."
        if self._FieldData is None or self._FieldData() is None:
            fdata = self.GetAttributes(ArrayAssociation.FIELD)
            self._FieldData = weakref.ref(fdata)
        return self._FieldData()

    def GetPoints(self):
        "Returns the points as a VTKCompositeDataArray instance."
        if self._Points is None or self._Points() is None:
            pts = []
            for ds in self:
                try:
                    _pts = ds.Points
                except AttributeError:
                    _pts = None

                if _pts is None:
                    pts.append(NoneArray)
                else:
                    pts.append(_pts)
            if len(pts) == 0 or all([a is NoneArray for a in pts]):
                cpts = NoneArray
            else:
                cpts = VTKCompositeDataArray(pts, dataset=self)
            self._Points = weakref.ref(cpts)
        return self._Points()

    PointData = property(GetPointData, None, None, "This property returns the point data of the dataset.")
    CellData = property(GetCellData, None, None, "This property returns the cell data of a dataset.")
    FieldData = property(GetFieldData, None, None, "This property returns the field data of a dataset.")
    Points = property(GetPoints, None, None, "This property returns the points of the dataset.")

class DataSet(DataObject):
    """This is a python friendly wrapper of a vtkDataSet that defines
    a few useful properties."""

    def GetPointData(self):
        "Returns the point data as a DataSetAttributes instance."
        return self.GetAttributes(ArrayAssociation.POINT)

    def GetCellData(self):
        "Returns the cell data as a DataSetAttributes instance."
        return self.GetAttributes(ArrayAssociation.CELL)

    PointData = property(GetPointData, None, None, "This property returns the point data of the dataset.")
    CellData = property(GetCellData, None, None, "This property returns the cell data of a dataset.")

class PointSet(DataSet):
    """This is a python friendly wrapper of a vtkPointSet that defines
    a few useful properties."""
    def GetPoints(self):
        """Returns the points as a VTKArray instance. Returns None if the
        dataset has implicit points."""
        if not self.VTKObject.GetPoints():
            return None
        array = vtkDataArrayToVTKArray(
            self.VTKObject.GetPoints().GetData(), self)
        array.Association = ArrayAssociation.POINT
        return array

    def SetPoints(self, pts):
        """Given a VTKArray instance, sets the points of the dataset."""
        from vtk.vtkCommonCore import vtkPoints
        if isinstance(pts, vtkPoints):
            p = pts
        else:
            pts = numpyTovtkDataArray(pts)
            p = vtkPoints()
            p.SetData(pts)
        self.VTKObject.SetPoints(p)

    Points = property(GetPoints, SetPoints, None, "This property returns the point coordinates of dataset.")

class PolyData(PointSet):
    """This is a python friendly wrapper of a vtkPolyData that defines
    a few useful properties."""

    def GetPolygons(self):
        """Returns the polys as a VTKArray instance."""
        if not self.VTKObject.GetPolys():
            return None
        return vtkDataArrayToVTKArray(
            self.VTKObject.GetPolys().GetData(), self)

    Polygons = property(GetPolygons, None, None, "This property returns the connectivity of polygons.")

class UnstructuredGrid(PointSet):
    """This is a python friendly wrapper of a vtkUnstructuredGrid that defines
    a few useful properties."""

    def GetCellTypes(self):
        """Returns the cell types as a VTKArray instance."""
        if not self.VTKObject.GetCellTypesArray():
            return None
        return vtkDataArrayToVTKArray(
            self.VTKObject.GetCellTypesArray(), self)

    def GetCellLocations(self):
        """Returns the cell locations as a VTKArray instance."""
        if not self.VTKObject.GetCellLocationsArray():
            return None
        return vtkDataArrayToVTKArray(
            self.VTKObject.GetCellLocationsArray(), self)

    def GetCells(self):
        """Returns the cells as a VTKArray instance."""
        if not self.VTKObject.GetCells():
            return None
        return vtkDataArrayToVTKArray(
            self.VTKObject.GetCells().GetData(), self)

    def SetCells(self, cellTypes, cellLocations, cells):
        """Given cellTypes, cellLocations, cells as VTKArrays,
        populates the unstructured grid data structures."""
        from vtk import VTK_ID_TYPE
        from vtk.vtkCommonDataModel import vtkCellArray
        cellTypes = numpyTovtkDataArray(cellTypes)
        cellLocations = numpyTovtkDataArray(cellLocations, array_type=VTK_ID_TYPE)
        cells = numpyTovtkDataArray(cells, array_type=VTK_ID_TYPE)
        ca = vtkCellArray()
        ca.SetCells(cellTypes.GetNumberOfTuples(), cells)
        self.VTKObject.SetCells(cellTypes, cellLocations, ca)

    CellTypes = property(GetCellTypes, None, None, "This property returns the types of cells.")
    CellLocations = property(GetCellLocations, None, None, "This property returns the locations of cells.")
    Cells = property(GetCells, None, None, "This property returns the connectivity of cells.")

def WrapDataObject(ds):
    """Returns a Numpy friendly wrapper of a vtkDataObject."""
    if ds.IsA("vtkPolyData"):
        return PolyData(ds)
    elif ds.IsA("vtkUnstructuredGrid"):
        return UnstructuredGrid(ds)
    elif ds.IsA("vtkPointSet"):
        return PointSet(ds)
    elif ds.IsA("vtkDataSet"):
        return DataSet(ds)
    elif ds.IsA("vtkCompositeDataSet"):
        return CompositeDataSet(ds)
    elif ds.IsA("vtkTable"):
        return Table(ds)
