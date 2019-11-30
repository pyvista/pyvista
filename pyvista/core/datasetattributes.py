import numpy
from vtk import buffer_shared
from vtk.vtkCommonKitPython import vtkDataArray, vtkAbstractArray
from vtk.numpy_interface.dataset_adapter import (VTKObjectWrapper, _make_tensor_array_contiguous,
                                                 numpyTovtkDataArray, ArrayAssociation,
                                                 VTKArray)
import vtk.util.numpy_support as numpy_support
from vtk.vtkCommonCore import vtkWeakReference


class DataSetAttributes(VTKObjectWrapper):
    """Python friendly wrapper of DataSetAttributes.
    Loosely based on dataset_adapter.DataSetAttributes."""
    def __init__(self, vtkobject, dataset, association):
        super().__init__(vtkobject=vtkobject)
        self._dataset = dataset
        self._association = association

    def __getitem__(self, key):
        """Implements the [] operator. Accepts an array name or index."""
        return self.get_array(key)

    def __setitem__(self, key, value):
        self.append(narray=value, name=key)

    def __delitem__(self, key):
        self.RemoveArray(key)

    def __contains__(self, item):
        return item in self.keys()

    def get_array(self, key):
        """Given an index or name, returns a VTKArray."""
        max_index = self.VTKObject.GetNumberOfArrays()
        if isinstance(key, int) and key >= self.VTKObject.GetNumberOfArrays():
            raise IndexError('Array index ({}) out of range [{}, {}]'.format(key, 0, max_index))
        vtkarray = self.VTKObject.GetArray(key)
        if not vtkarray:
            vtkarray = self.VTKObject.GetAbstractArray(key)
            return vtkarray if vtkarray else None
        array = pyvista_ndarray.from_vtk_data_array(vtkarray, dataset=self._dataset)
        array._association = self._association
        return array

    def append(self, narray, name):
        """Appends a new array to the dataset attributes."""
        if narray is None:
            return
        if self._association == ArrayAssociation.POINT:
            array_len = self._dataset.GetNumberOfPoints()
        elif self._association == ArrayAssociation.CELL:
            array_len = self._dataset.GetNumberOfCells()
        else:
            array_len = narray.shape[0] if isinstance(narray, numpy.ndarray) else 1

        # Fixup input array length:
        if not isinstance(narray, numpy.ndarray) or numpy.ndim(narray) == 0: # Scalar input
            tmparray = numpy.empty(array_len)
            tmparray.fill(narray)
            narray = tmparray
        elif narray.shape[0] != array_len: # Vector input
            components = 1
            for l in narray.shape:
                components *= l
            tmparray = numpy.empty((array_len, components))
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
            if (narray.strides[1] / size == 3 and narray.strides[2] / size == 1) or \
                (narray.strides[1] / size == 1 and narray.strides[2] / size == 3 and \
                 not narray.flags.contiguous):
                narray = narray.transpose(0, 2, 1)

        # If array is not contiguous, make a deep copy that is contiguous
        if not narray.flags.contiguous:
            narray = numpy.ascontiguousarray(narray)

        # Flatten array of matrices to array of vectors
        if len(shape) == 3:
            narray = narray.reshape(shape[0], shape[1]*shape[2])

        # this handle the case when an input array is directly appended on the
        # output. We want to make sure that the array added to the output is not
        # referring to the input dataset.
        copy = pyvista_ndarray(narray)
        try:
            copy.VTKObject = narray.VTKObject
        except AttributeError:
            pass
        arr = numpyTovtkDataArray(copy, name)
        self.VTKObject.AddArray(arr)

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



class pyvista_ndarray(VTKArray):
    """This is a sub-class of numpy ndarray that stores a
    reference to a vtk array as well as the owning dataset.
    The numpy array and vtk array should point to the same
    memory location."""

    def __new__(cls, ndarray, vtk_array=None, dataset=None):
        # Input array is an already formed ndarray instance
        obj = numpy.asarray(ndarray).view(cls)
        obj._association = ArrayAssociation.FIELD
        # add the new attributes to the created instance
        obj.VTKObject = vtk_array
        if dataset:
            obj._dataset = vtkWeakReference()
            # Unless ALL pyvista objects ARE vtk objects, OR, ALL pyvista objects ARE
            # VTKObjectWrappers, which object gets set matters.
            if isinstance(dataset, VTKObjectWrapper):
                obj._dataset.Set(dataset.VTKObject)
            else:
                obj._dataset.Set(dataset)
        return obj

    @classmethod
    def from_any(cls, obj, dtype=None, order=None, dataset=None):
        """Factory method to create a `pyvista_ndarray`` instance from an object.

        Parameters
        ----------
        obj : any
            Object to create numpy array from based on type.

        dataset : vtkDataSet, optional
            The vtkDataSet which vtk_data_array belongs to. Required to
            update or access a dataset when this pyvista_ndarray is updated.

        dtype : data-type, optional
            By default, the data-type is inferred from the input data.

        order : {‘C’, ‘F’}, optional
            Whether to use row-major (C-style) or column-major (Fortran-style) memory representation. Defaults to ‘C’.

        """
        if isinstance(obj, (vtkDataArray, vtkAbstractArray)):
            return cls.from_vtk_data_array(vtk_data_array=obj, dataset=dataset)
        elif isinstance(obj, (list, tuple, numpy.ndarray)):
            return cls.from_iter(a=obj, dtype=dtype, order=order)
        else:
            raise ValueError('Cannot create {} from type: {}'.format(
                type(cls).__name__, type(obj).__name__))

    @classmethod
    def from_vtk_data_array(cls, vtk_data_array, dataset=None):
        """Create a ``pyvista_ndarray`` instance from a vtk array.

        Parameters
        ----------
        vtk_data_array : vtkDataArray
            Array to copy.

        dataset : vtkDataSet, optional
            The vtkDataSet which vtk_data_array belongs to. Required to
            update or access a dataset when this pyvista_ndarray is updated.

        """
        narray = numpy_support.vtk_to_numpy(vtk_data_array)

        # Make arrays of 9 components into matrices. Also transpose
        # as VTK stores matrices in Fortran order
        shape = narray.shape
        if len(shape) == 2 and shape[1] == 9:
            narray = narray.reshape((shape[0], 3, 3)).transpose(0, 2, 1)
        return cls(narray, vtk_data_array, dataset=dataset)

    @classmethod
    def from_iter(cls, a, dtype=None, order=None):
        """Create a ``pyvista_ndarray`` instance from an iterable.

        Parameters
        ----------
        a : array_like
            Input data, in any form that can be converted to an array. 
            This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
            
        dtype : data-type, optional
            By default, the data-type is inferred from the input data.
            
        order : {‘C’, ‘F’}, optional
            Whether to use row-major (C-style) or column-major (Fortran-style) memory representation. Defaults to ‘C’.

        """
        return cls(numpy.asarray(a, dtype, order))
