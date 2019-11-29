import numpy
from vtk import buffer_shared
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper, _make_tensor_array_contiguous
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

    def __setitem__(self, key, value):
        if self[key] is dsa.NoneArray:
            self.append(narray=value, name=key)
        else:
            self.RemoveArray(key)
            self[key] = value


class pyvista_ndarray(numpy.ndarray):
    """Wraps vtkDataArray as an numpy.ndarray. Both this array and
    vtkDataArray point to the same memory location."""

    def __new__(cls, ndarray, vtk_array=None, dataset=None):
        # Input array is an already formed ndarray instance
        obj = numpy.asarray(ndarray).view(cls)
        obj._association = ArrayAssociation.FIELD
        # add the new attributes to the created instance
        obj.VTKObject = vtk_array
        if dataset:
            obj._dataset = vtkWeakReference()
            obj._dataset.Set(dataset.VTKObject)
        return obj

    def __getattr__(self, name):
        """Forwards unknown attribute requests to VTK array."""
        o = self.__dict__.get('VTKObject', None)
        if o is None:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                 (self.__class__.__name__, name))
        return getattr(o, name)

    def __array_finalize__(self, obj):
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

        self._association = getattr(obj, 'Association', None)
        self._dataset = getattr(obj, 'DataSet', None)

    def __array_wrap__(self, out_arr, context=None):
        if out_arr.shape == ():
            # Convert to scalar value
            return out_arr[()]
        return numpy.ndarray.__array_wrap__(self, out_arr, context)

    #TODO implement
    @classmethod
    def from_vtk_data_array(cls, vtk_data_array, dataset=None):
        """Create pyvista_ndarray from vtkDataArray"""
        narray = numpy_support.vtk_to_numpy(vtk_data_array)

        # Make arrays of 9 components into matrices. Also transpose
        # as VTK stores matrices in Fortran order
        shape = narray.shape
        if len(shape) == 2 and shape[1] == 9:
            narray = narray.reshape((shape[0], 3, 3)).transpose(0, 2, 1)

        return cls(narray, vtk_data_array, dataset=dataset)




