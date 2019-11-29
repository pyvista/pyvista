import numpy
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper
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
        array.Association = self.Association
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
        obj.Association = ArrayAssociation.FIELD
        # add the new attributes to the created instance
        obj.VTKObject = vtk_array
        if dataset:
            obj._dataset = vtkWeakReference()
            obj._dataset.Set(dataset.VTKObject)
        return obj

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




