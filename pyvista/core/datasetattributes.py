import numpy
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper
import vtk.util.numpy_support as numpy_support


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
        "Given an index or name, returns a VTKArray."
        if isinstance(key, int) and key >= self.VTKObject.GetNumberOfArrays():
            raise IndexError("array index out of range")
        vtkarray = self.VTKObject.get_array(key)
        if not vtkarray:
            vtkarray = self.VTKObject.GetAbstractArray(key)
            if vtkarray:
                return vtkarray
            return NoneArray
        array = vtkDataArrayToVTKArray(vtkarray, self.DataSet)
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
    pass



