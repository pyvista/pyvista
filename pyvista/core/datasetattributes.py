from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper


class DataSetAttributes(VTKObjectWrapper):
    """
    Python friendly wrapper of DataSetAttributes.
    Loosely based on dataset_adapter.DataSetAttributes.
    """
    def __init__(self, vtkobject, dataset, association):
        super().__init__(vtkobject=vtkobject)
        self._dataset = dataset
        self._association = association

    def __getitem__(self, key):
        """Implements the [] operator. Accepts an array name or index."""
        return self.GetArray(key)

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

    def __setitem__(self, key, value):
        if self[key] is dsa.NoneArray:
            self.append(narray=value, name=key)
        else:
            self.RemoveArray(key)
            self[key] = value


class pyvista_ndarray:
    pass



