from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper


class FieldData(VTKObjectWrapper):
    def __init__(self, vtk_field_data):
        super().__init__(vtkobject=vtk_field_data)
        self._arrays = {}

    def arrays(self):
        """Return all the point arrays."""
        pdata = self.VTKObject
        narr = pdata.GetNumberOfArrays()

        if self._arrays:
            if narr == len(self._arrays):
                return self._point_arrays

        self._arrays = {}

        for i in range(narr):
            name = pdata.GetArrayName(i)
            if not name:
                name = 'Point Array {}'.format(i)
                pdata.GetAbstractArray(i).SetName(name)
            self._point_arrays[name] = self._point_array(name)

        return self._point_arrays

    def arrays_from_field_data(self):
        """Generator which yields abstract arrays from a vtkFieldData object.

         Parameters
        ----------
        field_data : vtkFieldData
            An object of the type vtkFieldData, vtkCellData, or vtkPointData
        """
        for i in range(self.VTKObject.GetNumberOfArrays()):
            yield self.VTKObject.GetAbstractArray(i)


class PointData(FieldData):
    def __init__(self, vtk_point_data):
        super().__init__(vtk_field_data=vtk_point_data)


class CellData(FieldData):
    def __init__(self, vtk_cell_data):
        super().__init__(vtk_field_data=vtk_cell_data)
