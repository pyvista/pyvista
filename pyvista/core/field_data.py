from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper


class FieldData(VTKObjectWrapper):
    def __init__(self, vtk_field_data):
        super().__init__(vtkobject=vtk_field_data)


class PointData(FieldData):
    def __init__(self, vtk_point_data):
        super().__init__(vtk_field_data=vtk_point_data)


class CellData(FieldData):
    def __init__(self, vtk_cell_data):
        super().__init__(vtk_field_data=vtk_cell_data)
