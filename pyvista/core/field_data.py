from vtk.vtkCommonKitPython import vtkFieldData, vtkDataSetAttributes


class FieldData(vtkFieldData):
    def __init__(self, vtk_field_data=None):
        super().__init__()
        if vtk_field_data is not None:
            self.ShallowCopy(vtk_field_data)


class PointData(vtkDataSetAttributes, FieldData):
    def __init__(self, vtk_point_data):
        super().__init__(vtk_point_data)


class CellData(vtkDataSetAttributes, FieldData):
    def __init__(self, vtk_cell_data):
        super().__init__(vtk_cell_data)
