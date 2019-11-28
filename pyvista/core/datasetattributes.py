import vtk.numpy_interface.dataset_adapter as dsa

class DataSetAttributes(dsa.DataSetAttributes):
    def __init__(self, vtkobject, dataset, association):
        super().__init__(vtkobject, dataset, association)

    def __setitem__(self, key, value):
        if self[key] is dsa.NoneArray:
            self.append(narray=value, name=key)
        else:
            self.RemoveArray(key)
            self[key] = value

