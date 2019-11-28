import numpy as np
from pyvista.utilities import (CELL_DATA_FIELD, FIELD_DATA_FIELD,
                               POINT_DATA_FIELD, convert_array, get_array,
                               is_pyvista_dataset, parse_field_choice,
                               raise_not_matching, vtk_bit_array_to_char)
import vtk
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper


class FieldData(VTKObjectWrapper):
    def __init__(self, vtk_field_data):
        super().__init__(vtkobject=vtk_field_data)
        self._bool_array_names = set()
        self._np_arrays = {}

    #TODO, vtkArray/numpyarray conversion, or make a custom pyvista array which allows conversion
    def __setitem__(self, key, value):
        if value is None:
            raise ValueError('Cannot add None to arrays.')
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        self._np_arrays[key] = value
        self.Modified()

    def __getitem__(self, item):
        return self._np_arrays[item]

    def __iter__(self):
        for i in range(self.VTKObject.GetNumberOfArrays()):
            yield self.VTKObject.GetAbstractArray(i)

    def arrays(self):
        """Return all arrays."""
        pdata = self.VTKObject
        narr = pdata.GetNumberOfArrays()

        if self._np_arrays:
            if narr == len(self._np_arrays):
                return self._np_arrays

        self._np_arrays = {}

        for i in range(narr):
            name = pdata.GetArrayName(i)
            if not name:
                name = 'Point Array {}'.format(i)
                pdata.GetAbstractArray(i).SetName(name)
            self._np_arrays[name] = self._point_array(name)

        return self._np_arrays

    def add_array(self, scalars, name, deep=True):
        """Add field scalars to the mesh.

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars. Does not have to match number of points or
            numbers of cells.

        name : str
            Name of field scalars to add.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        # need to track which arrays are boolean as all boolean arrays
        # must be stored as uint8
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)
            self._field_bool_array_names.add(name)

        if not scalars.flags.c_contiguous:
            scalars = np.ascontiguousarray(scalars)

        vtkarr = convert_array(scalars, deep=deep)
        vtkarr.SetName(name)
        self.AddArray(vtkarr)
        self.Modified()

    def array(self, name):
        """Return field scalars of a vtk object.

        Parameters
        ----------
        name : str
            Name of field scalars to retrieve.

        Return
        ------
        scalars : np.ndarray
            Numpy array of scalars

        """
        vtkarr = self.GetAbstractArray(name)
        # numpy does not support bit array data types
        if isinstance(vtkarr, vtk.vtkBitArray):
            vtkarr = vtk_bit_array_to_char(vtkarr)
            self._bool_array_names.add(name)

        array = convert_array(vtkarr)
        if array.dtype == np.uint8 and name in self._bool_array_names:
            array = array.view(np.bool)
        return array


class DataSetAttributes(FieldData):
    def __init__(self, data_set_attributes):
        super().__init__(data_set_attributes)


class PointData(DataSetAttributes):
    def __init__(self, vtk_point_data):
        super().__init__(data_set_attributes=vtk_point_data)
        self._point_bool_array_names = set()

    def _add_point_array(self, scalars, name, deep=True):
        """Add point scalars to the mesh.

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars. Must match number of points.

        name : str
            Name of point scalars to add.

        set_active : bool, optional
            Sets the scalars to the active plotting scalars.  Default False.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        if scalars.shape[0] != self.n_points:
            raise ValueError('Number of scalars must match the number of points')

        # need to track which arrays are boolean as all boolean arrays
        # must be stored as uint8
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)
            self._point_bool_array_names.add(name)

        if not scalars.flags.c_contiguous:
            scalars = np.ascontiguousarray(scalars)

        vtkarr = convert_array(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetPointData().AddArray(vtkarr)
        self.Modified()


class CellData(DataSetAttributes):
    def __init__(self, vtk_cell_data):
        super().__init__(data_set_attributes=vtk_cell_data)
        self._cell_bool_array_names = set()

    def _add_cell_array(self, scalars, name, set_active=False, deep=True):
        """Add cell scalars to the vtk object.

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars.  Must match number of points.

        name : str
            Name of point scalars to add.

        set_active : bool, optional
            Sets the scalars to the active plotting scalars.  Default False.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        if scalars is None:
            raise TypeError('Empty array unable to be added')

        if not isinstance(scalars, np.ndarray):
            scalars = np.array(scalars)

        if scalars.shape[0] != self.n_cells:
            raise Exception('Number of scalars must match the number of cells (%d)'
                            % self.n_cells)

        if not scalars.flags.c_contiguous:
            raise AssertionError('Array must be contiguous')
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)
            self._cell_bool_array_names.add(name)

        vtkarr = convert_array(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetCellData().AddArray(vtkarr)
        self.Modified()
        if set_active or self.active_scalar_info[1] is None:
            self.GetCellData().SetActiveScalars(name)
            self._active_scalar_info = (CELL_DATA_FIELD, name)