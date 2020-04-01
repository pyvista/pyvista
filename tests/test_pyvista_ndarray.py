import numpy as np
from pytest import mark, raises
from pyvista import examples, UnstructuredGrid, pyvista_ndarray


class TestConstructors:
    @mark.parametrize('arr', [[], [0, 0, 0], [-1, -1], [False, False, True], [1.0, -2, 65]])
    def test_from_iter_list(self, arr):
        pv_arr = pyvista_ndarray.from_iter(arr)
        assert np.equal(arr, pv_arr).all()

    @mark.parametrize('arr', [(), (0, 0, 0), (-1, -1), (False, False, True), (1.0, -2, 65)])
    def test_from_iter_tuple(self, arr):
        pv_arr = pyvista_ndarray.from_iter(arr)
        assert np.equal(arr, pv_arr).all()

    @mark.parametrize('arr', [(), [], (0, 0), [0, 0], (False, True), [False, True], (1.0, -2, 65), [1.0, -2, 65]])
    def test_from_any(self, arr):
        pv_arr = pyvista_ndarray.from_any(arr)
        assert np.equal(arr, pv_arr).all()

    @mark.parametrize('var', [None, 1, False, -1.0])
    def test_from_any_should_fail_if_not_array(self, var):
        with raises(ValueError):
            pyvista_ndarray.from_any(var)


class TestSetItem:
    def test_setitem_should_change_value(self):
        grid = UnstructuredGrid(examples.hexbeamfile)
        vtk_arr = grid.GetCellData().GetScalars()

        pv_arr = pyvista_ndarray.from_vtk_data_array(vtk_data_array=vtk_arr, dataset=grid)
        pv_arr[0] = 99
        assert pv_arr[0] == 99

    def test_setitem_should_change_modified_time(self):
        grid = UnstructuredGrid(examples.hexbeamfile)
        vtk_arr = grid.GetCellData().GetScalars()

        pv_arr = pyvista_ndarray.from_vtk_data_array(vtk_data_array=vtk_arr, dataset=grid)
        last_modified_time = pv_arr.VTKObject.GetMTime()
        pv_arr[0] = 99
        assert last_modified_time < pv_arr.VTKObject.GetMTime()
