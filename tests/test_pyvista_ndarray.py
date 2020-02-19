import numpy as np
from pytest import mark, raises
from pyvista import pyvista_ndarray


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
