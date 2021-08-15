import numpy as np
import pyvista
from pyvista import examples


def test_eq_wrong_type(sphere):
    assert sphere != [1, 2, 3]


def test_uniform_eq():
    orig = examples.load_uniform()
    copy = orig.copy(deep=True)
    copy.origin = [1, 1, 1]
    assert orig != copy

    copy.origin = [0, 0, 0]
    assert orig == copy

    copy.point_arrays.clear()
    assert orig != copy


def test_polydata_eq(sphere):
    sphere.clear_arrays()
    sphere.point_arrays['data0'] = np.zeros(sphere.n_points)
    sphere.point_arrays['data1'] = np.arange(sphere.n_points)

    copy = sphere.copy(deep=True)
    assert sphere == copy

    copy.faces = [3, 0, 1, 2]
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.field_arrays['new'] = [1]
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.point_arrays['new'] = range(sphere.n_points)
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.cell_arrays['new'] = range(sphere.n_cells)
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.point_arrays.active_scalars_name = 'data1'
    assert sphere != copy
