import numpy as np
import pytest

from pyvista import LookupTable


@pytest.fixture
def table():
    return LookupTable()


def test_values(table):
    values = np.array([[0, 0, 0, 0], [255, 255, 255, 255]])
    table.values = values
    assert table.values.dtype == np.uint8
    assert np.allclose(table.values, values)


def test_apply_cmap(table):
    n_values = 5
    table.apply_cmap('reds', n_values=n_values)
    assert table.values.shape == (n_values, 4)
    assert table.n_values == n_values


def test_cmap_init():
    new_table = LookupTable('gray', n_values=2, flip=True)
    assert np.allclose([[254, 255, 255, 255], [0, 0, 0, 255]], new_table.values)
