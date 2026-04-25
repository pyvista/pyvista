"""Tests for non-spatially referenced objects"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core import _vtk_core as _vtk

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


def test_table_init(tmpdir):
    """Save some delimited text to a file and read it"""
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.csv'))
    nr, nc = 50, 3
    arrays = np.random.default_rng().random((nr, nc))

    # Create from 2D array
    table = pv.Table(arrays)
    assert table.n_rows == nr
    assert table.n_columns == nc
    assert table.n_arrays == nc

    assert len(table.row_arrays) == nc
    for i in range(nc):
        assert np.allclose(arrays[:, i], table[i])

    with pytest.raises(ValueError):  # noqa: PT011
        pv.Table(np.random.default_rng().random((100, 2, 3)))

    # Create from 1D array
    table = pv.Table(arrays[:, 0])
    assert table.n_rows == nr
    assert table.n_columns == 1

    assert len(table.row_arrays) == 1
    assert np.allclose(arrays[:, 0], table[0])

    # create from dictionary
    array_dict = {}
    for i in range(nc):
        array_dict[f'foo{i}'] = arrays[:, i].copy()
    table = pv.Table(array_dict)
    assert table.n_rows == nr
    assert table.n_columns == nc

    assert len(table.row_arrays) == nc
    for i in range(nc):
        assert np.allclose(arrays[:, i], table[f'foo{i}'])

    dataset = examples.load_hexbeam()
    array_dict = dict(dataset.point_data)
    table = pv.Table(array_dict)
    assert table.n_rows == dataset.n_points
    assert table.n_columns == len(array_dict)

    assert len(table.row_arrays) == len(array_dict)
    for name in table.keys():
        assert np.allclose(dataset[name], table[name])

    # Create from vtkTable object
    h = '\t'.join([f'a{i}' for i in range(nc)])
    np.savetxt(filename, arrays, delimiter='\t', header=h, comments='')

    reader = _vtk.vtkDelimitedTextReader()
    reader.SetFileName(filename)
    reader.DetectNumericColumnsOn()
    reader.SetFieldDelimiterCharacters('\t')
    reader.SetHaveHeaders(True)
    reader.Update()

    # Test init
    table = pv.Table(reader.GetOutput(), deep=True)
    assert isinstance(table, _vtk.vtkTable)
    assert isinstance(table, pv.Table)

    table = pv.Table(reader.GetOutput(), deep=False)
    assert isinstance(table, _vtk.vtkTable)
    assert isinstance(table, pv.Table)

    # Test wrap
    table = pv.wrap(reader.GetOutput())
    assert isinstance(table, _vtk.vtkTable)
    assert isinstance(table, pv.Table)

    assert table.n_rows == nr
    assert table.n_columns == nc

    assert len(table.row_arrays) == nc
    for i in range(nc):
        assert np.allclose(arrays[:, i], table[i])

    with pytest.raises(TypeError):
        pv.Table('foo')


def test_table_row_arrays():
    nr, nc = 50, 3
    arrays = np.random.default_rng().random((nr, nc))
    table = pv.Table()
    for i in range(nc):
        table[f'foo{i}'] = arrays[:, i]
    assert table.n_columns == nc
    assert table.n_rows == nr
    for i in range(nc):
        assert np.allclose(table[f'foo{i}'], arrays[:, i])
    # Multi component
    table = pv.Table()
    table['multi'] = arrays
    assert table.n_columns == 1
    assert table.n_rows == nr
    assert np.allclose(table[0], arrays)
    assert np.allclose(table['multi'], arrays)
    del table['multi']
    assert table.n_columns == 0

    dataset = examples.load_hexbeam()
    array_dict = dataset.point_data
    # Test dict methods
    table = pv.Table()
    table.update(array_dict)
    assert table.n_rows == dataset.n_points
    assert table.n_columns == len(array_dict)

    assert len(table.row_arrays) == len(array_dict)
    for name in table.keys():
        assert np.allclose(dataset[name], table[name])

    for i, array in enumerate(table.values()):
        name = table.keys()[i]
        assert np.allclose(dataset[name], array)

    for name, array in table.items():
        assert np.allclose(dataset[name], array)

    n = table.n_arrays
    array = table.pop(table.keys()[0])
    assert isinstance(array, np.ndarray)
    assert table.n_arrays == n - 1
    array = table.get(table.keys()[0])
    assert isinstance(array, np.ndarray)
    assert table.n_arrays == n - 1

    del table[table.keys()[0]]
    assert table.n_arrays == n - 2


def test_table_row_np_bool():
    n = 50
    table = pv.Table()
    bool_arr = np.zeros(n, np.bool_)
    table.row_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert table.row_arrays['bool_arr'].all()
    assert table._row_array('bool_arr').all()
    assert table._row_array('bool_arr').dtype == np.bool_


def test_table_row_uint8():
    n = 50
    table = pv.Table()
    arr = np.zeros(n, np.uint8)
    table.row_arrays['arr'] = arr
    arr[:] = np.arange(n)
    assert np.allclose(table.row_arrays['arr'], np.arange(n))


def test_table_repr():
    nr, nc = 50, 3
    arrays = np.random.default_rng().random((nr, nc))
    table = pv.Table(arrays)
    text = table._repr_html_()
    assert isinstance(text, str)
    text = table.__repr__()
    assert isinstance(text, str)
    text = table.__str__()
    assert isinstance(text, str)


@pytest.mark.skipif(pd is None, reason='Requires Pandas')
def test_table_pandas():
    nr, nc = 50, 3
    arrays = np.random.default_rng().random((nr, nc))
    df = pd.DataFrame()
    for i in range(nc):
        df[f'foo{i}'] = arrays[:, i].copy()
    table = pv.Table(df)
    assert table.n_rows == nr
    assert table.n_columns == nc
    for i in range(nc):
        assert np.allclose(table.row_arrays[f'foo{i}'], arrays[:, i])
    assert df.equals(table.to_pandas())


def test_table_iter():
    nr, nc = 50, 3
    arrays = np.random.default_rng().random((nr, nc))
    table = pv.Table(arrays)
    for i, array in enumerate(table):
        assert np.allclose(array, arrays[:, i])


@pytest.mark.parametrize('preference', ['row', None])
def test_get_data_range_table(preference):
    nr, nc = 50, 3
    arrays = np.random.default_rng().random((nr, nc))
    table = pv.Table(arrays)
    nanmin, nanmax = (
        table.get_data_range(preference=preference) if preference else table.get_data_range()
    )
    assert nanmin == np.nanmin(arrays[:, 0])
    assert nanmax == np.nanmax(arrays[:, 0])


def test_from_dict_raises(mocker: MockerFixture):
    m = mocker.MagicMock()
    m.ndim = 1
    with pytest.raises(
        ValueError, match=r'Dictionary must contain only NumPy arrays with maximum of 2D.'
    ):
        pv.Table(dict(a=m))


@pytest.mark.skipif(pa is None, reason='Requires pyarrow')
def test_table_to_arrow():
    table = pv.Table({'a': np.arange(5, dtype=np.int64), 'b': np.linspace(0, 1, 5)})
    arrow_table = table.to_arrow()
    assert isinstance(arrow_table, pa.Table)
    assert arrow_table.num_rows == 5
    assert arrow_table.schema.names == ['a', 'b']
    assert arrow_table.column('a').type == pa.int64()
    assert np.array_equal(arrow_table.column('a').to_numpy(), np.arange(5))


@pytest.mark.skipif(pa is None, reason='Requires pyarrow')
def test_table_arrow_c_stream_round_trip():
    table = pv.Table({'a': np.arange(5, dtype=np.int64), 'b': np.linspace(0, 1, 5)})
    consumed = pa.table(table)
    assert consumed.equals(table.to_arrow())


@pytest.mark.skipif(pa is None, reason='Requires pyarrow')
def test_table_arrow_c_stream_returns_pycapsule():
    table = pv.Table({'a': np.arange(3, dtype=np.int32)})
    capsule = table.__arrow_c_stream__()
    assert type(capsule).__name__ == 'PyCapsule'


@pytest.mark.skipif(pa is None, reason='Requires pyarrow')
def test_table_to_arrow_empty():
    table = pv.Table()
    arrow_table = table.to_arrow()
    assert arrow_table.num_rows == 0
    assert arrow_table.num_columns == 0


@pytest.mark.skipif(pa is None, reason='Requires pyarrow')
@pytest.mark.parametrize(
    'dtype',
    [np.int32, np.int64, np.uint8, np.float32, np.float64],
)
def test_table_to_arrow_preserves_dtype(dtype):
    expected = np.arange(5, dtype=dtype)
    table = pv.Table({'col': expected})
    arrow_table = table.to_arrow()
    assert arrow_table.column('col').type == pa.from_numpy_dtype(dtype)
    assert np.array_equal(arrow_table.column('col').to_numpy(), expected)


@pytest.mark.skipif(pa is None, reason='Requires pyarrow')
@pytest.mark.skipif(pd is None, reason='Requires pandas')
def test_table_to_arrow_matches_to_pandas():
    arrays = np.random.default_rng(seed=0).random((10, 3))
    table = pv.Table(arrays)
    pd.testing.assert_frame_equal(table.to_arrow().to_pandas(), table.to_pandas())
