"""
Tests for non-spatially referenced objects
"""
import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples

try:
    import pandas as pd
except ImportError:
    pd = None


def test_table_init(tmpdir):
    """Save some delimited text to a file and read it"""
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % 'csv'))
    nr, nc = 50, 3
    arrays = np.random.rand(nr, nc)

    # Create from 2D array
    table = pyvista.Table(arrays)
    assert table.n_rows == nr
    assert table.n_columns == nc
    assert table.n_arrays == nc

    assert len(table.row_arrays) == nc
    for i in range(nc):
        assert np.allclose(arrays[:,i], table[i])

    with pytest.raises(ValueError):
        pyvista.Table(np.random.rand(100))

    with pytest.raises(ValueError):
        pyvista.Table(np.random.rand(100, 2, 3))

    # create from dictionary
    array_dict = {}
    for i in range(nc):
        array_dict['foo{}'.format(i)] = arrays[:, i].copy()
    table = pyvista.Table(array_dict)
    assert table.n_rows == nr
    assert table.n_columns == nc

    assert len(table.row_arrays) == nc
    for i in range(nc):
        assert np.allclose(arrays[:,i], table['foo{}'.format(i)])

    dataset = examples.load_hexbeam()
    array_dict = dict(dataset.point_arrays)
    table = pyvista.Table(array_dict)
    assert table.n_rows == dataset.n_points
    assert table.n_columns == len(array_dict)

    assert len(table.row_arrays) == len(array_dict)
    for name in table.keys():
        assert np.allclose(dataset[name], table[name])

    # Create from vtkTable object
    h = '\t'.join(['a{}'.format(i) for i in range(nc)])
    np.savetxt(filename, arrays, delimiter='\t', header=h, comments='')

    reader = vtk.vtkDelimitedTextReader()
    reader.SetFileName(filename)
    reader.DetectNumericColumnsOn()
    reader.SetFieldDelimiterCharacters('\t')
    reader.SetHaveHeaders(True)
    reader.Update()

    # Test init
    table = pyvista.Table(reader.GetOutput(), deep=True)
    assert isinstance(table, vtk.vtkTable)
    assert isinstance(table, pyvista.Table)

    table = pyvista.Table(reader.GetOutput(), deep=False)
    assert isinstance(table, vtk.vtkTable)
    assert isinstance(table, pyvista.Table)

    # Test wrap
    table = pyvista.wrap(reader.GetOutput())
    assert isinstance(table, vtk.vtkTable)
    assert isinstance(table, pyvista.Table)

    assert table.n_rows == nr
    assert table.n_columns == nc

    assert len(table.row_arrays) == nc
    for i in range(nc):
        assert np.allclose(arrays[:,i], table[i])

    with pytest.raises(TypeError):
        pyvista.Table("foo")

    return


def test_table_row_arrays():
    nr, nc = 50, 3
    arrays = np.random.rand(nr, nc)
    table = pyvista.Table()
    for i in range(nc):
        table['foo{}'.format(i)] = arrays[:, i]
    assert table.n_columns == nc
    assert table.n_rows == nr
    for i in range(nc):
        assert np.allclose(table['foo{}'.format(i)], arrays[:, i])
    # Multi component
    table = pyvista.Table()
    table['multi'] = arrays
    assert table.n_columns == 1
    assert table.n_rows == nr
    assert np.allclose(table[0], arrays)
    assert np.allclose(table['multi'], arrays)
    del table['multi']
    assert table.n_columns == 0

    dataset = examples.load_hexbeam()
    array_dict = dataset.point_arrays
    # Test dict methods
    table = pyvista.Table()
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
    assert table.n_arrays == n-1
    array = table.get(table.keys()[0])
    assert isinstance(array, np.ndarray)
    assert table.n_arrays == n-1

    del table[table.keys()[0]]
    assert table.n_arrays == n-2

    return


def test_table_row_np_bool():
    n = 50
    table = pyvista.Table()
    bool_arr = np.zeros(n, np.bool_)
    table.row_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert table.row_arrays['bool_arr'].all()
    assert table._row_array('bool_arr').all()
    assert table._row_array('bool_arr').dtype == np.bool_


def test_table_row_uint8():
    n = 50
    table = pyvista.Table()
    arr = np.zeros(n, np.uint8)
    table.row_arrays['arr'] = arr
    arr[:] = np.arange(n)
    assert np.allclose(table.row_arrays['arr'], np.arange(n))


def test_table_repr():
    nr, nc = 50, 3
    arrays = np.random.rand(nr, nc)
    table = pyvista.Table(arrays)
    text = table._repr_html_()
    assert isinstance(text, str)
    text = table.__repr__()
    assert isinstance(text, str)
    text = table.__str__()
    assert isinstance(text, str)


@pytest.mark.skipif(pd is None, reason="Requires Pandas")
def test_table_pandas():
    nr, nc = 50, 3
    arrays = np.random.rand(nr, nc)
    df = pd.DataFrame()
    for i in range(nc):
        df['foo{}'.format(i)] = arrays[:, i].copy()
    table = pyvista.Table(df)
    assert table.n_rows == nr
    assert table.n_columns == nc
    for i in range(nc):
        assert np.allclose(table.row_arrays['foo{}'.format(i)], arrays[:, i])
    assert df.equals(table.to_pandas())


def test_table_iter():
    nr, nc = 50, 3
    arrays = np.random.rand(nr, nc)
    table = pyvista.Table(arrays)
    for i, array in enumerate(table):
        assert np.allclose(array, arrays[:, i])


def test_texture():
    texture = pyvista.Texture(examples.mapfile)
    assert texture is not None
    image = texture.to_image()
    assert isinstance(image, pyvista.UniformGrid)
    arr = texture.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] * arr.shape[1] == image.n_points
    texture.flip(0)
    texture.flip(1)
    texture = pyvista.Texture(examples.load_globe_texture())
    assert texture is not None
