"""Tests for non-spatially referenced objects."""
import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples
from pyvista.utilities import PyvistaDeprecationWarning

try:
    import pandas as pd
except ImportError:
    pd = None


def test_table_init(tmpdir):
    """Save some delimited text to a file and read it"""
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.csv'))
    nr, nc = 50, 3
    arrays = np.random.rand(nr, nc)

    # Create from 2D array
    table = pyvista.Table(arrays)
    assert table.n_rows == nr
    assert table.n_columns == nc
    assert table.n_arrays == nc

    assert len(table.row_arrays) == nc
    for i in range(nc):
        assert np.allclose(arrays[:, i], table[i])

    with pytest.raises(ValueError):
        pyvista.Table(np.random.rand(100, 2, 3))

    # Create from 1D array
    table = pyvista.Table(arrays[:, 0])
    assert table.n_rows == nr
    assert table.n_columns == 1

    assert len(table.row_arrays) == 1
    assert np.allclose(arrays[:, 0], table[0])

    # create from dictionary
    array_dict = {}
    for i in range(nc):
        array_dict[f'foo{i}'] = arrays[:, i].copy()
    table = pyvista.Table(array_dict)
    assert table.n_rows == nr
    assert table.n_columns == nc

    assert len(table.row_arrays) == nc
    for i in range(nc):
        assert np.allclose(arrays[:, i], table[f'foo{i}'])

    dataset = examples.load_hexbeam()
    array_dict = dict(dataset.point_data)
    table = pyvista.Table(array_dict)
    assert table.n_rows == dataset.n_points
    assert table.n_columns == len(array_dict)

    assert len(table.row_arrays) == len(array_dict)
    for name in table.keys():
        assert np.allclose(dataset[name], table[name])

    # Create from vtkTable object
    h = '\t'.join([f'a{i}' for i in range(nc)])
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
        assert np.allclose(arrays[:, i], table[i])

    with pytest.raises(TypeError):
        pyvista.Table("foo")

    return


def test_table_row_arrays():
    nr, nc = 50, 3
    arrays = np.random.rand(nr, nc)
    table = pyvista.Table()
    for i in range(nc):
        table[f'foo{i}'] = arrays[:, i]
    assert table.n_columns == nc
    assert table.n_rows == nr
    for i in range(nc):
        assert np.allclose(table[f'foo{i}'], arrays[:, i])
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
    array_dict = dataset.point_data
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
    assert table.n_arrays == n - 1
    array = table.get(table.keys()[0])
    assert isinstance(array, np.ndarray)
    assert table.n_arrays == n - 1

    del table[table.keys()[0]]
    assert table.n_arrays == n - 2

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
        df[f'foo{i}'] = arrays[:, i].copy()
    table = pyvista.Table(df)
    assert table.n_rows == nr
    assert table.n_columns == nc
    for i in range(nc):
        assert np.allclose(table.row_arrays[f'foo{i}'], arrays[:, i])
    assert df.equals(table.to_pandas())


def test_table_iter():
    nr, nc = 50, 3
    arrays = np.random.rand(nr, nc)
    table = pyvista.Table(arrays)
    for i, array in enumerate(table):
        assert np.allclose(array, arrays[:, i])


def test_texture_empty_init():
    texture = pyvista.Texture()
    assert texture.dimensions == (0, 0)
    assert texture.n_components == 0


def test_texture_grayscale():
    # verify a grayscale image can be created
    texture = pyvista.Texture(np.zeros((10, 10, 1), dtype=np.uint8))
    assert texture.dimensions == (10, 10)
    assert texture.n_components == 1


def test_texture():
    with pytest.raises(TypeError, match='Cannot create a pyvista.Texture from'):
        texture = pyvista.Texture(range(10))

    texture = pyvista.Texture(examples.mapfile)
    assert texture is not None
    image = texture.to_image()
    assert isinstance(image, pyvista.UniformGrid)

    with pytest.warns(PyvistaDeprecationWarning):
        texture.to_array()

    texture.repeat = True
    assert texture.repeat is True

    texture.repeat = False
    assert texture.repeat is False

    arr = texture.image_data
    assert isinstance(arr, pyvista.pyvista_ndarray)
    assert arr.shape[0] * arr.shape[1] == image.n_points

    texture = pyvista.Texture(examples.load_globe_texture())
    assert texture is not None


@pytest.mark.parametrize('inplace', [True, False])
def test_texture_flip(texture, inplace):
    orig_data = texture.image_data.copy()

    texture = texture.flip_x(inplace)
    assert not np.array_equal(orig_data, texture.image_data)
    texture = texture.flip_x(inplace)
    assert np.array_equal(orig_data, texture.image_data)

    texture = texture.flip_y(inplace)
    assert not np.array_equal(orig_data, texture.image_data)
    texture = texture.flip_y(inplace)
    assert np.array_equal(orig_data, texture.image_data)


@pytest.mark.parametrize('inplace', [True, False])
def test_texture_rotate(texture, inplace):
    orig_dim = texture.dimensions

    texture_rot = texture.rotate_ccw(inplace)
    assert texture_rot.dimensions == orig_dim[::-1]


def test_image_data():
    texture = pyvista.Texture()
    with pytest.raises(ValueError, match='Third dimension'):
        texture.image_data = np.zeros((10, 10, 2))


def test_texture_repr(texture):
    tex_repr = repr(texture)
    assert 'Cube Map:\tFalse' in tex_repr
    assert 'Components:\t3' in tex_repr
    assert 'Dimensions:\t300, 200\n' in tex_repr


def test_texture_from_images(image):
    texture = pyvista.Texture([image] * 6)
    assert texture.cube_map


def test_texture_to_grayscale(texture):
    bw_texture = texture.to_grayscale()
    assert bw_texture.n_components == 1
    assert bw_texture.dimensions == texture.dimensions
    assert bw_texture.image_data.dtype == np.uint8

    # no change when already converted
    assert bw_texture.to_grayscale() is bw_texture


def test_skybox():
    empty_texture = pyvista.Texture()
    with pytest.raises(ValueError, match='This texture is not a cube map'):
        empty_texture.to_skybox()

    texture = examples.load_globe_texture()
    texture.cube_map = False
    assert texture.cube_map is False

    texture.cube_map = True
    assert texture.cube_map is True

    skybox = texture.to_skybox()
    assert isinstance(skybox, vtk.vtkOpenGLSkybox)
