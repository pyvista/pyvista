import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples
from pyvista.utilities import PyVistaDeprecationWarning


def test_texture():
    with pytest.raises(TypeError, match='Cannot create a pyvista.Texture from'):
        texture = pyvista.Texture(range(10))

    texture = pyvista.Texture(examples.mapfile)
    assert texture is not None
    image = texture.to_image()
    assert isinstance(image, pyvista.UniformGrid)

    with pytest.warns(PyVistaDeprecationWarning):
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


def test_texture_empty_init():
    texture = pyvista.Texture()
    assert texture.dimensions == (0, 0)
    assert texture.n_components == 0


def test_texture_grayscale():
    # verify a grayscale image can be created
    texture = pyvista.Texture(np.zeros((10, 10, 1), dtype=np.uint8))
    assert texture.dimensions == (10, 10)
    assert texture.n_components == 1


@pytest.mark.parametrize('inplace', [False])
def test_texture_rotate_cw(texture, inplace):
    orig_dim = texture.dimensions
    orig_data = texture.image_data.copy()

    texture_rot = texture.rotate_cw(inplace)
    assert texture_rot.dimensions == orig_dim[::-1]
    assert np.allclose(np.rot90(orig_data), texture_rot.image_data)


@pytest.mark.parametrize('inplace', [False])
def test_texture_rotate_ccw(texture, inplace):
    orig_dim = texture.dimensions
    orig_data = texture.image_data.copy()

    texture_rot = texture.rotate_ccw(inplace)
    assert texture_rot.dimensions == orig_dim[::-1]
    assert np.allclose(np.rot90(orig_data, k=3), texture_rot.image_data)


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
    with pytest.raises(ValueError, match='consist of 6'):
        pyvista.Texture(['foo'] * 6)


def test_texture_to_grayscale(texture):
    bw_texture = texture.to_grayscale()
    assert bw_texture.n_components == 1
    assert bw_texture.dimensions == texture.dimensions
    assert bw_texture.image_data.dtype == np.uint8

    # no change when already converted
    assert bw_texture.to_grayscale() is bw_texture
