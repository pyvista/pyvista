import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import VTKVersionError
from pyvista.utilities.misc import PyVistaDeprecationWarning


def test_texture():
    with pytest.raises(TypeError, match='Cannot create a pyvista.Texture from'):
        texture = pv.Texture(range(10))

    texture = pv.Texture(examples.mapfile)
    assert texture is not None

    image = texture.to_image()
    assert isinstance(image, pv.UniformGrid)

    arr = texture.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] * arr.shape[1] == image.n_points

    # test texture from array

    texture = pv.Texture(examples.load_globe_texture())
    assert texture is not None


def test_texture_empty_init():
    texture = pv.Texture()
    assert texture.dimensions == (0, 0)
    assert texture.n_components == 0


def test_texture_grayscale_init():
    # verify a grayscale image can be created on init
    texture = pv.Texture(np.zeros((10, 10, 1), dtype=np.uint8))
    assert texture.dimensions == (10, 10)
    assert texture.n_components == 1


def test_from_array():
    texture = pv.Texture()
    with pytest.raises(ValueError, match='Third dimension'):
        texture._from_array(np.zeros((10, 10, 2)))

    with pytest.raises(ValueError, match='must be nn by nm'):
        texture._from_array(np.zeros(10))


def test_texture_rotate_cw(texture):
    orig_dim = texture.dimensions
    orig_data = texture.to_array()

    texture_rot = texture.rotate_cw()
    assert texture_rot.dimensions == orig_dim[::-1]
    assert np.allclose(np.rot90(orig_data), texture_rot.to_array())


def test_texture_rotate_ccw(texture):
    orig_dim = texture.dimensions
    orig_data = texture.to_array()

    texture_rot = texture.rotate_ccw()
    assert texture_rot.dimensions == orig_dim[::-1]
    assert np.allclose(np.rot90(orig_data, k=3), texture_rot.to_array())


def test_texture_from_images(image):
    texture = pv.Texture([image] * 6)
    assert texture.cube_map
    with pytest.raises(TypeError, match='pyvista.UniformGrid'):
        pv.Texture(['foo'] * 6)


def test_skybox():
    texture = examples.load_globe_texture()
    texture.cube_map = False
    assert texture.cube_map is False

    texture.cube_map = True
    assert texture.cube_map is True

    skybox = texture.to_skybox()
    assert isinstance(skybox, vtk.vtkOpenGLSkybox)


def test_flip_deprecated(texture):
    with pytest.warns(PyVistaDeprecationWarning, match='flip_x'):
        _ = texture.flip(0)


def test_flip_x(texture):
    flipped = texture.flip_x()
    assert flipped.dimensions == texture.dimensions
    assert not np.allclose(flipped.to_array(), texture.to_array())


def test_flip_y(texture):
    flipped = texture.flip_y()
    assert flipped.dimensions == texture.dimensions
    assert not np.allclose(flipped.to_array(), texture.to_array())


def test_texture_repr(texture):
    tex_repr = repr(texture)
    assert 'Components:   3' in tex_repr
    assert 'Cube Map:     False' in tex_repr
    assert 'Dimensions:   300, 200\n' in tex_repr


def test_interpolate(texture):
    assert isinstance(texture.interpolate, bool)
    for value in [True, False]:
        texture.interpolate = value
        assert texture.interpolate is value
        assert bool(texture.GetInterpolate()) is value


def test_mipmap(texture):
    assert isinstance(texture.mipmap, bool)
    for value in [True, False]:
        texture.mipmap = value
        assert texture.mipmap is value
        assert bool(texture.GetMipmap()) is value


def test_repeat(texture):
    assert isinstance(texture.repeat, bool)
    for value in [True, False]:
        texture.repeat = value
        assert texture.repeat is value
        assert bool(texture.GetRepeat()) is value


def test_wrap(texture):
    if pv.vtk_version_info < (9, 1):
        with pytest.raises(VTKVersionError):
            assert isinstance(texture.wrap, texture.WrapType)
        with pytest.raises(VTKVersionError):
            texture.wrap = texture.WrapType.CLAMP_TO_EDGE
    else:
        assert isinstance(texture.wrap, texture.WrapType)
        texture.wrap = texture.WrapType.CLAMP_TO_EDGE
        assert texture.wrap == texture.WrapType.CLAMP_TO_EDGE


def test_grayscale(texture):
    grayscale = texture.to_grayscale()
    assert grayscale.n_components == 1
    assert grayscale.dimensions == texture.dimensions

    gray_again = grayscale.to_grayscale()
    assert gray_again == grayscale
    assert gray_again is not grayscale  # equal and copy
