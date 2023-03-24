import numpy as np
import vtk

import pyvista as pv
from pyvista import examples


def test_texture():
    texture = pv.Texture(examples.mapfile)
    assert texture is not None
    image = texture.to_image()
    assert isinstance(image, pv.UniformGrid)
    arr = texture.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] * arr.shape[1] == image.n_points
    texture = pv.Texture(examples.load_globe_texture())
    assert texture is not None


def test_skybox():
    texture = examples.load_globe_texture()
    texture.cube_map = False
    assert texture.cube_map is False

    texture.cube_map = True
    assert texture.cube_map is True

    skybox = texture.to_skybox()
    assert isinstance(skybox, vtk.vtkOpenGLSkybox)


def test_texture_empty_init():
    texture = pv.Texture()
    assert texture.dimensions == (0, 0)
    assert texture.n_components == 0


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
    assert isinstance(texture.wrap, texture.WrapType)
