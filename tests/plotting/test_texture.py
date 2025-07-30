from __future__ import annotations

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import VTKVersionError
from pyvista.plotting.texture import numpy_to_texture


def test_texture():
    with pytest.raises(TypeError, match='Cannot create a pyvista.Texture from'):
        texture = pv.Texture(range(10))

    texture = pv.Texture(examples.mapfile)
    assert texture is not None

    image = texture.to_image()
    assert isinstance(image, pv.ImageData)

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


@pytest.mark.parametrize('n_components', [1, 3, 4])
def test_texture_color_mode(n_components):
    im = pv.ImageData(dimensions=(2, 3, 4))
    im['data'] = np.zeros((im.n_points, n_components))
    texture = pv.Texture(im)
    assert texture.n_components == n_components
    if n_components in [3, 4]:
        assert texture.color_mode == 'direct'
    else:
        assert texture.color_mode == 'map'


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
    with pytest.raises(TypeError, match='pyvista.ImageData'):
        pv.Texture(['foo'] * 6)


def test_skybox_example():
    texture = examples.load_globe_texture()
    texture.cube_map = False
    assert texture.cube_map is False

    texture.cube_map = True
    assert texture.cube_map is True

    skybox = texture.to_skybox()
    assert isinstance(skybox, vtk.vtkOpenGLSkybox)


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
    assert 'Dimensions:   300, 200' in tex_repr


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


def test_numpy_to_texture():
    tex_im = np.ones((1024, 1024, 3), dtype=np.float64) * 255
    with pytest.warns(UserWarning, match='np.uint8'):
        tex = numpy_to_texture(tex_im)
    assert isinstance(tex, pv.Texture)
    assert tex.to_array().dtype == np.uint8


@pytest.mark.parametrize('as_str', [True, False])
@pytest.mark.parametrize('ndim', [3, 4])
def test_save_ply_texture_array(sphere, ndim, as_str, tmpdir):
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.ply'))

    texture = np.ones((sphere.n_points, ndim), np.uint8)
    texture[:, 2] = np.arange(sphere.n_points)[::-1]
    if as_str:
        sphere.point_data['texture'] = texture
        sphere.save(filename, texture='texture')
    else:
        sphere.save(filename, texture=texture)

    mesh = pv.PolyData(filename)
    color_array_name = 'RGB' if ndim == 3 else 'RGBA'
    assert np.allclose(mesh[color_array_name], texture)


@pytest.mark.parametrize('as_str', [True, False])
def test_save_ply_texture_array_catch(sphere, as_str, tmpdir):
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.ply'))

    texture = np.ones((sphere.n_points, 3), np.float32)
    if as_str:
        sphere.point_data['texture'] = texture
        with pytest.raises(ValueError, match='Invalid datatype'):
            sphere.save(filename, texture='texture')
    else:
        with pytest.raises(ValueError, match='Invalid datatype'):
            sphere.save(filename, texture=texture)

    with pytest.raises(TypeError):
        sphere.save(filename, texture=[1, 2, 3])


def test_texture_coordinates():
    """Test adding texture coordinates"""
    # create a rectangle vertices
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0.5, 0],
            [0, 0.5, 0],
        ],
    )

    # mesh faces
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]]).astype(np.int8)

    # Create simple texture coordinates
    texture_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    # Create the poly data
    mesh = pv.PolyData(vertices, faces)
    # Attempt setting the texture coordinates
    mesh.active_texture_coordinates = texture_coordinates
    # now grab the texture coordinates
    foo = mesh.active_texture_coordinates
    assert np.allclose(foo, texture_coordinates)


def test_multiple_texture_coordinates():
    mesh = examples.load_airplane()
    mesh.texture_map_to_plane(inplace=True, name='tex_a', use_bounds=False)
    mesh.texture_map_to_plane(inplace=True, name='tex_b', use_bounds=True)
    assert not np.allclose(mesh['tex_a'], mesh['tex_b'])


def test_inplace_no_overwrite_texture_coordinates():
    mesh = pv.Box()
    truth = mesh.texture_map_to_plane(inplace=False)
    mesh.texture_map_to_sphere(inplace=True)
    test = mesh.texture_map_to_plane(inplace=True)
    assert np.allclose(truth.active_texture_coordinates, test.active_texture_coordinates)
