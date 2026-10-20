from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pyvista as pv


@pytest.mark.parametrize('n_components', [1, 3, 4])
@pytest.mark.parametrize('extension', ['.png', '.vti'])
@pytest.mark.parametrize('as_str', [False, True])
def test_texture_save_round_trip(tmp_path, n_components, extension, as_str):
    array = np.arange(3 * 5 * n_components, dtype=np.uint8).reshape(3, 5, n_components)
    texture = pv.Texture(array)
    filename = tmp_path / f'texture{extension}'

    texture.save(str(filename) if as_str else filename)

    restored = pv.Texture(pv.read(filename))
    assert restored.dimensions == texture.dimensions
    np.testing.assert_array_equal(restored.to_array(), array)
    np.testing.assert_array_equal(texture.to_array(), array)


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('compression', [None, 'zlib'])
def test_texture_save_options(tmp_path, binary, compression):
    texture = pv.Texture(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    filename = tmp_path / 'texture.vti'
    texture.save(filename, binary=binary, compression=compression)
    assert pv.read(filename) == texture.to_image()
    contents = filename.read_bytes()
    data_format = b'binary' if binary else b'ascii'
    assert b'format="' + data_format + b'"' in contents
    assert (b'compressor="vtkZLibDataCompressor"' in contents) == (compression == 'zlib')


@pytest.mark.parametrize('image', [None, pv.ImageData()])
@pytest.mark.parametrize('extension', ['.png', '.vti'])
def test_texture_save_empty(tmp_path, image, extension):
    filename = tmp_path / f'texture{extension}'
    with pytest.raises(ValueError, match='The texture is empty and has no image'):
        pv.Texture(image).save(filename)
    assert not filename.exists()


def test_texture_save_cube_map(tmp_path):
    image = pv.Texture(np.arange(18, dtype=np.uint8).reshape(2, 3, 3)).to_image()
    texture = pv.Texture([image] * 6)
    filename = tmp_path / 'texture.png'
    with pytest.raises(ValueError, match='Saving cubemap textures is not supported'):
        texture.save(filename)
    assert not filename.exists()


def test_texture_save_invalid_extension(tmp_path):
    texture = pv.Texture(np.zeros((2, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='Invalid file extension'):
        texture.save(tmp_path / 'texture.invalid')


def test_texture_save_missing_parent(tmp_path):
    texture = pv.Texture(np.zeros((2, 3, 3), dtype=np.uint8))
    with pytest.raises(FileNotFoundError, match='Parent directory does not exist'):
        texture.save(tmp_path / 'missing' / 'texture.png')


def test_texture_save_custom_writer(tmp_path):
    texture = pv.Texture(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    filename = tmp_path / 'texture.custom'

    @pv.register_writer('.custom')
    def write_image(image, path, *, value):
        assert isinstance(image, pv.ImageData)
        assert image == texture.to_image()
        Path(path).write_bytes(value)

    texture.save(filename, value=b'custom image writer')
    assert filename.read_bytes() == b'custom image writer'


def test_texture_save_unexpected_writer_kwargs(tmp_path):
    texture = pv.Texture(np.zeros((2, 3, 3), dtype=np.uint8))
    filename = tmp_path / 'texture.png'
    with pytest.raises(TypeError, match='unexpected keyword arguments'):
        texture.save(filename, unsupported=True)
    assert not filename.exists()
