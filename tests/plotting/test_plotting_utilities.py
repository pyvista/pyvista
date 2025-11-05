from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageSequence
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.plotting.helpers import view_vectors
from pyvista.report import GPUInfo
from pyvista.report import _get_render_window_class

HAS_IMAGEIO = bool(importlib.util.find_spec('imageio'))


@pytest.mark.skip_plotting
def test_gpuinfo(monkeypatch):
    gpuinfo = GPUInfo()
    _repr = gpuinfo.__repr__()
    _repr_html = gpuinfo._repr_html_()
    assert isinstance(_repr, str)
    assert len(_repr) > 1
    assert isinstance(_repr_html, str)
    assert len(_repr_html) > 1

    # test corrupted internal infos
    monkeypatch.setattr(
        'pyvista.report._get_cached_render_window_info.info',
        'foo',
        raising=False,
    )
    for func_name in ['renderer', 'version', 'vendor']:
        with pytest.raises(RuntimeError, match=func_name):
            getattr(gpuinfo, func_name)()

    match = 'Unable to parse rendering information for the vtkRenderWindow class name.'
    with pytest.raises(RuntimeError, match=match):
        _get_render_window_class()


@pytest.mark.skip_plotting
def test_ray_trace_plot():
    sphere = pv.Sphere(radius=0.5, theta_resolution=10, phi_resolution=10)
    points, ind = sphere.ray_trace(
        [0, 0, 0],
        [1, 1, 1],
        plot=True,
        first_point=True,
        off_screen=True,
    )
    assert np.any(points)
    assert np.any(ind)


@pytest.mark.skip_plotting
@pytest.mark.skip_check_gc
def test_plot_curvature():
    sphere = pv.Sphere(radius=0.5, theta_resolution=10, phi_resolution=10)
    sphere.plot_curvature(off_screen=True)


@pytest.mark.skip_plotting
@pytest.mark.skip_check_gc
def test_plot_curvature_pointset():
    grid = examples.load_structured()
    grid.plot_curvature(off_screen=True)


@pytest.mark.skip_plotting
def test_plot_boundaries():
    # make sure to plot an object that has boundaries
    pv.Cube().plot_boundaries(off_screen=True)


@pytest.mark.skip_plotting
@pytest.mark.parametrize('flip', [True, False])
@pytest.mark.parametrize('faces', [True, False])
def test_plot_normals(flip, faces):
    sphere = pv.Sphere(radius=0.5, theta_resolution=10, phi_resolution=10)
    sphere.plot_normals(off_screen=True, flip=flip, faces=faces)


def test_get_sg_image_scraper():
    scraper = pv._get_sg_image_scraper()
    assert isinstance(scraper, pv.Scraper)
    assert callable(scraper)


def test_skybox(tmpdir):
    path = str(tmpdir.mkdir('tmpdir'))
    sets = ['posx', 'negx', 'posy', 'negy', 'posz', 'negz']
    filenames = []
    for suffix in sets:
        image = Image.new('RGB', (10, 10))
        filename = str(Path(path) / suffix) + '.jpg'
        image.save(filename)
        filenames.append(filename)

    skybox = pv.cubemap(path)
    assert isinstance(skybox, pv.Texture)

    with pytest.raises(FileNotFoundError, match='Unable to locate'):
        pv.cubemap('')

    skybox = pv.cubemap_from_filenames(filenames)
    assert isinstance(skybox, pv.Texture)

    with pytest.raises(ValueError, match='must contain 6 paths'):
        pv.cubemap_from_filenames(image_paths=['/path'])


def test_view_vectors():
    views = ('xy', 'yx', 'xz', 'zx', 'yz', 'zy')

    for view in views:
        vec, viewup = view_vectors(view)
        assert isinstance(vec, np.ndarray)
        assert np.array_equal(vec.shape, (3,))
        assert isinstance(viewup, np.ndarray)
        assert np.array_equal(viewup.shape, (3,))

    with pytest.raises(ValueError, match='Unexpected value for direction'):
        view_vectors('invalid')


@pytest.fixture
def gif_file(tmpdir):
    filename = str(tmpdir.join('sample.gif'))

    pl = pv.Plotter(window_size=(300, 200))
    pl.open_gif(filename, palettesize=16, fps=1)

    mesh = pv.Sphere()
    opacity = mesh.points[:, 0]
    opacity -= opacity.min()
    opacity /= opacity.max()
    for color in ['red', 'blue', 'green']:
        pl.clear()
        pl.background_color = 'w'
        pl.add_mesh(mesh, color=color, opacity=opacity)
        pl.camera_position = 'xy'
        pl.write_frame()

    pl.close()
    return filename


@pytest.mark.skipif(not HAS_IMAGEIO, reason='Requires imageio')
def test_gif_reader(gif_file):
    reader = pv.get_reader(gif_file)
    assert isinstance(reader, pv.GIFReader)
    assert reader.path == gif_file
    reader.show_progress()

    grid = reader.read()
    assert grid.n_arrays == 3

    img = Image.open(gif_file)
    new_grid = pv.ImageData(dimensions=(img.size[0], img.size[1], 1))

    # load each frame to the grid
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        data = np.array(frame.convert('RGB').getdata(), dtype=np.uint8)
        data_name = f'frame{i}'
        new_grid.point_data.set_array(data, data_name)
        assert np.allclose(grid[data_name], new_grid[data_name])

    img.close()
