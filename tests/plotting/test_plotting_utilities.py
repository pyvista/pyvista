from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
from PIL import ImageSequence
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.plotting._plotting import _resolve_scalars_field
from pyvista.plotting._plotting import reduce_component_scalars
from pyvista.plotting.helpers import view_vectors
from pyvista.report import GPUInfo
from pyvista.report import _get_render_window_class
from tests.conftest import PILLOW_VERSION_INFO

HAS_IMAGEIO = bool(importlib.util.find_spec('imageio'))


@pytest.mark.skip_plotting
def test_gpuinfo(monkeypatch):
    gpuinfo = GPUInfo()
    _repr = repr(gpuinfo)
    _str = str(gpuinfo)
    _repr_html = gpuinfo._repr_html_()
    assert isinstance(_repr, str)
    assert _repr.startswith('<GPUInfo object at ')
    assert isinstance(_str, str)
    assert len(_str) > 1
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
        pillow_get_data = (
            Image.Image.get_flattened_data
            if PILLOW_VERSION_INFO >= (12, 1)
            else Image.Image.getdata
        )
        data = np.array(pillow_get_data(frame.convert('RGB')), dtype=np.uint8)
        data_name = f'frame{i}'
        new_grid.point_data.set_array(data, data_name)
        assert np.allclose(grid[data_name], new_grid[data_name])

    img.close()


def test_resolve_scalars_field_raises_on_mismatch():
    """Unit-test the shared ``_resolve_scalars_field`` helper's raise branch.

    The ``add_mesh`` happy path never reaches this branch; the caller
    pre-checks that ``shape[0] in (n_points, n_cells)`` before calling.
    Exercise it directly so the error message stays covered.
    """
    sphere = pv.Sphere()
    with pytest.raises(ValueError, match='Length of scalars array'):
        _resolve_scalars_field(np.zeros(42, dtype=np.float32), sphere, 'point')


@pytest.mark.parametrize(
    ('component', 'error_type', 'match'),
    [
        ('not-an-int', TypeError, 'component must be None or an integer'),
        (-1, ValueError, 'nonnegative'),
        (9, ValueError, 'less than the'),
    ],
)
def test_reduce_component_scalars_invalid(component, error_type, match):
    """Invalid ``component`` values raise from the shared reduction helper."""
    scalars = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(error_type, match=match):
        reduce_component_scalars(scalars, 'vec', component)


def test_reduce_component_scalars_norm_path():
    """``component=None`` reduces via ``np.linalg.norm`` and synthesizes
    the ``-normed`` derived name."""
    vec = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    reduced, name = reduce_component_scalars(vec, 'u', None)
    assert name == 'u-normed'
    np.testing.assert_allclose(reduced, [5.0, 5.0])


def test_reduce_component_scalars_component_int():
    """Picking an integer ``component`` extracts the column and
    synthesizes the ``-<component>`` derived name."""
    vec = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    reduced, name = reduce_component_scalars(vec, 'u', 1)
    assert name == 'u-1'
    np.testing.assert_allclose(reduced, [2.0, 5.0])


def test_resolve_scalars_field_returns_cell():
    """Cell-length scalars resolve to ``'cell'`` without the caller
    needing to pass ``preference='cell'``."""
    sphere = pv.Sphere()
    result = _resolve_scalars_field(np.zeros(sphere.n_cells, dtype=np.float32), sphere, 'point')
    assert result == 'cell'


def test_resolve_scalars_field_prefers_hint_when_ambiguous():
    """When the array length matches both ``n_points`` and ``n_cells``
    the helper falls back to the caller-provided ``preference``.

    Real pyvista meshes rarely have ``n_points == n_cells``; use a
    ``Mock`` to isolate the disambiguation branch.
    """
    mesh = Mock(n_points=10, n_cells=10)
    scalars = np.zeros(10, dtype=np.float32)
    assert _resolve_scalars_field(scalars, mesh, 'point') == 'point'
    assert _resolve_scalars_field(scalars, mesh, 'cell') == 'cell'


def test_add_mesh_raw_numpy_mismatched_length_raises():
    """``add_mesh`` with raw numpy scalars of mismatched length raises clearly.

    Covers the downstream ``raise_not_matching`` path from
    ``mapper._configure_scalars_mode``. The length falls through
    ``plotter.add_mesh``'s Block A (shape[0] doesn't match points/cells)
    and is raveled by ``mapper.set_scalars`` to a 1D array whose size
    still doesn't match, tripping the final validation.
    """
    sphere = pv.Sphere()
    pl = pv.Plotter()
    with pytest.raises(ValueError, match='Number of scalars'):
        pl.add_mesh(sphere, scalars=np.zeros(42, dtype=np.float32))
