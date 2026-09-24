from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
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
from pyvista.plotting.utilities.gl_checks import _QT_GUI_MODULES
from pyvista.plotting.utilities.gl_checks import _gl_platform_from_maps
from pyvista.plotting.utilities.gl_checks import _loaded_gl_platform
from pyvista.plotting.utilities.gl_checks import _offscreen_probe_render_window
from pyvista.plotting.utilities.gl_checks import _process_uses_egl
from pyvista.plotting.utilities.gl_checks import _qt_platform_name
from pyvista.plotting.utilities.gl_checks import uses_egl
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
def test_plot_curvature():
    sphere = pv.Sphere(radius=0.5, theta_resolution=10, phi_resolution=10)
    sphere.plot_curvature(off_screen=True)


@pytest.mark.skip_plotting
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

    with pytest.raises(ValueError, match="view 'invalid' is not valid"):
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


@pytest.mark.parametrize(
    ('vectors', 'component', 'expected_name', 'expected'),
    [
        ([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], None, 'u-normed', [5.0, 5.0]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 1, 'u-1', [2.0, 5.0]),
    ],
    ids=['norm', 'component'],
)
def test_reduce_component_scalars(vectors, component, expected_name, expected):
    """``component=None`` norms the vectors; an integer picks that column."""
    vec = np.array(vectors, dtype=np.float32)
    reduced, name = reduce_component_scalars(vec, 'u', component)
    assert name == expected_name
    np.testing.assert_allclose(reduced, expected)


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


@pytest.fixture
def no_loaded_gl(monkeypatch):
    """Hide the GL libraries this test process has mapped from the probes."""
    monkeypatch.setattr('pyvista.plotting.utilities.gl_checks._loaded_gl_platform', lambda: None)


@pytest.mark.usefixtures('no_loaded_gl')
def test_offscreen_probe_render_window(monkeypatch):
    """GL probes must not create GLX windows inside a Wayland session.

    Making a GLX context current in a process already using EGL (e.g. Qt on
    the native wayland platform) aborts the process, so under Wayland the
    probe window must be EGL-based. See pyvista/pyvistaqt#445.
    """
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.delenv('VTK_DEFAULT_OPENGL_WINDOW', raising=False)
    default_cls = type(_offscreen_probe_render_window())
    assert issubclass(default_cls, pv._vtk.vtkRenderWindow)

    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    monkeypatch.delenv('DISPLAY', raising=False)
    if not pv._vtk.has_attr('vtkEGLRenderWindow'):
        pytest.skip('VTK build lacks vtkEGLRenderWindow')
    assert isinstance(_offscreen_probe_render_window(), pv._vtk.vtkEGLRenderWindow)

    # an explicit VTK_DEFAULT_OPENGL_WINDOW override wins over the heuristic
    monkeypatch.setenv('VTK_DEFAULT_OPENGL_WINDOW', 'vtkXOpenGLRenderWindow')
    sentinel = object()
    monkeypatch.setattr(pv._vtk, 'vtkRenderWindow', lambda: sentinel)
    assert _offscreen_probe_render_window() is sentinel


@pytest.mark.usefixtures('no_loaded_gl')
def test_uses_egl_wayland(monkeypatch):
    """uses_egl must not instantiate the factory render window under Wayland.

    Construction (and destruction) of the default GLX window aborts a process
    already using EGL. See pyvista/pyvistaqt#445.
    """
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('VTK_DEFAULT_OPENGL_WINDOW', raising=False)
    has_x = pv._vtk.has_attr('vtkXOpenGLRenderWindow')
    assert uses_egl() is not has_x

    monkeypatch.setenv('VTK_DEFAULT_OPENGL_WINDOW', 'vtkEGLRenderWindow')
    assert uses_egl() is True
    monkeypatch.setenv('VTK_DEFAULT_OPENGL_WINDOW', 'vtkXOpenGLRenderWindow')
    assert uses_egl() is False


def test_uses_egl_loaded_egl(monkeypatch):
    """A mapped libEGL answers uses_egl even when VTK was built with X."""
    monkeypatch.setattr('pyvista.plotting.utilities.gl_checks._loaded_gl_platform', lambda: 'egl')
    monkeypatch.setattr(pv._vtk, 'has_attr', lambda name: name == 'vtkXOpenGLRenderWindow')
    monkeypatch.delenv('VTK_DEFAULT_OPENGL_WINDOW', raising=False)
    for name in _QT_GUI_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert uses_egl() is True


class _FakeApp:
    def __init__(self, platform_name):
        self._platform_name = platform_name

    def platformName(self):  # noqa: N802
        return self._platform_name


def _fake_binding(monkeypatch, module_name, platform_name):
    """Install a stand-in Qt binding exposing QGuiApplication.instance()."""
    module = ModuleType(module_name)
    app = None if platform_name is None else _FakeApp(platform_name)
    module.QGuiApplication = type('QGuiApplication', (), {'instance': staticmethod(lambda: app)})
    monkeypatch.setitem(sys.modules, module_name, module)


@pytest.mark.parametrize('module_name', _QT_GUI_MODULES)
def test_qt_platform_name(monkeypatch, module_name):
    """A live Qt application reports the platform it actually connected to."""
    _fake_binding(monkeypatch, module_name, 'xcb')
    assert _qt_platform_name() == 'xcb'


def test_qt_platform_name_no_application(monkeypatch):
    """An imported binding with no application running answers nothing."""
    for name in _QT_GUI_MODULES[1:]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    _fake_binding(monkeypatch, _QT_GUI_MODULES[0], None)
    assert _qt_platform_name() is None


@pytest.mark.parametrize(
    ('wayland_display', 'display', 'platform_name', 'loaded', 'expected'),
    [
        # The case this exists for: QT_QPA_PLATFORM=xcb runs Qt through
        # XWayland inside a Wayland session, so the process's GL is GLX even
        # though a compositor is running.
        ('wayland-0', ':0', 'xcb', None, False),
        ('wayland-0', ':0', 'wayland', None, True),
        # Qt outranks the mapped libraries, which it may load for either.
        ('wayland-0', ':0', 'xcb', 'egl', False),
        # A host other than Qt, e.g. GTK on native Wayland with XWayland's
        # DISPLAY set, is known by the GL library it has loaded.
        ('wayland-0', ':0', None, 'egl', True),
        ('wayland-0', None, None, 'glx', False),
        # With neither, VTK's factory uses GLX whenever DISPLAY is set, e.g.
        # a Jupyter kernel in a Wayland session.
        ('wayland-0', ':0', None, None, False),
        ('wayland-0', None, None, None, True),
        (None, None, None, None, False),
        # A Qt application on X11 with no compositor at all.
        (None, ':0', 'xcb', None, False),
    ],
)
def test_process_uses_egl(monkeypatch, wayland_display, display, platform_name, loaded, expected):
    """A running Qt application outranks the mapped libraries and the session variables."""
    monkeypatch.setattr('pyvista.plotting.utilities.gl_checks._loaded_gl_platform', lambda: loaded)
    for var, value in (('WAYLAND_DISPLAY', wayland_display), ('DISPLAY', display)):
        monkeypatch.delenv(var, raising=False)
        if value is not None:
            monkeypatch.setenv(var, value)
    for name in _QT_GUI_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    if platform_name is not None:
        _fake_binding(monkeypatch, _QT_GUI_MODULES[0], platform_name)
    assert _process_uses_egl() is expected


def test_offscreen_probe_follows_qt_platform(monkeypatch):
    """Qt on xcb inside a Wayland session must not get an EGL probe window.

    An EGL render window cannot make its context current in a GLX process, so
    every probe render logs ``Unable to eglMakeCurrent`` with EGL_BAD_ACCESS.
    """
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    monkeypatch.delenv('VTK_DEFAULT_OPENGL_WINDOW', raising=False)
    if not pv._vtk.has_attr('vtkEGLRenderWindow'):
        pytest.skip('VTK build lacks vtkEGLRenderWindow')

    _fake_binding(monkeypatch, _QT_GUI_MODULES[0], 'xcb')
    assert not isinstance(_offscreen_probe_render_window(), pv._vtk.vtkEGLRenderWindow)

    _fake_binding(monkeypatch, _QT_GUI_MODULES[0], 'wayland')
    assert isinstance(_offscreen_probe_render_window(), pv._vtk.vtkEGLRenderWindow)


_EGL_LINE = '7f00-7f01 r--p 00000000 103:02 1 /usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0\n'
_GLX_LINE = '7f02-7f03 r--p 00000000 103:02 2 /usr/lib/x86_64-linux-gnu/libGLX_mesa.so.0\n'
_GL_LINE = '7f04-7f05 r--p 00000000 103:02 3 /usr/lib/x86_64-linux-gnu/libGL.so.1.7.0\n'


@pytest.mark.parametrize(
    ('maps', 'expected'),
    [
        (_EGL_LINE, 'egl'),
        (_GLX_LINE, 'glx'),
        (_EGL_LINE + _GLX_LINE, None),
        ('', None),
        # libGL alone is linked by EGL programs too, so it decides nothing
        (_GL_LINE, None),
        (_GL_LINE + _EGL_LINE, 'egl'),
    ],
)
def test_gl_platform_from_maps(maps, expected):
    """Exactly one of libEGL and libGLX mapped names the GL platform."""
    assert _gl_platform_from_maps(maps) == expected


def test_loaded_gl_platform(monkeypatch, tmp_path):
    """The process's memory map is read, and its absence answers nothing."""
    maps = tmp_path / 'maps'
    maps.write_text(_EGL_LINE, encoding='utf-8')
    monkeypatch.setattr('pyvista.plotting.utilities.gl_checks._PROC_MAPS', maps)
    assert _loaded_gl_platform() == 'egl'

    # any file can be mapped, so paths need not be valid UTF-8
    maps.write_bytes(
        b'7f06-7f07 r--p 00000000 103:02 4 /data/r\xe9sum\xe9.bin\n' + _EGL_LINE.encode()
    )
    assert _loaded_gl_platform() == 'egl'

    monkeypatch.setattr('pyvista.plotting.utilities.gl_checks._PROC_MAPS', tmp_path / 'missing')
    assert _loaded_gl_platform() is None
