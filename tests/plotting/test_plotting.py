"""This test module tests any functionality that requires plotting.

See the image regression notes in doc/extras/developer_notes.rst

"""

from __future__ import annotations

import inspect
import io
import os
import pathlib
from pathlib import Path
import re
import time
from types import FunctionType
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import get_args

import numpy as np
from PIL import Image
import pytest

import pyvista as pv
from pyvista import demos
from pyvista import examples
from pyvista.core.errors import DeprecationError
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.plotting import BackgroundPlotter
from pyvista.plotting import QtDeprecationError
from pyvista.plotting import QtInteractor
from pyvista.plotting import _vtk
from pyvista.plotting.axes_assembly import ScaleModeOptions
from pyvista.plotting.colors import matplotlib_default_colors
from pyvista.plotting.errors import InvalidCameraError
from pyvista.plotting.errors import RenderWindowUnavailable
from pyvista.plotting.plotter import SUPPORTED_FORMATS
from pyvista.plotting.texture import numpy_to_texture
from pyvista.plotting.utilities import algorithms
from tests.core.test_imagedata_filters import labeled_image  # noqa: F401
from tests.examples.test_cell_examples import cell_example_functions

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import ItemsView

    from pytest_mock import MockerFixture

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


HAS_IMAGEIO = True
try:
    import imageio
except ModuleNotFoundError:
    HAS_IMAGEIO = False

try:
    import imageio_ffmpeg

    imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    if HAS_IMAGEIO:
        imageio.plugins.ffmpeg.download()
    else:
        raise


THIS_PATH = pathlib.Path(__file__).parent.absolute()


def using_mesa():
    """Determine if using mesa."""
    pl = pv.Plotter(notebook=False, off_screen=True)
    pl.show(auto_close=False)
    gpu_info = pl.render_window.ReportCapabilities()
    pl.close()

    regex = re.compile('OpenGL version string:(.+)\n')
    return 'Mesa' in regex.findall(gpu_info)[0]


# always set on Windows CI
# These tests fail with mesa opengl on windows
skip_mesa = pytest.mark.skipif(using_mesa(), reason='Does not display correctly within OSMesa')
skip_windows_mesa = skip_mesa and pytest.mark.skip_windows(
    'Does not display correctly within OSMesa on Windows'
)
skip_lesser_9_3_X = pytest.mark.needs_vtk_version(  # noqa: N816
    9, 3, reason='Functions not implemented before 9.3.X'
)
skip_lesser_9_4_X = pytest.mark.needs_vtk_version(  # noqa: N816
    9, 4, reason='Functions not implemented before 9.4.X or invalid results prior'
)
skip_lesser_9_4_X_depth_peeling = pytest.mark.needs_vtk_version(  # noqa: N816
    9, 4, reason='Depth peeling unstable on CI before 9.4.X'
)

CI_WINDOWS = os.environ.get('CI_WINDOWS', 'false').lower() == 'true'


@pytest.fixture(autouse=True)
def verify_image_cache_wrapper(verify_image_cache):
    return verify_image_cache


@pytest.fixture
def no_images_to_verify(verify_image_cache_wrapper):
    verify_image_cache_wrapper.allow_useless_fixture = True
    yield verify_image_cache_wrapper
    assert (n_calls := verify_image_cache_wrapper.n_calls) == 0, (
        f'No images were expected to be generated, but got {n_calls}'
    )


@pytest.fixture
def multicomp_poly():
    """Create a dataset with vector values on points and cells."""
    data = pv.Plane(direction=(0, 0, -1))

    vector_values_points = np.empty((data.n_points, 3))
    vector_values_points[:, 0] = np.arange(data.n_points)
    vector_values_points[:, 1] = np.arange(data.n_points)[::-1]
    vector_values_points[:, 2] = 0

    vector_values_cells = np.empty((data.n_cells, 3))
    vector_values_cells[:, 0] = np.arange(data.n_cells)
    vector_values_cells[:, 1] = np.arange(data.n_cells)[::-1]
    vector_values_cells[:, 2] = 0

    data['vector_values_points'] = vector_values_points
    data['vector_values_cells'] = vector_values_cells
    return data


@pytest.mark.usefixtures('no_images_to_verify')
def test_pyvista_qt_raises():
    match = re.escape(QtDeprecationError.message.format(*[BackgroundPlotter.__name__] * 4))
    with pytest.raises(QtDeprecationError, match=match):
        BackgroundPlotter()

    match = re.escape(QtDeprecationError.message.format(*[QtInteractor.__name__] * 4))
    with pytest.raises(QtDeprecationError, match=match):
        QtInteractor()


@pytest.mark.usefixtures('no_images_to_verify')
def test_plotting_module_raises(mocker: MockerFixture):
    from pyvista.plotting import plotting

    m = mocker.patch.object(plotting, 'inspect')
    m.getattr_static.side_effect = AttributeError

    match = re.escape(
        'Module `pyvista.plotting.plotting` has been deprecated and we could not automatically '
        'find `foo`'
    )
    with pytest.raises(AttributeError, match=match):
        plotting.foo  # noqa: B018


def test_import_gltf(verify_image_cache):
    # image cache created with 9.0.20210612.dev0
    verify_image_cache.high_variance_test = True

    filename = str(Path(THIS_PATH) / '..' / 'example_files' / 'Box.glb')
    pl = pv.Plotter()

    with pytest.raises(FileNotFoundError):
        pl.import_gltf('not a file')

    pl.import_gltf(filename)
    assert np.allclose(pl.bounds, (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
    pl.show()


def test_export_gltf(tmpdir, sphere, airplane, hexbeam, verify_image_cache):
    # image cache created with 9.0.20210612.dev0
    verify_image_cache.high_variance_test = True
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.gltf'))

    pl = pv.Plotter()
    pl.add_mesh(sphere, smooth_shading=True)
    pl.add_mesh(airplane)
    pl.add_mesh(hexbeam)  # to check warning
    with pytest.warns(UserWarning, match='Plotter contains non-PolyData datasets'):
        pl.export_gltf(filename)

    pl_import = pv.Plotter()
    pl_import.import_gltf(filename)
    pl_import.show()

    with pytest.raises(RuntimeError, match='This plotter has been closed'):
        pl_import.export_gltf(filename)


def test_import_vrml():
    filename = str(Path(THIS_PATH) / '..' / 'example_files' / 'Box.wrl')

    match = (
        'VRML files must be imported directly into a Plotter. '
        'See `pyvista.Plotter.import_vrml` for details.'
    )
    with pytest.raises(ValueError, match=match):
        pv.read(filename)

    pl = pv.Plotter()

    with pytest.raises(FileNotFoundError):
        pl.import_vrml('not a file')

    pl.import_vrml(filename)
    assert np.allclose(pl.bounds, (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
    pl.show()


def test_export_vrml(tmpdir, sphere):
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.wrl'))

    pl = pv.Plotter()
    pl.add_mesh(sphere, smooth_shading=True)
    pl.export_vrml(filename)

    pl_import = pv.Plotter()
    pl_import.import_vrml(filename)
    pl_import.show()

    with pytest.raises(RuntimeError, match='This plotter has been closed'):
        pl_import.export_vrml(filename)


def test_import_3ds():
    filename = examples.download_3ds.download_iflamigm()
    pl = pv.Plotter()

    with pytest.raises(FileNotFoundError, match='Unable to locate'):
        pl.import_3ds('not a file')

    pl.import_3ds(filename)
    assert np.allclose(
        pl.bounds,
        (
            -5.379246234893799,
            5.364696979522705,
            -1.9769330024719238,
            2.731842041015625,
            -7.883847236633301,
            5.437096118927002,
        ),
    )
    pl.show()


def test_import_obj():
    download_obj_file = examples.download_room_surface_mesh(load=False)
    pl = pv.Plotter()

    with pytest.raises(FileNotFoundError, match='Unable to locate'):
        pl.import_obj('not a file')

    pl.import_obj(download_obj_file)
    assert np.allclose(pl.bounds, (-10.0, 10.0, 0.0, 4.5, -10.0, 10.0))
    pl.show()


def test_import_obj_with_texture():
    filename = examples.download_doorman(load=False)
    pl = pv.Plotter()
    pl.import_obj(filename)
    pl.show(cpos='xy')


@pytest.mark.skip_windows
@pytest.mark.skipif(CI_WINDOWS, reason='Windows CI testing segfaults on pbr')
def test_pbr(sphere, verify_image_cache):
    """Test PBR rendering"""
    verify_image_cache.high_variance_test = True

    texture = examples.load_globe_texture()
    texture.mipmap = True
    texture.interpolate = True

    pl = pv.Plotter(lighting=None)
    pl.set_environment_texture(texture)
    pl.add_light(pv.Light())
    pl.add_mesh(
        sphere,
        color='w',
        pbr=True,
        metallic=0.8,
        roughness=0.2,
        smooth_shading=True,
        diffuse=1,
    )
    pl.add_mesh(
        pv.Sphere(center=(0, 0, 1)),
        color='w',
        pbr=True,
        metallic=0.0,
        roughness=1.0,
        smooth_shading=True,
        diffuse=1,
    )
    pl.show()


@pytest.mark.parametrize('resample', [True, 0.5])
def test_set_environment_texture_cubemap(resample, verify_image_cache):
    """Test set_environment_texture with a cubemap."""
    # Skip due to large variance
    verify_image_cache.windows_skip_image_cache = True
    verify_image_cache.macos_skip_image_cache = True

    pl = pv.Plotter(lighting=None)
    texture = examples.download_cubemap_park()
    pl.set_environment_texture(texture, is_srgb=True, resample=resample)
    pl.camera_position = 'xy'
    pl.camera.zoom(0.7)
    _ = pl.add_mesh(pv.Sphere(), pbr=True, roughness=0.1, metallic=0.5)
    pl.show()


@pytest.mark.skip_windows
@pytest.mark.skip_mac('MacOS CI fails when downloading examples')
def test_remove_environment_texture_cubemap(sphere):
    """Test remove_environment_texture with a cubemap."""
    texture = examples.download_sky_box_cube_map()

    pl = pv.Plotter()
    pl.set_environment_texture(texture)
    pl.add_mesh(sphere, color='w', pbr=True, metallic=0.8, roughness=0.2)
    pl.remove_environment_texture()
    pl.show()


def test_plot_pyvista_ndarray(sphere):
    # verify we can plot pyvista_ndarray
    pv.plot(sphere.points)

    pl = pv.Plotter()
    pl.add_points(sphere.points)
    pl.add_points(sphere.points + 1)
    pl.show()


def test_plot_increment_point_size():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
    pl = pv.Plotter()
    pl.add_points(points + 1)
    pl.add_lines(points)
    pl.increment_point_size_and_line_width(5)
    pl.show()


def test_plot_update(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show(auto_close=False)
    pl.update()
    time.sleep(0.1)
    pl.update()
    pl.update(force_redraw=True)
    pl.close()


@pytest.mark.parametrize('anti_aliasing', [True, 'msaa', False])
def test_plot(sphere, tmpdir, verify_image_cache, anti_aliasing):
    verify_image_cache.high_variance_test = True
    verify_image_cache.macos_skip_image_cache = True
    verify_image_cache.windows_skip_image_cache = True

    tmp_dir = tmpdir.mkdir('tmpdir2')
    filename = str(tmp_dir.join('tmp.png'))
    scalars = np.arange(sphere.n_points)
    cpos, img = pv.plot(
        sphere,
        full_screen=True,
        text='this is a sphere',
        show_bounds=True,
        color='r',
        style='surface',
        line_width=2,
        scalars=scalars,
        flip_scalars=True,
        cmap='bwr',
        interpolate_before_map=True,
        screenshot=filename,
        return_img=True,
        return_cpos=True,
        anti_aliasing=anti_aliasing,
    )
    assert isinstance(cpos, pv.CameraPosition)
    assert isinstance(img, np.ndarray)
    assert Path(filename).is_file()

    verify_image_cache.skip = True
    filename = pathlib.Path(str(tmp_dir.join('tmp2.png')))
    pv.plot(sphere, screenshot=filename)

    # Ensure it added a PNG extension by default
    assert filename.with_suffix('.png').is_file()

    # test invalid extension
    filename = pathlib.Path(str(tmp_dir.join('tmp3.foo')))
    with pytest.raises(ValueError):  # noqa: PT011
        pv.plot(sphere, screenshot=filename)


def test_plot_helper_volume(uniform, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    uniform.plot(
        volume=True,
        parallel_projection=True,
        show_scalar_bar=False,
        show_grid=True,
    )


def test_plot_helper_two_datasets(sphere, airplane):
    pv.plot([sphere, airplane])


def test_plot_helper_two_volumes(uniform, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    grid = uniform.copy()
    grid.origin = (0, 0, 10)
    pv.plot(
        [uniform, grid],
        volume=True,
        show_scalar_bar=False,
    )


def test_plot_volume_ugrid(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    # Handle UnsutructuredGrid directly
    grid = examples.load_hexbeam()
    pl = pv.Plotter()
    pl.add_volume(grid, scalars='sample_point_scalars')
    pl.show()

    # Handle 3D structured grid
    grid = examples.load_uniform().cast_to_structured_grid()
    pl = pv.Plotter()
    pl.add_volume(grid, scalars='Spatial Point Data')
    pl.show()

    # Make sure PolyData fails
    mesh = pv.Sphere()
    mesh['scalars'] = mesh.points[:, 1]
    pl = pv.Plotter()
    with pytest.raises(TypeError):
        pl.add_volume(mesh, scalars='scalars')
    pl.close()

    # Make sure 2D StructuredGrid fails
    mesh = examples.load_structured()  # wavy surface
    mesh['scalars'] = mesh.points[:, 1]
    pl = pv.Plotter()
    with pytest.raises(ValueError):  # noqa: PT011
        pl.add_volume(mesh, scalars='scalars')
    pl.close()


def test_plot_return_cpos(sphere):
    cpos = sphere.plot(return_cpos=True)
    assert isinstance(cpos, pv.CameraPosition)
    assert sphere.plot(return_cpos=False) is None


def test_add_title(verify_image_cache):
    verify_image_cache.high_variance_test = True
    pl = pv.Plotter()
    pl.add_title('Plot Title')
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_plot_invalid_style(sphere):
    with pytest.raises(ValueError):  # noqa: PT011
        pv.plot(sphere, style='not a style')


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.parametrize(
    ('interaction', 'kwargs'),
    [
        ('trackball', {}),
        ('trackball_actor', {}),
        ('image', {}),
        ('joystick', {}),
        ('joystick_actor', {}),
        ('zoom', {}),
        ('terrain', {}),
        ('terrain', {'mouse_wheel_zooms': True, 'shift_pans': True}),
        ('rubber_band', {}),
        ('rubber_band_2d', {}),
    ],
)
def test_interactor_style(sphere, interaction, kwargs):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    getattr(pl, f'enable_{interaction}_style')(**kwargs)
    assert pl.iren._style_class is not None
    pl.close()


@pytest.mark.usefixtures('no_images_to_verify')
def test_lighting_disable_3_lights():
    with pytest.raises(DeprecationError):
        pv.Plotter().disable_3_lights()


def test_lighting_enable_three_lights(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)

    pl.enable_3_lights()
    lights = pl.renderer.lights
    assert len(lights) == 3
    for light in lights:
        assert light.on

    assert lights[0].intensity == 1.0
    assert lights[1].intensity == 0.6
    assert lights[2].intensity == 0.5

    pl.show()


def test_lighting_add_manual_light(sphere):
    pl = pv.Plotter(lighting=None)
    pl.add_mesh(sphere)

    # test manual light addition
    light = pv.Light()
    pl.add_light(light)
    assert pl.renderer.lights == [light]

    # failing case
    with pytest.raises(TypeError):
        pl.add_light('invalid')

    pl.show()


def test_lighting_remove_manual_light(sphere):
    pl = pv.Plotter(lighting=None)
    pl.add_mesh(sphere)
    pl.add_light(pv.Light())

    # test light removal
    pl.remove_all_lights()
    assert not pl.renderer.lights

    pl.show()


def test_lighting_subplots(sphere):
    pl = pv.Plotter(shape='1|1')
    pl.add_mesh(sphere)
    renderers = pl.renderers

    light = pv.Light()
    pl.remove_all_lights()
    for renderer in renderers:
        assert not renderer.lights

    pl.subplot(0)
    pl.add_light(light, only_active=True)
    assert renderers[0].lights
    assert not renderers[1].lights
    pl.add_light(light, only_active=False)
    assert renderers[0].lights
    assert renderers[1].lights
    pl.subplot(1)
    pl.add_mesh(pv.Sphere())
    pl.remove_all_lights(only_active=True)
    assert renderers[0].lights
    assert not renderers[1].lights

    pl.show()


def test_lighting_init_light_kit(sphere):
    pl = pv.Plotter(lighting='light kit')
    pl.add_mesh(sphere)
    lights = pl.renderer.lights
    assert len(lights) == 5
    assert lights[0].light_type == pv.Light.HEADLIGHT
    for light in lights[1:]:
        assert light.light_type == light.CAMERA_LIGHT
    pl.show()


def test_lighting_init_three_lights(sphere):
    pl = pv.Plotter(lighting='three lights')
    pl.add_mesh(sphere)
    lights = pl.renderer.lights
    assert len(lights) == 3
    for light in lights:
        assert light.light_type == light.CAMERA_LIGHT
    pl.show()


def test_lighting_init_none(sphere):
    # ``None`` already tested above
    pl = pv.Plotter(lighting='none')
    pl.add_mesh(sphere)
    lights = pl.renderer.lights
    assert not lights
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_lighting_init_invalid():
    with pytest.raises(ValueError):  # noqa: PT011
        pv.Plotter(lighting='invalid')


@pytest.mark.usefixtures('no_images_to_verify')
def test_plotter_shape_invalid():
    # wrong size
    with pytest.raises(ValueError):  # noqa: PT011
        pv.Plotter(shape=(1,))
    # not positive
    with pytest.raises(ValueError):  # noqa: PT011
        pv.Plotter(shape=(1, 0))
    with pytest.raises(ValueError):  # noqa: PT011
        pv.Plotter(shape=(0, 2))
    # not a sequence
    with pytest.raises(TypeError):
        pv.Plotter(shape={1, 2})


def test_plot_bounds_axes_with_no_data():
    pl = pv.Plotter()
    pl.show_bounds()
    pl.show()


def test_plot_show_grid(sphere):
    pl = pv.Plotter()

    with pytest.raises(ValueError, match='Value of location'):
        pl.show_grid(location='foo')
    with pytest.raises(TypeError, match='location must be a string'):
        pl.show_grid(location=10)
    with pytest.raises(ValueError, match='Value of tick'):
        pl.show_grid(ticks='foo')
    with pytest.raises(TypeError, match='must be a string'):
        pl.show_grid(ticks=10)

    pl.show_grid()  # Add mesh after to make sure bounds update
    pl.add_mesh(sphere)
    pl.show()


@skip_mesa
def test_plot_show_grid_with_mesh(hexbeam, plane, verify_image_cache):
    """Show the grid bounds for a specific mesh."""
    verify_image_cache.macos_skip_image_cache = True

    hexbeam.clear_data()
    pl = pv.Plotter()
    pl.add_mesh(hexbeam, style='wireframe')
    pl.add_mesh(plane)
    pl.show_grid(mesh=plane, show_zlabels=False, show_zaxis=False)
    pl.show()


@pytest.mark.parametrize('use_3d_text', [True, False])
@pytest.mark.parametrize('font_size', [12, 24])
def test_plot_show_grid_font_size(sphere, use_3d_text, font_size):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show_grid(use_3d_text=use_3d_text, font_size=font_size)
    pl.show()


cpos_param = [
    [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)],
    [-1, 2, -5],  # trigger view vector
    [1.0, 2.0, 3.0],
]
cpos_param.extend(pv.plotting.renderer.Renderer.CAMERA_STR_ATTR_MAP)


@pytest.mark.parametrize('cpos', cpos_param)
def test_set_camera_position(cpos, sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.camera_position = cpos
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.parametrize(
    'cpos',
    [
        [(2.0, 5.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)],
        [-1, 2],
        [(1, 2, 3)],
        'notvalid',
    ],
)
def test_set_camera_position_invalid(cpos, sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    with pytest.raises(InvalidCameraError):
        pl.camera_position = cpos


@pytest.mark.usefixtures('no_images_to_verify')
def test_parallel_projection():
    pl = pv.Plotter()
    assert isinstance(pl.parallel_projection, bool)


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.parametrize('state', [True, False])
def test_set_parallel_projection(state):
    pl = pv.Plotter()
    pl.parallel_projection = state
    assert pl.parallel_projection == state


@pytest.mark.usefixtures('no_images_to_verify')
def test_parallel_scale():
    pl = pv.Plotter()
    assert isinstance(pl.parallel_scale, float)


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.parametrize('value', [1, 1.5, 0.3, 10])
def test_set_parallel_scale(value):
    pl = pv.Plotter()
    pl.parallel_scale = value
    assert pl.parallel_scale == value


@pytest.mark.usefixtures('no_images_to_verify')
def test_set_parallel_scale_invalid():
    pl = pv.Plotter()
    with pytest.raises(TypeError):
        pl.parallel_scale = 'invalid'


@pytest.mark.usefixtures('no_images_to_verify')
def test_plot_no_active_scalars(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)

    def _test_update_scalars_with_invalid_array():
        pl.update_scalars(np.arange(5))
        if pv._version.version_info[:2] > (0, 46):
            msg = 'Convert error this method'
            raise RuntimeError(msg)
        if pv._version.version_info[:2] > (0, 47):
            msg = 'Remove this method'
            raise RuntimeError(msg)

    def _test_update_scalars_with_valid_array():
        pl.update_scalars(np.arange(sphere.n_faces_strict))
        if pv._version.version_info[:2] > (0, 46):
            msg = 'Convert error this method'
            raise RuntimeError(msg)
        if pv._version.version_info[:2] > (0, 47):
            msg = 'Remove this method'
            raise RuntimeError(msg)

    with (
        pytest.raises(ValueError, match='Number of scalars'),
        pytest.warns(
            PyVistaDeprecationWarning,
            match='This method is deprecated and will be removed in a future version',
        ),
    ):
        _test_update_scalars_with_invalid_array()
    with (
        pytest.raises(ValueError, match='No active scalars'),
        pytest.warns(
            PyVistaDeprecationWarning,
            match='This method is deprecated and will be removed in a future version',
        ),
    ):
        _test_update_scalars_with_valid_array()


def test_plot_show_bounds(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show_bounds(
        show_xaxis=False,
        show_yaxis=False,
        show_zaxis=False,
        show_xlabels=False,
        show_ylabels=False,
        show_zlabels=False,
        use_2d=True,
    )
    pl.show()


def test_plot_label_fmt(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    fmt = '%.3f' if pv.vtk_version_info < (9, 6, 0) else '{:.3f}'
    pl.show_bounds(xtitle='My X', fmt=fmt)
    pl.show()


@pytest.mark.parametrize('grid', [True, 'both', 'front', 'back'])
@pytest.mark.parametrize('location', ['all', 'origin', 'outer', 'front', 'back'])
@pytest.mark.usefixtures('verify_image_cache')
def test_plot_show_bounds_params(grid, location):
    pl = pv.Plotter()
    pl.add_mesh(pv.Cone())
    pl.show_bounds(grid=grid, ticks='inside', location=location)
    pl.show_bounds(grid=grid, ticks='outside', location=location)
    pl.show_bounds(grid=grid, ticks='both', location=location)
    pl.show()


def test_plot_silhouette_non_poly(hexbeam):
    pl = pv.Plotter()
    pl.add_mesh(hexbeam, show_scalar_bar=False)
    pl.add_silhouette(hexbeam, line_width=10)
    pl.show()


def test_plot_no_silhouette(tri_cylinder):
    # silhouette=False
    pl = pv.Plotter()
    pl.add_mesh(tri_cylinder)
    assert len(list(pl.renderer.GetActors())) == 1  # only cylinder
    pl.show()


def test_plot_silhouette(tri_cylinder):
    # silhouette=True and default properties
    pl = pv.Plotter()
    pl.add_mesh(tri_cylinder, silhouette=True)
    actors = list(pl.renderer.GetActors())
    assert len(actors) == 2  # cylinder + silhouette
    actor = actors[0]  # get silhouette actor
    props = actor.GetProperty()
    assert props.GetColor() == pv.global_theme.silhouette.color
    assert props.GetOpacity() == pv.global_theme.silhouette.opacity
    assert props.GetLineWidth() == pv.global_theme.silhouette.line_width
    pl.show()


def test_plot_silhouette_method(tri_cylinder):
    pl = pv.Plotter()

    pl.add_mesh(tri_cylinder)
    assert len(pl.renderer.actors) == 1  # cylinder

    actor = pl.add_silhouette(tri_cylinder)
    assert isinstance(actor, pv.Actor)
    assert len(pl.renderer.actors) == 2  # cylinder + silhouette

    props = actor.prop
    assert props.color == pv.global_theme.silhouette.color
    assert props.opacity == pv.global_theme.silhouette.opacity
    assert props.line_width == pv.global_theme.silhouette.line_width
    pl.show()


def test_plot_silhouette_options(tri_cylinder):
    # cover other properties
    pl = pv.Plotter()
    pl.add_mesh(tri_cylinder, silhouette=dict(decimate=0.5, feature_angle=20))
    pl.show()


def test_plotter_scale(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.set_scale(10, 10, 15)
    assert pl.scale == [10, 10, 15]
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.set_scale(5.0)
    pl.set_scale(yscale=6.0)
    pl.set_scale(zscale=9.0)
    assert pl.scale == [5.0, 6.0, 9.0]
    pl.show()

    pl = pv.Plotter()
    pl.scale = [1.0, 4.0, 2.0]
    assert pl.scale == [1.0, 4.0, 2.0]
    pl.add_mesh(sphere)
    pl.show()


def test_plot_add_scalar_bar(sphere, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    sphere['test_scalars'] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_scalar_bar(
        label_font_size=10,
        title_font_size=20,
        title='woa',
        interactive=True,
        vertical=True,
    )
    pl.add_scalar_bar(background_color='white', n_colors=256)
    assert isinstance(pl.scalar_bar, _vtk.vtkScalarBarActor)
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_plot_invalid_add_scalar_bar():
    pl = pv.Plotter()
    with pytest.raises(AttributeError):
        pl.add_scalar_bar()


@pytest.mark.usefixtures('no_images_to_verify')
def test_add_scalar_bar_with_unconstrained_font_size(sphere):
    sphere['test_scalars'] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    actor = pl.add_scalar_bar(unconstrained_font_size=True)
    assert actor.GetUnconstrainedFontSize()


def test_plot_list():
    sphere_a = pv.Sphere(center=(0, 0, 0), radius=0.75)
    sphere_b = pv.Sphere(center=(1, 0, 0), radius=0.5)
    sphere_c = pv.Sphere(center=(2, 0, 0), radius=0.25)
    pv.plot([sphere_a, sphere_b, sphere_c], color='tan')


@pytest.mark.usefixtures('no_images_to_verify')
def test_add_lines_invalid():
    pl = pv.Plotter()
    with pytest.raises(TypeError):
        pl.add_lines(range(10))


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.skipif(not HAS_IMAGEIO, reason='Requires imageio')
def test_open_gif_invalid():
    pl = pv.Plotter()
    with pytest.raises(ValueError):  # noqa: PT011
        pl.open_gif('file.abs')


@pytest.mark.skipif(not HAS_IMAGEIO, reason='Requires imageio')
def test_make_movie(sphere, tmpdir, verify_image_cache):
    verify_image_cache.skip = True

    # Make temporary file
    filename = str(tmpdir.join('tmp.mp4'))

    movie_sphere = sphere.copy()
    movie_sphere['scalars'] = np.random.default_rng().random(movie_sphere.n_faces_strict)

    pl = pv.Plotter()
    pl.open_movie(filename)
    actor = pl.add_axes_at_origin()
    pl.remove_actor(actor, reset_camera=False, render=True)
    pl.add_mesh(movie_sphere, scalars='scalars')
    pl.show(auto_close=False, window_size=[304, 304])
    pl.set_focus([0, 0, 0])
    for _ in range(3):  # limiting number of frames to write for speed
        pl.write_frame()
        random_points = np.random.default_rng().random(movie_sphere.points.shape)
        movie_sphere.points[:] = random_points * 0.01 + movie_sphere.points * 0.99
        movie_sphere.points[:] -= movie_sphere.points.mean(0)
        scalars = np.random.default_rng().random(movie_sphere.n_faces_strict)
        movie_sphere['scalars'] = scalars

    # remove file
    pl.close()
    Path(filename).unlink()  # verifies that the plotter has closed


def test_add_legend(sphere):
    pl = pv.Plotter()
    with pytest.raises(TypeError):
        pl.add_mesh(sphere, label=2)
    pl.add_mesh(sphere)
    with pytest.raises(ValueError):  # noqa: PT011
        pl.add_legend()
    legend_labels = [['sphere', 'r']]
    pl.add_legend(labels=legend_labels, border=True, bcolor=None, size=[0.1, 0.1])
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_legend_invalid_face(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    legend_labels = [['sphere', 'r']]
    face = 'invalid_face'
    with pytest.raises(ValueError):  # noqa: PT011
        pl.add_legend(
            labels=legend_labels,
            border=True,
            bcolor=None,
            size=[0.1, 0.1],
            face=face,
        )


def test_legend_subplots(sphere, cube):
    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(sphere, color='blue', smooth_shading=True, label='Sphere')
    assert pl.legend is None
    pl.add_legend(bcolor='w')
    assert isinstance(pl.legend, _vtk.vtkActor2D)

    pl.subplot(0, 1)
    pl.add_mesh(cube, color='r', label='Cube')
    assert pl.legend is None
    pl.add_legend(bcolor='w')
    assert isinstance(pl.legend, _vtk.vtkActor2D)

    pl.show()


def test_add_axes_twice():
    pl = pv.Plotter()
    pl.add_axes()
    pl.add_axes(interactive=True)
    pl.show()


def test_hide_axes():
    pl = pv.Plotter()
    pl.add_axes()
    pl.hide_axes()
    pl.show()


def test_add_axes_parameters():
    pl = pv.Plotter()
    pl.add_axes()
    pl.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.7,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16),
        viewport=(0, 0, 0.4, 0.4),
    )
    pl.show()


def test_show_axes_all():
    pl = pv.Plotter()
    pl.show_axes_all()
    pl.show()


def test_hide_axes_all():
    pl = pv.Plotter()
    pl.hide_axes_all()
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_isometric_view_interactive(sphere):
    plotter_iso = pv.Plotter()
    plotter_iso.add_mesh(sphere)
    plotter_iso.camera_position = 'xy'
    cpos_old = plotter_iso.camera_position
    plotter_iso.isometric_view_interactive()
    assert plotter_iso.camera_position != cpos_old


def test_add_point_labels():
    pl = pv.Plotter()

    # cannot use random points with image regression
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    n = points.shape[0]

    with pytest.raises(ValueError):  # noqa: PT011
        pl.add_point_labels(points, range(n - 1))

    pl.add_point_labels(points, range(n), show_points=True, point_color='r', point_size=10)
    pl.add_point_labels(
        points - 1,
        range(n),
        show_points=False,
        point_color='r',
        point_size=10,
    )
    pl.show()


@pytest.mark.parametrize('always_visible', [False, True])
def test_add_point_labels_always_visible(always_visible):
    # just make sure it runs without exception
    pl = pv.Plotter()
    pl.add_point_labels(
        np.array([[0.0, 0.0, 0.0]]),
        ['hello world'],
        always_visible=always_visible,
    )
    pl.show()


@pytest.mark.parametrize('shape', [None, 'rect', 'rounded_rect'])
@pytest.mark.usefixtures('verify_image_cache')
def test_add_point_labels_shape(shape):
    pl = pv.Plotter()
    pl.add_point_labels(np.array([[0.0, 0.0, 0.0]]), ['hello world'], shape=shape)
    pl.show()


@pytest.mark.parametrize('justification_horizontal', ['left', 'center', 'right'])
@pytest.mark.parametrize('justification_vertical', ['bottom', 'center', 'top'])
def test_add_point_labels_justification(justification_horizontal, justification_vertical):
    pl = pv.Plotter()
    pl.add_point_labels(
        np.array([[0.0, 0.0, 0.0]]),
        ['hello world'],
        justification_horizontal=justification_horizontal,
        justification_vertical=justification_vertical,
        shape_opacity=0.0,
        background_color='grey',
        background_opacity=1.0,
    )
    pl.show()


def test_set_background():
    pl = pv.Plotter()
    pl.set_background('k')
    pl.background_color = 'yellow'
    pl.set_background([0, 0, 0], top=[1, 1, 1])  # Gradient
    _ = pl.background_color
    pl.show()

    pl = pv.Plotter(shape=(1, 2))
    pl.set_background('orange')
    for renderer in pl.renderers:
        assert renderer.GetBackground() == pv.Color('orange')
    pl.show()

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 1)
    pl.set_background('orange', all_renderers=False)
    assert pl.renderers[0].GetBackground() != pv.Color('orange')
    assert pl.renderers[1].GetBackground() == pv.Color('orange')
    pl.show()


def test_add_points():
    pl = pv.Plotter()

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    n = points.shape[0]

    pl.add_points(
        points,
        scalars=np.arange(n),
        cmap=None,
        flip_scalars=True,
        show_scalar_bar=False,
    )
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_key_press_event():
    pl = pv.Plotter()
    pl.key_press_event(None, None)
    pl.close()


@pytest.mark.usefixtures('no_images_to_verify')
def test_enable_picking_gc():
    pl = pv.Plotter()
    sphere = pv.Sphere()
    pl.add_mesh(sphere)
    pl.enable_cell_picking()
    pl.close()


@pytest.mark.usefixtures('no_images_to_verify')
def test_left_button_down():
    pl = pv.Plotter()

    attr = 'GetRenderFramebuffer'
    if hasattr(renwin := pl.render_window, attr):
        if not getattr(renwin, attr)().GetFBOIndex():
            # This only fails for VTK<9.2.3
            with pytest.raises(ValueError, match='Invoking helper with no framebuffer'):
                pl.left_button_down(None, None)
    else:
        pl.left_button_down(None, None)
    pl.close()


def test_show_axes():
    pl = pv.Plotter()
    pl.show_axes()
    pl.show()


def test_plot_cell_data(sphere, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    pl = pv.Plotter()
    scalars = np.arange(sphere.n_faces_strict)
    pl.add_mesh(
        sphere,
        interpolate_before_map=True,
        scalars=scalars,
        n_colors=10,
        rng=sphere.n_faces_strict,
        show_scalar_bar=False,
    )
    pl.show()


def test_plot_clim(sphere):
    pl = pv.Plotter()
    scalars = np.arange(sphere.n_faces_strict)
    pl.add_mesh(
        sphere,
        interpolate_before_map=True,
        scalars=scalars,
        n_colors=5,
        clim=10,
        show_scalar_bar=False,
    )
    assert pl.mapper.GetScalarRange() == (-10, 10)
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_invalid_n_arrays(sphere):
    pl = pv.Plotter()
    with pytest.raises(ValueError):  # noqa: PT011
        pl.add_mesh(sphere, scalars=np.arange(10))


def test_plot_arrow():
    cent = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    pv.plot_arrows(cent, direction)


def test_plot_arrows():
    cent = np.array([[0, 0, 0], [1, 0, 0]])
    direction = np.array([[1, 1, 1], [-1, -1, -1]])
    pv.plot_arrows(cent, direction)


def test_add_arrows():
    vector = np.array([1, 0, 0])
    center = np.array([0, 0, 0])
    pl = pv.Plotter()
    pl.add_arrows(cent=center, direction=vector, mag=2.2, color='#009900')
    pl.show()


def test_axes():
    pl = pv.Plotter()
    pl.add_orientation_widget(pv.Cube(), color='b')
    pl.add_mesh(pv.Cube())
    pl.show()


def test_box_axes(verify_image_cache):
    """Test deprecated function and make sure we remove it by v0.48."""
    verify_image_cache.skip = True

    pl = pv.Plotter()

    def _test_add_axes_box():
        pl.add_axes(box=True)
        if pv._version.version_info[:2] > (0, 47):
            msg = 'Calling this should raise an error'
            raise RuntimeError(msg)
        if pv._version.version_info[:2] > (0, 48):
            msg = 'Remove this function'
            raise RuntimeError(msg)

    with pytest.warns(
        pv.PyVistaDeprecationWarning,
        match='`box` is deprecated. Use `add_box_axes` or `add_color_box_axes` method instead.',
    ):
        _test_add_axes_box()
    pl.close()


def test_box_axes_color_box():
    pl = pv.Plotter()

    def _test_add_axes_color_box():
        pl.add_axes(box=True, box_args={'color_box': True})
        if pv._version.version_info[:2] > (0, 47):
            msg = 'Convert error this function'
            raise RuntimeError(msg)
        if pv._version.version_info[:2] > (0, 48):
            msg = 'Remove this function'
            raise RuntimeError(msg)

    with pytest.warns(
        pv.PyVistaDeprecationWarning,
        match='`box` is deprecated. Use `add_box_axes` or `add_color_box_axes` method instead.',
    ):
        _test_add_axes_color_box()
    pl.add_mesh(pv.Sphere())
    pl.show()


def test_add_box_axes():
    pl = pv.Plotter()
    pl.add_orientation_widget(pv.Sphere(), color='b')
    pl.add_box_axes()
    pl.add_mesh(pv.Sphere())
    pl.show()


def test_add_north_arrow():
    pl = pv.Plotter()
    pl.add_north_arrow_widget(viewport=(0, 0, 0.5, 0.5))
    pl.add_mesh(pv.Arrow(direction=(0, 1, 0)))
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_screenshot(tmpdir):
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    img = pl.screenshot(transparent_background=False)
    assert np.any(img)
    img_again = pl.screenshot()
    assert np.any(img_again)
    filename = str(tmpdir.mkdir('tmpdir').join('export-graphic.svg'))
    pl.save_graphic(filename)

    # test window and array size
    w, h = 20, 10
    img = pl.screenshot(transparent_background=False, window_size=(w, h))
    assert img.shape == (h, w, 3)
    img = pl.screenshot(transparent_background=True, window_size=(w, h))
    assert img.shape == (h, w, 4)

    # check error before first render
    pl = pv.Plotter(off_screen=False)
    pl.add_mesh(pv.Sphere())
    with pytest.raises(RuntimeError):
        pl.screenshot()


@pytest.mark.usefixtures('no_images_to_verify')
def test_screenshot_scaled():
    # FYI: no regression tests because show() is not called
    factor = 2
    pl = pv.Plotter(image_scale=factor)
    width, height = pl.window_size
    pl.add_mesh(pv.Sphere())
    img = pl.screenshot(transparent_background=False)
    assert np.any(img)
    assert img.shape == (width * factor, height * factor, 3)
    img_again = pl.screenshot(scale=3)
    assert np.any(img_again)
    assert img_again.shape == (width * 3, height * 3, 3)
    assert pl.image_scale == factor, 'image_scale leaked from screenshot context'
    img = pl.image
    assert img.shape == (width * factor, height * factor, 3)

    w, h = 20, 10
    factor = 4
    pl.image_scale = factor
    img = pl.screenshot(transparent_background=False, window_size=(w, h))
    assert img.shape == (h * factor, w * factor, 3)

    img = pl.screenshot(transparent_background=True, window_size=(w, h), scale=5)
    assert img.shape == (h * 5, w * 5, 4)
    assert pl.image_scale == factor, 'image_scale leaked from screenshot context'

    with pytest.raises(ValueError):  # noqa: PT011
        pl.image_scale = 0.5

    pl.close()


@pytest.mark.usefixtures('no_images_to_verify')
def test_screenshot_altered_window_size(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)

    pl.window_size = (800, 800)
    a = pl.screenshot()
    assert a.shape == (800, 800, 3)
    # pl.show(auto_close=False)  # for image regression test

    pl.window_size = (1000, 1000)
    b = pl.screenshot()
    assert b.shape == (1000, 1000, 3)
    # pl.show(auto_close=False)  # for image regression test

    d = pl.screenshot(window_size=(600, 600))
    assert d.shape == (600, 600, 3)
    # pl.show()  # for image regression test

    pl.close()


def test_screenshot_bytes():
    # Test screenshot to bytes object
    buffer = io.BytesIO()
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Sphere())
    pl.show(screenshot=buffer)
    buffer.seek(0)
    im = Image.open(buffer)
    assert im.format == 'PNG'


@pytest.mark.usefixtures('no_images_to_verify')
def test_screenshot_rendering(tmpdir):
    pl = pv.Plotter()
    pl.add_mesh(examples.load_airplane(), smooth_shading=True)
    filename = str(tmpdir.mkdir('tmpdir').join('export-graphic.svg'))
    assert pl._first_time
    pl.save_graphic(filename)
    assert not pl._first_time


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.parametrize('ext', SUPPORTED_FORMATS)
def test_save_screenshot(tmpdir, sphere, ext):
    filename = str(tmpdir.mkdir('tmpdir').join('tmp' + ext))
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.screenshot(filename)
    assert Path(filename).is_file()
    assert pathlib.Path(filename).stat().st_size


def test_scalars_by_name(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    pl = pv.Plotter()
    data = examples.load_uniform()
    pl.add_mesh(data, scalars='Spatial Cell Data')
    pl.show()


def test_multi_block_plot(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    multi = pv.MultiBlock()
    multi.append(examples.load_rectilinear())
    uni = examples.load_uniform()
    arr = np.random.default_rng().random(uni.n_cells)
    uni.cell_data.set_array(arr, 'Random Data')
    multi.append(uni)
    # And now add a data set without the desired array and a NULL component
    multi.append(examples.load_airplane())

    # missing data should still plot
    multi.plot(scalars='Random Data')

    multi.plot(multi_colors=True)


def test_clear(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.clear()
    pl.show()


def test_plot_texture():
    """Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = examples.load_globe_texture()
    pl = pv.Plotter()
    pl.add_mesh(globe, texture=texture)
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.skipif(not HAS_IMAGEIO, reason='Requires imageio')
def test_plot_numpy_texture():
    """Text adding a np.ndarray texture to a plot"""
    globe = examples.load_globe()
    texture_np = np.asarray(imageio.v2.imread(examples.mapfile))
    pl = pv.Plotter()
    pl.add_mesh(globe, texture=texture_np)


@pytest.mark.skipif(not HAS_IMAGEIO, reason='Requires imageio')
def test_read_texture_from_numpy():
    """Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = numpy_to_texture(imageio.v2.imread(examples.mapfile))
    pl = pv.Plotter()
    pl.add_mesh(globe, texture=texture)
    pl.show()


def _make_rgb_dataset(dtype: str, return_composite: bool, scalars: str):
    def _dtype_convert_func(dtype):
        # Convert color to the specified dtype
        if dtype == 'float':

            def as_dtype(color: tuple[float, float, float]):
                return pv.Color(color).float_rgb
        elif dtype == 'int':

            def as_dtype(color: tuple[float, float, float]):
                return pv.Color(color).int_rgb
        elif dtype == 'uint8':

            def as_dtype(color: tuple[float, float, float]):
                return np.array(pv.Color(color).int_rgb, dtype=np.uint8)
        else:
            raise NotImplementedError

        return as_dtype

    def _make_polys():
        # Create 3 separate PolyData with one quad cell each
        faces = [4, 0, 1, 2, 3]
        poly1 = pv.PolyData(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]], faces
        )
        poly2 = pv.PolyData(
            [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], faces
        )
        poly3 = pv.PolyData(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]], faces
        )
        return poly1, poly2, poly3

    poly1, poly2, poly3 = _make_polys()
    dtype_convert_func = _dtype_convert_func(dtype)

    RED = dtype_convert_func((1.0, 0.0, 0.0))
    GREEN = dtype_convert_func((0.0, 1.0, 0.0))
    BLUE = dtype_convert_func((0.0, 0.0, 1.0))

    # Color the polydata cells
    poly1[scalars] = [RED]
    poly2[scalars] = [GREEN]
    poly3[scalars] = [BLUE]

    dataset = pv.MultiBlock((poly1, poly2, poly3))

    # Make sure the dataset is as expected
    if return_composite:
        assert isinstance(dataset, pv.MultiBlock)
        assert all(np.dtype(block[scalars].dtype) == np.dtype(dtype) for block in dataset)
        assert all(block.array_names == [scalars] for block in dataset)
    else:
        # Merge and return
        dataset = pv.merge(dataset)
        assert isinstance(dataset, pv.PolyData)
        assert np.dtype(dataset[scalars].dtype) == np.dtype(dtype)
        assert dataset.array_names == [scalars]
    return dataset


# check_gc fails for polydata (suspected memory leak with pv.merge)
@pytest.mark.skip_check_gc
@pytest.mark.parametrize('composite', [True, False], ids=['composite', 'polydata'])
@pytest.mark.parametrize('dtype', ['float', 'int', 'uint8'])
def test_plot_rgb(composite, dtype):
    scalars = 'face_colors'
    dataset = _make_rgb_dataset(dtype, return_composite=composite, scalars=scalars)

    pl = pv.Plotter()
    actor = pl.add_mesh(dataset, scalars=scalars, rgb=True)
    actor.prop.lighting = False
    pl.show()


# check_gc fails for polydata (suspected memory leak with pv.merge)
@pytest.mark.skip_check_gc
@pytest.mark.parametrize('scalars', ['_rgb', '_rgba'])
@pytest.mark.parametrize('composite', [True, False], ids=['composite', 'polydata'])
def test_plot_rgb_implicit(composite, scalars):
    dataset = _make_rgb_dataset(dtype='uint8', return_composite=composite, scalars=scalars)

    pl = pv.Plotter()
    actor = pl.add_mesh(dataset)
    actor.prop.lighting = False
    pl.show()


def test_vector_array_with_points(multicomp_poly):
    """Test using vector valued data with and without component arg."""
    # test no component argument
    pl = pv.Plotter()
    pl.add_mesh(multicomp_poly, scalars='vector_values_points')
    pl.camera_position = 'xy'
    pl.camera.tight()
    pl.show()

    # test component argument
    pl = pv.Plotter()
    pl.add_mesh(multicomp_poly, scalars='vector_values_points', component=0)
    pl.camera_position = 'xy'
    pl.camera.tight()
    pl.show()


@skip_windows_mesa
def test_vector_array_with_cells(multicomp_poly):
    """Test using vector valued data with and without component arg."""
    pl = pv.Plotter()
    pl.add_mesh(multicomp_poly, scalars='vector_values_cells')
    pl.camera_position = 'xy'
    pl.camera.tight()
    pl.show()

    # test component argument
    pl = pv.Plotter()
    pl.add_mesh(multicomp_poly, scalars='vector_values_cells', component=0)
    pl.camera_position = 'xy'
    pl.camera.tight()
    pl.show()


def test_vector_array(multicomp_poly):
    """Test using vector valued data for image regression."""
    pl = pv.Plotter(shape=(2, 2))
    pl.subplot(0, 0)
    pl.add_mesh(multicomp_poly, scalars='vector_values_points', show_scalar_bar=False)
    pl.camera_position = 'xy'
    pl.camera.tight()
    pl.subplot(0, 1)
    pl.add_mesh(multicomp_poly.copy(), scalars='vector_values_points', component=0)
    pl.subplot(1, 0)
    pl.add_mesh(multicomp_poly.copy(), scalars='vector_values_points', component=1)
    pl.subplot(1, 1)
    pl.add_mesh(multicomp_poly.copy(), scalars='vector_values_points', component=2)
    pl.link_views()
    pl.show()


@skip_windows_mesa
def test_vector_plotting_doesnt_modify_data(multicomp_poly):
    """Test that the operations in plotting do not modify the data in the mesh."""
    copy_vector_values_points = multicomp_poly['vector_values_points'].copy()
    copy_vector_values_cells = multicomp_poly['vector_values_cells'].copy()

    # test that adding a vector with no component parameter to a Plotter instance
    # does not modify it.
    pl = pv.Plotter()
    pl.add_mesh(multicomp_poly, scalars='vector_values_points')
    pl.show()
    assert np.array_equal(multicomp_poly['vector_values_points'], copy_vector_values_points)

    pl = pv.Plotter()
    pl.add_mesh(multicomp_poly, scalars='vector_values_cells')
    pl.show()
    assert np.array_equal(multicomp_poly['vector_values_cells'], copy_vector_values_cells)

    # test that adding a vector with a component parameter to a Plotter instance
    # does not modify it.
    pl = pv.Plotter()
    pl.add_mesh(multicomp_poly, scalars='vector_values_points', component=0)
    pl.show()
    assert np.array_equal(multicomp_poly['vector_values_points'], copy_vector_values_points)

    pl = pv.Plotter()
    pl.add_mesh(multicomp_poly, scalars='vector_values_cells', component=0)
    pl.show()
    assert np.array_equal(multicomp_poly['vector_values_cells'], copy_vector_values_cells)


@pytest.mark.usefixtures('no_images_to_verify')
def test_vector_array_fail_with_incorrect_component(multicomp_poly):
    """Test failure modes of component argument."""
    pl = pv.Plotter()

    # Non-Integer
    with pytest.raises(TypeError):
        pl.add_mesh(multicomp_poly, scalars='vector_values_points', component=1.5)

    # Component doesn't exist
    pl = pv.Plotter()
    with pytest.raises(ValueError):  # noqa: PT011
        pl.add_mesh(multicomp_poly, scalars='vector_values_points', component=3)

    # Component doesn't exist
    pl = pv.Plotter()
    with pytest.raises(ValueError):  # noqa: PT011
        pl.add_mesh(multicomp_poly, scalars='vector_values_points', component=-1)


def test_camera(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.view_isometric()
    pl.reset_camera()
    pl.view_xy()
    pl.view_xz()
    pl.view_yz()
    pl.add_mesh(examples.load_uniform(), reset_camera=True, culling=True)
    pl.view_xy(negative=True)
    pl.view_xz(negative=True)
    pl.view_yz(negative=True)
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.camera.zoom(5)
    pl.camera.up = 0, 0, 10
    pl.show()


def test_multi_renderers():
    pl = pv.Plotter(shape=(2, 2))

    pl.subplot(0, 0)
    pl.add_text('Render Window 0', font_size=30)
    sphere = pv.Sphere()
    pl.add_mesh(sphere, scalars=sphere.points[:, 2], show_scalar_bar=False)
    pl.add_scalar_bar('Z', vertical=True)

    pl.subplot(0, 1)
    pl.add_text('Render Window 1', font_size=30)
    pl.add_mesh(pv.Cube(), show_edges=True)

    pl.subplot(1, 0)
    pl.add_text('Render Window 2', font_size=30)
    pl.add_mesh(pv.Arrow(), color='y', show_edges=True)

    pl.subplot(1, 1)
    pl.add_text('Render Window 3', position=(0.0, 0.0), font_size=30, viewport=True)
    pl.add_mesh(pv.Cone(), color='g', show_edges=True, culling=True)
    pl.add_bounding_box(render_lines_as_tubes=True, line_width=5)
    pl.show_bounds(all_edges=True)

    pl.update_bounds_axes()
    pl.show()


def test_multi_renderers_subplot_ind_2x1():
    # Test subplot indices (2 rows by 1 column)
    pl = pv.Plotter(shape=(2, 1))
    # First row
    pl.subplot(0, 0)
    pl.add_mesh(pv.Sphere())
    # Second row
    pl.subplot(1, 0)
    pl.add_mesh(pv.Cube())
    pl.show()


def test_multi_renderers_subplot_ind_1x2():
    # Test subplot indices (1 row by 2 columns)
    pl = pv.Plotter(shape=(1, 2))
    # First column
    pl.subplot(0, 0)
    pl.add_mesh(pv.Sphere())
    # Second column
    pl.subplot(0, 1)
    pl.add_mesh(pv.Cube())
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_multi_renderers_bad_indices():
    # Test bad indices
    pl = pv.Plotter(shape=(1, 2))
    with pytest.raises(IndexError):
        pl.subplot(1, 0)


def test_multi_renderers_subplot_ind_3x1():
    # Test subplot 3 on left, 1 on right
    pl = pv.Plotter(shape='3|1')
    # First column
    pl.subplot(0)
    pl.add_mesh(pv.Sphere())
    pl.subplot(1)
    pl.add_mesh(pv.Cube())
    pl.subplot(2)
    pl.add_mesh(pv.Cylinder())
    pl.subplot(3)
    pl.add_mesh(pv.Cone())
    pl.show()


def test_multi_renderers_subplot_ind_3x1_splitting_pos():
    # Test subplot 3 on top, 1 on bottom
    pl = pv.Plotter(shape='3/1', splitting_position=0.5)
    # First column
    pl.subplot(0)
    pl.add_mesh(pv.Sphere())
    pl.subplot(1)
    pl.add_mesh(pv.Cube())
    pl.subplot(2)
    pl.add_mesh(pv.Cylinder())
    pl.subplot(3)
    pl.add_mesh(pv.Cone())
    pl.show()


def test_multi_renderers_subplot_ind_1x3():
    # Test subplot 3 on bottom, 1 on top
    pl = pv.Plotter(shape='1|3')
    # First column
    pl.subplot(0)
    pl.add_mesh(pv.Sphere())
    pl.subplot(1)
    pl.add_mesh(pv.Cube())
    pl.subplot(2)
    pl.add_mesh(pv.Cylinder())
    pl.subplot(3)
    pl.add_mesh(pv.Cone())
    pl.show()


def test_subplot_groups():
    pl = pv.Plotter(shape=(3, 3), groups=[(1, [1, 2]), (np.s_[:], 0)])
    pl.subplot(0, 0)
    pl.add_mesh(pv.Sphere())
    pl.subplot(0, 1)
    pl.add_mesh(pv.Cube())
    pl.subplot(0, 2)
    pl.add_mesh(pv.Arrow())
    pl.subplot(1, 1)
    pl.add_mesh(pv.Cylinder())
    pl.subplot(2, 1)
    pl.add_mesh(pv.Cone())
    pl.subplot(2, 2)
    pl.add_mesh(pv.Box())
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_subplot_groups_fail():
    # Test group overlap
    with pytest.raises(ValueError):  # noqa: PT011
        # Partial overlap
        pv.Plotter(shape=(3, 3), groups=[([1, 2], [0, 1]), ([0, 1], [1, 2])])
    with pytest.raises(ValueError):  # noqa: PT011
        # Full overlap (inner)
        pv.Plotter(shape=(4, 4), groups=[(np.s_[:], np.s_[:]), ([1, 2], [1, 2])])
    with pytest.raises(ValueError):  # noqa: PT011
        # Full overlap (outer)
        pv.Plotter(shape=(4, 4), groups=[(1, [1, 2]), ([0, 3], np.s_[:])])


@pytest.mark.skip_windows
def test_link_views(sphere):
    pl = pv.Plotter(shape=(1, 4))
    pl.subplot(0, 0)
    pl.add_mesh(sphere, smooth_shading=False, show_edges=False)
    pl.subplot(0, 1)
    pl.add_mesh(sphere, smooth_shading=True, show_edges=False)
    pl.subplot(0, 2)
    pl.add_mesh(sphere, smooth_shading=False, show_edges=True)
    pl.subplot(0, 3)
    pl.add_mesh(sphere, smooth_shading=True, show_edges=True)
    with pytest.raises(TypeError):
        pl.link_views(views='foo')
    pl.link_views([0, 1])
    pl.link_views()
    with pytest.raises(TypeError):
        pl.unlink_views(views='foo')
    pl.unlink_views([0, 1])
    pl.unlink_views(2)
    pl.unlink_views()
    pl.show()


@pytest.mark.skip_windows
@pytest.mark.usefixtures('verify_image_cache')
def test_link_views_camera_set():
    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(pv.Cone())
    assert not pl.renderer.camera_set
    pl.subplot(0, 1)
    pl.add_mesh(pv.Cube())
    assert not pl.renderer.camera_set
    pl.link_views()  # make sure the default isometric view is used
    for renderer in pl.renderers:
        assert not renderer.camera_set
    pl.show()

    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(pv.Cone())
    pl.subplot(0, 1)
    pl.add_mesh(pv.Cube())
    pl.link_views()
    pl.unlink_views()
    for renderer in pl.renderers:
        assert not renderer.camera_set
    pl.show()

    wavelet = pv.Wavelet().clip('x')
    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(wavelet, color='red')
    pl.subplot(0, 1)
    pl.add_mesh(wavelet, color='red')
    pl.link_views()
    pl.camera_position = pv.CameraPosition(
        position=(55.0, 16, 31), focal_point=(-5.0, 0.0, 0.0), viewup=(-0.22, 0.97, -0.09)
    )
    pl.show()


def test_orthographic_slicer(uniform):
    uniform.set_active_scalars('Spatial Cell Data')
    slices = uniform.slice_orthogonal()

    # Orthographic Slicer
    pl = pv.Plotter(shape=(2, 2))

    pl.subplot(1, 1)
    pl.add_mesh(slices, clim=uniform.get_data_range())
    pl.add_axes()
    pl.enable()

    pl.subplot(0, 0)
    pl.add_mesh(slices['XY'])
    pl.view_xy()
    pl.disable()

    pl.subplot(0, 1)
    pl.add_mesh(slices['XZ'])
    pl.view_xz(negative=True)
    pl.disable()

    pl.subplot(1, 0)
    pl.add_mesh(slices['YZ'])
    pl.view_yz()
    pl.disable()

    pl.show()


def test_remove_actor(uniform):
    pl = pv.Plotter()
    pl.add_mesh(uniform.copy(), name='data')
    pl.add_mesh(uniform.copy(), name='data')
    pl.add_mesh(uniform.copy(), name='data')
    pl.show()


def test_add_mesh_remove_existing_actor(verify_image_cache, uniform):
    """Test remove_existing_actor parameter for add_mesh method."""
    verify_image_cache.skip = True
    pl = pv.Plotter()
    actor1 = pl.add_mesh(uniform.copy(), name='test_mesh1')
    actor2 = pl.add_mesh(uniform.copy(), name='test_mesh2', remove_existing_actor=False)
    actors = list(pl.renderer.actors.values())
    assert actor1 in actors
    assert actor2 in actors


def test_image_properties() -> None:
    mesh = examples.load_uniform()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.show(auto_close=False)  # DO NOT close plotter
    # Get RGB image
    _ = pl.image
    # Get the depth image
    _ = pl.get_image_depth()
    pl.close()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.show(store_image_depth=True)  # close plotter
    # Get RGB image
    _ = pl.image
    # verify property matches method while testing both available
    assert np.allclose(pl.image_depth, pl.get_image_depth(), equal_nan=True)
    pl.close()

    # gh-920
    rr = np.array([[-0.5, -0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, -0.5, 1]])
    tris = np.array([[3, 0, 2, 1], [3, 2, 0, 3]])
    mesh = pv.PolyData(rr, tris)
    pl = pv.Plotter()
    pl.add_mesh(mesh, color=True)
    with pytest.warns(pv.VTKExecutionWarning):
        pl.renderer.camera_position = 0.0, 0.0, 1.0
    pl.renderer.ResetCamera()
    pl.enable_parallel_projection()
    assert pl.renderer.camera_set
    pl.show(interactive=False, auto_close=False)
    img = pl.get_image_depth(fill_value=0.0)
    rng = np.ptp(img)
    assert 3.8 < rng < 3.9, rng  # 3.8460655 in testing
    pl.close()


@pytest.mark.skip_check_gc
@pytest.mark.parametrize('enable_parallel_projection', [True, False])
def test_image_depth_parallel_projection(enable_parallel_projection):
    # Create depth image
    pl = pv.Plotter()
    pl.add_mesh(pv.ParametricRandomHills(), color=True)
    if enable_parallel_projection:
        pl.enable_parallel_projection()
    pl.show(store_image_depth=True, auto_close=False)
    zval = pl.get_image_depth()
    pl.clear_actors()

    # Plot depth image
    image = pv.ImageData(dimensions=(*zval.shape, 1))
    image['Distance To Camera'] = np.flipud(zval).T.ravel(order='F')
    pl.add_mesh(image)
    pl.view_xy()
    pl.camera.tight()
    pl.show()


def test_image_depth_raise(sphere: pv.PolyData, verify_image_cache) -> None:
    """Ensure a RuntimeError is raised when not storing image_depth."""
    verify_image_cache.skip = True

    pl = pv.Plotter()
    pl.add_mesh(sphere, color='w')
    pl.show()

    with pytest.raises(RuntimeError, match='store_image_depth=True'):
        pl.get_image_depth()


def test_volume_rendering_from_helper(uniform, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    uniform.plot(volume=True, opacity='linear')


@skip_windows_mesa  # due to opacity
def test_volume_rendering_from_plotter(uniform):
    pl = pv.Plotter()
    pl.add_volume(uniform, opacity='sigmoid', cmap='jet', n_colors=15)
    pl.show()


@skip_windows_mesa  # due to opacity
@pytest.mark.skip_check_gc("vtkWeakReference not gc'd on Python 3.14 vtk dev wheels")
def test_volume_rendering_rectilinear(uniform):
    grid = uniform.cast_to_rectilinear_grid()

    pl = pv.Plotter()
    pl.add_volume(grid, opacity='sigmoid', cmap='jet', n_colors=15)
    pl.show()

    pl = pv.Plotter()
    pl.add_volume(grid)
    pl.show()

    pl = pv.Plotter()
    with pytest.raises(TypeError):
        pl.add_volume(grid, mapper='fixed_point')
    pl.close()


@skip_windows_mesa  # due to opacity
@pytest.mark.parametrize('mapper', ['fixed_point', 'gpu', 'open_gl', 'smart'])
def test_volume_rendering_mappers_image_data(mapper):
    image = pv.ImageData(dimensions=(50, 50, 50))
    image['scalars'] = -image.x

    pl = pv.Plotter()
    pl.add_volume(image, mapper=mapper)
    pl.show()


@pytest.mark.skip_windows
def test_multiblock_volume_rendering(uniform):
    ds_a = uniform.copy()
    ds_b = uniform.copy()
    ds_b.origin = (9.0, 0.0, 0.0)
    ds_c = uniform.copy()
    ds_c.origin = (0.0, 9.0, 0.0)
    ds_d = uniform.copy()
    ds_d.origin = (9.0, 9.0, 0.0)

    data = pv.MultiBlock(
        dict(
            a=ds_a,
            b=ds_b,
            c=ds_c,
            d=ds_d,
        ),
    )
    data['a'].rename_array('Spatial Point Data', 'a')
    data['b'].rename_array('Spatial Point Data', 'b')
    data['c'].rename_array('Spatial Point Data', 'c')
    data['d'].rename_array('Spatial Point Data', 'd')
    data.plot(volume=True, multi_colors=True)


def test_array_volume_rendering(uniform, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    arr = uniform['Spatial Point Data'].reshape(uniform.dimensions)
    pv.plot(arr, volume=True, opacity='linear')


def test_plot_compare_four():
    # Really just making sure no errors are thrown
    mesh = examples.load_uniform()
    data_a = mesh.contour()
    data_b = mesh.threshold_percent(0.5)
    data_c = mesh.decimate_boundary(0.5)
    data_d = mesh.glyph(scale=False, orient=False)
    pv.plot_compare_four(
        data_a,
        data_b,
        data_c,
        data_d,
        display_kwargs={'color': 'w'},
    )


@skip_lesser_9_4_X_depth_peeling
def test_plot_depth_peeling():
    mesh = examples.load_airplane()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.enable_depth_peeling()
    pl.disable_depth_peeling()
    pl.show()


@pytest.mark.skip_windows('No testing on windows for EDL')
def test_plot_eye_dome_lighting_plot(airplane):
    airplane.plot(eye_dome_lighting=True)


@pytest.mark.skip_windows('No testing on windows for EDL')
def test_plot_eye_dome_lighting_plotter(airplane):
    pl = pv.Plotter()
    pl.add_mesh(airplane)
    pl.enable_eye_dome_lighting()
    pl.show()


@pytest.mark.skip_windows('No testing on windows for EDL')
def test_plot_eye_dome_lighting_enable_disable(airplane):
    pl = pv.Plotter()
    pl.add_mesh(airplane)
    pl.enable_eye_dome_lighting()
    pl.disable_eye_dome_lighting()
    pl.show()


@pytest.mark.skip_windows
def test_opacity_by_array_direct(plane):
    # test with opacity parm as an array, both cell and point sized
    plane_shift = plane.translate((0, 0, 1), inplace=False)
    pl = pv.Plotter()
    pl.add_mesh(plane, color='b', opacity=np.linspace(0, 1, plane.n_points), show_edges=True)
    pl.add_mesh(
        plane_shift,
        color='r',
        opacity=np.linspace(0, 1, plane.n_cells),
        show_edges=True,
    )
    pl.show()


@skip_windows_mesa
def test_opacity_by_array(uniform):
    # Test with opacity array
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['opac'] = opac
    pl = pv.Plotter()
    pl.add_mesh(uniform, scalars='Spatial Point Data', opacity='opac')
    pl.show()


@skip_windows_mesa
def test_opacity_by_array_uncertainty(uniform):
    # Test with uncertainty array (transparency)
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['unc'] = opac
    pl = pv.Plotter()
    pl.add_mesh(uniform, scalars='Spatial Point Data', opacity='unc', use_transparency=True)
    pl.show()


def test_opacity_by_array_user_transform(uniform, verify_image_cache):
    verify_image_cache.high_variance_test = True

    uniform['Spatial Point Data'] /= uniform['Spatial Point Data'].max()

    # Test with user defined transfer function
    opacities = [0, 0.2, 0.9, 0.2, 0.1]
    pl = pv.Plotter()
    pl.add_mesh(uniform, scalars='Spatial Point Data', opacity=opacities)
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_opacity_mismatched_fail(uniform):
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['unc'] = opac

    # Test using mismatched arrays
    pl = pv.Plotter()
    with pytest.raises(ValueError):  # noqa: PT011
        # cell scalars vs point opacity
        pl.add_mesh(uniform, scalars='Spatial Cell Data', opacity='unc')


@skip_windows_mesa
def test_opacity_by_array_preference():
    tetra = pv.Tetrahedron()  # 4 points, 4 cells
    opacities = np.linspace(0.2, 0.8, tetra.n_points)
    tetra.clear_data()
    tetra.point_data['scalars'] = tetra.cell_data['scalars'] = np.arange(tetra.n_points)
    tetra.point_data['opac'] = tetra.cell_data['opac'] = opacities

    # test opacity by key
    pl = pv.Plotter()
    pl.add_mesh(tetra.copy(), opacity='opac', preference='cell')
    pl.add_mesh(tetra.translate((2, 0, 0), inplace=False), opacity='opac', preference='point')
    pl.close()

    # test opacity by array
    pl = pv.Plotter()
    pl.add_mesh(tetra.copy(), opacity=opacities, preference='cell')
    pl.add_mesh(tetra.translate((2, 0, 0), inplace=False), opacity=opacities, preference='point')
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.parametrize('mapping', [None, True, object()])
def test_opacity_transfer_functions_raises(mapping):
    with pytest.raises(
        TypeError,
        match=re.escape(f'Transfer function type ({type(mapping)}) not understood'),
    ):
        pv.opacity_transfer_function(mapping, n_colors=10)


@pytest.mark.usefixtures('no_images_to_verify')
def test_opacity_transfer_functions():
    n = 256
    mapping = pv.opacity_transfer_function('linear', n)
    assert len(mapping) == n
    mapping = pv.opacity_transfer_function('sigmoid_10', n)
    assert len(mapping) == n
    mapping = pv.opacity_transfer_function('foreground', n)
    assert len(mapping) == n
    mapping = pv.opacity_transfer_function('foreground', 5)
    assert np.array_equal(mapping, [0, 255, 255, 255, 255])
    with pytest.raises(ValueError):  # noqa: PT011
        mapping = pv.opacity_transfer_function('foo', n)
    with pytest.raises(RuntimeError):
        mapping = pv.opacity_transfer_function(np.linspace(0, 1, 2 * n), n)
    foo = np.linspace(0, n, n)
    mapping = pv.opacity_transfer_function(foo, n)
    assert np.allclose(foo, mapping)
    foo = [0, 0.2, 0.9, 0.2, 0.1]
    mapping = pv.opacity_transfer_function(foo, n, interpolate=False)
    assert len(mapping) == n
    foo = [3, 5, 6, 10]
    mapping = pv.opacity_transfer_function(foo, n)
    assert len(mapping) == n


@skip_windows_mesa
@pytest.mark.parametrize(
    'opacity',
    [
        'sigmoid',
        'sigmoid_1',
        'sigmoid_2',
        'sigmoid_3',
        'sigmoid_4',
        'sigmoid_5',
        'sigmoid_6',
        'sigmoid_7',
        'sigmoid_8',
        'sigmoid_9',
        'sigmoid_10',
        'sigmoid_15',
        'sigmoid_20',
    ],
)
def test_plot_sigmoid_opacity_transfer_functions(uniform, opacity):
    pl = pv.Plotter()
    pl.add_volume(uniform, opacity=opacity)
    pl.show()


def test_closing_and_mem_cleanup(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    verify_image_cache.skip = True
    n = 5
    for _ in range(n):
        for _ in range(n):
            pl = pv.Plotter()
            for k in range(n):
                pl.add_mesh(pv.Sphere(radius=k))
            pl.show()
        pv.close_all()


def test_above_below_scalar_range_annotations():
    pl = pv.Plotter()
    pl.add_mesh(
        examples.load_uniform(),
        clim=[100, 500],
        cmap='viridis',
        below_color='blue',
        above_color='red',
    )
    pl.show()


def test_user_annotations_scalar_bar_mesh(uniform):
    pl = pv.Plotter()
    pl.add_mesh(uniform, annotations={100.0: 'yum'})
    pl.show()


def test_fixed_font_size_annotation_text_scaling_off():
    pl = pv.Plotter()
    sargs = {'title_font_size': 12, 'label_font_size': 10}
    pl.add_mesh(
        examples.load_uniform(),
        clim=[100, 500],
        cmap='viridis',
        below_color='blue',
        above_color='red',
        annotations={300.0: 'yum'},
        scalar_bar_args=sargs,
    )
    pl.show()


def test_user_annotations_scalar_bar_volume(uniform, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    pl = pv.Plotter()
    pl.add_volume(uniform, scalars='Spatial Point Data', annotations={100.0: 'yum'})
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_user_matrix_volume(uniform):
    shear = np.eye(4)
    shear[0, 1] = 1

    pl = pv.Plotter()
    volume = pl.add_volume(uniform, user_matrix=shear)
    np.testing.assert_almost_equal(volume.user_matrix, shear)

    match = 'Shape must be one of [(3, 3), (4, 4)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        pl.add_volume(uniform, user_matrix=np.eye(5))

    with pytest.raises(TypeError):
        pl.add_volume(uniform, user_matrix='invalid')


@pytest.mark.usefixtures('no_images_to_verify')
def test_user_matrix_mesh(sphere):
    shear = np.eye(4)
    shear[0, 1] = 1

    pl = pv.Plotter()
    actor = pl.add_mesh(sphere, user_matrix=shear)
    np.testing.assert_almost_equal(actor.user_matrix, shear)

    match = 'Shape must be one of [(3, 3), (4, 4)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        pl.add_mesh(sphere, user_matrix=np.eye(5))

    with pytest.raises(TypeError):
        pl.add_mesh(sphere, user_matrix='invalid')


def test_user_matrix_silhouette(airplane):
    matrix = [[-1, 0, 0, 1], [0, 1, 0, 2], [0, 0, -1, 3], [0, 0, 0, 1]]
    pl = pv.Plotter()
    pl.add_mesh(
        airplane,
        silhouette=dict(line_width=10),
        user_matrix=matrix,
    )
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_scalar_bar_args_unmodified_add_mesh(sphere):
    sargs = {'vertical': True}
    sargs_copy = sargs.copy()

    pl = pv.Plotter()
    pl.add_mesh(sphere, scalar_bar_args=sargs)

    assert sargs == sargs_copy


@pytest.mark.usefixtures('no_images_to_verify')
def test_scalar_bar_args_unmodified_add_volume(uniform):
    sargs = {'vertical': True}
    sargs_copy = sargs.copy()

    pl = pv.Plotter()
    pl.add_volume(uniform, scalar_bar_args=sargs)

    assert sargs == sargs_copy


def test_plot_string_array(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    mesh = examples.load_uniform()
    labels = np.empty(mesh.n_cells, dtype='<U10')
    labels[:] = 'High'
    labels[mesh['Spatial Cell Data'] < 300] = 'Medium'
    labels[mesh['Spatial Cell Data'] < 100] = 'Low'
    mesh['labels'] = labels
    pl = pv.Plotter()
    pl.add_mesh(mesh, scalars='labels')
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_fail_plot_table():
    """Make sure tables cannot be plotted"""
    table = pv.Table(np.random.default_rng().random((50, 3)))
    with pytest.raises(TypeError):
        pv.plot(table)
    pl = pv.Plotter()
    with pytest.raises(TypeError):
        pl.add_mesh(table)


@pytest.mark.usefixtures('no_images_to_verify')
def test_bad_keyword_arguments():
    """Make sure bad keyword arguments raise an error"""
    mesh = examples.load_uniform()
    with pytest.raises(TypeError):
        pv.plot(mesh, foo=5)
    with pytest.raises(TypeError):
        pv.plot(mesh, scalar=mesh.active_scalars_name)
    pl = pv.Plotter()
    with pytest.raises(TypeError):
        pl.add_mesh(mesh, scalar=mesh.active_scalars_name)
    pl = pv.Plotter()
    with pytest.raises(TypeError):
        pl.add_mesh(mesh, foo='bad')


def test_cmap_list(sphere, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    n = sphere.n_points
    scalars = np.empty(n)
    scalars[: n // 3] = 0
    scalars[n // 3 : 2 * n // 3] = 1
    scalars[2 * n // 3 :] = 2

    with pytest.raises(TypeError):
        sphere.plot(scalars=scalars, cmap=['red', None, 'blue'])

    sphere.plot(scalars=scalars, cmap=['red', 'green', 'blue'])


def test_default_name_tracking():
    N = 10
    color = 'tan'

    pl = pv.Plotter()
    for i in range(N):
        for j in range(N):
            center = (i, j, 0)
            mesh = pv.Sphere(center=center)
            pl.add_mesh(mesh, color=color)
    n_made_it = len(pl.renderer._actors)
    pl.show()
    assert n_made_it == N**2

    # release attached scalars
    mesh.ReleaseData()
    del mesh


def test_add_background_image_global(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_background_image(examples.mapfile, as_global=True)
    pl.show()


def test_add_background_image_not_global(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_background_image(examples.mapfile, as_global=False)
    pl.show()


def test_add_background_image_subplots(airplane):
    pl = pv.Plotter(shape=(2, 2))
    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.add_mesh(airplane)
    pl.subplot(1, 1)
    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.add_mesh(airplane)
    pl.remove_background_image()

    # should error out as there's no background
    with pytest.raises(RuntimeError):
        pl.remove_background_image()

    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.show()


@pytest.mark.parametrize(
    'face',
    ['-Z', '-Y', '-X', '+Z', '+Y', '+X'],
)
def test_add_floor(face):
    box = pv.Box((-100.0, -90.0, 20.0, 40.0, 100, 105)).outline()
    pl = pv.Plotter()
    pl.add_mesh(box, color='k')
    pl.add_floor(face=face, color='red', opacity=1.0)
    pl.show()


def test_add_remove_floor(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_floor(color='b', line_width=2, lighting=True)
    pl.add_bounding_box()  # needed for update_bounds_axes
    assert len(pl.renderer._floors) == 1
    pl.add_mesh(pv.Sphere(radius=1.0))
    pl.update_bounds_axes()
    assert len(pl.renderer._floors) == 1
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_floor(color='b', line_width=2, lighting=True)
    pl.remove_floors()
    assert not pl.renderer._floors
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_reset_camera_clipping_range(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)

    # get default clipping range
    default_clipping_range = pl.camera.clipping_range

    # make sure we assign something different than default
    assert default_clipping_range != (10, 100)

    # set clipping range to some random numbers and make sure
    # assignment is successful
    pl.camera.clipping_range = (10, 100)
    assert pl.camera.clipping_range == (10, 100)

    pl.reset_camera_clipping_range()
    assert pl.camera.clipping_range == default_clipping_range
    assert pl.camera.clipping_range != (10, 100)


@pytest.mark.usefixtures('no_images_to_verify')
def test_index_vs_loc():
    # first: 2d grid
    pl = pv.Plotter(shape=(2, 3))
    # index_to_loc valid cases
    vals = [0, 2, 4]
    expecteds = [(0, 0), (0, 2), (1, 1)]
    for val, expected in zip(vals, expecteds, strict=True):
        assert tuple(pl.renderers.index_to_loc(val)) == expected
    # loc_to_index valid cases
    vals = [(0, 0), (0, 2), (1, 1)]
    expecteds = [0, 2, 4]
    for val, expected in zip(vals, expecteds, strict=True):
        assert pl.renderers.loc_to_index(val) == expected
        assert pl.renderers.loc_to_index(expected) == expected

    # indexing failing cases
    with pytest.raises(TypeError):
        pl.renderers.index_to_loc(1.5)
    with pytest.raises(IndexError):
        pl.renderers.index_to_loc(-1)
    with pytest.raises(TypeError):
        pl.renderers.index_to_loc((1, 2))
    with pytest.raises(IndexError):
        pl.renderers.loc_to_index((-1, 0))
    with pytest.raises(IndexError):
        pl.renderers.loc_to_index((0, -1))
    with pytest.raises(TypeError):
        pl.renderers.loc_to_index({1, 2})
    with pytest.raises(ValueError):  # noqa: PT011
        pl.renderers.loc_to_index((1, 2, 3))

    # set active_renderer fails
    with pytest.raises(IndexError):
        pl.renderers.set_active_renderer(0, -1)

    # then: "1d" grid
    pl = pv.Plotter(shape='2|3')
    # valid cases
    for val in range(5):
        assert pl.renderers.index_to_loc(val) == val
        assert pl.renderers.index_to_loc(np.int_(val)) == val
        assert pl.renderers.loc_to_index(val) == val
        assert pl.renderers.loc_to_index(np.int_(val)) == val


def test_interactive_update():
    # Regression test for #1053
    pl = pv.Plotter()
    pl.show(interactive_update=True)
    assert isinstance(pl.iren.interactor, _vtk.vtkRenderWindowInteractor)
    pl.close()

    pl = pv.Plotter()
    with pytest.warns(UserWarning, match=r'The plotter will close immediately automatically'):
        pl.show(auto_close=True, interactive_update=True)


@pytest.mark.usefixtures('no_images_to_verify')
def test_where_is():
    pl = pv.Plotter(shape=(2, 2))
    pl.subplot(0, 0)
    pl.add_mesh(pv.Box(), name='box')
    pl.subplot(0, 1)
    pl.add_mesh(pv.Sphere(), name='sphere')
    pl.subplot(1, 0)
    pl.add_mesh(pv.Box(), name='box')
    pl.subplot(1, 1)
    pl.add_mesh(pv.Cone(), name='cone')
    places = pl.where_is('box')
    assert isinstance(places, list)
    for loc in places:
        assert isinstance(loc, tuple)


def test_log_scale(uniform):
    pl = pv.Plotter()
    pl.add_mesh(uniform, log_scale=True, clim=[-1, uniform.get_data_range()[1]])
    pl.show()


@pytest.mark.parametrize('point', [(-0.5, -0.5, 0), np.array([[-0.5], [-0.5], [0]])])
def test_set_focus(point):
    plane = pv.Plane()
    pl = pv.Plotter()
    pl.add_mesh(plane, color='tan', show_edges=True)
    pl.set_focus(point)  # focus on corner of the plane
    pl.show()


@pytest.mark.parametrize('vector', [(1.0, 1.0, 1.0), np.array([[-0.5], [-0.5], [0]])])
def test_set_viewup(verify_image_cache, vector):
    verify_image_cache.high_variance_test = True

    plane = pv.Plane()
    plane_higher = pv.Plane(center=(0, 0, 1), i_size=0.5, j_size=0.5)
    pl = pv.Plotter()
    pl.add_mesh(plane, color='tan', show_edges=False)
    pl.add_mesh(plane_higher, color='red', show_edges=False)
    pl.set_viewup(vector)
    pl.show()


def test_plot_shadows():
    pl = pv.Plotter(lighting=None)

    # add several planes
    for plane_y in [2, 5, 10]:
        screen = pv.Plane(center=(0, plane_y, 0), direction=(0, -1, 0), i_size=5, j_size=5)
        pl.add_mesh(screen, color='white')

    light = pv.Light(
        position=(0, 0, 0),
        focal_point=(0, 1, 0),
        color='cyan',
        intensity=15,
        cone_angle=15,
        positional=True,
        show_actor=True,
        attenuation_values=(2, 0, 0),
    )

    pl.add_light(light)
    pl.view_vector((1, -2, 2))

    # verify disabling shadows when not enabled does nothing
    pl.disable_shadows()

    pl.enable_shadows()

    # verify shadows can safely be enabled twice
    pl.enable_shadows()

    pl.show()


def test_plot_shadows_enable_disable():
    """Test shadows are added and removed properly"""
    pl = pv.Plotter(lighting=None)

    # add several planes
    for plane_y in [2, 5, 10]:
        screen = pv.Plane(center=(0, plane_y, 0), direction=(0, -1, 0), i_size=5, j_size=5)
        pl.add_mesh(screen, color='white')

    light = pv.Light(
        position=(0, 0, 0),
        focal_point=(0, 1, 0),
        color='cyan',
        intensity=15,
        cone_angle=15,
    )
    light.positional = True
    light.attenuation_values = (2, 0, 0)
    light.show_actor()

    pl.add_light(light)
    pl.view_vector((1, -2, 2))

    # add and remove and verify that the light passes through all via
    # image cache
    pl.enable_shadows()
    pl.disable_shadows()

    pl.show()


def test_plot_lighting_change_positional_true_false(sphere):
    light = pv.Light(positional=True, show_actor=True)

    pl = pv.Plotter(lighting=None)
    pl.add_light(light)
    light.positional = False
    pl.add_mesh(sphere)
    pl.show()


def test_plot_lighting_change_positional_false_true(sphere):
    light = pv.Light(positional=False, show_actor=True)

    pl = pv.Plotter(lighting=None)

    pl.add_light(light)
    light.positional = True
    pl.add_mesh(sphere)
    pl.show()


def test_plotter_image():
    pl = pv.Plotter()
    wsz = tuple(pl.window_size)
    pl.show()
    assert pl.image.shape[:2] == wsz


def test_scalar_cell_priorities():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1.5, 1, 0], [0, 0, 1]])
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2], [3, 0, 1, 3], [3, 1, 2, 3]])
    mesh = pv.PolyData(vertices, faces)
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]

    mesh.cell_data['colors'] = colors
    pl = pv.Plotter()
    pl.add_mesh(mesh, scalars='colors', rgb=True, preference='cell')
    pl.show()

    c = pv.Cone()
    c.cell_data['ids'] = list(range(c.n_cells))
    c.plot()


def test_collision_plot(verify_image_cache):
    """Verify rgba arrays automatically plot"""
    verify_image_cache.windows_skip_image_cache = True
    sphere0 = pv.Sphere()
    sphere1 = pv.Sphere(radius=0.6, center=(-1, 0, 0))
    col, _n_contacts = sphere0.collision(sphere1, generate_scalars=True)

    pl = pv.Plotter()
    pl.add_mesh(col)
    pl.camera_position = 'zy'
    pl.show()


@pytest.mark.skip_mac('MacOS CI fails when downloading examples')
def test_chart_plot():
    """Basic test to verify chart plots correctly"""
    # Chart 1 (bottom left)
    chart_bl = pv.Chart2D(size=(0.4, 0.4), loc=(0.05, 0.05))
    chart_bl.background_color = 'tab:purple'
    chart_bl.x_range = [np.pi / 2, 3 * np.pi / 2]
    chart_bl.y_axis.margin = 20
    chart_bl.y_axis.tick_locations = [-1, 0, 1]
    chart_bl.y_axis.tick_labels = ['Small', 'Medium', 'Large']
    chart_bl.y_axis.tick_size += 10
    chart_bl.y_axis.tick_labels_offset += 12
    chart_bl.y_axis.pen.width = 10
    chart_bl.grid = True
    x = np.linspace(0, 2 * np.pi, 50)
    y = np.cos(x) * (-1) ** np.arange(len(x))
    hidden_plot = chart_bl.line(x, y, color='k', width=40)
    hidden_plot.visible = False  # Make sure plot visibility works
    chart_bl.bar(x, y, color='#33ff33')

    # Chart 2 (bottom right)
    chart_br = pv.Chart2D(size=(0.4, 0.4), loc=(0.55, 0.05))
    chart_br.background_texture = examples.load_globe_texture()
    chart_br.active_border_color = 'r'
    chart_br.border_width = 5
    chart_br.border_style = '-.'
    chart_br.hide_axes()
    x = np.linspace(0, 1, 50)
    y = np.sin(6.5 * x - 1)
    chart_br.scatter(x, y, color='y', size=15, style='o', label='Invisible label')
    chart_br.legend_visible = False  # Check legend visibility

    # Chart 3 (top left)
    chart_tl = pv.Chart2D(size=(0.4, 0.4), loc=(0.05, 0.55))
    chart_tl.active_background_color = (0.8, 0.8, 0.2)
    chart_tl.title = 'Exponential growth'
    chart_tl.x_label = 'X axis'
    chart_tl.y_label = 'Y axis'
    chart_tl.y_axis.log_scale = True
    x = np.arange(6)
    y = 10**x
    chart_tl.line(x, y, color='tab:green', width=5, style='--')
    removed_plot = chart_tl.area(x, y, color='k')
    chart_tl.remove_plot(removed_plot)  # Make sure plot removal works

    # Chart 4 (top right)
    chart_tr = pv.Chart2D(size=(0.4, 0.4), loc=(0.55, 0.55))
    x = [0, 1, 2, 3, 4]
    ys = [[0, 1, 2, 3, 4], [1, 0, 1, 0, 1], [6, 4, 5, 3, 2]]
    chart_tr.stack(x, ys, colors='citrus', labels=['Segment 1', 'Segment 2', 'Segment 3'])
    chart_tr.legend_visible = True

    # Hidden chart (make sure chart visibility works)
    hidden_chart = pv.ChartPie([3, 4, 5])
    hidden_chart.visible = False

    # Removed chart (make sure chart removal works)
    removed_chart = pv.ChartBox([[1, 2, 3]])

    pl = pv.Plotter(window_size=(1000, 1000))
    pl.background_color = 'w'
    pl.add_chart(chart_bl, chart_br, chart_tl, chart_tr, hidden_chart, removed_chart)
    pl.remove_chart(removed_chart)
    pl.set_chart_interaction([chart_br, chart_tl])
    pl.show()


def test_chart_matplotlib_plot(verify_image_cache):
    """Test integration with matplotlib"""
    # Seeing CI failures for Conda job that need to be addressed
    verify_image_cache.high_variance_test = True

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    # First, create the matplotlib figure
    # use tight layout to keep axis labels visible on smaller figures
    fig, ax = plt.subplots(tight_layout=True)
    alphas = [0.5 + i for i in range(5)]
    betas = [*reversed(alphas)]
    N = int(1e4)
    data = [rng.beta(alpha, beta, N) for alpha, beta in zip(alphas, betas, strict=True)]
    labels = [
        f'$\\alpha={alpha:.1f}\\,;\\,\\beta={beta:.1f}$'
        for alpha, beta in zip(alphas, betas, strict=True)
    ]
    ax.violinplot(data)
    ax.set_xticks(np.arange(1, 1 + len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title('$B(\\alpha, \\beta)$')

    # Next, embed the figure into a pv plotting window
    pl = pv.Plotter()
    pl.background_color = 'w'
    chart = pv.ChartMPL(fig)
    pl.add_chart(chart)
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_get_charts():
    """Test that the get_charts method is retuning a list of charts"""
    chart = pv.Chart2D()
    pl = pv.Plotter()
    pl.add_chart(chart)

    charts = pl.renderer.get_charts()
    assert len(charts) == 1
    assert chart is charts[0]


def test_add_remove_background(sphere):
    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(sphere, color='w')
    pl.add_background_image(examples.mapfile, as_global=False)
    pl.subplot(0, 1)
    pl.add_mesh(sphere, color='w')
    pl.add_background_image(examples.mapfile, as_global=False)
    pl.remove_background_image()
    pl.show()


@pytest.mark.parametrize(
    'background',
    [examples.mapfile, Path(examples.mapfile), 'blue'],
    ids=['str', 'Path', 'color'],
)
def test_plot_mesh_background(background):
    globe = examples.load_globe()
    globe.plot(texture=pv.Texture(examples.mapfile), background=background)


@pytest.mark.usefixtures('no_images_to_verify')
def test_plot_mesh_background_raises():
    globe = examples.load_globe()
    match = 'Background must be color-like or a file path. Got {} instead.'
    with pytest.raises(TypeError, match=match):
        globe.plot(texture=pv.Texture(examples.mapfile), background={})


def test_plot_zoom(sphere):
    # it's difficult to verify that zoom actually worked since we
    # can't get the output with cpos or verify the image cache matches
    sphere.plot(zoom=2)


def test_splitting():
    nut = examples.load_nut()
    nut['sample_data'] = nut.points[:, 2]

    # feature angle of 50 will smooth the outer edges of the nut but not the inner.
    nut.plot(
        smooth_shading=True,
        split_sharp_edges=True,
        feature_angle=50,
        show_scalar_bar=False,
    )


@pytest.mark.parametrize('smooth_shading', [True, False])
@pytest.mark.parametrize('use_custom_normals', [True, False])
def test_plot_normals_smooth_shading(sphere, use_custom_normals, smooth_shading):
    sphere = pv.Sphere(phi_resolution=5, theta_resolution=5)
    sphere.clear_data()

    if use_custom_normals:
        normals = [[0, 0, -1]] * sphere.n_points
        sphere.point_data.active_normals = normals

    sphere.plot_normals(show_mesh=True, color='red', smooth_shading=smooth_shading)


@pytest.mark.skip_mac('This is a flaky test on MacOS')
def test_splitting_active_cells(cube):
    cube.cell_data['cell_id'] = range(cube.n_cells)
    cube = cube.triangulate().subdivide(1)
    cube.plot(
        smooth_shading=True,
        split_sharp_edges=True,
        show_scalar_bar=False,
    )


def test_add_cursor():
    sphere = pv.Sphere()
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_cursor()
    pl.show()


def test_enable_stereo_render(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    pl = pv.Plotter()
    pl.add_mesh(pv.Cube())
    pl.camera.distance = 0.1
    pl.enable_stereo_render()
    pl.show()


def test_disable_stereo_render():
    pl = pv.Plotter()
    pl.add_mesh(pv.Cube())
    pl.camera.distance = 0.1
    pl.enable_stereo_render()
    pl.disable_stereo_render()
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_orbit_on_path(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere, show_edges=True)
    pl.orbit_on_path(step=0.01, progress_bar=True)
    pl.close()


def test_rectlinear_edge_case(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    # ensure that edges look like square edges regardless of the dtype of X
    xrng = np.arange(-10, 10, 5)
    yrng = np.arange(-10, 10, 5)
    zrng = [1]
    rec_grid = pv.RectilinearGrid(xrng, yrng, zrng)
    rec_grid.plot(show_edges=True, cpos='xy')


def test_pointset_plot(pointset):
    pointset.plot()

    pl = pv.Plotter()
    pl.add_mesh(pointset, scalars=range(pointset.n_points), show_scalar_bar=False)
    pl.show()


def test_pointset_plot_as_points(pointset):
    pl = pv.Plotter()
    pl.add_points(pointset, scalars=range(pointset.n_points), show_scalar_bar=False)
    pl.show()


def test_pointset_plot_vtk():
    pointset = _vtk.vtkPointSet()
    points = pv.vtk_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    pointset.SetPoints(points)

    pl = pv.Plotter()
    pl.add_mesh(pointset, color='red', point_size=25)
    pl.show()


def test_pointset_plot_as_points_vtk():
    pointset = _vtk.vtkPointSet()
    points = pv.vtk_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    pointset.SetPoints(points)

    pl = pv.Plotter()
    pl.add_points(pointset, color='red', point_size=25)
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
@pytest.mark.skipif(not HAS_IMAGEIO, reason='Requires imageio')
def test_write_gif(sphere, tmpdir):
    basename = 'write_gif.gif'
    path = str(tmpdir.join(basename))
    pl = pv.Plotter()
    pl.open_gif(path)
    pl.add_mesh(sphere)
    pl.write_frame()
    pl.close()

    # assert file exists and is not empty
    assert Path(path).is_file()
    assert Path(path).stat().st_size


def test_ruler():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.add_ruler([-0.6, -0.6, 0], [0.6, -0.6, 0], font_size_factor=1.2)
    pl.view_xy()
    pl.show()


def test_ruler_number_labels():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.add_ruler([-0.6, -0.6, 0], [0.6, -0.6, 0], font_size_factor=1.2, number_labels=2)
    pl.view_xy()
    pl.show()


def test_legend_scale(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_legend_scale(color='red')
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_legend_scale(color='red', xy_label_mode=True)
    pl.view_xy()
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_legend_scale(
        xy_label_mode=True,
        bottom_axis_visibility=False,
        left_axis_visibility=False,
        right_axis_visibility=False,
        top_axis_visibility=False,
    )
    pl.view_xy()
    pl.show()


def test_plot_complex_value(plane, verify_image_cache):
    """Test plotting complex data."""
    verify_image_cache.windows_skip_image_cache = True
    data = np.arange(plane.n_points, dtype=np.complex128)
    data += np.linspace(0, 1, plane.n_points) * -1j

    with pytest.warns(np.exceptions.ComplexWarning, match='Casting complex'):
        plane.plot(scalars=data)

    pl = pv.Plotter()
    with pytest.warns(np.exceptions.ComplexWarning, match='Casting complex'):
        pl.add_mesh(plane, scalars=data, show_scalar_bar=True)
    pl.show()


def test_screenshot_notebook(tmpdir):
    tmp_dir = tmpdir.mkdir('tmpdir2')
    filename = str(tmp_dir.join('tmp.png'))

    pl = pv.Plotter(notebook=True)
    pl.theme.jupyter_backend = 'static'
    pl.add_mesh(pv.Cone())
    pl.show(screenshot=filename)
    pl.close()

    assert Path(filename).is_file()


def test_culling_frontface(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere, culling='frontface')
    pl.show()


def test_add_text():
    pl = pv.Plotter()
    pl.add_text('Upper Left', position='upper_left', font_size=25, color='blue')
    pl.add_text('Center', position=(0.5, 0.5), viewport=True, orientation=-90)
    pl.show()


@pytest.mark.needs_vtk_version(9, 4, 0)
def test_add_text_latex():
    """Test LaTeX symbols."""
    pl = pv.Plotter()
    pl.add_text(r'$\rho$', position='upper_left', font_size=150, color='blue')
    pl.show()


def test_add_text_font_file():
    pl = pv.Plotter()
    font_file = str(Path(__file__).parent / 'fonts/Mplus2-Regular.ttf')
    pl.add_text('左上', position='upper_left', font_size=25, color='blue', font_file=font_file)
    pl.add_text(
        '中央',
        position=(0.5, 0.5),
        viewport=True,
        orientation=-90,
        font_file=font_file,
    )
    pl.show()


@skip_windows_mesa
def test_plot_categories_int(sphere):
    sphere['data'] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars='data', categories=5, lighting=False)
    pl.show()


@skip_windows_mesa
def test_plot_categories_true(sphere):
    sphere['data'] = np.linspace(0, 5, sphere.n_points, dtype=int)
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars='data', categories=True, lighting=False)
    pl.show()


@pytest.mark.skip_windows
def test_depth_of_field():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere(), show_edges=True)
    pl.enable_depth_of_field()
    pl.show()


def test_blurring():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere(), show_edges=True)
    pl.add_blurring()
    pl.show()


@skip_mesa
def test_ssaa_pass():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere(), show_edges=True)
    pl.enable_anti_aliasing('ssaa')
    pl.show()


@skip_windows_mesa
def test_ssao_pass(verify_image_cache):
    verify_image_cache.macos_skip_image_cache = True
    ugrid = pv.ImageData(dimensions=(2, 2, 2)).to_tetrahedra(5).explode()
    pl = pv.Plotter()
    pl.add_mesh(ugrid)

    pl.enable_ssao()
    pl.show()

    with pytest.raises(RuntimeError, match=r'The renderer has been closed.'):
        pl.disable_ssao()


@skip_mesa
def test_ssao_pass_from_helper(verify_image_cache):
    verify_image_cache.macos_skip_image_cache = True  # high variance (~1000) on MacOS 15
    ugrid = pv.ImageData(dimensions=(2, 2, 2)).to_tetrahedra(5).explode()

    ugrid.plot(ssao=True)


@pytest.mark.skip_windows
def test_many_multi_pass(verify_image_cache):
    verify_image_cache.high_variance_test = True

    pl = pv.Plotter(lighting=None)
    pl.add_mesh(pv.Sphere(), show_edges=True)
    pl.add_light(pv.Light(position=(0, 0, 10)))
    pl.enable_anti_aliasing('ssaa')
    pl.enable_depth_of_field()
    pl.add_blurring()
    pl.enable_shadows()
    pl.enable_eye_dome_lighting()
    pl.show()


def test_plot_composite_many_options(multiblock_poly):
    # add composite data
    for block in multiblock_poly:
        # use np.uint8 for coverage of non-standard datatypes
        block['data'] = np.arange(block.n_points, dtype=np.uint8)

    pl = pv.Plotter()
    pl.add_composite(
        multiblock_poly,
        scalars='data',
        annotations={94: 'foo', 162: 'bar'},
        above_color='k',
        below_color='w',
        clim=[64, 192],
        log_scale=True,
        flip_scalars=True,
        label='my composite',
    )
    pl.add_legend()
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_plot_composite_raise(sphere, multiblock_poly):
    pl = pv.Plotter()
    with pytest.raises(TypeError, match='Must be a composite dataset'):
        pl.add_composite(sphere)
    with pytest.raises(TypeError, match='must be a string for'):
        pl.add_composite(multiblock_poly, scalars=range(10))


def test_plot_composite_lookup_table(multiblock_poly, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    lut = pv.LookupTable('Greens', n_values=8)
    pl = pv.Plotter()
    pl.add_composite(multiblock_poly, scalars='data_b', cmap=lut)
    pl.show()


def test_plot_composite_preference_cell(multiblock_poly, verify_image_cache):
    """Show that we will plot cell data if both point and cell exist in all."""
    verify_image_cache.windows_skip_image_cache = True

    # use the first two datasets as the third is missing scalars
    multiblock_poly[:2].plot(preference='cell')


@pytest.mark.skip_windows('Test fails on Windows because of opacity')
@skip_lesser_9_4_X
def test_plot_composite_poly_scalars_opacity(multiblock_poly):
    pl = pv.Plotter()

    _actor, mapper = pl.add_composite(
        multiblock_poly,
        scalars='data_a',
        nan_color='green',
        color_missing_with_nan=True,
        smooth_shading=True,
        show_edges=True,
        cmap='bwr',
    )
    mapper.block_attr[1].color = 'blue'
    mapper.block_attr[1].opacity = 0.5

    pl.camera_position = 'xy'

    pl.show()


@skip_lesser_9_4_X
def test_plot_composite_poly_scalars_cell(multiblock_poly, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    pl = pv.Plotter()

    _actor, mapper = pl.add_composite(
        multiblock_poly,
        scalars='cell_data',
    )
    mapper.block_attr[1].color = 'blue'

    pl.camera_position = 'xy'
    pl.show()


def test_plot_composite_poly_no_scalars(multiblock_poly):
    pl = pv.Plotter()

    _actor, mapper = pl.add_composite(
        multiblock_poly,
        color='red',
        lighting=False,
    )

    # Note: set the camera position before making the blocks invisible
    # VTK ignores invisible blocks when computing camera bounds.
    pl.camera_position = 'xy'
    mapper.block_attr[2].color = 'blue'
    mapper.block_attr[3].visible = False

    pl.show()


@skip_windows_mesa
def test_plot_composite_poly_component_norm(multiblock_poly):
    for ii, block in enumerate(multiblock_poly):
        data = block.compute_normals().point_data['Normals']
        data[:, ii] *= 2
        block['data'] = data

    pl = pv.Plotter()
    pl.add_composite(multiblock_poly, scalars='data', cmap='bwr')
    pl.show()


def test_plot_composite_poly_component_single(multiblock_poly):
    for block in multiblock_poly:
        data = block.compute_normals().point_data['Normals']
        block['data'] = data

    pl = pv.Plotter()
    with pytest.raises(ValueError, match='must be nonnegative'):
        pl.add_composite(multiblock_poly, scalars='data', component=-1)
    with pytest.raises(TypeError, match='None or an integer'):
        pl.add_composite(multiblock_poly, scalars='data', component='apple')

    pl.add_composite(multiblock_poly, scalars='data', component=1)
    pl.show()


def test_plot_composite_poly_component_nested_multiblock(multiblock_poly, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    for block in multiblock_poly:
        data = block.compute_normals().point_data['Normals']
        block['data'] = data

    multiblock_poly2 = multiblock_poly.copy()
    for block in multiblock_poly2:
        block.points += np.array([0, 0, 1])

    multimulti = pv.MultiBlock([multiblock_poly, multiblock_poly2])

    pl = pv.Plotter()
    pl.add_composite(multimulti, scalars='data', style='points', clim=[0.99, 1.01], copy_mesh=True)
    pl.add_composite(multimulti, scalars='data', component=1, copy_mesh=True)
    pl.show()


def test_plot_composite_poly_complex(multiblock_poly):
    # add composite data
    for block in multiblock_poly:
        data = np.arange(block.n_points) + np.arange(block.n_points) * 1j
        block['data'] = data

    # make a multi_multi for better coverage
    multi_multi = pv.MultiBlock([multiblock_poly, multiblock_poly])

    pl = pv.Plotter()
    with pytest.warns(np.exceptions.ComplexWarning, match='Casting complex'):
        pl.add_composite(multi_multi, scalars='data')
    pl.show()


def test_plot_composite_rgba(multiblock_poly):
    # add composite data
    for i, block in enumerate(multiblock_poly):
        rgba_value = np.zeros((block.n_points, 3), dtype=np.uint8)
        rgba_value[:, i] = np.linspace(0, 255, block.n_points)
        block['data'] = rgba_value

    pl = pv.Plotter()
    with pytest.raises(ValueError, match='3/4 in shape'):
        pl.add_composite(multiblock_poly, scalars='all_data', rgba=True)
    pl.add_composite(multiblock_poly, scalars='data', rgba=True)
    pl.show()


# Ignore VTK error during teardown: ERROR: Bad table range: [inf, -inf] vtkLookupTable.cxx
@pytest.mark.skip_catch_vtk_errors
def test_plot_composite_bool(multiblock_poly, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    # add in bool data
    for block in multiblock_poly:
        block['scalars'] = np.zeros(block.n_points, dtype=bool)
        block['scalars'][::2] = 1

    pl = pv.Plotter()
    pl.add_composite(multiblock_poly, scalars='scalars')
    pl.show()


@pytest.mark.usefixtures('no_images_to_verify')
def test_export_obj(tmpdir, sphere):
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.obj'))

    pl = pv.Plotter()
    pl.add_mesh(sphere, smooth_shading=True)

    with pytest.raises(ValueError, match=r'end with ".obj"'):
        pl.export_obj('badfilename')

    pl.export_obj(filename)

    # Check that the object file has been written
    assert Path(filename).exists()

    # Check that when we close the plotter, the adequate error is raised
    pl.close()
    with pytest.raises(RuntimeError, match=r'This plotter must still have a render window open.'):
        pl.export_obj(filename)


def test_multi_plot_scalars(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    res = 5
    plane = pv.Plane(j_resolution=res, i_resolution=res, direction=(0, 0, -1))
    plane.clear_data()
    kek = np.arange(res + 1)
    kek = np.tile(kek, (res + 1, 1))
    u = kek.flatten().copy()
    v = kek.T.flatten().copy()

    plane.point_data['u'] = u
    plane.point_data['v'] = v

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_text('"u" point scalars')
    pl.add_mesh(plane, scalars='u', copy_mesh=True)
    pl.subplot(0, 1)
    pl.add_text('"v" point scalars')
    pl.add_mesh(plane, scalars='v', copy_mesh=True)
    pl.show()


@skip_windows_mesa
def test_bool_scalars(sphere):
    sphere['scalars'] = np.zeros(sphere.n_points, dtype=bool)
    sphere['scalars'][::2] = 1
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show()


@pytest.mark.skip_windows('Test fails on Windows because of pbr')
# pbr required
def test_property_pbr(verify_image_cache):
    verify_image_cache.macos_skip_image_cache = True
    prop = pv.Property(interpolation='pbr', metallic=1.0)
    prop.plot()


def test_tight_square(noise_2d):
    noise_2d.plot(
        window_size=[800, 200],
        show_scalar_bar=False,
        cpos='xy',
        zoom='tight',
    )


@skip_windows_mesa  # due to opacity
def test_plot_cell():
    grid = examples.cells.Tetrahedron()
    examples.plot_cell(grid)


@pytest.mark.parametrize(
    ('line_width', 'point_size', 'font_size', 'normals_scale'),
    [(5, 30, 20, 0.1), (10, 80, 50, 0.25)],
)
def test_plot_cell_kwargs(line_width, point_size, font_size, normals_scale, verify_image_cache):
    # Skip since variance is too high across operating systems
    verify_image_cache.macos_skip_image_cache = True
    verify_image_cache.windows_skip_image_cache = True

    grid = examples.cells.Polyhedron()
    examples.plot_cell(
        grid,
        show_normals=True,
        point_size=point_size,
        font_size=font_size,
        line_width=line_width,
        normals_scale=normals_scale,
    )


@skip_windows_mesa  # due to opacity
@pytest.mark.parametrize('wrong_orientation', [True, False])
def test_plot_cell_polyhedron(wrong_orientation):
    points = [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]]
    cells = [4, 3, 0, 2, 1, 3, 0, 1, 3, 3, 0, 3, 2, 3, 1, 2, 3]
    if wrong_orientation:
        # Swap two ids
        id1 = cells[2]
        cells[2] = cells[3]
        cells[3] = id1
    cells = [len(cells), *cells]
    polyhedron = pv.UnstructuredGrid(cells, [pv.CellType.POLYHEDRON], points)
    examples.plot_cell(polyhedron, show_normals=True)


@pytest.mark.needs_vtk_version(9, 5, 0, reason='Merge order differs with older vtk')
def test_plot_cell_multiple_cell_types(verify_image_cache):
    verify_image_cache.high_variance_test = True
    cell3d = examples.cells.Polyhedron()
    cell2d = examples.cells.Quadrilateral().translate((2, -2, 0))
    grid = cell2d + cell3d
    examples.plot_cell(grid, show_normals=True)


def test_tight_square_padding():
    grid = pv.ImageData(dimensions=(200, 100, 1))
    grid['data'] = np.arange(grid.n_points)
    pl = pv.Plotter(window_size=(150, 150))
    pl.add_mesh(grid, show_scalar_bar=False)
    pl.camera_position = 'xy'
    pl.camera.tight(padding=0.05)
    # limit to widest dimension
    assert np.allclose(pl.window_size, [150, 75])
    pl.show()


def test_tight_tall():
    grid = pv.ImageData(dimensions=(100, 200, 1))
    grid['data'] = np.arange(grid.n_points)
    pl = pv.Plotter(window_size=(150, 150))
    pl.add_mesh(grid, show_scalar_bar=False)
    pl.camera_position = 'xy'
    with pytest.raises(ValueError, match='can only be "tight"'):
        pl.camera.zoom('invalid')
    pl.camera.tight()
    # limit to widest dimension
    assert np.allclose(pl.window_size, [75, 150], rtol=1)
    pl.show()


def test_tight_wide():
    grid = pv.ImageData(dimensions=(200, 100, 1))
    grid['data'] = np.arange(grid.n_points)
    pl = pv.Plotter(window_size=(150, 150))
    pl.add_mesh(grid, show_scalar_bar=False)
    pl.camera_position = 'xy'
    pl.camera.tight()
    # limit to widest dimension
    assert np.allclose(pl.window_size, [150, 75])
    pl.show()


@pytest.mark.parametrize('view', ['xy', 'yx', 'xz', 'zx', 'yz', 'zy'])
@pytest.mark.parametrize('negative', [False, True])
def test_tight_direction(view, negative, colorful_tetrahedron):
    """Test camera.tight() with various views like xy."""
    pl = pv.Plotter()
    pl.add_mesh(colorful_tetrahedron, scalars='colors', rgb=True, preference='cell')
    pl.camera.tight(view=view, negative=negative)
    pl.add_axes()
    pl.show()


def test_tight_multiple_objects():
    pl = pv.Plotter()
    pl.add_mesh(
        pv.Cone(center=(0.0, -2.0, 0.0), direction=(0.0, -1.0, 0.0), height=1.0, radius=1.0),
    )
    pl.add_mesh(pv.Sphere(center=(0.0, 0.0, 0.0)))
    pl.camera.tight()
    pl.add_axes()
    pl.show()


def test_backface_params():
    mesh = pv.ParametricCatalanMinimal()

    with pytest.raises(TypeError, match=r'pyvista.Property or a dict'):
        mesh.plot(backface_params='invalid')

    params = dict(color='blue', smooth_shading=True)
    backface_params = dict(color='red', specular=1.0, specular_power=50.0)
    backface_prop = pv.Property(**backface_params)

    # check Property can be passed
    pl = pv.Plotter()
    pl.add_mesh(mesh, **params, backface_params=backface_prop)
    pl.close()

    # check and cache dict
    pl = pv.Plotter()
    pl.add_mesh(mesh, **params, backface_params=backface_params)
    pl.view_xz()
    pl.show()


def test_remove_bounds_axes(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    actor = pl.show_bounds(grid='front', location='outer')
    assert isinstance(actor, _vtk.vtkActor)
    pl.remove_bounds_axes()
    pl.show()


def test_charts_sin():
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x)
    chart = pv.Chart2D()
    chart.scatter(x, y)
    chart.line(x, y, color='r')
    chart.show()


def test_lookup_table():
    lut = pv.LookupTable('viridis')
    lut.n_values = 8
    lut.below_range_color = 'black'
    lut.above_range_color = 'grey'
    lut.nan_color = 'r'
    lut.nan_opacity = 0.5

    lut.plot()


def test_lookup_table_nan_hidden():
    lut = pv.LookupTable('viridis')
    lut.n_values = 8
    lut.below_range_color = 'black'
    lut.above_range_color = 'grey'
    lut.nan_opacity = 0

    lut.plot()


def test_lookup_table_above_below_opacity():
    lut = pv.LookupTable('viridis')
    lut.n_values = 8
    lut.below_range_color = 'blue'
    lut.below_range_opacity = 0.5
    lut.above_range_color = 'green'
    lut.above_range_opacity = 0.5
    lut.nan_color = 'r'
    lut.nan_opacity = 0.5

    lut.plot()


@skip_windows_mesa
@skip_lesser_9_4_X_depth_peeling
def test_plot_nan_color(uniform):
    arg = uniform.active_scalars < uniform.active_scalars.mean()
    uniform.active_scalars[arg] = np.nan
    # NaN values should be hidden
    pl = pv.Plotter()
    pl.add_mesh(uniform, nan_opacity=0)
    pl.enable_depth_peeling()
    pl.show()
    # nan annotation should appear on scalar bar
    pl = pv.Plotter()
    pl.add_mesh(
        uniform,
        nan_opacity=0.5,
        nan_color='green',
        scalar_bar_args=dict(nan_annotation=True),
    )
    pl.enable_depth_peeling()
    pl.show()


@skip_lesser_9_4_X_depth_peeling
def test_plot_above_below_color(uniform):
    mean = uniform.active_scalars.mean()
    clim = (mean - mean / 2, mean + mean / 2)

    lut = pv.LookupTable('viridis')
    lut.n_values = 8
    lut.below_range_color = 'blue'
    lut.below_range_opacity = 0.5
    lut.above_range_color = 'green'
    lut.above_range_opacity = 0.5
    lut.scalar_range = clim

    pl = pv.Plotter()
    pl.add_mesh(uniform, cmap=lut, scalar_bar_args={'above_label': '', 'below_label': ''})
    pl.enable_depth_peeling()
    pl.show()


def test_plotter_lookup_table(sphere, verify_image_cache):
    # Image regression test fails within OSMesa on Windows
    verify_image_cache.windows_skip_image_cache = True

    lut = pv.LookupTable('Reds')
    lut.n_values = 3
    lut.scalar_range = (sphere.points[:, 2].min(), sphere.points[:, 2].max())
    sphere.plot(scalars=sphere.points[:, 2], cmap=lut)


@skip_windows_mesa  # due to opacity
@pytest.mark.skip_check_gc("vtkTypeUInt8Array not gc'd on Python 3.14 vtk dev wheels")
def test_plotter_volume_lookup_table(uniform):
    uniform.set_active_scalars('Spatial Point Data')

    lut = pv.LookupTable()
    lut.apply_cmap('coolwarm', 255)
    lut.apply_opacity('linear')
    lut.scalar_range = uniform.get_data_range()

    pl = pv.Plotter()
    pl.add_volume(uniform, cmap=lut)
    pl.show()


@skip_windows_mesa  # due to opacity
@pytest.mark.skip_check_gc
def test_plotter_volume_lookup_table_reactive(uniform):
    """Ensure that changes to the underlying lookup table are reflected by the volume property."""
    uniform.set_active_scalars('Spatial Point Data')

    pl = pv.Plotter()
    actor = pl.add_volume(uniform, cmap='viridis', clim=[0, uniform.n_points // 2])
    actor.mapper.lookup_table.apply_cmap('coolwarm', 255)
    actor.mapper.lookup_table.apply_opacity('sigmoid')
    actor.mapper.lookup_table.scalar_range = [0, uniform.n_points]
    pl.render()
    pl.show()

    # Test switching out the lookup table
    pl = pv.Plotter()
    actor = pl.add_volume(
        uniform, cmap='viridis', clim=[0, uniform.n_points // 2], show_scalar_bar=False
    )

    lut = pv.LookupTable()
    lut.apply_cmap('coolwarm', 255)
    actor.prop.apply_lookup_table(lut)
    lut.apply_opacity('sigmoid')
    lut.scalar_range = [0, uniform.n_points]
    pl.render()
    pl.show()


@skip_windows_mesa  # due to opacity
def test_plotter_volume_log_scale(uniform):
    uniform.clear_data()
    uniform['data'] = np.logspace(1, 5, uniform.n_points)

    pl = pv.Plotter()
    pl.add_volume(uniform, scalars='data', log_scale=True)
    pl.show()


@skip_windows_mesa  # due to opacity
def test_plotter_volume_add_scalars(uniform):
    uniform.clear_data()
    pl = pv.Plotter()
    pl.add_volume(uniform, scalars=uniform.z, show_scalar_bar=False)
    pl.show()


@skip_windows_mesa  # due to opacity
def test_plotter_volume_add_scalars_log_scale(uniform):
    uniform.clear_data()
    pl = pv.Plotter()

    # for below zero to trigger the edge case
    scalars = uniform.z - 0.01
    assert any(scalars < 0), 'need negative values to test log_scale entrirely'
    pl.add_volume(uniform, scalars=scalars, show_scalar_bar=True, log_scale=True)
    pl.show()


@skip_windows_mesa  # due to opacity
def test_plotter_volume_opacity_n_colors():
    # See https://github.com/pyvista/pyvista/issues/5505
    grid = pv.ImageData(dimensions=(9, 9, 9))
    grid['scalars'] = -grid.x

    pl = pv.Plotter()
    pl.add_volume(grid, opacity='linear', n_colors=128)
    pl.show()

    pl = pv.Plotter()
    pl.add_volume(grid, opacity='linear', n_colors=5)
    pl.show()


@skip_windows_mesa  # due to opacity
def test_plotter_volume_clim():
    # Validate that we can use clim with volume rendering
    grid = pv.ImageData(dimensions=(9, 9, 9))
    grid['scalars'] = np.arange(grid.n_points)

    pl = pv.Plotter()
    pl.add_volume(grid, clim=[0, grid.n_points], show_scalar_bar=True)
    pl.show()

    pl = pv.Plotter()
    pl.add_volume(grid, clim=[grid.n_points * 0.25, grid.n_points * 0.75], show_scalar_bar=True)
    pl.show()

    # Validate that we can change clim on the mapper
    pl = pv.Plotter()
    actor = pl.add_volume(grid, clim=[0, grid.n_points], show_scalar_bar=True)
    actor.mapper.scalar_range = [grid.n_points * 0.25, grid.n_points * 0.75]
    pl.show()


@skip_windows_mesa  # due to opacity
def test_plotter_volume_clim_uint():
    # Validate that add_volume does not set 0-255 as the default clim for uint8 data
    # for example the `load_frog_tissues` dataset is uint8 with values 0-29 and we want
    # add_volume to automatically set the clim to 0-29 as that is the valid range
    # Let's validate this with a toy dataset:
    volume = pv.ImageData(dimensions=(3, 3, 3))
    volume['data'] = np.arange(volume.n_points).astype(np.uint8)

    pl = pv.Plotter()
    actor = pl.add_volume(volume, show_scalar_bar=True)
    pl.show()
    assert actor.mapper.scalar_range == (0, np.prod(volume.dimensions) - 1)


def test_plot_actor(sphere):
    pl = pv.Plotter()
    actor = pl.add_mesh(sphere, lighting=False, color='b', show_edges=True)
    actor.plot()


def test_wireframe_color(sphere):
    sphere.plot(lighting=False, color='b', style='wireframe')


@pytest.mark.parametrize('direction', ['xy', 'yx', 'xz', 'zx', 'yz', 'zy'])
@pytest.mark.parametrize('negative', [False, True])
def test_view_xyz(direction, negative, colorful_tetrahedron):
    """Test various methods like view_xy."""
    pl = pv.Plotter()
    pl.add_mesh(colorful_tetrahedron, scalars='colors', rgb=True, preference='cell')
    getattr(pl, f'view_{direction}')(negative=negative)
    pl.add_axes()
    pl.show()


@pytest.mark.skip_windows
def test_plot_points_gaussian(sphere):
    sphere.plot(
        color='r',
        style='points_gaussian',
        render_points_as_spheres=False,
        point_size=20,
        opacity=0.5,
    )


@pytest.mark.skip_windows
def test_plot_points_gaussian_scalars(sphere):
    sphere.plot(
        scalars=sphere.points[:, 2],
        style='points_gaussian',
        render_points_as_spheres=False,
        point_size=20,
        opacity=0.5,
        show_scalar_bar=False,
    )


@pytest.mark.skip_windows
def test_plot_points_gaussian_as_spheres(sphere):
    sphere.plot(
        color='b',
        style='points_gaussian',
        render_points_as_spheres=True,
        emissive=True,
        point_size=20,
        opacity=0.5,
    )


@pytest.mark.skip_windows
def test_plot_points_gaussian_scale(sphere):
    sphere['z'] = sphere.points[:, 2] * 0.1
    pl = pv.Plotter()
    actor = pl.add_mesh(
        sphere,
        style='points_gaussian',
        render_points_as_spheres=True,
        emissive=False,
        show_scalar_bar=False,
    )
    actor.mapper.scale_array = 'z'
    pl.view_xz()
    pl.show()


@skip_windows_mesa  # due to opacity
def test_plot_show_vertices(sphere, hexbeam, multiblock_all):
    sphere.plot(
        color='w',
        show_vertices=True,
        point_size=20,
        lighting=False,
        render_points_as_spheres=True,
        vertex_style='points',
        vertex_opacity=0.1,
        vertex_color='b',
    )

    hexbeam.plot(
        color='w',
        opacity=0.5,
        show_vertices=True,
        point_size=20,
        lighting=True,
        render_points_as_spheres=True,
        vertex_style='points',
        vertex_color='r',
    )

    multiblock_all.plot(
        color='w',
        show_vertices=True,
        point_size=3,
        render_points_as_spheres=True,
    )


def test_remove_vertices_actor(sphere):
    # Test remove by name
    pl = pv.Plotter()
    pl.add_mesh(
        sphere,
        color='w',
        show_vertices=True,
        point_size=20,
        lighting=False,
        vertex_style='points',
        vertex_color='b',
        name='sphere',
    )
    pl.remove_actor('sphere')
    pl.show()
    # Test remove by Actor
    pl = pv.Plotter()
    actor = pl.add_mesh(
        sphere,
        color='w',
        show_vertices=True,
        point_size=20,
        lighting=False,
        vertex_style='points',
        vertex_color='b',
        name='sphere',
    )
    pl.remove_actor(actor)
    pl.show()


@pytest.mark.skip_windows
def test_add_point_scalar_labels_fmt(verify_image_cache):
    # parallel on GitHub hosted sometimes has high image error
    verify_image_cache.macos_skip_image_cache = True

    mesh = examples.load_uniform().slice()
    pl = pv.Plotter()
    pl.add_mesh(mesh, scalars='Spatial Point Data', show_edges=True)
    fmt = '%.3f' if pv.vtk_version_info < (9, 6, 0) else '{:.3f}'
    pl.add_point_scalar_labels(mesh, 'Spatial Point Data', point_size=20, font_size=36, fmt=fmt)
    pl.camera_position = pv.CameraPosition(
        position=(7, 4, 5), focal_point=(4.4, 7.0, 7.2), viewup=(0.8, 0.5, 0.25)
    )
    pl.show()


def test_plot_individual_cell(hexbeam):
    hexbeam.get_cell(0).plot(color='b')


def test_add_point_scalar_labels_list():
    pl = pv.Plotter()

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    with pytest.raises(TypeError):
        pl.add_point_scalar_labels(points=False, labels=labels)
    with pytest.raises(TypeError):
        pl.add_point_scalar_labels(points=points, labels=False)

    pl.add_point_scalar_labels(points, labels)
    pl.show()


def test_plot_algorithm_cone():
    algo = pv.ConeSource()
    algo.SetResolution(10)

    pl = pv.Plotter()
    pl.add_mesh(algo, color='red')
    pl.show(auto_close=False)
    # Use low resolution so it appears in image regression tests easily
    algo.SetResolution(3)
    pl.show()

    # Bump resolution and plot with silhouette
    algo.SetResolution(8)
    pl = pv.Plotter()
    pl.add_mesh(algo, color='red', silhouette=True)
    pl.show()


@skip_windows_mesa
def test_plot_algorithm_scalars():
    name, name2 = 'foo', 'bar'
    mesh = pv.Wavelet()
    mesh.point_data[name] = np.arange(mesh.n_points)
    mesh.cell_data[name2] = np.arange(mesh.n_cells)
    assert mesh.active_scalars_name != name
    assert mesh.active_scalars_name != name2

    alg = _vtk.vtkGeometryFilter()
    alg.SetInputDataObject(mesh)

    pl = pv.Plotter()
    pl.add_mesh(alg, scalars=name)
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(alg, scalars=name2)
    pl.show()


def test_algorithm_add_points():
    algo = _vtk.vtkRTAnalyticSource()

    pl = pv.Plotter()
    pl.add_points(algo)
    pl.show()


def test_algorithm_add_point_labels():
    algo = pv.ConeSource()
    elev = _vtk.vtkElevationFilter()
    elev.SetInputConnection(algo.GetOutputPort())
    elev.SetLowPoint(0, 0, -1)
    elev.SetHighPoint(0, 0, 1)

    pl = pv.Plotter()
    pl.add_point_labels(elev, 'Elevation', always_visible=False)
    pl.show()


def test_pointset_to_polydata_algorithm(pointset):
    alg = _vtk.vtkElevationFilter()
    alg.SetInputDataObject(pointset)

    pl = pv.Plotter()
    pl.add_mesh(alg, scalars='Elevation')
    pl.show()

    assert isinstance(alg.GetOutputDataObject(0), _vtk.vtkPointSet)


def test_add_ids_algorithm():
    algo = _vtk.vtkCubeSource()

    alg = algorithms.add_ids_algorithm(algo)

    pl = pv.Plotter()
    pl.add_mesh(alg, scalars='point_ids')
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(alg, scalars='cell_ids')
    pl.show()

    result = pv.wrap(alg.GetOutputDataObject(0))
    assert 'point_ids' in result.point_data
    assert 'cell_ids' in result.cell_data


@skip_windows_mesa
def test_plot_volume_rgba(uniform):
    with pytest.raises(ValueError, match='dimensions'):
        uniform.plot(volume=True, scalars=np.empty((uniform.n_points, 1, 1)))

    scalars = uniform.points - (uniform.origin)
    scalars /= scalars.max()
    scalars = np.hstack((scalars, scalars[::-1, -1].reshape(-1, 1) ** 2))
    scalars *= 255

    with pytest.raises(ValueError, match='datatype'):
        uniform.plot(volume=True, scalars=scalars)

    scalars = scalars.astype(np.uint8)
    uniform.plot(volume=True, scalars=scalars)

    pl = pv.Plotter()
    with pytest.warns(UserWarning, match='Ignoring custom opacity'):
        pl.add_volume(uniform, scalars=scalars, opacity='sigmoid_10')
    pl.show()


def test_plot_window_size_context():
    pl = pv.Plotter()
    pl.add_mesh(pv.Cube())
    with pl.window_size_context((200, 200)):
        pl.show()

    pl.close()
    with pytest.warns(UserWarning, match='Attempting to set window_size'):
        with pl.window_size_context((200, 200)):
            pass


def test_color_cycler():
    pv.global_theme.color_cycler = 'default'
    pl = pv.Plotter()
    a0 = pl.add_mesh(pv.Cone(center=(0, 0, 0)))
    a1 = pl.add_mesh(pv.Cube(center=(1, 0, 0)))
    a2 = pl.add_mesh(pv.Sphere(center=(1, 1, 0)))
    a3 = pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))
    pl.show()
    assert a0.prop.color.hex_rgb == matplotlib_default_colors[0]
    assert a1.prop.color.hex_rgb == matplotlib_default_colors[1]
    assert a2.prop.color.hex_rgb == matplotlib_default_colors[2]
    assert a3.prop.color.hex_rgb == matplotlib_default_colors[3]

    pv.global_theme.color_cycler = ['red', 'green', 'blue']
    pl = pv.Plotter()
    a0 = pl.add_mesh(pv.Cone(center=(0, 0, 0)))  # red
    a1 = pl.add_mesh(pv.Cube(center=(1, 0, 0)))  # green
    a2 = pl.add_mesh(pv.Sphere(center=(1, 1, 0)))  # blue
    a3 = pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))  # red again
    pl.show()

    assert a0.prop.color.name == 'red'
    assert a1.prop.color.name == 'green'
    assert a2.prop.color.name == 'blue'
    assert a3.prop.color.name == 'red'

    # Make sure all solid color matching theme default again
    pv.global_theme.color_cycler = None
    pl = pv.Plotter()
    a0 = pl.add_mesh(pv.Cone(center=(0, 0, 0)))
    a1 = pl.add_mesh(pv.Cube(center=(1, 0, 0)))
    pl.show()

    assert a0.prop.color.hex_rgb == pv.global_theme.color.hex_rgb
    assert a1.prop.color.hex_rgb == pv.global_theme.color.hex_rgb

    pl = pv.Plotter()
    with pytest.raises(ValueError):  # noqa: PT011
        pl.set_color_cycler('foo')
    with pytest.raises(TypeError):
        pl.set_color_cycler(5)


def test_color_cycler_true():
    pv.global_theme.color_cycler = 'default'
    a = pv.Wavelet().clip(invert=True)
    b = pv.Wavelet().clip(invert=False)

    pl = pv.Plotter()
    a0 = pl.add_mesh(a, color=True)
    a1 = pl.add_mesh(b, color=True)
    pl.show()

    assert a0.prop.color.hex_rgb == matplotlib_default_colors[0]
    assert a1.prop.color.hex_rgb == matplotlib_default_colors[1]


def test_plotter_render_callback():
    n_ren = [0]

    def callback(this_pl):
        assert isinstance(this_pl, pv.Plotter)
        n_ren[0] += 1

    pl = pv.Plotter()
    pl.add_on_render_callback(callback, render_event=True)
    assert len(pl._on_render_callbacks) == 0
    pl.add_on_render_callback(callback, render_event=False)
    assert len(pl._on_render_callbacks) == 1
    pl.show()
    assert n_ren[0] == 1  # if two, render_event not respected
    pl.clear_on_render_callbacks()
    assert len(pl._on_render_callbacks) == 0


def test_plot_texture_alone(texture):
    """Test plotting directly from the Texture class."""
    texture.plot()


def test_plot_texture_flip_x(texture):
    """Test Texture.flip_x."""
    texture.flip_x().plot()


def test_plot_texture_flip_y(texture):
    """Test Texture.flip_y."""
    texture.flip_y().plot()


@pytest.mark.skipif(CI_WINDOWS, reason='Windows CI testing segfaults on pbr')
@pytest.mark.needs_vtk_version(less_than=(9, 3), reason='This is broken on VTK 9.3')
def test_plot_cubemap_alone(cubemap, verify_image_cache):
    """Test plotting directly from the Texture class."""
    verify_image_cache.high_variance_test = True
    cubemap.plot()


@pytest.mark.skip_egl(reason='Render window will be current with offscreen builds of VTK.')
def test_not_current(verify_image_cache):
    verify_image_cache.skip = True

    pl = pv.Plotter()
    assert not pl.render_window.IsCurrent()
    with pytest.raises(RenderWindowUnavailable, match='current'):
        pl._check_has_ren_win()
    pl.show(auto_close=False)
    pl._make_render_window_current()
    pl._check_has_ren_win()
    pl.close()


@pytest.mark.parametrize('name', ['default', 'all', 'matplotlib', 'warm'])
def test_color_cycler_names(name):
    pl = pv.Plotter()
    pl.set_color_cycler(name)
    a0 = pl.add_mesh(pv.Cone(center=(0, 0, 0)))
    a1 = pl.add_mesh(pv.Cube(center=(1, 0, 0)))
    a2 = pl.add_mesh(pv.Sphere(center=(1, 1, 0)))
    a3 = pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))
    pl.show()
    assert a0.prop.color.hex_rgb != pv.global_theme.color.hex_rgb
    assert a1.prop.color.hex_rgb != pv.global_theme.color.hex_rgb
    assert a2.prop.color.hex_rgb != pv.global_theme.color.hex_rgb
    assert a3.prop.color.hex_rgb != pv.global_theme.color.hex_rgb


def test_scalar_bar_actor_removal(sphere):
    # verify that when removing an actor we also remove the
    # corresponding scalar bar

    sphere['scalars'] = sphere.points[:, 2]

    pl = pv.Plotter()
    actor = pl.add_mesh(sphere, show_scalar_bar=True)
    assert list(pl.scalar_bars.keys()) == ['scalars']
    pl.remove_actor(actor)
    assert len(pl.scalar_bars) == 0
    pl.show()


def test_update_scalar_bar_range(sphere):
    sphere['z'] = sphere.points[:, 2]
    minmax = sphere.bounds[2:4]  # ymin, ymax
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars='z')

    # automatic mapper lookup works
    pl.update_scalar_bar_range(minmax)
    # named mapper lookup works
    pl.update_scalar_bar_range(minmax, name='z')
    # missing name raises
    with pytest.raises(ValueError, match='not valid/not found in this plotter'):
        pl.update_scalar_bar_range(minmax, name='invalid')
    pl.show()


def test_add_remove_scalar_bar(sphere):
    """Verify a scalar bar can be added and removed."""
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars=sphere.points[:, 2], show_scalar_bar=False)

    # verify that the number of slots is restored
    init_slots = len(pl._scalar_bar_slots)
    pl.add_scalar_bar(interactive=True)
    pl.remove_scalar_bar()
    assert len(pl._scalar_bar_slots) == init_slots
    pl.show()


@pytest.mark.parametrize('geometry_type', [*pv.AxesGeometrySource.GEOMETRY_TYPES, 'custom'])
def test_axes_geometry_shaft_type_tip_type(geometry_type):
    if geometry_type == 'custom':
        geometry_type = pv.ParametricConicSpiral()
    pv.AxesGeometrySource(
        shaft_length=0.4,
        shaft_radius=0.05,
        tip_radius=0.1,
        shaft_type=geometry_type,
        tip_type=geometry_type,
    ).output.plot()


POSITION = (-0.5, -0.5, 1)
ORIENTATION = (10, 20, 30)
SCALE = (1.5, 2, 2.5)
ORIGIN = (2, 1.5, 1)
actor = pv.Actor()
actor.position = POSITION
actor.orientation = ORIENTATION
actor.scale = SCALE
actor.origin = ORIGIN
USER_MATRIX = pv.array_from_vtkmatrix(actor.GetMatrix())

XYZ_ASSEMBLY_TEST_CASES = dict(
    default={},
    position=dict(position=POSITION),
    orientation=dict(orientation=ORIENTATION),
    scale=dict(scale=SCALE),
    origin=dict(origin=ORIGIN, orientation=ORIENTATION),
    user_matrix=dict(user_matrix=USER_MATRIX),
)


@pytest.mark.parametrize(
    'test_kwargs',
    XYZ_ASSEMBLY_TEST_CASES.values(),
    ids=XYZ_ASSEMBLY_TEST_CASES.keys(),
)
@pytest.mark.parametrize(
    ('assembly', 'obj_kwargs'),
    [
        (pv.AxesAssembly, {}),
        (pv.AxesAssemblySymmetric, dict(label_size=25)),
        (pv.PlanesAssembly, dict(opacity=1)),
    ],
    ids=['Axes', 'AxesSymmetric', 'Planes'],
)
def test_xyz_assembly(test_kwargs, assembly, obj_kwargs, verify_image_cache):
    verify_image_cache.high_variance_test = True
    pl = pv.Plotter()
    assembly = assembly(**test_kwargs, **obj_kwargs, label_color='white')
    pl.add_actor(assembly)
    if isinstance(assembly, pv.PlanesAssembly):
        assembly.camera = pl.camera
    if test_kwargs:
        # Add second axes at the origin for visual reference
        pl.add_axes_at_origin(x_color='black', y_color='black', z_color='black', labels_off=True)
    pl.show()


@pytest.mark.parametrize(
    'assembly',
    [pv.AxesAssembly, pv.AxesAssemblySymmetric, pv.PlanesAssembly],
    ids=['Axes', 'AxesSymmetric', 'Planes'],
)
def test_xyz_assembly_show_labels_false(assembly):
    pl = pv.Plotter()
    assembly = assembly(show_labels=False)
    pl.add_actor(assembly)
    if isinstance(assembly, pv.PlanesAssembly):
        assembly.camera = pl.camera
    pl.show()


@pytest.mark.parametrize('scale_mode', get_args(ScaleModeOptions))
@pytest.mark.parametrize('symmetric', [True, False])
def test_axes_assembly_scale_mode(scale_mode, symmetric):
    cls = pv.AxesAssemblySymmetric if symmetric else pv.AxesAssembly
    scale = (0.4, 1.0, 2.5)
    axes_scale = cls(position=(-1, 2, 1), scale=scale, scale_mode=scale_mode)
    matrix = pv.Transform().scale(scale).matrix
    axes_matrix = cls(user_matrix=matrix, scale_mode=scale_mode)

    pl = pv.Plotter()
    pl.add_actor(axes_scale)
    pl.add_actor(axes_matrix)
    pl.enable_parallel_projection()
    pl.show()


@pytest.mark.parametrize('relative_position', [(0, 0, -0.5), (0, 0, 0.5)], ids=['bottom', 'top'])
def test_label_prop3d(relative_position):
    dataset = pv.Cone(direction=(0, 0, 1))
    actor = pv.Actor(mapper=pv.DataSetMapper(dataset=dataset))
    actor.user_matrix = USER_MATRIX

    label = pv.Label(text='TEXT', size=100, relative_position=relative_position)
    label.prop.justification_horizontal = 'center'
    label.user_matrix = USER_MATRIX

    pl = pv.Plotter()
    pl.add_actor(label)
    pl.add_actor(actor)
    pl.show()


def test_axes_actor_default_colors():
    axes = pv.AxesActor()
    axes.shaft_type = pv.AxesActor.ShaftType.CYLINDER

    pl = pv.Plotter()
    pl.add_actor(axes)
    pl.camera.zoom(1.5)
    pl.show()


def test_axes_actor_properties():
    axes = pv.Axes()
    axes_actor = axes.axes_actor
    axes_actor.shaft_type = pv.AxesActor.ShaftType.CYLINDER
    axes_actor.tip_type = pv.AxesActor.TipType.SPHERE
    axes_actor.x_label = 'U'
    axes_actor.y_label = 'V'
    axes_actor.z_label = 'W'

    # Test actor properties using color
    x_color = (1.0, 0.0, 1.0)  # magenta
    y_color = (1.0, 1.0, 0.0)  # yellow
    z_color = (0.0, 1.0, 1.0)  # cyan

    axes_actor.x_axis_shaft_properties.color = x_color
    axes_actor.x_axis_tip_properties.color = x_color

    axes_actor.y_axis_shaft_properties.color = y_color
    axes_actor.y_axis_tip_properties.color = y_color

    axes_actor.z_axis_shaft_properties.color = z_color
    axes_actor.z_axis_tip_properties.color = z_color

    pl = pv.Plotter()
    pl.add_actor(axes_actor)
    pl.camera.zoom(1.5)
    pl.show()


def test_show_bounds_no_labels():
    pl = pv.Plotter()
    pl.add_mesh(pv.Cone())
    pl.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        show_xlabels=False,
        show_ylabels=False,
        show_zlabels=False,
        xtitle='Easting',
        ytitle='Northing',
        ztitle='Elevation',
    )
    pl.camera_position = pv.CameraPosition(
        position=(1.97, 1.89, 1.66),
        focal_point=(0.05, -0.05, 0.00),
        viewup=(-0.36, -0.36, 0.85),
    )
    pl.show()


def test_show_bounds_n_labels():
    pl = pv.Plotter()
    pl.add_mesh(pv.Cone())
    pl.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        n_xlabels=2,
        n_ylabels=2,
        n_zlabels=2,
        xtitle='Easting',
        ytitle='Northing',
        ztitle='Elevation',
    )
    pl.camera_position = pv.CameraPosition(
        position=(1.97, 1.89, 1.66),
        focal_point=(0.05, -0.05, 0.00),
        viewup=(-0.36, -0.36, 0.85),
    )
    pl.show()


@skip_lesser_9_3_X
def test_radial_gradient_background():
    pl = pv.Plotter()
    pl.set_background('white', right='black')
    pl.show()

    pl = pv.Plotter()
    pl.set_background('white', side='black')
    pl.show()

    pl = pv.Plotter()
    pl.set_background('white', corner='black')
    pl.show()

    pl = pv.Plotter()
    with pytest.raises(ValueError):  # noqa: PT011
        pl.set_background('white', top='black', right='black')


@pytest.mark.usefixtures('no_images_to_verify')
def test_no_empty_meshes():
    pl = pv.Plotter()
    with pytest.raises(ValueError, match='Empty meshes'):
        pl.add_mesh(pv.PolyData())


@pytest.mark.skipif(CI_WINDOWS, reason='Windows CI testing fatal exception: access violation')
def test_voxelize_volume():
    mesh = examples.download_cow()
    cpos = [(15, 3, 15), (0, 0, 0), (0, 0, 0)]

    # Create an equal density voxel volume and plot the result.
    with pytest.warns(pv.PyVistaDeprecationWarning):
        vox = pv.voxelize_volume(mesh, density=0.15)
    vox.plot(scalars='InsideMesh', show_edges=True, cpos=cpos)

    # Create a voxel volume from unequal density dimensions and plot result.
    with pytest.warns(pv.PyVistaDeprecationWarning):
        vox = pv.voxelize_volume(mesh, density=[0.15, 0.15, 0.5])
    vox.plot(scalars='InsideMesh', show_edges=True, cpos=cpos)


def test_enable_custom_trackball_style():
    def setup_plot():
        mesh = pv.Cube()
        mesh['face_id'] = np.arange(6)
        pl = pv.Plotter()
        # mostly use the settings from `enable_2d_style`
        # but also test environment_rotate
        pl.enable_custom_trackball_style(
            left='pan',
            middle='spin',
            right='dolly',
            shift_left='dolly',
            control_left='spin',
            shift_middle='dolly',
            control_middle='pan',
            shift_right='environment_rotate',
            control_right='rotate',
        )
        pl.enable_parallel_projection()
        pl.add_mesh(mesh, scalars='face_id', show_scalar_bar=False)
        return pl

    # baseline, image
    pl = setup_plot()
    pl.show()

    start = (100, 100)
    pan = rotate = env_rotate = (150, 150)
    spin = (100, 150)
    dolly = (100, 25)

    # Compare all images to baseline
    # - Panning moves up and left
    # - Spinning rotates while fixing the view direction
    # - Dollying zooms out
    # - Rotating rotates freely without fixing view direction

    # left click pans, image 1
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._mouse_left_button_press(*start)
    pl.iren._mouse_left_button_release(*pan)
    pl.close()

    # middle click spins, image 2
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._mouse_middle_button_press(*start)
    pl.iren._mouse_middle_button_release(*spin)
    pl.close()

    # right click dollys, image 3
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._mouse_right_button_press(*start)
    pl.iren._mouse_right_button_release(*dolly)
    pl.close()

    # ctrl left click spins, image 4
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._control_key_press()
    pl.iren._mouse_left_button_press(*start)
    pl.iren._mouse_left_button_release(*spin)
    pl.iren._control_key_release()
    pl.close()

    # shift left click dollys, image 5
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._shift_key_press()
    pl.iren._mouse_left_button_press(*start)
    pl.iren._mouse_left_button_release(*dolly)
    pl.iren._shift_key_release()
    pl.close()

    # ctrl middle click pans, image 6
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._control_key_press()
    pl.iren._mouse_middle_button_press(*start)
    pl.iren._mouse_middle_button_release(*pan)
    pl.iren._control_key_release()
    pl.close()

    # shift middle click dollys, image 7
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._shift_key_press()
    pl.iren._mouse_middle_button_press(*start)
    pl.iren._mouse_middle_button_release(*dolly)
    pl.iren._shift_key_release()
    pl.close()

    # ctrl right click rotates, image 8
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._control_key_press()
    pl.iren._mouse_right_button_press(*start)
    pl.iren._mouse_right_button_release(*rotate)
    pl.iren._control_key_release()
    pl.close()

    # shift right click environment rotate, image 9
    # does nothing here
    pl = setup_plot()
    pl.show(auto_close=False)
    pl.iren._shift_key_press()
    pl.iren._mouse_right_button_press(*start)
    pl.iren._mouse_right_button_release(*env_rotate)
    pl.iren._shift_key_release()
    pl.close()


def test_create_axes_orientation_box():
    actor = pv.create_axes_orientation_box(
        line_width=4,
        text_scale=0.53,
        edge_color='red',
        x_color='k',
        y_color=None,
        z_color=None,
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        color_box=False,
        labels_off=False,
        opacity=1.0,
        show_text_edges=True,
    )
    pl = pv.Plotter()
    _ = pl.add_actor(actor)
    pl.show()


_TypeType = TypeVar('_TypeType', bound=type)


def _get_module_members(module: ModuleType, typ: _TypeType) -> dict[str, _TypeType]:
    """Get all members of a specified type which are defined locally inside a module."""

    def is_local(obj):
        return type(obj) is typ and obj.__module__ == module.__name__

    return dict(inspect.getmembers(module, predicate=is_local))


def _get_module_functions(module: ModuleType):
    """Get all functions defined locally inside a module."""
    return _get_module_members(module, typ=FunctionType)


def _get_default_kwargs(call: Callable) -> dict[str, Any]:
    """Get all args/kwargs and their default value"""
    params = dict(inspect.signature(call).parameters)
    # Get default value for positional or keyword args
    return {
        key: val.default
        for key, val in params.items()
        if val.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }


def _has_param(call: Callable, param: str) -> bool:
    kwargs = _get_default_kwargs(call)
    if param in kwargs:
        # Param is valid if it is explicitly named in function signature
        return True
    else:
        # Try adding param as a new kwarg
        kwargs[param] = None
        try:
            call(**kwargs)
        except TypeError as ex:
            # Param is not valid only if a kwarg TypeError is raised
            return 'unexpected keyword argument' not in repr(ex)
        else:
            return True


def _get_default_param_value(call: Callable, param: str) -> Any:
    return _get_default_kwargs(call)[param]


def _generate_direction_object_functions() -> ItemsView[str, FunctionType]:
    """Generate a list of geometric or parametric object functions which have a direction."""
    geo_functions = _get_module_functions(pv.core.geometric_objects)
    para_functions = _get_module_functions(pv.core.parametric_objects)
    functions: dict[str, FunctionType] = {**geo_functions, **para_functions}

    # Only keep functions with capitalized first letter
    # Only keep functions which accept `normal` or `direction` param
    functions = {
        name: func
        for name, func in functions.items()
        if name[0].isupper() and (_has_param(func, 'direction') or _has_param(func, 'normal'))
    }
    # Remove Spline from test case (if present).
    if 'Spline' in functions.keys():
        functions.pop('Spline')
    # Add a separate test for vtk < 9.3
    functions['Capsule_legacy'] = functions['Capsule']
    actual_names = functions.keys()
    expected_names = [
        'Arrow',
        'Capsule',
        'Capsule_legacy',
        'CircularArcFromNormal',
        'Cone',
        'Cylinder',
        'CylinderStructured',
        'Disc',
        'ParametricBohemianDome',
        'ParametricBour',
        'ParametricBoy',
        'ParametricCatalanMinimal',
        'ParametricConicSpiral',
        'ParametricCrossCap',
        'ParametricDini',
        'ParametricEllipsoid',
        'ParametricEnneper',
        'ParametricFigure8Klein',
        'ParametricHenneberg',
        'ParametricKlein',
        'ParametricKuen',
        'ParametricMobius',
        'ParametricPluckerConoid',
        'ParametricPseudosphere',
        'ParametricRandomHills',
        'ParametricRoman',
        'ParametricSuperEllipsoid',
        'ParametricSuperToroid',
        'ParametricTorus',
        'Plane',
        'Polygon',
        'SolidSphere',
        'SolidSphereGeneric',
        'Sphere',
        'Text3D',
    ]

    assert sorted(actual_names) == sorted(expected_names)
    return functions.items()


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'direction_obj_test_case' in metafunc.fixturenames:
        functions = _generate_direction_object_functions()
        positive_cases = [(name, func, 'pos') for name, func in functions]
        negative_cases = [(name, func, 'neg') for name, func in functions]
        test_cases = [*positive_cases, *negative_cases]

        # Name test cases using object name and direction
        ids = [f'{case[0]}-{case[2]}' for case in test_cases]
        metafunc.parametrize('direction_obj_test_case', test_cases, ids=ids)


def test_direction_objects(direction_obj_test_case):
    name, func, direction = direction_obj_test_case
    positive_dir = direction == 'pos'

    # Add required args if needed
    kwargs = {}
    if name == 'CircularArcFromNormal':
        kwargs['center'] = (0, 0, 0)
    elif name == 'Text3D':
        kwargs['string'] = 'Text3D'

    # Test Capsule separately based on vtk version
    if 'Capsule' in name:
        legacy_vtk = pv.vtk_version_info < (9, 3)
        if (legacy_vtk and 'legacy' not in name) or (not legacy_vtk and 'legacy' in name):
            pytest.xfail(
                'Test capsule separately for different vtk versions. Expected to fail if testing '
                'with wrong version.',
            )

    direction_param_name = None

    def _create_object(_direction=None):
        nonlocal direction_param_name
        try:
            # Create using `direction` param
            direction_param_name = 'direction'
            obj = func(**kwargs) if _direction is None else func(direction=_direction, **kwargs)

        except TypeError:
            # Create using `normal` param
            direction_param_name = 'normal'
            obj = func(**kwargs) if _direction is None else func(normal=_direction, **kwargs)

        # Add scalars tied to point IDs as visual markers of object orientation
        scalars = np.arange(obj.n_points)
        obj['scalars'] = scalars % 32

        return obj

    text_kwargs = dict(font_size=10)
    axes_kwargs = dict(viewport=(0, 0, 1.0, 1.0))

    pl = pv.Plotter(shape=(2, 2))

    pl.subplot(0, 0)
    pl.add_mesh(_create_object())
    pl.add_text(name, **text_kwargs)
    pl.add_axes()

    direction = (1, 0, 0) if positive_dir else (-1, 0, 0)
    obj = _create_object(_direction=direction)
    pl.subplot(1, 0)
    pl.add_mesh(obj)
    pl.add_text(f'{direction_param_name}={direction}', **text_kwargs)
    pl.view_yz()
    pl.add_axes(**axes_kwargs)

    direction = (0, 1, 0) if positive_dir else (0, -1, 0)
    obj = _create_object(_direction=direction)
    pl.subplot(1, 1)
    pl.add_mesh(obj)
    pl.add_text(f'{direction_param_name}={direction}', **text_kwargs)
    pl.view_zx()
    pl.add_axes(**axes_kwargs)

    direction = (0, 0, 1) if positive_dir else (0, 0, -1)
    obj = _create_object(_direction=direction)
    pl.subplot(0, 1)
    pl.add_mesh(obj)
    pl.add_text(f'{direction_param_name}={direction}', **text_kwargs)
    pl.view_xy()
    pl.add_axes(**axes_kwargs)

    pl.show()


@pytest.mark.needs_vtk_version(9, 3, 0)
@pytest.mark.parametrize('orient_faces', [True, False])
def test_contour_labels_orient_faces(labeled_image, orient_faces):  # noqa: F811
    if pv.vtk_version_info >= (9, 6, 0) and orient_faces is False:
        # This bug was fixed in VTK 9.6
        pytest.xfail('The faces are oriented correctly, even when orient_faces=False')
    with pytest.warns(pv.PyVistaDeprecationWarning):
        contour = labeled_image.contour_labels(background_value=5, orient_faces=orient_faces)
    contour.clear_data()
    contour.plot_normals()


@pytest.fixture
def _allow_empty_mesh():
    # setup
    flag = pv.global_theme.allow_empty_mesh
    pv.global_theme.allow_empty_mesh = True
    yield
    # teardown
    pv.global_theme.allow_empty_mesh = flag


@pytest.fixture
def _show_edges():
    # setup
    flag = pv.global_theme.show_edges
    pv.global_theme.show_edges = True
    yield
    # teardown
    pv.global_theme.show_edges = flag


@pytest.mark.usefixtures('_allow_empty_mesh', '_show_edges')
@pytest.mark.parametrize(
    ('select_inputs', 'select_outputs'),
    [(None, None), (None, 2), (2, 2)],
    ids=['in_None-out_None', 'in_None-out_2', 'in_2-out_2'],
)
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_boundary_style(
    labeled_image,  # noqa: F811
    select_inputs,
    select_outputs,
):
    def plot_boundary_labels(mesh_):
        # Split labeled boundaries for regions 2 and 5
        values = [[2, 0], [2, 5], [5, 0]]
        label_meshes = mesh_.split_values(
            values,
            component_mode='multi',
        )
        assert label_meshes.n_blocks <= len(values)
        pl.add_mesh(label_meshes[0], color='red', label=str(values[0]))
        pl.add_mesh(label_meshes[1], color='lime', label=str(values[1]))
        pl.add_mesh(label_meshes[2], color='blue', label=str(values[2]))

    def _generate_mesh(style):
        with pytest.warns(pv.PyVistaDeprecationWarning):
            mesh = labeled_image.contour_labels(
                boundary_style=style,
                **test_kwargs,
                **fixed_kwargs,
            )
        # Shrink mesh to help reveal cells hidden behind other cells
        return mesh.shrink(0.7)

    # Remove one foreground point from the fixture to simplify plots
    labeled_image.active_scalars[19] = 0

    fixed_kwargs = dict(
        smoothing_distance=0.3,
        output_mesh_type='quads',
        orient_faces=False,
        simplify_output=False,
    )

    test_kwargs = dict(
        select_inputs=select_inputs,
        select_outputs=select_outputs,
    )

    # Create meshes to plot
    EXTERNAL, ALL, INTERNAL = 'external', 'all', 'internal'
    external_mesh = _generate_mesh(EXTERNAL)
    all_mesh = _generate_mesh(ALL)
    internal_mesh = _generate_mesh(INTERNAL)

    # Offset to fit in a single frame
    external_mesh.points += (0, 0, 1)
    internal_mesh.points += (0, 0, -1)

    pl = pv.Plotter()

    plot_boundary_labels(external_mesh)
    pl.add_text(EXTERNAL, position='upper_left')

    plot_boundary_labels(all_mesh)
    pl.add_text(ALL, position='left_edge')

    plot_boundary_labels(internal_mesh)
    pl.add_text(INTERNAL, position='lower_left')

    pl.camera_position = pv.CameraPosition(
        position=(5, 4, 3.5), focal_point=(1, 1, 1), viewup=(0.0, 0.0, 1.0)
    )
    pl.show(return_cpos=True)


@pytest.mark.parametrize(
    ('smoothing_distance', 'smoothing_scale'),
    [(0, None), (None, 0), (5, 0.5), (5, 1)],
    ids=[
        'dist_0-scale_None',
        'dist_None-scale_0',
        'dist_5-scale_0.5',
        'dist_5-scale_1',
    ],
)
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_smoothing_constraint(
    labeled_image,  # noqa: F811
    smoothing_distance,
    smoothing_scale,
):
    # Scale spacing for visualization
    labeled_image.spacing = (10, 10, 10)

    with pytest.warns(pv.PyVistaDeprecationWarning):
        mesh = labeled_image.contour_labels(
            'all',
            smoothing_distance=smoothing_distance,
            smoothing_scale=smoothing_scale,
            pad_background=False,
            orient_faces=False,
        )

    # Translate so origin is in bottom left corner
    mesh.points -= np.array(mesh.bounds)[[0, 2, 4]]

    # Add box of fixed size for scale
    box = pv.Box(bounds=(0, 10, 0, 10, 0, 10)).extract_all_edges()
    pl = pv.Plotter()
    pl.add_mesh(mesh, show_scalar_bar=False)
    pl.add_mesh(box)

    # Configure plot to enable showing one side of the mesh to visualize
    # the scale of the smoothing applied by the smoothing constraints
    pl.enable_parallel_projection()
    pl.view_yz()
    pl.show_grid()
    pl.reset_camera()
    pl.camera.zoom(1.5)
    pl.show()


@pytest.mark.usefixtures('_show_edges')
@pytest.mark.parametrize('smoothing', [True, False])
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_compare_select_inputs_select_outputs(
    labeled_image,  # noqa: F811
    smoothing,
):
    common_kwargs = dict(
        smoothing=smoothing,
        smoothing_distance=0.8,
        output_mesh_type='quads',
        orient_faces=False,
    )
    with pytest.warns(pv.PyVistaDeprecationWarning):
        mesh_select_inputs = labeled_image.contour_labels(select_inputs=2, **common_kwargs)
    with pytest.warns(pv.PyVistaDeprecationWarning):
        mesh_select_outputs = labeled_image.contour_labels(select_outputs=2, **common_kwargs)

    pl = pv.Plotter()
    pl.add_mesh(mesh_select_inputs, color='red', opacity=0.7)
    pl.add_mesh(mesh_select_outputs, color='blue', opacity=0.7)
    pl.view_xy()
    pl.show()


@pytest.mark.skip_windows('Windows colors all plane cells red (bug?)')
@pytest.mark.parametrize('normal_sign', ['+', '-'])
@pytest.mark.parametrize('plane', ['yz', 'zx', 'xy'])
def test_orthogonal_planes_source_normals(normal_sign, plane):
    plane_source = pv.OrthogonalPlanesSource(normal_sign=normal_sign, resolution=2)
    output = plane_source.output
    plane = output[plane]
    plane['_rgb'] = [
        pv.Color('red').float_rgb,
        pv.Color('green').float_rgb,
        pv.Color('blue').float_rgb,
        pv.Color('yellow').float_rgb,
    ]
    plane.plot_normals(mag=0.8, color='white', lighting=False, show_edges=True)


@pytest.mark.skip_check_gc  # gc fails, suspected memory leak with merge
@pytest.mark.parametrize('distance', [(1, 1, 1), (-1, -1, -1)], ids=['+', '-'])
def test_orthogonal_planes_source_push(distance):
    source = pv.OrthogonalPlanesSource()
    source.push(distance)
    planes = pv.merge(source.output, merge_points=False)
    planes.plot_normals()


# Add skips since Plane's edges differ (e.g. triangles instead of quads)
@pytest.mark.skip_windows
@pytest.mark.parametrize(
    'resolution',
    [(10, 1, 1), (1, 10, 1), (1, 1, 10)],
    ids=['x_resolution', 'y_resolution', 'z_resolution'],
)
def test_orthogonal_planes_source_resolution(resolution):
    plane_source = pv.OrthogonalPlanesSource(resolution=resolution)
    plane_source.output.plot(show_edges=True, line_width=5, lighting=False)


@pytest.mark.skip_windows
@pytest.mark.parametrize(
    ('name', 'value'),
    [
        (None, None),
        ('shrink_factor', 0.1),
        ('shrink_factor', 1.0),
        ('shrink_factor', 2),
        ('explode_factor', 0.0),
        ('explode_factor', 0.5),
        ('explode_factor', -0.5),
        ('frame_width', 0.1),
        ('frame_width', 1.0),
    ],
)
def test_cube_faces_source(name, value):
    kwargs = {name: value} if name is not None else {}
    cube_faces_source = pv.CubeFacesSource(**kwargs, x_length=1, y_length=2, z_length=3)
    pv.merge(cube_faces_source.output, merge_points=False).plot_normals(
        mag=0.5, show_edges=True, line_width=3, edge_color='red'
    )


def test_planes_assembly():
    pl = pv.Plotter()
    actor = pv.PlanesAssembly()
    pl.add_actor(actor)
    actor.camera = pl.camera
    pl.add_axes()
    pl.show()


# Difference in clipping generates error of approx 500
@pytest.mark.parametrize('label_offset', [0.05, 0, -0.05])
@pytest.mark.parametrize(
    ('label_kwarg', 'camera_position'),
    [('x_label', 'yz'), ('y_label', 'zx'), ('z_label', 'xy')],
)
@pytest.mark.parametrize(('label_mode', 'label_size'), [('2D', 25), ('3D', 40)])
def test_planes_assembly_label_position(
    label_kwarg, camera_position, label_mode, label_size, label_offset
):
    pl = pv.Plotter()

    for edge in ('right', 'top', 'left', 'bottom'):
        for position in (-1, -0.5, 0, 0.5, 1):
            actor = pv.PlanesAssembly(
                labels=['', '', ''],
                opacity=0.01,
                label_edge=edge,
                label_position=position,
                label_mode=label_mode,
                label_offset=label_offset,
                label_size=label_size,
            )
            label_name = str(position) + edge[0].upper()
            setattr(actor, label_kwarg, label_name)
            pl.add_actor(actor)
            actor.camera = pl.camera
    pl.camera_position = camera_position
    pl.add_axes_at_origin()
    pl.show()


BOUNDS = (-50, 50, -10, 30, -80, 80)


@pytest.mark.parametrize(
    'bounds',
    [BOUNDS, BOUNDS * np.array(0.01)],
)
@pytest.mark.parametrize('label_size', [25, 50])
def test_planes_assembly_label_size(bounds, label_size):
    pl = pv.Plotter()
    labels = ['FIRST ', 'SECOND ', 'THIRD ']
    common_kwargs = dict(bounds=bounds, label_size=label_size, opacity=0.1)
    for label_mode in ['2D', '3D']:
        actor = pv.PlanesAssembly(
            x_label=labels[0] + label_mode,
            y_label=labels[1] + label_mode,
            z_label=labels[2] + label_mode,
            label_mode=label_mode,
            label_color='white' if label_mode == '3D' else 'black',
            **common_kwargs,
        )
        pl.add_actor(actor)
        actor.camera = pl.camera
    pl.show()


@pytest.fixture
def oblique_cone():
    return pv.examples.download_oblique_cone()


@pytest.mark.skip_mac(
    'Barely exceeds error threshold (slightly different rendering).', machine='arm64'
)
@pytest.mark.parametrize('box_style', ['outline', 'face', 'frame'])
def test_bounding_box(oblique_cone, box_style):
    pl = pv.Plotter()
    box = oblique_cone.bounding_box(box_style)
    oriented_box = oblique_cone.bounding_box(box_style, oriented=True)

    pl.add_mesh(oblique_cone)
    pl.add_mesh(box, color='red', opacity=0.5, line_width=5)
    pl.add_mesh(oriented_box, color='blue', opacity=0.5, line_width=5)
    pl.show()


@pytest.mark.parametrize('operator', ['or', 'and', 'ior', 'iand'])
def test_bitwise_and_or_of_polydata(operator):
    radius = 0.5
    shift = [0.25, 0.25, 0.25]
    kwargs = dict(theta_resolution=10, phi_resolution=10)
    sphere = pv.Sphere(radius=radius, **kwargs)
    sphere_shifted = pv.Sphere(radius=radius, center=shift, **kwargs)
    # Expand the wireframe ever so slightly to avoid rendering artifacts
    wireframe = pv.Sphere(radius=radius + 0.001, **kwargs).extract_all_edges()
    wireframe_shifted = pv.Sphere(
        radius=radius + 0.001, center=shift, **kwargs
    ).extract_all_edges()

    if operator == 'or':
        result = sphere | sphere_shifted
    elif operator == 'and':
        result = sphere & sphere_shifted
    elif operator == 'ior':
        result = sphere.copy()
        result |= sphere_shifted
    elif operator == 'iand':
        result = sphere.copy()
        result &= sphere_shifted
    pl = pv.Plotter()
    pl.add_mesh(wireframe, color='r', line_width=2)
    pl.add_mesh(wireframe_shifted, color='b', line_width=2)
    pl.add_mesh(result, color='lightblue')
    pl.camera_position = 'xz'
    pl.show()


def test_plot_logo():
    logo_plotter = demos.plot_logo(window_size=(400, 300), just_return_plotter=True)
    logo_plotter.show()


@skip_mesa
def test_plot_wireframe_style():
    sphere = pv.Sphere()
    sphere.plot(style='wireframe')


@pytest.mark.parametrize('as_multiblock', ['as_multiblock', None])
@pytest.mark.parametrize('return_clipped', ['return_clipped', None])
def test_clip_multiblock_crinkle(return_clipped, as_multiblock):
    return_clipped = bool(return_clipped)
    as_multiblock = bool(as_multiblock)

    mesh = examples.download_bunny_coarse()
    if as_multiblock:
        mesh = pv.MultiBlock([mesh])

    clipped = mesh.clip('x', crinkle=True, return_clipped=return_clipped)
    if isinstance(clipped, tuple):
        clipped = pv.MultiBlock(clipped)
        clipped[0].translate((-0.1, 0, 0), inplace=True)

    pl = pv.Plotter()
    pl.add_mesh(clipped, show_edges=True)
    pl.view_xy()
    pl.show()


@pytest.mark.parametrize('as_multiblock', ['as_multiblock', None])
def test_clip_box_crinkle(as_multiblock):
    as_multiblock = bool(as_multiblock)

    mesh = examples.download_bunny_coarse()
    if as_multiblock:
        mesh = pv.MultiBlock([mesh])
    bounds = mesh.bounds
    x_size, _, _ = mesh.bounds_size
    bounds_right = (
        bounds.x_min,
        bounds.x_min + x_size / 2,
        bounds.y_min,
        bounds.y_max,
        bounds.z_min,
        bounds.z_max,
    )
    bounds_left = (
        bounds.x_min + x_size / 2,
        bounds.x_max,
        bounds.y_min,
        bounds.y_max,
        bounds.z_min,
        bounds.z_max,
    )
    clipped_right = mesh.clip_box(bounds_right, crinkle=True)
    clipped_left = mesh.clip_box(bounds_left, crinkle=True)
    clipped_right.translate((0.1, 0, 0), inplace=True)

    pl = pv.Plotter()
    pl.add_mesh(clipped_right, show_edges=True)
    pl.add_mesh(clipped_left, show_edges=True)
    pl.view_xy()
    pl.show()


def test_box():
    box = pv.Box(level=0)
    box['cell_data'] = np.arange(box.n_cells)

    box_multi = pv.Box(level=[0, 1, 2]).translate((-3, 3, 0))
    box_multi['cell_data'] = np.arange(box_multi.n_cells)

    pl = pv.Plotter()
    pl.add_mesh(box_multi, show_edges=True, cmap='turbo')
    pl.add_mesh(box, show_edges=True, cmap='turbo')
    pl.add_point_labels(box_multi.points, np.arange(box_multi.n_points))
    pl.add_point_labels(box.points, np.arange(box.n_points))
    pl.show()


def test_partitioned_dataset(sphere):
    mesh = pv.PartitionedDataSet([sphere])
    mesh.plot()


@pytest.mark.parametrize('cell_example', cell_example_functions)
def test_cell_examples_normals(cell_example, verify_image_cache):
    if cell_example is examples.cells.Empty:
        pytest.skip('nothing to plot')
    if cell_example in [
        examples.cells.BiQuadraticQuadraticWedge,
        examples.cells.QuadraticLinearWedge,
        examples.cells.QuadraticWedge,
    ] and pv.vtk_version_info < (9, 4, 0):
        pytest.xfail('point ordering changed in newer VTK')

    # Skip since variance is too high
    verify_image_cache.macos_skip_image_cache = True
    verify_image_cache.windows_skip_image_cache = True

    grid = cell_example()
    if next(grid.cell).dimension == 2:
        # Ensure normals of 2D cells point in z-direction for consistency
        normal = grid.extract_surface(algorithm=None).cell_normals.mean(axis=0)
        assert np.allclose(normal, (0.0, 0.0, 1.0))
    examples.plot_cell(grid, show_normals=True)


@pytest.mark.parametrize('data', ['point', 'cell'])
def test_hide_cells(data):
    grid = examples.load_explicit_structured().resize(bounds=(-1, 1, -1, 1, -1, 1))
    if data == 'cell':
        grid.cell_data['scalars'] = range(grid.n_cells)
        clim_max = grid.n_cells
    else:
        grid.point_data['scalars'] = range(grid.n_points)
        clim_max = grid.n_points

    kwargs = dict(show_edges=True, show_grid=True, clim=[0, clim_max])

    grid.plot(**kwargs)

    grid = grid.hide_cells(range(60, 120))
    grid.plot(**kwargs)

    grid = grid.cast_to_unstructured_grid()
    grid.plot(**kwargs)


def test_hide_cells_no_scalars():
    grid = examples.load_explicit_structured().resize(bounds=(-1, 1, -1, 1, -1, 1))
    grid = grid.hide_cells(range(80, 120))
    grid = grid.cast_to_unstructured_grid()
    # Test plotting still works with ghost cells active
    assert grid.active_scalars_name == _vtk.vtkDataSetAttributes.GhostArrayName()
    grid.plot(color='w', show_edges=True, show_grid=True)
