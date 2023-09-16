"""
This test module tests any functionality that requires plotting.

See the image regression notes in doc/extras/developer_notes.rst

"""
import io
import os
import pathlib
import platform
import re
import time

from PIL import Image
import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import DeprecationError, PyVistaDeprecationWarning
from pyvista.plotting import check_math_text_support
from pyvista.plotting.colors import matplotlib_default_colors
from pyvista.plotting.errors import InvalidCameraError, RenderWindowUnavailable
from pyvista.plotting.opts import InterpolationType, RepresentationType
from pyvista.plotting.plotter import SUPPORTED_FORMATS
from pyvista.plotting.texture import numpy_to_texture
from pyvista.plotting.utilities import algorithms
from pyvista.plotting.utilities.gl_checks import uses_egl

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting

HAS_IMAGEIO = True
try:
    import imageio
except ModuleNotFoundError:
    HAS_IMAGEIO = False


ffmpeg_failed = False
try:
    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as err:
        if HAS_IMAGEIO:
            imageio.plugins.ffmpeg.download()
        else:
            raise err
except:  # noqa: E722
    ffmpeg_failed = True


THIS_PATH = pathlib.Path(__file__).parent.absolute()


def using_mesa():
    """Determine if using mesa."""
    pl = pv.Plotter(notebook=False, off_screen=True)
    pl.show(auto_close=False)
    gpu_info = pl.render_window.ReportCapabilities()
    pl.close()

    regex = re.compile("OpenGL version string:(.+)\n")
    return "Mesa" in regex.findall(gpu_info)[0]


# always set on Windows CI
# These tests fail with mesa opengl on windows
skip_windows = pytest.mark.skipif(os.name == 'nt', reason='Test fails on Windows')
skip_windows_mesa = pytest.mark.skipif(
    using_mesa() and os.name == 'nt', reason='Does not display correctly within OSMesa on Windows'
)
skip_9_1_0 = pytest.mark.needs_vtk_version(9, 1, 0)
skip_9_0_X = pytest.mark.skipif(pv.vtk_version_info < (9, 1), reason="Flaky on 9.0.X")
skip_lesser_9_0_X = pytest.mark.skipif(
    pv.vtk_version_info < (9, 1), reason="Functions not implemented before 9.0.X"
)

CI_WINDOWS = os.environ.get('CI_WINDOWS', 'false').lower() == 'true'

skip_mac = pytest.mark.skipif(
    platform.system() == 'Darwin', reason='MacOS CI fails when downloading examples'
)
skip_mac_flaky = pytest.mark.skipif(
    platform.system() == 'Darwin', reason='This is a flaky test on MacOS'
)
skip_mesa = pytest.mark.skipif(using_mesa(), reason='Does not display correctly within OSMesa')


@pytest.fixture(autouse=True)
def verify_image_cache_wrapper(verify_image_cache):
    return verify_image_cache


@pytest.fixture()
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


def test_import_gltf(verify_image_cache):
    # image cache created with 9.0.20210612.dev0
    verify_image_cache.high_variance_test = True

    filename = os.path.join(THIS_PATH, '..', 'example_files', 'Box.glb')
    pl = pv.Plotter()

    with pytest.raises(FileNotFoundError):
        pl.import_gltf('not a file')

    pl.import_gltf(filename)
    pl.show()


def test_export_gltf(tmpdir, sphere, airplane, hexbeam, verify_image_cache):
    # image cache created with 9.0.20210612.dev0
    verify_image_cache.high_variance_test = True
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.gltf'))

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
    filename = os.path.join(THIS_PATH, '..', 'example_files', 'Box.wrl')
    pl = pv.Plotter()

    with pytest.raises(FileNotFoundError):
        pl.import_vrml('not a file')

    pl.import_vrml(filename)
    pl.show()


def test_export_vrml(tmpdir, sphere, airplane, hexbeam):
    filename = str(tmpdir.mkdir("tmpdir").join("tmp.wrl"))

    pl = pv.Plotter()
    pl.add_mesh(sphere, smooth_shading=True)
    pl.export_vrml(filename)

    pl_import = pv.Plotter()
    pl_import.import_vrml(filename)
    pl_import.show()

    with pytest.raises(RuntimeError, match="This plotter has been closed"):
        pl_import.export_vrml(filename)


@skip_windows
@pytest.mark.skipif(CI_WINDOWS, reason="Windows CI testing segfaults on pbr")
def test_pbr(sphere, verify_image_cache):
    """Test PBR rendering"""
    verify_image_cache.high_variance_test = True

    texture = examples.load_globe_texture()

    pl = pv.Plotter(lighting=None)
    pl.set_environment_texture(texture)
    pl.add_light(pv.Light())
    pl.add_mesh(
        sphere, color='w', pbr=True, metallic=0.8, roughness=0.2, smooth_shading=True, diffuse=1
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


@skip_windows
@skip_mac
def test_set_environment_texture_cubemap(sphere, verify_image_cache):
    """Test set_environment_texture with a cubemap."""
    verify_image_cache.high_variance_test = True

    texture = examples.download_sky_box_cube_map()

    pl = pv.Plotter(lighting=None)
    pl.set_environment_texture(texture)
    pl.add_mesh(sphere, color='w', pbr=True, metallic=0.8, roughness=0.2)

    # VTK flipped the Z axis for the cubemap between 9.1 and 9.2
    verify_image_cache.skip = pv.vtk_version_info > (9, 1)
    pl.show()


@skip_windows
@skip_mac
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

    plotter = pv.Plotter()
    plotter.add_points(sphere.points)
    plotter.add_points(sphere.points + 1)
    plotter.show()


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


@pytest.mark.parametrize('anti_aliasing', [True, "msaa", False])
def test_plot(sphere, tmpdir, verify_image_cache, anti_aliasing):
    verify_image_cache.high_variance_test = True
    verify_image_cache.macos_skip_image_cache = True
    verify_image_cache.windows_skip_image_cache = True

    tmp_dir = tmpdir.mkdir("tmpdir2")
    filename = str(tmp_dir.join('tmp.png'))
    scalars = np.arange(sphere.n_points)
    cpos, img = pv.plot(
        sphere,
        full_screen=True,
        text='this is a sphere',
        show_bounds=True,
        color='r',
        style='wireframe',
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
    assert os.path.isfile(filename)

    verify_image_cache.skip = True
    filename = pathlib.Path(str(tmp_dir.join('tmp2.png')))
    pv.plot(sphere, screenshot=filename)

    # Ensure it added a PNG extension by default
    assert filename.with_suffix(".png").is_file()

    # test invalid extension
    with pytest.raises(ValueError):
        filename = pathlib.Path(str(tmp_dir.join('tmp3.foo')))
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
    with pytest.raises(ValueError):
        pl.add_volume(mesh, scalars='scalars')
    pl.close()


def test_plot_return_cpos(sphere):
    cpos = sphere.plot(return_cpos=True)
    assert isinstance(cpos, pv.CameraPosition)
    assert sphere.plot(return_cpos=False) is None


def test_add_title(verify_image_cache):
    verify_image_cache.high_variance_test = True
    plotter = pv.Plotter()
    plotter.add_title('Plot Title')
    plotter.show()


def test_plot_invalid_style(sphere):
    with pytest.raises(ValueError):
        pv.plot(sphere, style='not a style')


@pytest.mark.parametrize(
    'interaction, kwargs',
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
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    getattr(plotter, f'enable_{interaction}_style')(**kwargs)
    assert plotter.iren._style_class is not None
    plotter.close()


def test_lighting_disable_3_lights():
    with pytest.raises(DeprecationError):
        pv.Plotter().disable_3_lights()


def test_lighting_enable_three_lights(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)

    plotter.enable_3_lights()
    lights = plotter.renderer.lights
    assert len(lights) == 3
    for light in lights:
        assert light.on

    assert lights[0].intensity == 1.0
    assert lights[1].intensity == 0.6
    assert lights[2].intensity == 0.5

    plotter.show()


def test_lighting_add_manual_light(sphere):
    plotter = pv.Plotter(lighting=None)
    plotter.add_mesh(sphere)

    # test manual light addition
    light = pv.Light()
    plotter.add_light(light)
    assert plotter.renderer.lights == [light]

    # failing case
    with pytest.raises(TypeError):
        plotter.add_light('invalid')

    plotter.show()


def test_lighting_remove_manual_light(sphere):
    plotter = pv.Plotter(lighting=None)
    plotter.add_mesh(sphere)
    plotter.add_light(pv.Light())

    # test light removal
    plotter.remove_all_lights()
    assert not plotter.renderer.lights

    plotter.show()


def test_lighting_subplots(sphere):
    plotter = pv.Plotter(shape='1|1')
    plotter.add_mesh(sphere)
    renderers = plotter.renderers

    light = pv.Light()
    plotter.remove_all_lights()
    for renderer in renderers:
        assert not renderer.lights

    plotter.subplot(0)
    plotter.add_light(light, only_active=True)
    assert renderers[0].lights and not renderers[1].lights
    plotter.add_light(light, only_active=False)
    assert renderers[0].lights and renderers[1].lights
    plotter.subplot(1)
    plotter.add_mesh(pv.Sphere())
    plotter.remove_all_lights(only_active=True)
    assert renderers[0].lights and not renderers[1].lights

    plotter.show()


def test_lighting_init_light_kit(sphere):
    plotter = pv.Plotter(lighting='light kit')
    plotter.add_mesh(sphere)
    lights = plotter.renderer.lights
    assert len(lights) == 5
    assert lights[0].light_type == pv.Light.HEADLIGHT
    for light in lights[1:]:
        assert light.light_type == light.CAMERA_LIGHT
    plotter.show()


def test_lighting_init_three_lights(sphere):
    plotter = pv.Plotter(lighting='three lights')
    plotter.add_mesh(sphere)
    lights = plotter.renderer.lights
    assert len(lights) == 3
    for light in lights:
        assert light.light_type == light.CAMERA_LIGHT
    plotter.show()


def test_lighting_init_none(sphere):
    # ``None`` already tested above
    plotter = pv.Plotter(lighting='none')
    plotter.add_mesh(sphere)
    lights = plotter.renderer.lights
    assert not lights
    plotter.show()


def test_lighting_init_invalid():
    with pytest.raises(ValueError):
        pv.Plotter(lighting='invalid')


def test_plotter_shape_invalid():
    # wrong size
    with pytest.raises(ValueError):
        pv.Plotter(shape=(1,))
    # not positive
    with pytest.raises(ValueError):
        pv.Plotter(shape=(1, 0))
    with pytest.raises(ValueError):
        pv.Plotter(shape=(0, 2))
    # not a sequence
    with pytest.raises(TypeError):
        pv.Plotter(shape={1, 2})


def test_plot_bounds_axes_with_no_data():
    plotter = pv.Plotter()
    plotter.show_bounds()
    plotter.show()


def test_plot_show_grid(sphere):
    plotter = pv.Plotter()

    with pytest.raises(ValueError, match='Value of location'):
        plotter.show_grid(location='foo')
    with pytest.raises(TypeError, match='location must be a string'):
        plotter.show_grid(location=10)
    with pytest.raises(ValueError, match='Value of tick'):
        plotter.show_grid(ticks='foo')
    with pytest.raises(TypeError, match='must be a string'):
        plotter.show_grid(ticks=10)

    plotter.show_grid()  # Add mesh after to make sure bounds update
    plotter.add_mesh(sphere)
    plotter.show()


@skip_mesa
def test_plot_show_grid_with_mesh(hexbeam, plane, verify_image_cache):
    """Show the grid bounds for a specific mesh."""
    verify_image_cache.macos_skip_image_cache = True

    hexbeam.clear_data()
    plotter = pv.Plotter()
    plotter.add_mesh(hexbeam, style='wireframe')
    plotter.add_mesh(plane)
    plotter.show_grid(mesh=plane, show_zlabels=False, show_zaxis=False)
    plotter.show()


cpos_param = [
    [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)],
    [-1, 2, -5],  # trigger view vector
    [1.0, 2.0, 3.0],
]
cpos_param.extend(pv.plotting.renderer.Renderer.CAMERA_STR_ATTR_MAP)


@pytest.mark.parametrize('cpos', cpos_param)
def test_set_camera_position(cpos, sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.camera_position = cpos
    plotter.show()


@pytest.mark.parametrize(
    'cpos', [[(2.0, 5.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)], [-1, 2], [(1, 2, 3)], 'notvalid']
)
def test_set_camera_position_invalid(cpos, sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    with pytest.raises(InvalidCameraError):
        plotter.camera_position = cpos


def test_parallel_projection():
    plotter = pv.Plotter()
    assert isinstance(plotter.parallel_projection, bool)


@pytest.mark.parametrize("state", [True, False])
def test_set_parallel_projection(state):
    plotter = pv.Plotter()
    plotter.parallel_projection = state
    assert plotter.parallel_projection == state


def test_parallel_scale():
    plotter = pv.Plotter()
    assert isinstance(plotter.parallel_scale, float)


@pytest.mark.parametrize("value", [1, 1.5, 0.3, 10])
def test_set_parallel_scale(value):
    plotter = pv.Plotter()
    plotter.parallel_scale = value
    assert plotter.parallel_scale == value


def test_set_parallel_scale_invalid():
    plotter = pv.Plotter()
    with pytest.raises(TypeError):
        plotter.parallel_scale = "invalid"


def test_plot_no_active_scalars(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    with pytest.raises(ValueError):
        plotter.update_scalars(np.arange(5))
    with pytest.raises(ValueError):
        plotter.update_scalars(np.arange(sphere.n_faces))


def test_plot_show_bounds(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.show_bounds(
        show_xaxis=False,
        show_yaxis=False,
        show_zaxis=False,
        show_xlabels=False,
        show_ylabels=False,
        show_zlabels=False,
        use_2d=True,
    )
    plotter.show()


def test_plot_label_fmt(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.show_bounds(xtitle='My X', fmt=r'%.3f')
    plotter.show()


@pytest.mark.parametrize('grid', [True, 'both', 'front', 'back'])
@pytest.mark.parametrize('location', ['all', 'origin', 'outer', 'front', 'back'])
def test_plot_show_bounds_params(grid, location):
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Cone())
    plotter.show_bounds(grid=grid, ticks='inside', location=location)
    plotter.show_bounds(grid=grid, ticks='outside', location=location)
    plotter.show_bounds(grid=grid, ticks='both', location=location)
    plotter.show()


def test_plot_silhouette_non_poly(hexbeam):
    plotter = pv.Plotter()
    plotter.add_mesh(hexbeam, show_scalar_bar=False)
    plotter.add_silhouette(hexbeam, line_width=10)
    plotter.show()


def test_plot_no_silhouette(tri_cylinder):
    # silhouette=False
    plotter = pv.Plotter()
    plotter.add_mesh(tri_cylinder)
    assert len(list(plotter.renderer.GetActors())) == 1  # only cylinder
    plotter.show()


def test_plot_silhouette(tri_cylinder):
    # silhouette=True and default properties
    plotter = pv.Plotter()
    plotter.add_mesh(tri_cylinder, silhouette=True)
    actors = list(plotter.renderer.GetActors())
    assert len(actors) == 2  # cylinder + silhouette
    actor = actors[0]  # get silhouette actor
    props = actor.GetProperty()
    assert props.GetColor() == pv.global_theme.silhouette.color
    assert props.GetOpacity() == pv.global_theme.silhouette.opacity
    assert props.GetLineWidth() == pv.global_theme.silhouette.line_width
    plotter.show()


def test_plot_silhouette_method(tri_cylinder):
    plotter = pv.Plotter()

    plotter.add_mesh(tri_cylinder)
    assert len(plotter.renderer.actors) == 1  # cylinder

    actor = plotter.add_silhouette(tri_cylinder)
    assert isinstance(actor, pv.Actor)
    assert len(plotter.renderer.actors) == 2  # cylinder + silhouette

    props = actor.prop
    assert props.color == pv.global_theme.silhouette.color
    assert props.opacity == pv.global_theme.silhouette.opacity
    assert props.line_width == pv.global_theme.silhouette.line_width
    plotter.show()

    params = {'line_width': 5, 'opacity': 0.5}
    with pytest.warns(PyVistaDeprecationWarning, match='`params` is deprecated'):
        actor = plotter.add_silhouette(tri_cylinder, params=params)
    assert actor.prop.line_width == params['line_width']
    assert actor.prop.opacity == params['opacity']


def test_plot_silhouette_options(tri_cylinder):
    # cover other properties
    plotter = pv.Plotter()
    plotter.add_mesh(tri_cylinder, silhouette=dict(decimate=0.5, feature_angle=20))
    plotter.show()


def test_plotter_scale(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.set_scale(10, 10, 15)
    assert plotter.scale == [10, 10, 15]
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.set_scale(5.0)
    plotter.set_scale(yscale=6.0)
    plotter.set_scale(zscale=9.0)
    assert plotter.scale == [5.0, 6.0, 9.0]
    plotter.show()

    plotter = pv.Plotter()
    plotter.scale = [1.0, 4.0, 2.0]
    assert plotter.scale == [1.0, 4.0, 2.0]
    plotter.add_mesh(sphere)
    plotter.show()


def test_plot_add_scalar_bar(sphere, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    sphere['test_scalars'] = sphere.points[:, 2]
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_scalar_bar(
        label_font_size=10, title_font_size=20, title='woa', interactive=True, vertical=True
    )
    plotter.add_scalar_bar(background_color='white', n_colors=256)
    assert isinstance(plotter.scalar_bar, vtk.vtkScalarBarActor)
    plotter.show()


def test_plot_invalid_add_scalar_bar():
    with pytest.raises(AttributeError):
        plotter = pv.Plotter()
        plotter.add_scalar_bar()


def test_plot_list():
    sphere_a = pv.Sphere(0.5)
    sphere_b = pv.Sphere(1.0)
    sphere_c = pv.Sphere(2.0)
    pv.plot([sphere_a, sphere_b, sphere_c], style='wireframe')


def test_add_lines_invalid():
    plotter = pv.Plotter()
    with pytest.raises(TypeError):
        plotter.add_lines(range(10))


@pytest.mark.skipif(not HAS_IMAGEIO, reason="Requires imageio")
def test_open_gif_invalid():
    plotter = pv.Plotter()
    with pytest.raises(ValueError):
        plotter.open_gif('file.abs')


@pytest.mark.skipif(ffmpeg_failed, reason="Requires imageio-ffmpeg")
@pytest.mark.skipif(not HAS_IMAGEIO, reason="Requires imageio")
def test_make_movie(sphere, tmpdir, verify_image_cache):
    verify_image_cache.skip = True

    # Make temporary file
    filename = str(tmpdir.join('tmp.mp4'))

    movie_sphere = sphere.copy()
    plotter = pv.Plotter()
    plotter.open_movie(filename)
    actor = plotter.add_axes_at_origin()
    plotter.remove_actor(actor, reset_camera=False, render=True)
    plotter.add_mesh(movie_sphere, scalars=np.random.random(movie_sphere.n_faces))
    plotter.show(auto_close=False, window_size=[304, 304])
    plotter.set_focus([0, 0, 0])
    for _ in range(3):  # limiting number of frames to write for speed
        plotter.write_frame()
        random_points = np.random.random(movie_sphere.points.shape)
        movie_sphere.points[:] = random_points * 0.01 + movie_sphere.points * 0.99
        movie_sphere.points[:] -= movie_sphere.points.mean(0)
        scalars = np.random.random(movie_sphere.n_faces)
        plotter.update_scalars(scalars)

    # remove file
    plotter.close()
    os.remove(filename)  # verifies that the plotter has closed


def test_add_legend(sphere):
    plotter = pv.Plotter()
    with pytest.raises(TypeError):
        plotter.add_mesh(sphere, label=2)
    plotter.add_mesh(sphere)
    with pytest.raises(ValueError):
        plotter.add_legend()
    legend_labels = [['sphere', 'r']]
    plotter.add_legend(labels=legend_labels, border=True, bcolor=None, size=[0.1, 0.1])
    plotter.show()


def test_legend_circle_face(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    legend_labels = [['sphere', 'r']]
    face = "circle"
    _ = plotter.add_legend(
        labels=legend_labels, border=True, bcolor=None, size=[0.1, 0.1], face=face
    )
    plotter.show()


def test_legend_rectangle_face(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    legend_labels = [['sphere', 'r']]
    face = "rectangle"
    _ = plotter.add_legend(
        labels=legend_labels, border=True, bcolor=None, size=[0.1, 0.1], face=face
    )
    plotter.show()


def test_legend_invalid_face(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    legend_labels = [['sphere', 'r']]
    face = "invalid_face"
    with pytest.raises(ValueError):
        plotter.add_legend(
            labels=legend_labels, border=True, bcolor=None, size=[0.1, 0.1], face=face
        )


def test_legend_subplots(sphere, cube):
    plotter = pv.Plotter(shape=(1, 2))
    plotter.add_mesh(sphere, 'blue', smooth_shading=True, label='Sphere')
    assert plotter.legend is None
    plotter.add_legend(bcolor='w')
    assert isinstance(plotter.legend, vtk.vtkActor2D)

    plotter.subplot(0, 1)
    plotter.add_mesh(cube, 'r', label='Cube')
    assert plotter.legend is None
    plotter.add_legend(bcolor='w')
    assert isinstance(plotter.legend, vtk.vtkActor2D)

    plotter.show()


def test_add_axes_twice():
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_axes(interactive=True)
    plotter.show()


def test_hide_axes():
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.hide_axes()
    plotter.show()


def test_add_axes_parameters():
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.7,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16),
        viewport=(0, 0, 0.4, 0.4),
    )
    plotter.show()


def test_show_axes_all():
    plotter = pv.Plotter()
    plotter.show_axes_all()
    plotter.show()


def test_hide_axes_all():
    plotter = pv.Plotter()
    plotter.hide_axes_all()
    plotter.show()


def test_isometric_view_interactive(sphere):
    plotter_iso = pv.Plotter()
    plotter_iso.add_mesh(sphere)
    plotter_iso.camera_position = 'xy'
    cpos_old = plotter_iso.camera_position
    plotter_iso.isometric_view_interactive()
    assert not plotter_iso.camera_position == cpos_old


def test_add_point_labels():
    plotter = pv.Plotter()

    # cannot use random points with image regression
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    n = points.shape[0]

    with pytest.raises(ValueError):
        plotter.add_point_labels(points, range(n - 1))

    plotter.add_point_labels(points, range(n), show_points=True, point_color='r', point_size=10)
    plotter.add_point_labels(
        points - 1, range(n), show_points=False, point_color='r', point_size=10
    )
    plotter.show()


@pytest.mark.parametrize('always_visible', [False, True])
def test_add_point_labels_always_visible(always_visible):
    # just make sure it runs without exception
    plotter = pv.Plotter()
    plotter.add_point_labels(
        np.array([[0.0, 0.0, 0.0]]), ['hello world'], always_visible=always_visible
    )
    plotter.show()


def test_set_background():
    plotter = pv.Plotter()
    plotter.set_background('k')
    plotter.background_color = "yellow"
    plotter.set_background([0, 0, 0], top=[1, 1, 1])  # Gradient
    plotter.background_color
    plotter.show()

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('orange')
    for renderer in plotter.renderers:
        assert renderer.GetBackground() == pv.Color('orange')
    plotter.show()

    plotter = pv.Plotter(shape=(1, 2))
    plotter.subplot(0, 1)
    plotter.set_background('orange', all_renderers=False)
    assert plotter.renderers[0].GetBackground() != pv.Color('orange')
    assert plotter.renderers[1].GetBackground() == pv.Color('orange')
    plotter.show()


def test_add_points():
    plotter = pv.Plotter()

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    n = points.shape[0]

    plotter.add_points(
        points, scalars=np.arange(n), cmap=None, flip_scalars=True, show_scalar_bar=False
    )
    plotter.show()


def test_key_press_event():
    plotter = pv.Plotter()
    plotter.key_press_event(None, None)
    plotter.close()


def test_enable_picking_gc():
    plotter = pv.Plotter()
    sphere = pv.Sphere()
    plotter.add_mesh(sphere)
    plotter.enable_cell_picking()
    plotter.close()


def test_left_button_down():
    plotter = pv.Plotter()
    if (
        hasattr(plotter.ren_win, 'GetOffScreenFramebuffer')
        and not plotter.ren_win.GetOffScreenFramebuffer().GetFBOIndex()
    ):
        # This only fails for VTK<9.2.3
        with pytest.raises(ValueError):
            plotter.left_button_down(None, None)
    else:
        plotter.left_button_down(None, None)
    plotter.close()


def test_show_axes():
    plotter = pv.Plotter()
    plotter.show_axes()
    plotter.show()


def test_plot_cell_data(sphere, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    plotter = pv.Plotter()
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(
        sphere,
        interpolate_before_map=True,
        scalars=scalars,
        n_colors=10,
        rng=sphere.n_faces,
        show_scalar_bar=False,
    )
    plotter.show()


def test_plot_clim(sphere):
    plotter = pv.Plotter()
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(
        sphere,
        interpolate_before_map=True,
        scalars=scalars,
        n_colors=5,
        clim=10,
        show_scalar_bar=False,
    )
    assert plotter.mapper.GetScalarRange() == (-10, 10)
    plotter.show()


def test_invalid_n_arrays(sphere):
    with pytest.raises(ValueError):
        plotter = pv.Plotter()
        plotter.add_mesh(sphere, scalars=np.arange(10))
        plotter.show()


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
    plotter = pv.Plotter()
    plotter.add_arrows(cent=center, direction=vector, mag=2.2, color="#009900")
    plotter.show()


def test_axes():
    plotter = pv.Plotter()
    plotter.add_orientation_widget(pv.Cube(), color='b')
    plotter.add_mesh(pv.Cube())
    plotter.show()


def test_box_axes():
    plotter = pv.Plotter()
    plotter.add_axes(box=True)
    plotter.add_mesh(pv.Sphere())
    plotter.show()


def test_box_axes_color_box():
    plotter = pv.Plotter()
    plotter.add_axes(box=True, box_args={'color_box': True})
    plotter.add_mesh(pv.Sphere())
    plotter.show()


def test_screenshot(tmpdir):
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Sphere())
    img = plotter.screenshot(transparent_background=False)
    assert np.any(img)
    img_again = plotter.screenshot()
    assert np.any(img_again)
    filename = str(tmpdir.mkdir("tmpdir").join('export-graphic.svg'))
    plotter.save_graphic(filename)

    # test window and array size
    w, h = 20, 10
    img = plotter.screenshot(transparent_background=False, window_size=(w, h))
    assert img.shape == (h, w, 3)
    img = plotter.screenshot(transparent_background=True, window_size=(w, h))
    assert img.shape == (h, w, 4)

    # check error before first render
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(pv.Sphere())
    with pytest.raises(RuntimeError):
        plotter.screenshot()


def test_screenshot_scaled():
    # FYI: no regression tests because show() is not called
    factor = 2
    plotter = pv.Plotter(image_scale=factor)
    width, height = plotter.window_size
    plotter.add_mesh(pv.Sphere())
    img = plotter.screenshot(transparent_background=False)
    assert np.any(img)
    assert img.shape == (width * factor, height * factor, 3)
    img_again = plotter.screenshot(scale=3)
    assert np.any(img_again)
    assert img_again.shape == (width * 3, height * 3, 3)
    assert plotter.image_scale == factor, 'image_scale leaked from screenshot context'
    img = plotter.image
    assert img.shape == (width * factor, height * factor, 3)

    w, h = 20, 10
    factor = 4
    plotter.image_scale = factor
    img = plotter.screenshot(transparent_background=False, window_size=(w, h))
    assert img.shape == (h * factor, w * factor, 3)

    img = plotter.screenshot(transparent_background=True, window_size=(w, h), scale=5)
    assert img.shape == (h * 5, w * 5, 4)
    assert plotter.image_scale == factor, 'image_scale leaked from screenshot context'

    with pytest.raises(ValueError):
        plotter.image_scale = 0.5

    plotter.close()


def test_screenshot_altered_window_size(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)

    plotter.window_size = (800, 800)
    a = plotter.screenshot()
    assert a.shape == (800, 800, 3)
    # plotter.show(auto_close=False)  # for image regression test

    plotter.window_size = (1000, 1000)
    b = plotter.screenshot()
    assert b.shape == (1000, 1000, 3)
    # plotter.show(auto_close=False)  # for image regression test

    d = plotter.screenshot(window_size=(600, 600))
    assert d.shape == (600, 600, 3)
    # plotter.show()  # for image regression test

    plotter.close()


def test_screenshot_bytes():
    # Test screenshot to bytes object
    buffer = io.BytesIO()
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere())
    plotter.show(screenshot=buffer)
    buffer.seek(0)
    im = Image.open(buffer)
    assert im.format == 'PNG'


def test_screenshot_rendering(tmpdir):
    plotter = pv.Plotter()
    plotter.add_mesh(examples.load_airplane(), smooth_shading=True)
    filename = str(tmpdir.mkdir("tmpdir").join('export-graphic.svg'))
    assert plotter._first_time
    plotter.save_graphic(filename)
    assert not plotter._first_time


@pytest.mark.parametrize('ext', SUPPORTED_FORMATS)
def test_save_screenshot(tmpdir, sphere, ext):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp' + ext))
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.screenshot(filename)
    assert os.path.isfile(filename)
    assert pathlib.Path(filename).stat().st_size


def test_scalars_by_name(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    plotter = pv.Plotter()
    data = examples.load_uniform()
    plotter.add_mesh(data, scalars='Spatial Cell Data')
    plotter.show()


def test_multi_block_plot(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    multi = pv.MultiBlock()
    multi.append(examples.load_rectilinear())
    uni = examples.load_uniform()
    arr = np.random.rand(uni.n_cells)
    uni.cell_data.set_array(arr, 'Random Data')
    multi.append(uni)
    # And now add a data set without the desired array and a NULL component
    multi.append(examples.load_airplane())

    # missing data should still plot
    multi.plot(scalars='Random Data')

    multi.plot(multi_colors=True)


def test_clear(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.clear()
    plotter.show()


def test_plot_texture():
    """Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = examples.load_globe_texture()
    plotter = pv.Plotter()
    plotter.add_mesh(globe, texture=texture)
    plotter.show()


@pytest.mark.skipif(not HAS_IMAGEIO, reason="Requires imageio")
def test_plot_numpy_texture():
    """Text adding a np.ndarray texture to a plot"""
    globe = examples.load_globe()
    texture_np = np.asarray(imageio.imread(examples.mapfile))
    plotter = pv.Plotter()
    plotter.add_mesh(globe, texture=texture_np)


@pytest.mark.skipif(not HAS_IMAGEIO, reason="Requires imageio")
def test_read_texture_from_numpy():
    """Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = numpy_to_texture(imageio.imread(examples.mapfile))
    plotter = pv.Plotter()
    plotter.add_mesh(globe, texture=texture)
    plotter.show()


def test_plot_rgb():
    """Test adding a texture to a plot"""
    cube = pv.Cube()
    cube.clear_data()
    x_face_color = (255, 0, 0)
    y_face_color = (0, 255, 0)
    z_face_color = (0, 0, 255)
    face_colors = np.array(
        [
            x_face_color,
            x_face_color,
            y_face_color,
            y_face_color,
            z_face_color,
            z_face_color,
        ],
        dtype=np.uint8,
    )
    cube.cell_data['face_colors'] = face_colors
    plotter = pv.Plotter()
    plotter.add_mesh(cube, scalars='face_colors', rgb=True)
    plotter.show()


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
    pl.add_mesh(multicomp_poly, scalars="vector_values_points", show_scalar_bar=False)
    pl.camera_position = 'xy'
    pl.camera.tight()
    pl.subplot(0, 1)
    pl.add_mesh(multicomp_poly.copy(), scalars="vector_values_points", component=0)
    pl.subplot(1, 0)
    pl.add_mesh(multicomp_poly.copy(), scalars="vector_values_points", component=1)
    pl.subplot(1, 1)
    pl.add_mesh(multicomp_poly.copy(), scalars="vector_values_points", component=2)
    pl.link_views()
    pl.show()


def test_vector_plotting_doesnt_modify_data(multicomp_poly):
    """Test that the operations in plotting do not modify the data in the mesh."""

    copy_vector_values_points = multicomp_poly["vector_values_points"].copy()
    copy_vector_values_cells = multicomp_poly["vector_values_cells"].copy()

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


def test_vector_array_fail_with_incorrect_component(multicomp_poly):
    """Test failure modes of component argument."""
    p = pv.Plotter()

    # Non-Integer
    with pytest.raises(TypeError):
        p.add_mesh(multicomp_poly, scalars='vector_values_points', component=1.5)
        p.show()

    # Component doesn't exist
    p = pv.Plotter()
    with pytest.raises(ValueError):
        p.add_mesh(multicomp_poly, scalars='vector_values_points', component=3)
        p.show()

    # Component doesn't exist
    p = pv.Plotter()
    with pytest.raises(ValueError):
        p.add_mesh(multicomp_poly, scalars='vector_values_points', component=-1)
        p.show()


def test_camera(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.view_xy()
    plotter.view_xz()
    plotter.view_yz()
    plotter.add_mesh(examples.load_uniform(), reset_camera=True, culling=True)
    plotter.view_xy(True)
    plotter.view_xz(True)
    plotter.view_yz(True)
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.camera.zoom(5)
    plotter.camera.up = 0, 0, 10
    plotter.show()


def test_multi_renderers():
    plotter = pv.Plotter(shape=(2, 2))

    plotter.subplot(0, 0)
    plotter.add_text('Render Window 0', font_size=30)
    sphere = pv.Sphere()
    plotter.add_mesh(sphere, scalars=sphere.points[:, 2], show_scalar_bar=False)
    plotter.add_scalar_bar('Z', vertical=True)

    plotter.subplot(0, 1)
    plotter.add_text('Render Window 1', font_size=30)
    plotter.add_mesh(pv.Cube(), show_edges=True)

    plotter.subplot(1, 0)
    plotter.add_text('Render Window 2', font_size=30)
    plotter.add_mesh(pv.Arrow(), color='y', show_edges=True)

    plotter.subplot(1, 1)
    plotter.add_text('Render Window 3', position=(0.0, 0.0), font_size=30, viewport=True)
    plotter.add_mesh(pv.Cone(), color='g', show_edges=True, culling=True)
    plotter.add_bounding_box(render_lines_as_tubes=True, line_width=5)
    plotter.show_bounds(all_edges=True)

    plotter.update_bounds_axes()
    plotter.show()


def test_multi_renderers_subplot_ind_2x1():
    # Test subplot indices (2 rows by 1 column)
    plotter = pv.Plotter(shape=(2, 1))
    # First row
    plotter.subplot(0, 0)
    plotter.add_mesh(pv.Sphere())
    # Second row
    plotter.subplot(1, 0)
    plotter.add_mesh(pv.Cube())
    plotter.show()


def test_multi_renderers_subplot_ind_1x2():
    # Test subplot indices (1 row by 2 columns)
    plotter = pv.Plotter(shape=(1, 2))
    # First column
    plotter.subplot(0, 0)
    plotter.add_mesh(pv.Sphere())
    # Second column
    plotter.subplot(0, 1)
    plotter.add_mesh(pv.Cube())
    plotter.show()


def test_multi_renderers_bad_indices():
    with pytest.raises(IndexError):
        # Test bad indices
        plotter = pv.Plotter(shape=(1, 2))
        plotter.subplot(0, 0)
        plotter.add_mesh(pv.Sphere())
        plotter.subplot(1, 0)
        plotter.add_mesh(pv.Cube())
        plotter.show()


def test_multi_renderers_subplot_ind_3x1():
    # Test subplot 3 on left, 1 on right
    plotter = pv.Plotter(shape='3|1')
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pv.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pv.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pv.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pv.Cone())
    plotter.show()


def test_multi_renderers_subplot_ind_3x1_splitting_pos():
    # Test subplot 3 on top, 1 on bottom
    plotter = pv.Plotter(shape='3/1', splitting_position=0.5)
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pv.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pv.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pv.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pv.Cone())
    plotter.show()


def test_multi_renderers_subplot_ind_1x3():
    # Test subplot 3 on bottom, 1 on top
    plotter = pv.Plotter(shape='1|3')
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pv.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pv.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pv.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pv.Cone())
    plotter.show()


def test_subplot_groups():
    plotter = pv.Plotter(shape=(3, 3), groups=[(1, [1, 2]), (np.s_[:], 0)])
    plotter.subplot(0, 0)
    plotter.add_mesh(pv.Sphere())
    plotter.subplot(0, 1)
    plotter.add_mesh(pv.Cube())
    plotter.subplot(0, 2)
    plotter.add_mesh(pv.Arrow())
    plotter.subplot(1, 1)
    plotter.add_mesh(pv.Cylinder())
    plotter.subplot(2, 1)
    plotter.add_mesh(pv.Cone())
    plotter.subplot(2, 2)
    plotter.add_mesh(pv.Box())
    plotter.show()


def test_subplot_groups_fail():
    # Test group overlap
    with pytest.raises(ValueError):
        # Partial overlap
        pv.Plotter(shape=(3, 3), groups=[([1, 2], [0, 1]), ([0, 1], [1, 2])])
    with pytest.raises(ValueError):
        # Full overlap (inner)
        pv.Plotter(shape=(4, 4), groups=[(np.s_[:], np.s_[:]), ([1, 2], [1, 2])])
    with pytest.raises(ValueError):
        # Full overlap (outer)
        pv.Plotter(shape=(4, 4), groups=[(1, [1, 2]), ([0, 3], np.s_[:])])


@skip_windows
def test_link_views(sphere):
    plotter = pv.Plotter(shape=(1, 4))
    plotter.subplot(0, 0)
    plotter.add_mesh(sphere, smooth_shading=False, show_edges=False)
    plotter.subplot(0, 1)
    plotter.add_mesh(sphere, smooth_shading=True, show_edges=False)
    plotter.subplot(0, 2)
    plotter.add_mesh(sphere, smooth_shading=False, show_edges=True)
    plotter.subplot(0, 3)
    plotter.add_mesh(sphere, smooth_shading=True, show_edges=True)
    with pytest.raises(TypeError):
        plotter.link_views(views='foo')
    plotter.link_views([0, 1])
    plotter.link_views()
    with pytest.raises(TypeError):
        plotter.unlink_views(views='foo')
    plotter.unlink_views([0, 1])
    plotter.unlink_views(2)
    plotter.unlink_views()
    plotter.show()


@skip_windows
def test_link_views_camera_set(sphere, verify_image_cache):
    p = pv.Plotter(shape=(1, 2))
    p.add_mesh(pv.Cone())
    assert not p.renderer.camera_set
    p.subplot(0, 1)
    p.add_mesh(pv.Cube())
    assert not p.renderer.camera_set
    p.link_views()  # make sure the default isometric view is used
    for renderer in p.renderers:
        assert not renderer.camera_set
    p.show()

    p = pv.Plotter(shape=(1, 2))
    p.add_mesh(pv.Cone())
    p.subplot(0, 1)
    p.add_mesh(pv.Cube())
    p.link_views()
    p.unlink_views()
    for renderer in p.renderers:
        assert not renderer.camera_set
    p.show()

    wavelet = pv.Wavelet().clip('x')
    p = pv.Plotter(shape=(1, 2))
    p.add_mesh(wavelet, color='red')
    p.subplot(0, 1)
    p.add_mesh(wavelet, color='red')
    p.link_views()
    p.camera_position = [(55.0, 16, 31), (-5.0, 0.0, 0.0), (-0.22, 0.97, -0.09)]
    p.show()


def test_orthographic_slicer(uniform):
    uniform.set_active_scalars('Spatial Cell Data')
    slices = uniform.slice_orthogonal()

    # Orthographic Slicer
    p = pv.Plotter(shape=(2, 2))

    p.subplot(1, 1)
    p.add_mesh(slices, clim=uniform.get_data_range())
    p.add_axes()
    p.enable()

    p.subplot(0, 0)
    p.add_mesh(slices['XY'])
    p.view_xy()
    p.disable()

    p.subplot(0, 1)
    p.add_mesh(slices['XZ'])
    p.view_xz(negative=True)
    p.disable()

    p.subplot(1, 0)
    p.add_mesh(slices['YZ'])
    p.view_yz()
    p.disable()

    p.show()


def test_remove_actor(uniform):
    plotter = pv.Plotter()
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.show()


def test_image_properties():
    mesh = examples.load_uniform()
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.show(auto_close=False)  # DO NOT close plotter
    # Get RGB image
    _ = p.image
    # Get the depth image
    _ = p.get_image_depth()
    p.close()
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.show()  # close plotter
    # Get RGB image
    _ = p.image
    # verify property matches method while testing both available
    assert np.allclose(p.image_depth, p.get_image_depth(), equal_nan=True)
    p.close()

    # gh-920
    rr = np.array([[-0.5, -0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, -0.5, 1]])
    tris = np.array([[3, 0, 2, 1], [3, 2, 0, 3]])
    mesh = pv.PolyData(rr, tris)
    p = pv.Plotter()
    p.add_mesh(mesh, color=True)
    p.renderer.camera_position = (0.0, 0.0, 1.0)
    p.renderer.ResetCamera()
    p.enable_parallel_projection()
    assert p.renderer.camera_set
    p.show(interactive=False, auto_close=False)
    img = p.get_image_depth(fill_value=0.0)
    rng = np.ptp(img)
    assert 0.3 < rng < 0.4, rng  # 0.3313504 in testing
    p.close()


def test_volume_rendering_from_helper(uniform, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    uniform.plot(volume=True, opacity='linear')


@skip_windows_mesa  # due to opacity
def test_volume_rendering_from_plotter(uniform):
    plotter = pv.Plotter()
    plotter.add_volume(uniform, opacity='sigmoid', cmap='jet', n_colors=15)
    plotter.show()


@skip_windows_mesa  # due to opacity
@skip_9_0_X
def test_volume_rendering_rectilinear(uniform):
    grid = uniform.cast_to_rectilinear_grid()

    plotter = pv.Plotter()
    plotter.add_volume(grid, opacity='sigmoid', cmap='jet', n_colors=15)
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_volume(grid)
    plotter.show()

    plotter = pv.Plotter()
    with pytest.raises(TypeError):
        plotter.add_volume(grid, mapper='fixed_point')
    plotter.close()


@skip_windows
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
        )
    )
    data['a'].rename_array('Spatial Point Data', 'a')
    data['b'].rename_array('Spatial Point Data', 'b')
    data['c'].rename_array('Spatial Point Data', 'c')
    data['d'].rename_array('Spatial Point Data', 'd')
    data.plot(volume=True, multi_colors=True)


def test_array_volume_rendering(uniform, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    arr = uniform["Spatial Point Data"].reshape(uniform.dimensions)
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


def test_plot_depth_peeling():
    mesh = examples.load_airplane()
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.enable_depth_peeling()
    p.disable_depth_peeling()
    p.show()


@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_plot(airplane):
    airplane.plot(eye_dome_lighting=True)


@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_plotter(airplane):
    p = pv.Plotter()
    p.add_mesh(airplane)
    p.enable_eye_dome_lighting()
    p.show()


@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_enable_disable(airplane):
    p = pv.Plotter()
    p.add_mesh(airplane)
    p.enable_eye_dome_lighting()
    p.disable_eye_dome_lighting()
    p.show()


@skip_windows
def test_opacity_by_array_direct(plane, verify_image_cache):
    # VTK regression 9.0.1 --> 9.1.0
    verify_image_cache.high_variance_test = True

    # test with opacity parm as an array, both cell and point sized
    plane_shift = plane.translate((0, 0, 1), inplace=False)
    pl = pv.Plotter()
    pl.add_mesh(plane, color='b', opacity=np.linspace(0, 1, plane.n_points), show_edges=True)
    pl.add_mesh(plane_shift, color='r', opacity=np.linspace(0, 1, plane.n_cells), show_edges=True)
    pl.show()


def test_opacity_by_array(uniform):
    # Test with opacity array
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['opac'] = opac
    p = pv.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity='opac')
    p.show()


def test_opacity_by_array_uncertainty(uniform):
    # Test with uncertainty array (transparency)
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['unc'] = opac
    p = pv.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity='unc', use_transparency=True)
    p.show()


def test_opacity_by_array_user_transform(uniform, verify_image_cache):
    verify_image_cache.high_variance_test = True

    uniform['Spatial Point Data'] /= uniform['Spatial Point Data'].max()

    # Test with user defined transfer function
    opacities = [0, 0.2, 0.9, 0.2, 0.1]
    p = pv.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity=opacities)
    p.show()


def test_opacity_mismatched_fail(uniform):
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['unc'] = opac

    # Test using mismatched arrays
    p = pv.Plotter()
    with pytest.raises(ValueError):
        # cell scalars vs point opacity
        p.add_mesh(uniform, scalars='Spatial Cell Data', opacity='unc')


def test_opacity_by_array_preference():
    tetra = pv.Tetrahedron()  # 4 points, 4 cells
    opacities = np.linspace(0.2, 0.8, tetra.n_points)
    tetra.clear_data()
    tetra.point_data['scalars'] = tetra.cell_data['scalars'] = np.arange(tetra.n_points)
    tetra.point_data['opac'] = tetra.cell_data['opac'] = opacities

    # test opacity by key
    p = pv.Plotter()
    p.add_mesh(tetra.copy(), opacity='opac', preference='cell')
    p.add_mesh(tetra.translate((2, 0, 0), inplace=False), opacity='opac', preference='point')
    p.close()

    # test opacity by array
    p = pv.Plotter()
    p.add_mesh(tetra.copy(), opacity=opacities, preference='cell')
    p.add_mesh(tetra.translate((2, 0, 0), inplace=False), opacity=opacities, preference='point')
    p.show()


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
    with pytest.raises(ValueError):
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


def test_closing_and_mem_cleanup(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    verify_image_cache.skip = True
    n = 5
    for _ in range(n):
        for _ in range(n):
            p = pv.Plotter()
            for k in range(n):
                p.add_mesh(pv.Sphere(radius=k))
            p.show()
        pv.close_all()


def test_above_below_scalar_range_annotations():
    p = pv.Plotter()
    p.add_mesh(
        examples.load_uniform(),
        clim=[100, 500],
        cmap='viridis',
        below_color='blue',
        above_color='red',
    )
    p.show()


def test_user_annotations_scalar_bar_mesh(uniform):
    p = pv.Plotter()
    p.add_mesh(uniform, annotations={100.0: 'yum'})
    p.show()


def test_fixed_font_size_annotation_text_scaling_off():
    p = pv.Plotter()
    sargs = {'title_font_size': 12, 'label_font_size': 10}
    p.add_mesh(
        examples.load_uniform(),
        clim=[100, 500],
        cmap='viridis',
        below_color='blue',
        above_color='red',
        annotations={300.0: 'yum'},
        scalar_bar_args=sargs,
    )
    p.show()


def test_user_annotations_scalar_bar_volume(uniform, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    p = pv.Plotter()
    p.add_volume(uniform, scalars='Spatial Point Data', annotations={100.0: 'yum'})
    p.show()


def test_scalar_bar_args_unmodified_add_mesh(sphere):
    sargs = {"vertical": True}
    sargs_copy = sargs.copy()

    p = pv.Plotter()
    p.add_mesh(sphere, scalar_bar_args=sargs)

    assert sargs == sargs_copy


def test_scalar_bar_args_unmodified_add_volume(uniform):
    sargs = {"vertical": True}
    sargs_copy = sargs.copy()

    p = pv.Plotter()
    p.add_volume(uniform, scalar_bar_args=sargs)

    assert sargs == sargs_copy


def test_plot_string_array(verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    mesh = examples.load_uniform()
    labels = np.empty(mesh.n_cells, dtype='<U10')
    labels[:] = 'High'
    labels[mesh['Spatial Cell Data'] < 300] = 'Medium'
    labels[mesh['Spatial Cell Data'] < 100] = 'Low'
    mesh['labels'] = labels
    p = pv.Plotter()
    p.add_mesh(mesh, scalars='labels')
    p.show()


def test_fail_plot_table():
    """Make sure tables cannot be plotted"""
    table = pv.Table(np.random.rand(50, 3))
    with pytest.raises(TypeError):
        pv.plot(table)
    with pytest.raises(TypeError):
        plotter = pv.Plotter()
        plotter.add_mesh(table)


def test_bad_keyword_arguments():
    """Make sure bad keyword arguments raise an error"""
    mesh = examples.load_uniform()
    with pytest.raises(TypeError):
        pv.plot(mesh, foo=5)
    with pytest.raises(TypeError):
        pv.plot(mesh, scalar=mesh.active_scalars_name)
    with pytest.raises(TypeError):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalar=mesh.active_scalars_name)
        plotter.show()
    with pytest.raises(TypeError):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, foo="bad")
        plotter.show()


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
    color = "tan"

    p = pv.Plotter()
    for i in range(N):
        for j in range(N):
            center = (i, j, 0)
            mesh = pv.Sphere(center=center)
            p.add_mesh(mesh, color=color)
    n_made_it = len(p.renderer._actors)
    p.show()
    assert n_made_it == N**2

    # release attached scalars
    mesh.ReleaseData()
    del mesh


def test_add_background_image_global(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_background_image(examples.mapfile, as_global=True)
    plotter.show()


def test_add_background_image_not_global(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_background_image(examples.mapfile, as_global=False)
    plotter.show()


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


def test_add_remove_floor(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_floor(color='b', line_width=2, lighting=True)
    pl.add_bounding_box()  # needed for update_bounds_axes
    assert len(pl.renderer._floors) == 1
    pl.add_mesh(pv.Sphere(1.0))
    pl.update_bounds_axes()
    assert len(pl.renderer._floors) == 1
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_floor(color='b', line_width=2, lighting=True)
    pl.remove_floors()
    assert not pl.renderer._floors
    pl.show()


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


def test_index_vs_loc():
    # first: 2d grid
    pl = pv.Plotter(shape=(2, 3))
    # index_to_loc valid cases
    vals = [0, 2, 4]
    expecteds = [(0, 0), (0, 2), (1, 1)]
    for val, expected in zip(vals, expecteds):
        assert tuple(pl.renderers.index_to_loc(val)) == expected
    # loc_to_index valid cases
    vals = [(0, 0), (0, 2), (1, 1)]
    expecteds = [0, 2, 4]
    for val, expected in zip(vals, expecteds):
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
    with pytest.raises(ValueError):
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
    p = pv.Plotter()
    p.show(interactive_update=True)
    assert isinstance(p.iren.interactor, vtk.vtkRenderWindowInteractor)
    p.close()

    p = pv.Plotter()
    with pytest.warns(UserWarning):
        p.show(auto_close=True, interactive_update=True)


def test_where_is():
    plotter = pv.Plotter(shape=(2, 2))
    plotter.subplot(0, 0)
    plotter.add_mesh(pv.Box(), name='box')
    plotter.subplot(0, 1)
    plotter.add_mesh(pv.Sphere(), name='sphere')
    plotter.subplot(1, 0)
    plotter.add_mesh(pv.Box(), name='box')
    plotter.subplot(1, 1)
    plotter.add_mesh(pv.Cone(), name='cone')
    places = plotter.where_is('box')
    assert isinstance(places, list)
    for loc in places:
        assert isinstance(loc, tuple)


def test_log_scale():
    mesh = examples.load_uniform()
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, log_scale=True)
    plotter.show()


def test_set_focus():
    plane = pv.Plane()
    p = pv.Plotter()
    p.add_mesh(plane, color="tan", show_edges=True)
    p.set_focus((-0.5, -0.5, 0))  # focus on corner of the plane
    p.show()


def test_set_viewup(verify_image_cache):
    verify_image_cache.high_variance_test = True

    plane = pv.Plane()
    plane_higher = pv.Plane(center=(0, 0, 1), i_size=0.5, j_size=0.5)
    p = pv.Plotter()
    p.add_mesh(plane, color="tan", show_edges=False)
    p.add_mesh(plane_higher, color="red", show_edges=False)
    p.set_viewup((1.0, 1.0, 1.0))
    p.show()


def test_plot_shadows():
    plotter = pv.Plotter(lighting=None)

    # add several planes
    for plane_y in [2, 5, 10]:
        screen = pv.Plane(center=(0, plane_y, 0), direction=(0, -1, 0), i_size=5, j_size=5)
        plotter.add_mesh(screen, color='white')

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

    plotter.add_light(light)
    plotter.view_vector((1, -2, 2))

    # verify disabling shadows when not enabled does nothing
    plotter.disable_shadows()

    plotter.enable_shadows()

    # verify shadows can safely be enabled twice
    plotter.enable_shadows()

    plotter.show()


def test_plot_shadows_enable_disable():
    """Test shadows are added and removed properly"""
    plotter = pv.Plotter(lighting=None)

    # add several planes
    for plane_y in [2, 5, 10]:
        screen = pv.Plane(center=(0, plane_y, 0), direction=(0, -1, 0), i_size=5, j_size=5)
        plotter.add_mesh(screen, color='white')

    light = pv.Light(
        position=(0, 0, 0), focal_point=(0, 1, 0), color='cyan', intensity=15, cone_angle=15
    )
    light.positional = True
    light.attenuation_values = (2, 0, 0)
    light.show_actor()

    plotter.add_light(light)
    plotter.view_vector((1, -2, 2))

    # add and remove and verify that the light passes through all via
    # image cache
    plotter.enable_shadows()
    plotter.disable_shadows()

    plotter.show()


def test_plot_lighting_change_positional_true_false(sphere):
    light = pv.Light(positional=True, show_actor=True)

    plotter = pv.Plotter(lighting=None)
    plotter.add_light(light)
    light.positional = False
    plotter.add_mesh(sphere)
    plotter.show()


def test_plot_lighting_change_positional_false_true(sphere):
    light = pv.Light(positional=False, show_actor=True)

    plotter = pv.Plotter(lighting=None)

    plotter.add_light(light)
    light.positional = True
    plotter.add_mesh(sphere)
    plotter.show()


def test_plotter_image():
    plotter = pv.Plotter()
    wsz = tuple(plotter.window_size)
    plotter.show()
    assert plotter.image.shape[:2] == wsz


def test_scalar_cell_priorities():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1.5, 1, 0], [0, 0, 1]])
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2], [3, 0, 1, 3], [3, 1, 2, 3]])
    mesh = pv.PolyData(vertices, faces)
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]

    mesh.cell_data['colors'] = colors
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='colors', rgb=True, preference='cell')
    plotter.show()


def test_collision_plot(verify_image_cache):
    """Verify rgba arrays automatically plot"""
    verify_image_cache.windows_skip_image_cache = True
    sphere0 = pv.Sphere()
    sphere1 = pv.Sphere(radius=0.6, center=(-1, 0, 0))
    col, n_contacts = sphere0.collision(sphere1, generate_scalars=True)

    plotter = pv.Plotter()
    plotter.add_mesh(col)
    plotter.camera_position = 'zy'
    plotter.show()


@skip_mac
@pytest.mark.needs_vtk_version(9, 2, 0)
def test_chart_plot():
    """Basic test to verify chart plots correctly"""
    # Chart 1 (bottom left)
    chart_bl = pv.Chart2D(size=(0.4, 0.4), loc=(0.05, 0.05))
    chart_bl.background_color = "tab:purple"
    chart_bl.x_range = [np.pi / 2, 3 * np.pi / 2]
    chart_bl.y_axis.margin = 20
    chart_bl.y_axis.tick_locations = [-1, 0, 1]
    chart_bl.y_axis.tick_labels = ["Small", "Medium", "Large"]
    chart_bl.y_axis.tick_size += 10
    chart_bl.y_axis.tick_labels_offset += 12
    chart_bl.y_axis.pen.width = 10
    chart_bl.grid = True
    x = np.linspace(0, 2 * np.pi, 50)
    y = np.cos(x) * (-1) ** np.arange(len(x))
    hidden_plot = chart_bl.line(x, y, color="k", width=40)
    hidden_plot.visible = False  # Make sure plot visibility works
    chart_bl.bar(x, y, color="#33ff33")

    # Chart 2 (bottom right)
    chart_br = pv.Chart2D(size=(0.4, 0.4), loc=(0.55, 0.05))
    chart_br.background_texture = examples.load_globe_texture()
    chart_br.active_border_color = "r"
    chart_br.border_width = 5
    chart_br.border_style = "-."
    chart_br.hide_axes()
    x = np.linspace(0, 1, 50)
    y = np.sin(6.5 * x - 1)
    chart_br.scatter(x, y, color="y", size=15, style="o", label="Invisible label")
    chart_br.legend_visible = False  # Check legend visibility

    # Chart 3 (top left)
    chart_tl = pv.Chart2D(size=(0.4, 0.4), loc=(0.05, 0.55))
    chart_tl.active_background_color = (0.8, 0.8, 0.2)
    chart_tl.title = "Exponential growth"
    chart_tl.x_label = "X axis"
    chart_tl.y_label = "Y axis"
    chart_tl.y_axis.log_scale = True
    x = np.arange(6)
    y = 10**x
    chart_tl.line(x, y, color="tab:green", width=5, style="--")
    removed_plot = chart_tl.area(x, y, color="k")
    chart_tl.remove_plot(removed_plot)  # Make sure plot removal works

    # Chart 4 (top right)
    chart_tr = pv.Chart2D(size=(0.4, 0.4), loc=(0.55, 0.55))
    x = [0, 1, 2, 3, 4]
    ys = [[0, 1, 2, 3, 4], [1, 0, 1, 0, 1], [6, 4, 5, 3, 2]]
    chart_tr.stack(x, ys, colors="citrus", labels=["Segment 1", "Segment 2", "Segment 3"])
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


@skip_9_1_0
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
    data = [rng.beta(alpha, beta, N) for alpha, beta in zip(alphas, betas)]
    labels = [
        f"$\\alpha={alpha:.1f}\\,;\\,\\beta={beta:.1f}$" for alpha, beta in zip(alphas, betas)
    ]
    ax.violinplot(data)
    ax.set_xticks(np.arange(1, 1 + len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title("$B(\\alpha, \\beta)$")

    # Next, embed the figure into a pv plotting window
    pl = pv.Plotter()
    pl.background_color = "w"
    chart = pv.ChartMPL(fig)
    pl.add_chart(chart)
    pl.show()


def test_add_remove_background(sphere):
    plotter = pv.Plotter(shape=(1, 2))
    plotter.add_mesh(sphere, color='w')
    plotter.add_background_image(examples.mapfile, as_global=False)
    plotter.subplot(0, 1)
    plotter.add_mesh(sphere, color='w')
    plotter.add_background_image(examples.mapfile, as_global=False)
    plotter.remove_background_image()
    plotter.show()


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


@skip_mac_flaky
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
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_cursor()
    plotter.show()


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


@skip_9_1_0
def test_pointset_plot(pointset):
    pointset.plot()

    pl = pv.Plotter()
    pl.add_mesh(pointset, scalars=range(pointset.n_points), show_scalar_bar=False)
    pl.show()


@skip_9_1_0
def test_pointset_plot_as_points(pointset):
    pl = pv.Plotter()
    pl.add_points(pointset, scalars=range(pointset.n_points), show_scalar_bar=False)
    pl.show()


@skip_9_1_0
def test_pointset_plot_vtk():
    pointset = vtk.vtkPointSet()
    points = pv.vtk_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    pointset.SetPoints(points)

    pl = pv.Plotter()
    pl.add_mesh(pointset, color='red', point_size=25)
    pl.show()


@skip_9_1_0
def test_pointset_plot_as_points_vtk():
    pointset = vtk.vtkPointSet()
    points = pv.vtk_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    pointset.SetPoints(points)

    pl = pv.Plotter()
    pl.add_points(pointset, color='red', point_size=25)
    pl.show()


@pytest.mark.skipif(not HAS_IMAGEIO, reason="Requires imageio")
def test_write_gif(sphere, tmpdir):
    basename = 'write_gif.gif'
    path = str(tmpdir.join(basename))
    pl = pv.Plotter()
    pl.open_gif(path)
    pl.add_mesh(sphere)
    pl.write_frame()
    pl.close()

    # assert file exists and is not empty
    assert os.path.isfile(path)
    assert os.path.getsize(path)


def test_ruler():
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Sphere())
    plotter.add_ruler([-0.6, -0.6, 0], [0.6, -0.6, 0], font_size_factor=1.2)
    plotter.view_xy()
    plotter.show()


def test_legend_scale(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_legend_scale(color='red')
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_legend_scale(color='red', xy_label_mode=True)
    plotter.view_xy()
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_legend_scale(
        xy_label_mode=True,
        bottom_axis_visibility=False,
        left_axis_visibility=False,
        right_axis_visibility=False,
        top_axis_visibility=False,
    )
    plotter.view_xy()
    plotter.show()


def test_plot_complex_value(plane, verify_image_cache):
    """Test plotting complex data."""
    verify_image_cache.windows_skip_image_cache = True
    data = np.arange(plane.n_points, dtype=np.complex128)
    data += np.linspace(0, 1, plane.n_points) * -1j
    with pytest.warns(np.ComplexWarning):
        plane.plot(scalars=data)

    pl = pv.Plotter()
    with pytest.warns(np.ComplexWarning):
        pl.add_mesh(plane, scalars=data, show_scalar_bar=True)
    pl.show()


def test_screenshot_notebook(tmpdir):
    tmp_dir = tmpdir.mkdir("tmpdir2")
    filename = str(tmp_dir.join('tmp.png'))

    pl = pv.Plotter(notebook=True)
    pl.theme.jupyter_backend = 'static'
    pl.add_mesh(pv.Cone())
    pl.show(screenshot=filename)

    assert os.path.isfile(filename)


def test_culling_frontface(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere, culling='frontface')
    pl.show()


def test_add_text():
    plotter = pv.Plotter()
    plotter.add_text("Upper Left", position='upper_left', font_size=25, color='blue')
    plotter.add_text("Center", position=(0.5, 0.5), viewport=True, orientation=-90)
    plotter.show()


@pytest.mark.skipif(
    not check_math_text_support(),
    reason='VTK and Matplotlib version incompatibility. For VTK<=9.2.2, MathText requires matplotlib<3.6',
)
def test_add_text_latex():
    """Test LaTeX symbols.

    For VTK<=9.2.2, this requires matplotlib<3.6
    """
    plotter = pv.Plotter()
    plotter.add_text(r'$\rho$', position='upper_left', font_size=150, color='blue')
    plotter.show()


def test_add_text_font_file():
    plotter = pv.Plotter()
    font_file = os.path.join(os.path.dirname(__file__), "fonts/Mplus2-Regular.ttf")
    plotter.add_text("左上", position='upper_left', font_size=25, color='blue', font_file=font_file)
    plotter.add_text("中央", position=(0.5, 0.5), viewport=True, orientation=-90, font_file=font_file)
    plotter.show()


def test_plot_categories_int(sphere):
    sphere['data'] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars='data', categories=5, lighting=False)
    pl.show()


def test_plot_categories_true(sphere):
    sphere['data'] = np.linspace(0, 5, sphere.n_points, dtype=int)
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars='data', categories=True, lighting=False)
    pl.show()


@skip_windows
@skip_9_0_X
def test_depth_of_field():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere(), show_edges=True)
    pl.enable_depth_of_field()
    pl.show()


@skip_9_0_X
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
def test_ssao_pass():
    ugrid = pv.ImageData(dimensions=(2, 2, 2)).to_tetrahedra(5).explode()
    pl = pv.Plotter()
    pl.add_mesh(ugrid)

    pl.enable_ssao()
    pl.show(auto_close=False)

    # ensure this fails when ssao disabled
    pl.disable_ssao()
    with pytest.raises(RuntimeError):
        pl.show()


@skip_mesa
def test_ssao_pass_from_helper():
    ugrid = pv.ImageData(dimensions=(2, 2, 2)).to_tetrahedra(5).explode()

    ugrid.plot(ssao=True)


@skip_windows
def test_many_multi_pass():
    pl = pv.Plotter(lighting=None)
    pl.add_mesh(pv.Sphere(), show_edges=True)
    pl.add_light(pv.Light(position=(0, 0, 10)))
    pl.enable_anti_aliasing('ssaa')
    pl.enable_depth_of_field()
    pl.add_blurring()
    pl.enable_shadows()
    pl.enable_eye_dome_lighting()


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


def test_plot_composite_raise(sphere, multiblock_poly):
    pl = pv.Plotter()
    with pytest.raises(TypeError, match='Must be a composite dataset'):
        pl.add_composite(sphere)
    with pytest.raises(TypeError, match='must be a string for'):
        pl.add_composite(multiblock_poly, scalars=range(10))
    with pytest.warns(PyVistaDeprecationWarning, match='categories'):
        with pytest.raises(TypeError, match='must be an int'):
            pl.add_composite(multiblock_poly, categories='abc')


def test_plot_composite_categories(multiblock_poly):
    pl = pv.Plotter()
    with pytest.warns(PyVistaDeprecationWarning, match='categories'):
        pl.add_composite(multiblock_poly, scalars='data_b', categories=5)
    pl.show()


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


@skip_windows  # because of opacity
def test_plot_composite_poly_scalars_opacity(multiblock_poly, verify_image_cache):
    pl = pv.Plotter()

    actor, mapper = pl.add_composite(
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

    # 9.0.3 has a bug where VTK changes the edge visibility on blocks that are
    # also opaque. Don't verify the image of that version.
    verify_image_cache.skip = pv.vtk_version_info == (9, 0, 3)
    pl.show()


def test_plot_composite_poly_scalars_cell(multiblock_poly, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True
    pl = pv.Plotter()

    actor, mapper = pl.add_composite(
        multiblock_poly,
        scalars='cell_data',
    )
    mapper.block_attr[1].color = 'blue'

    pl.camera_position = 'xy'
    pl.show()


def test_plot_composite_poly_no_scalars(multiblock_poly):
    pl = pv.Plotter()

    actor, mapper = pl.add_composite(
        multiblock_poly,
        color='red',
        lighting=False,
    )

    # Note: set the camera position before making the blocks invisible to be
    # consistent between 9.0.3 and 9.1+
    #
    # 9.0.3 still considers invisible blocks when determining camera bounds, so
    # there will be some empty space where the invisible block is for 9.0.3,
    # while 9.1.0 ignores invisible blocks when computing camera bounds.
    pl.camera_position = 'xy'
    mapper.block_attr[2].color = 'blue'
    mapper.block_attr[3].visible = False

    pl.show()


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
    with pytest.warns(np.ComplexWarning, match='Casting complex'):
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


def test_plot_composite_bool(multiblock_poly, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    # add in bool data
    for i, block in enumerate(multiblock_poly):
        block['scalars'] = np.zeros(block.n_points, dtype=bool)
        block['scalars'][::2] = 1

    pl = pv.Plotter()
    pl.add_composite(multiblock_poly, scalars='scalars')
    pl.show()


def test_export_obj(tmpdir, sphere):
    filename = str(tmpdir.mkdir("tmpdir").join("tmp.obj"))

    pl = pv.Plotter()
    pl.add_mesh(sphere, smooth_shading=True)

    with pytest.raises(ValueError, match='end with ".obj"'):
        pl.export_obj('badfilename')

    pl.export_obj(filename)

    # Check that the object file has been written
    assert os.path.exists(filename)

    # Check that when we close the plotter, the adequate error is raised
    pl.close()
    with pytest.raises(RuntimeError, match='This plotter must still have a render window open.'):
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


def test_bool_scalars(sphere):
    sphere['scalars'] = np.zeros(sphere.n_points, dtype=bool)
    sphere['scalars'][::2] = 1
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.show()


@skip_windows  # because of pbr
@skip_9_1_0  # pbr required
def test_property_pbr(verify_image_cache):
    verify_image_cache.macos_skip_image_cache = True
    prop = pv.Property(interpolation='pbr', metallic=1.0)

    # VTK flipped the Z axis for the cubemap between 9.1 and 9.2
    verify_image_cache.skip = pv.vtk_version_info < (9, 2)
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
    pl.add_mesh(colorful_tetrahedron, scalars="colors", rgb=True, preference="cell")
    pl.camera.tight(view=view, negative=negative)
    pl.add_axes()
    pl.show()


def test_tight_multiple_objects():
    pl = pv.Plotter()
    pl.add_mesh(
        pv.Cone(center=(0.0, -2.0, 0.0), direction=(0.0, -1.0, 0.0), height=1.0, radius=1.0)
    )
    pl.add_mesh(pv.Sphere(center=(0.0, 0.0, 0.0)))
    pl.camera.tight()
    pl.add_axes()
    pl.show()


def test_backface_params():
    mesh = pv.ParametricCatalanMinimal()

    with pytest.raises(TypeError, match="pyvista.Property or a dict"):
        mesh.plot(backface_params="invalid")

    params = dict(color="blue", smooth_shading=True)
    backface_params = dict(color="red", specular=1.0, specular_power=50.0)
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
    assert isinstance(actor, vtk.vtkActor)
    pl.remove_bounds_axes()
    pl.show()


@skip_9_1_0
def test_charts_sin():
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x)
    chart = pv.Chart2D()
    chart.scatter(x, y)
    chart.line(x, y, 'r')
    chart.show()


def test_lookup_table(verify_image_cache):
    lut = pv.LookupTable('viridis')
    lut.n_values = 8
    lut.below_range_color = 'black'
    lut.above_range_color = 'grey'
    lut.nan_color = 'r'
    lut.nan_opacity = 0.5

    # There are minor variations within 9.0.3 that slightly invalidate the
    # image cache.
    verify_image_cache.skip = pv.vtk_version_info == (9, 0, 3)
    lut.plot()


def test_lookup_table_nan_hidden(verify_image_cache):
    lut = pv.LookupTable('viridis')
    lut.n_values = 8
    lut.below_range_color = 'black'
    lut.above_range_color = 'grey'
    lut.nan_opacity = 0

    # There are minor variations within 9.0.3 that slightly invalidate the
    # image cache.
    verify_image_cache.skip = pv.vtk_version_info == (9, 0, 3)
    lut.plot()


def test_lookup_table_above_below_opacity(verify_image_cache):
    lut = pv.LookupTable('viridis')
    lut.n_values = 8
    lut.below_range_color = 'blue'
    lut.below_range_opacity = 0.5
    lut.above_range_color = 'green'
    lut.above_range_opacity = 0.5
    lut.nan_color = 'r'
    lut.nan_opacity = 0.5

    # There are minor variations within 9.0.3 that slightly invalidate the
    # image cache.
    verify_image_cache.skip = pv.vtk_version_info == (9, 0, 3)
    lut.plot()


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
        uniform, nan_opacity=0.5, nan_color='green', scalar_bar_args=dict(nan_annotation=True)
    )
    pl.enable_depth_peeling()
    pl.show()


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
def test_plotter_volume_lookup_table(uniform):
    lut = pv.LookupTable()
    lut.alpha_range = (0, 1)
    pl = pv.Plotter()
    pl.add_volume(uniform, scalars='Spatial Point Data', cmap=lut)
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
    pl.add_mesh(colorful_tetrahedron, scalars="colors", rgb=True, preference="cell")
    getattr(pl, f"view_{direction}")(negative=negative)
    pl.add_axes()
    pl.show()


@skip_windows
def test_plot_points_gaussian(sphere):
    sphere.plot(
        color='r',
        style='points_gaussian',
        render_points_as_spheres=False,
        point_size=20,
        opacity=0.5,
    )


@skip_windows
def test_plot_points_gaussian_scalars(sphere):
    sphere.plot(
        scalars=sphere.points[:, 2],
        style='points_gaussian',
        render_points_as_spheres=False,
        point_size=20,
        opacity=0.5,
        show_scalar_bar=False,
    )


@skip_windows
def test_plot_points_gaussian_as_spheres(sphere):
    sphere.plot(
        color='b',
        style='points_gaussian',
        render_points_as_spheres=True,
        point_size=20,
        opacity=0.5,
    )


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


@skip_windows
def test_add_point_scalar_labels_fmt():
    mesh = examples.load_uniform().slice()
    p = pv.Plotter()
    p.add_mesh(mesh, scalars="Spatial Point Data", show_edges=True)
    p.add_point_scalar_labels(mesh, "Spatial Point Data", point_size=20, font_size=36, fmt='%.3f')
    p.camera_position = [(7, 4, 5), (4.4, 7.0, 7.2), (0.8, 0.5, 0.25)]
    p.show()


def test_plot_individual_cell(hexbeam):
    hexbeam.get_cell(0).plot(color='b')


def test_add_point_scalar_labels_list():
    plotter = pv.Plotter()

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    with pytest.raises(TypeError):
        plotter.add_point_scalar_labels(points=False, labels=labels)
    with pytest.raises(TypeError):
        plotter.add_point_scalar_labels(points=points, labels=False)

    plotter.add_point_scalar_labels(points, labels)
    plotter.show()


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

    alg = vtk.vtkGeometryFilter()
    alg.SetInputDataObject(mesh)

    pl = pv.Plotter()
    pl.add_mesh(alg, scalars=name)
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(alg, scalars=name2)
    pl.show()


def test_algorithm_add_points():
    algo = vtk.vtkRTAnalyticSource()

    pl = pv.Plotter()
    pl.add_points(algo)
    pl.show()


@skip_9_1_0
def test_algorithm_add_point_labels():
    algo = pv.ConeSource()
    elev = vtk.vtkElevationFilter()
    elev.SetInputConnection(algo.GetOutputPort())
    elev.SetLowPoint(0, 0, -1)
    elev.SetHighPoint(0, 0, 1)

    pl = pv.Plotter()
    pl.add_point_labels(elev, 'Elevation', always_visible=False)
    pl.show()


@skip_9_1_0
def test_pointset_to_polydata_algorithm(pointset):
    alg = vtk.vtkElevationFilter()
    alg.SetInputDataObject(pointset)

    pl = pv.Plotter()
    pl.add_mesh(alg, scalars='Elevation')
    pl.show()

    assert isinstance(alg.GetOutputDataObject(0), vtk.vtkPointSet)


def test_add_ids_algorithm():
    algo = vtk.vtkCubeSource()

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


def test_plot_window_size_context(sphere):
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
    with pytest.raises(ValueError):
        pl.set_color_cycler('foo')
    with pytest.raises(TypeError):
        pl.set_color_cycler(5)


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


@pytest.mark.needs_vtk_version(9, 2, 0)
@pytest.mark.skipif(CI_WINDOWS, reason="Windows CI testing segfaults on pbr")
def test_plot_cubemap_alone(cubemap):
    """Test plotting directly from the Texture class."""
    cubemap.plot()


@pytest.mark.skipif(
    uses_egl(), reason="Render window will be current with offscreen builds of VTK."
)
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


@skip_lesser_9_0_X
def test_axes_actor_properties():
    axes = pv.Axes()
    axes_actor = axes.axes_actor

    axes_actor.x_axis_shaft_properties.color = (1, 1, 1)
    assert axes_actor.x_axis_shaft_properties.color == (1, 1, 1)
    axes_actor.y_axis_shaft_properties.metallic = 0.2
    assert axes_actor.y_axis_shaft_properties.metallic == 0.2
    axes_actor.z_axis_shaft_properties.roughness = 0.3
    assert axes_actor.z_axis_shaft_properties.roughness == 0.3

    axes_actor.x_axis_tip_properties.anisotropy = 0.4
    assert axes_actor.x_axis_tip_properties.anisotropy == 0.4
    axes_actor.x_axis_tip_properties.anisotropy_rotation = 0.4
    assert axes_actor.x_axis_tip_properties.anisotropy_rotation == 0.4
    axes_actor.y_axis_tip_properties.lighting = False
    assert not axes_actor.y_axis_tip_properties.lighting
    axes_actor.z_axis_tip_properties.interpolation_model = InterpolationType.PHONG
    assert axes_actor.z_axis_tip_properties.interpolation_model == InterpolationType.PHONG

    axes_actor.x_axis_shaft_properties.index_of_refraction = 1.5
    assert axes_actor.x_axis_shaft_properties.index_of_refraction == 1.5
    axes_actor.y_axis_shaft_properties.opacity = 0.6
    assert axes_actor.y_axis_shaft_properties.opacity == 0.6
    axes_actor.z_axis_shaft_properties.shading = False
    assert not axes_actor.z_axis_shaft_properties.shading

    axes_actor.x_axis_tip_properties.representation = RepresentationType.POINTS
    assert axes_actor.x_axis_tip_properties.representation == RepresentationType.POINTS

    axes.axes_actor.shaft_type = pv.AxesActor.ShaftType.CYLINDER
    pl = pv.Plotter()
    pl.add_actor(axes_actor)
    pl.show()


def test_show_bounds_no_labels():
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Cone())
    plotter.show_bounds(
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
    plotter.camera_position = [(1.97, 1.89, 1.66), (0.05, -0.05, 0.00), (-0.36, -0.36, 0.85)]
    plotter.show()


def test_show_bounds_n_labels():
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Cone())
    plotter.show_bounds(
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
    plotter.camera_position = [(1.97, 1.89, 1.66), (0.05, -0.05, 0.00), (-0.36, -0.36, 0.85)]
    plotter.show()
