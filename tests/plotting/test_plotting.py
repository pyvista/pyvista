"""
See the image regression notes in docs/extras/developer_notes.rst
"""
import platform
import warnings
import inspect
import pathlib
import os
from weakref import proxy
from pathlib import Path

import imageio
import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples
from pyvista.plotting import system_supports_plotting
from pyvista.plotting.plotting import SUPPORTED_FORMATS


ffmpeg_failed = False
try:
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        imageio.plugins.ffmpeg.download()
except:
    ffmpeg_failed = True

OFF_SCREEN = True
VTK9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9

sphere = pyvista.Sphere()
sphere_b = pyvista.Sphere(1.0)
sphere_c = pyvista.Sphere(2.0)


# Reset image cache with new images
glb_reset_image_cache = False
IMAGE_CACHE_DIR = os.path.join(Path(__file__).parent.absolute(), 'image_cache')
if not os.path.isdir(IMAGE_CACHE_DIR):
    os.mkdir(IMAGE_CACHE_DIR)

skip_no_plotting = pytest.mark.skipif(not system_supports_plotting(),
                                      reason="Test requires system to support plotting")

# IMAGE warning/error thresholds (assumes using use_vtk)
IMAGE_REGRESSION_ERROR = 500  # major differences
IMAGE_REGRESSION_WARNING = 200  # minor differences


# this must be a session fixture to ensure this runs before any other test
@pytest.fixture(scope="session", autouse=True)
def get_cmd_opt(pytestconfig):
    global glb_reset_image_cache, glb_ignore_image_cache
    glb_reset_image_cache = pytestconfig.getoption('reset_image_cache')
    glb_ignore_image_cache = pytestconfig.getoption('ignore_image_cache')


def verify_cache_image(plotter):
    """Either store or validate an image.

    This is function should only be called within a pytest
    environment.  Pass it to either the ``Plotter.show()`` or the
    ``pyvista.plot()`` functions as the before_close_callback keyword
    arg.

    Assign this only once for each test you'd like to validate the
    previous image of.  This will not work with parameterized tests.

    Example Usage:
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.show(before_close_callback=verify_cache_image)

    """
    global glb_reset_image_cache, glb_ignore_image_cache

    # Image cache is only valid for VTK9 on Linux
    if not VTK9 or platform.system() != 'Linux':
        return

    # since each test must contain a unique name, we can simply
    # use the function test to name the image
    stack = inspect.stack()
    test_name = None
    for item in stack:
        if item.function == 'check_gc':
            return
        if item.function[:5] == 'test_':
            test_name = item.function
            break

    if test_name is None:
        raise RuntimeError('Unable to identify calling test function.  This function '
                           'should only be used within a pytest environment.')

    # cached image name
    image_filename = os.path.join(IMAGE_CACHE_DIR, test_name[5:] + '.png')

    # simply save the last screenshot if it doesn't exist of the cache
    # is being reset.
    if glb_reset_image_cache or not os.path.isfile(image_filename):
        return plotter.screenshot(image_filename)

    if glb_ignore_image_cache:
        return

    # otherwise, compare with the existing cached image
    error = pyvista.compare_images(image_filename, plotter)
    if error > IMAGE_REGRESSION_ERROR:
        raise RuntimeError('Exceeded image regression error of '
                           f'{IMAGE_REGRESSION_ERROR} with an image error of '
                           f'{error}')
    if error > IMAGE_REGRESSION_WARNING:
        warnings.warn('Exceeded image regression warning of '
                      f'{IMAGE_REGRESSION_WARNING} with an image error of '
                      f'{error}')


@skip_no_plotting
def test_plot(tmpdir):
    tmp_dir = tmpdir.mkdir("tmpdir2")
    filename = str(tmp_dir.join('tmp.png'))
    scalars = np.arange(sphere.n_points)
    cpos, img = pyvista.plot(sphere,
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
                             before_close_callback=verify_cache_image)
    assert isinstance(cpos, pyvista.CameraPosition)
    assert isinstance(img, np.ndarray)
    assert os.path.isfile(filename)

    filename = pathlib.Path(str(tmp_dir.join('tmp2.png')))
    cpos = pyvista.plot(sphere, screenshot=filename)

    # Ensure it added a PNG extension by default
    assert filename.with_suffix(".png").is_file()

    # test invalid extension
    with pytest.raises(ValueError):
        filename = pathlib.Path(str(tmp_dir.join('tmp3.foo')))
        pyvista.plot(sphere, screenshot=filename)


@skip_no_plotting
def test_add_title():
    plotter = pyvista.Plotter()
    plotter.add_title('Plot Title')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_invalid_style():
    with pytest.raises(ValueError):
        pyvista.plot(sphere, style='not a style')


@skip_no_plotting
def test_interactor_style():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    interactions = (
        'trackball',
        'trackball_actor',
        'image',
        'joystick',
        'zoom',
        'terrain',
        'rubber_band',
        'rubber_band_2d',
    )
    for interaction in interactions:
        getattr(plotter, f'enable_{interaction}_style')()
        assert plotter._style_class is not None
    plotter.close()


@skip_no_plotting
def test_lighting():
    plotter = pyvista.Plotter()

    # test default disable_3_lights()
    plotter.disable_3_lights()
    lights = plotter.renderer.lights
    assert len(lights) == 5
    for light in lights:
        assert light.on

    plotter.enable_3_lights()
    lights = plotter.renderer.lights
    assert len(lights) == 3
    for light in lights:
        assert light.on

    assert lights[0].intensity == 1.0
    assert lights[1].intensity == 0.6
    assert lights[2].intensity == 0.5

    # test manual light addition
    light = pyvista.Light()
    plotter.add_light(light)
    assert plotter.renderer.lights[-1] is light

    # test light removal
    plotter.remove_all_lights()
    assert not plotter.renderer.lights

    # failing case
    with pytest.raises(TypeError):
        plotter.add_light('invalid')

    plotter.close()


@skip_no_plotting
def test_lighting_subplots():
    plotter = pyvista.Plotter(shape='1|1')
    renderers = plotter.renderers

    light = pyvista.Light()
    plotter.remove_all_lights()
    for renderer in renderers:
        assert not renderer.lights
    plotter.subplot(0)
    plotter.add_light(light, only_active=True)
    assert renderers[0].lights and not renderers[1].lights
    plotter.add_light(light, only_active=False)
    assert renderers[0].lights and renderers[1].lights
    plotter.subplot(1)
    plotter.remove_all_lights(only_active=True)
    assert renderers[0].lights and not renderers[1].lights

    plotter.close()

@skip_no_plotting
def test_lighting_init():
    plotter = pyvista.Plotter(lighting='light kit')
    lights = plotter.renderer.lights
    assert len(lights) == 5
    assert lights[0].light_type == pyvista.Light.HEADLIGHT
    for light in lights[1:]:
        assert light.light_type == light.CAMERA_LIGHT
    plotter.close()

    plotter = pyvista.Plotter(lighting='three lights')
    lights = plotter.renderer.lights
    assert len(lights) == 3
    for light in lights:
        assert light.light_type == light.CAMERA_LIGHT
    plotter.close()

    for no_lighting in 'none', None:
        plotter = pyvista.Plotter(lighting=no_lighting)
        lights = plotter.renderer.lights
        assert not lights
        plotter.close()

    # invalid input
    with pytest.raises(ValueError):
        pyvista.Plotter(lighting='invalid')


@skip_no_plotting
def test_plotter_shape_invalid():
    # wrong size
    with pytest.raises(ValueError):
        pyvista.Plotter(shape=(1,))
    # not positive
    with pytest.raises(ValueError):
        pyvista.Plotter(shape=(1, 0))
    with pytest.raises(ValueError):
        pyvista.Plotter(shape=(0, 2))
    # not a sequence
    with pytest.raises(TypeError):
        pyvista.Plotter(shape={1, 2})


@skip_no_plotting
def test_plot_bounds_axes_with_no_data():
    plotter = pyvista.Plotter()
    plotter.show_bounds()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_show_grid():
    plotter = pyvista.Plotter()
    plotter.show_grid()
    plotter.add_mesh(sphere)
    plotter.show(before_close_callback=verify_cache_image)


cpos_param = [[(2.0, 5.0, 13.0),
              (0.0, 0.0, 0.0),
              (-0.7, -0.5, 0.3)],
             [-1, 2, -5],  # trigger view vector
             [1.0, 2.0, 3.0],
]
cpos_param.extend(pyvista.plotting.Renderer.CAMERA_STR_ATTR_MAP)
@skip_no_plotting
@pytest.mark.parametrize('cpos', cpos_param)
def test_set_camera_position(cpos, sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.camera_position = cpos
    plotter.show()


@skip_no_plotting
@pytest.mark.parametrize('cpos', [[(2.0, 5.0),
                                   (0.0, 0.0, 0.0),
                                   (-0.7, -0.5, 0.3)],
                                  [-1, 2],
                                  [(1,2,3)],
                                  'notvalid'])
def test_set_camera_position_invalid(cpos, sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    with pytest.raises(pyvista.core.errors.InvalidCameraError):
        plotter.camera_position = cpos

@skip_no_plotting
def test_parallel_projection():
    plotter = pyvista.Plotter()
    assert isinstance(plotter.parallel_projection, bool)

@skip_no_plotting
@pytest.mark.parametrize("state", [True, False])
def test_set_parallel_projection(state):
    plotter = pyvista.Plotter()
    plotter.parallel_projection = state
    assert plotter.parallel_projection == state

@skip_no_plotting
def test_parallel_scale():
    plotter = pyvista.Plotter()
    assert isinstance(plotter.parallel_scale, float)

@skip_no_plotting
@pytest.mark.parametrize("value", [1, 1.5, 0.3, 10])
def test_set_parallel_scale(value):
    plotter = pyvista.Plotter()
    plotter.parallel_scale = value
    assert plotter.parallel_scale == value

@skip_no_plotting
def test_set_parallel_scale_invalid():
    plotter = pyvista.Plotter()
    with pytest.raises(TypeError):
        plotter.parallel_scale = "invalid"

@skip_no_plotting
def test_plot_no_active_scalars():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    with pytest.raises(ValueError):
        plotter.update_scalars(np.arange(5))
    with pytest.raises(ValueError):
        plotter.update_scalars(np.arange(sphere.n_faces))


@skip_no_plotting
def test_plot_show_bounds():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.show_bounds(show_xaxis=False,
                        show_yaxis=False,
                        show_zaxis=False,
                        show_xlabels=False,
                        show_ylabels=False,
                        show_zlabels=False,
                        use_2d=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_label_fmt():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.show_bounds(xlabel='My X', fmt=r'%.3f')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.parametrize('grid', [True, 'both', 'front', 'back'])
@pytest.mark.parametrize('location', ['all', 'origin', 'outer', 'front', 'back'])
def test_plot_show_bounds_params(grid, location):
    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.Cube())
    plotter.show_bounds(grid=grid, ticks='inside', location=location)
    plotter.show_bounds(grid=grid, ticks='outside', location=location)
    plotter.show_bounds(grid=grid, ticks='both', location=location)
    plotter.show()


@skip_no_plotting
def test_plotter_scale():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.set_scale(10, 10, 10)
    assert plotter.scale == [10, 10, 10]
    plotter.set_scale(5.0)
    plotter.set_scale(yscale=6.0)
    plotter.set_scale(zscale=9.0)
    assert plotter.scale == [5.0, 6.0, 9.0]
    plotter.scale = [1.0, 4.0, 2.0]
    assert plotter.scale == [1.0, 4.0, 2.0]
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_add_scalar_bar():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_scalar_bar(label_font_size=10, title_font_size=20, title='woa',
                           interactive=True, vertical=True)
    plotter.add_scalar_bar(background_color='white', n_colors=256)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_invalid_add_scalar_bar():
    with pytest.raises(AttributeError):
        plotter = pyvista.Plotter()
        plotter.add_scalar_bar()


@skip_no_plotting
def test_plot_list():
    pyvista.plot([sphere, sphere_b, sphere_c],
                 style='wireframe',
                 before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_lines_invalid():
    plotter = pyvista.Plotter()
    with pytest.raises(TypeError):
        plotter.add_lines(range(10))


@skip_no_plotting
def test_open_gif_invalid():
    plotter = pyvista.Plotter()
    with pytest.raises(ValueError):
        plotter.open_gif('file.abs')


@pytest.mark.skipif(ffmpeg_failed, reason="Requires imageio-ffmpeg")
@skip_no_plotting
def test_make_movie():
    # Make temporary file
    filename = os.path.join(pyvista.USER_DATA_PATH, 'tmp.mp4')

    movie_sphere = sphere.copy()
    plotter = pyvista.Plotter()
    plotter.open_movie(filename)
    actor = plotter.add_axes_at_origin()
    plotter.remove_actor(actor, reset_camera=False, render=True)
    plotter.add_mesh(movie_sphere,
                     scalars=np.random.random(movie_sphere.n_faces))
    plotter.show(auto_close=False, window_size=[304, 304])
    plotter.set_focus([0, 0, 0])
    for i in range(3):  # limiting number of frames to write for speed
        plotter.write_frame()
        random_points = np.random.random(movie_sphere.points.shape)
        movie_sphere.points[:] = random_points*0.01 + movie_sphere.points*0.99
        movie_sphere.points[:] -= movie_sphere.points.mean(0)
        scalars = np.random.random(movie_sphere.n_faces)
        plotter.update_scalars(scalars)

    # remove file
    plotter.close()
    os.remove(filename)  # verifies that the plotter has closed


@skip_no_plotting
def test_add_legend():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    with pytest.raises(ValueError):
        plotter.add_legend()
    legend_labels = [['sphere', 'r']]
    plotter.add_legend(labels=legend_labels, border=True, bcolor=None,
                       size=[0.1, 0.1])
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_axes_twice():
    plotter = pyvista.Plotter()
    plotter.add_axes()
    plotter.add_axes(interactive=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_point_labels():
    plotter = pyvista.Plotter()

    # cannot use random points with image regression
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [0.5, 0.5, 0.5],
                       [1, 1, 1]])
    n = points.shape[0]

    with pytest.raises(ValueError):
        plotter.add_point_labels(points, range(n - 1))

    plotter.add_point_labels(points, range(n), show_points=True, point_color='r', point_size=10)
    plotter.add_point_labels(points - 1, range(n), show_points=False, point_color='r',
                             point_size=10)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.parametrize('always_visible', [False, True])
def test_add_point_labels_always_visible(always_visible):
    # just make sure it runs without exception
    plotter = pyvista.Plotter()
    plotter.add_point_labels(
        np.array([[0, 0, 0]]), ['hello world'], always_visible=always_visible)
    plotter.show()


@skip_no_plotting
def test_set_background():
    plotter = pyvista.Plotter()
    plotter.set_background('k')
    plotter.background_color = "yellow"
    plotter.set_background([0, 0, 0], top=[1, 1, 1])  # Gradient
    plotter.background_color
    plotter.show()

    plotter = pyvista.Plotter(shape=(1, 2))
    plotter.set_background('orange')
    for renderer in plotter.renderers:
        assert renderer.GetBackground() == pyvista.parse_color('orange')
    plotter.show()

    plotter = pyvista.Plotter(shape=(1, 2))
    plotter.subplot(0, 1)
    plotter.set_background('orange', all_renderers=False)
    assert plotter.renderers[0].GetBackground() != pyvista.parse_color('orange')
    assert plotter.renderers[1].GetBackground() == pyvista.parse_color('orange')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_points():
    plotter = pyvista.Plotter()

    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [0.5, 0.5, 0.5],
                       [1, 1, 1]])
    n = points.shape[0]

    plotter.add_points(points, scalars=np.arange(n), cmap=None, flip_scalars=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_key_press_event():
    plotter = pyvista.Plotter()
    plotter.key_press_event(None, None)
    plotter.close()


@skip_no_plotting
def test_enable_picking_gc():
    plotter = pyvista.Plotter()
    sphere = pyvista.Sphere()
    plotter.add_mesh(sphere)
    plotter.enable_cell_picking()
    plotter.close()


@skip_no_plotting
def test_left_button_down():
    plotter = pyvista.Plotter()
    if VTK9:
        with pytest.raises(ValueError):
            plotter.left_button_down(None, None)
    else:
        plotter.left_button_down(None, None)
    plotter.close()


@skip_no_plotting
def test_show_axes():
    plotter = pyvista.Plotter()
    plotter.show_axes()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_update():
    plotter = pyvista.Plotter()
    plotter.update()


@skip_no_plotting
def test_plot_cell_arrays():
    plotter = pyvista.Plotter()
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(sphere, interpolate_before_map=True, scalars=scalars,
                     n_colors=5, rng=10)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_clim():
    plotter = pyvista.Plotter()
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(sphere, interpolate_before_map=True, scalars=scalars,
                     n_colors=5, clim=10)
    plotter.show(before_close_callback=verify_cache_image)
    assert plotter.mapper.GetScalarRange() == (-10, 10)


@skip_no_plotting
def test_invalid_n_arrays():
    with pytest.raises(ValueError):
        plotter = pyvista.Plotter()
        plotter.add_mesh(sphere, scalars=np.arange(10))
        plotter.show()


@skip_no_plotting
def test_plot_arrow():
    cent = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    pyvista.plot_arrows(cent, direction, before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_arrows():
    cent = np.array([[0, 0, 0],
                     [1, 0, 0]])
    direction = np.array([[1, 1, 1],
                          [-1, -1, -1]])
    pyvista.plot_arrows(cent, direction, before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_arrows():
    vector = np.array([1, 0, 0])
    center = np.array([0, 0, 0])
    plotter = pyvista.Plotter()
    plotter.add_arrows(cent=center, direction=vector, mag=2.2, color="#009900")
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_axes():
    plotter = pyvista.Plotter()
    plotter.add_orientation_widget(pyvista.Cube())
    plotter.add_mesh(pyvista.Cube())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_box_axes():
    plotter = pyvista.Plotter()
    plotter.add_axes(box=True, box_args={'color_box': True})
    plotter.add_mesh(pyvista.Sphere())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_screenshot(tmpdir):
    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.Sphere())
    img = plotter.screenshot(transparent_background=False)
    assert np.any(img)
    img_again = plotter.screenshot()
    assert np.any(img_again)
    filename = str(tmpdir.mkdir("tmpdir").join('export-graphic.svg'))
    plotter.save_graphic(filename)

    # checking if plotter closes
    ref = proxy(plotter)
    plotter.close()

    try:
        ref
    except:
        raise RuntimeError('Plotter did not close')


@skip_no_plotting
@pytest.mark.parametrize('ext', SUPPORTED_FORMATS)
def test_save_screenshot(tmpdir, sphere, ext):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp' + ext))
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.screenshot(filename)
    assert os.path.isfile(filename)
    assert Path(filename).stat().st_size


@skip_no_plotting
def test_scalars_by_name():
    plotter = pyvista.Plotter()
    data = examples.load_uniform()
    plotter.add_mesh(data, scalars='Spatial Cell Data')
    plotter.show(before_close_callback=verify_cache_image)


def test_themes():
    old_rcParms = dict(pyvista.rcParams)  # must cache old rcParams
    pyvista.set_plot_theme('paraview')
    pyvista.set_plot_theme('document')
    pyvista.set_plot_theme('night')
    pyvista.set_plot_theme('default')
    for key, value in old_rcParms.items():
        pyvista.rcParams[key] = value


@skip_no_plotting
def test_multi_block_plot():
    multi = pyvista.MultiBlock()
    multi.append(examples.load_rectilinear())
    uni = examples.load_uniform()
    arr = np.random.rand(uni.n_cells)
    uni.cell_arrays.append(arr, 'Random Data')
    multi.append(uni)
    # And now add a data set without the desired array and a NULL component
    multi[3] = examples.load_airplane()
    with pytest.raises(ValueError):
        # The scalars are not available in all datasets so raises ValueError
        multi.plot(scalars='Random Data', multi_colors=True)
    multi.plot(multi_colors=True, before_close_callback=verify_cache_image)


@skip_no_plotting
def test_clear():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.clear()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_texture():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = examples.load_globe_texture()
    plotter = pyvista.Plotter()
    plotter.add_mesh(globe, texture=texture)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_texture_associated():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    plotter = pyvista.Plotter()
    plotter.add_mesh(globe, texture=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_read_texture_from_numpy():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = pyvista.numpy_to_texture(imageio.imread(examples.mapfile))
    plotter = pyvista.Plotter()
    plotter.add_mesh(globe, texture=texture)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_rgb():
    """"Test adding a texture to a plot"""
    cube = pyvista.Cube()
    cube.clear_arrays()
    x_face_color = (255, 0, 0)
    y_face_color = (0, 255, 0)
    z_face_color = (0, 0, 255)
    face_colors = np.array([x_face_color,
                            x_face_color,
                            y_face_color,
                            y_face_color,
                            z_face_color,
                            z_face_color,
                            ], dtype=np.uint8)
    cube.cell_arrays['face_colors'] = face_colors
    plotter = pyvista.Plotter()
    plotter.add_mesh(cube, scalars='face_colors', rgb=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_multi_component_array():
    """"Test adding a texture to a plot"""
    image = pyvista.UniformGrid((3, 3, 3))

    # fixed this to allow for image regression testing
    # image['array'] = np.random.randn(*image.dimensions).ravel(order='f')
    image['array'] = np.array([-0.2080155 ,  0.45258783,  1.03826775,
                                0.38214289,  0.69745718, -2.04209996,
                                0.7361947 , -1.59777205,  0.74254271,
                               -0.27793002, -1.5788904 , -0.71479534,
                               -0.93487136, -0.95082609, -0.64480196,
                               -1.79935993, -0.9481572 , -0.34988819,
                                0.17934252,  0.30425682, -1.31709916,
                                0.02550247, -0.27620985,  0.89869448,
                               -0.13012903,  1.05667384,  1.52085349])

    plotter = pyvista.Plotter()
    plotter.add_mesh(image, scalars='array')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_camera():
    plotter = pyvista.Plotter()
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
    plotter.show(before_close_callback=verify_cache_image)
    plotter.camera_position = None

    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.camera.zoom(5)
    plotter.camera.up([0, 0, 10])
    plotter.show()


@skip_no_plotting
def test_multi_renderers():
    plotter = pyvista.Plotter(shape=(2, 2))

    plotter.subplot(0, 0)
    plotter.add_text('Render Window 0', font_size=30)
    sphere = pyvista.Sphere()
    plotter.add_mesh(sphere, scalars=sphere.points[:, 2])
    plotter.add_scalar_bar('Z', vertical=True)

    plotter.subplot(0, 1)
    plotter.add_text('Render Window 1', font_size=30)
    plotter.add_mesh(pyvista.Cube(), show_edges=True)

    plotter.subplot(1, 0)
    plotter.add_text('Render Window 2', font_size=30)
    plotter.add_mesh(pyvista.Arrow(), color='y', show_edges=True)

    plotter.subplot(1, 1)
    plotter.add_text('Render Window 3', position=(0., 0.),
                     font_size=30, viewport=True)
    plotter.add_mesh(pyvista.Cone(), color='g', show_edges=True,
                     culling=True)
    plotter.add_bounding_box(render_lines_as_tubes=True, line_width=5)
    plotter.show_bounds(all_edges=True)

    plotter.update_bounds_axes()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_multi_renderers_subplot_ind_2x1():

    # Test subplot indices (2 rows by 1 column)
    plotter = pyvista.Plotter(shape=(2, 1))
    # First row
    plotter.subplot(0,0)
    plotter.add_mesh(pyvista.Sphere())
    # Second row
    plotter.subplot(1,0)
    plotter.add_mesh(pyvista.Cube())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_multi_renderers_subplot_ind_1x2():
    # Test subplot indices (1 row by 2 columns)
    plotter = pyvista.Plotter(shape=(1, 2))
    # First column
    plotter.subplot(0, 0)
    plotter.add_mesh(pyvista.Sphere())
    # Second column
    plotter.subplot(0, 1)
    plotter.add_mesh(pyvista.Cube())
    plotter.show(before_close_callback=verify_cache_image)

@skip_no_plotting
def test_multi_renderers_bad_indices():
    with pytest.raises(IndexError):
        # Test bad indices
        plotter = pyvista.Plotter(shape=(1, 2))
        plotter.subplot(0, 0)
        plotter.add_mesh(pyvista.Sphere())
        plotter.subplot(1, 0)
        plotter.add_mesh(pyvista.Cube())
        plotter.show()


@skip_no_plotting
def test_multi_renderers_subplot_ind_3x1():
    # Test subplot 3 on left, 1 on right
    plotter = pyvista.Plotter(shape='3|1')
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pyvista.Cone())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_multi_renderers_subplot_ind_1x3():
    # Test subplot 3 on bottom, 1 on top
    plotter = pyvista.Plotter(shape='1|3')
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pyvista.Cone())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_subplot_groups():
    plotter = pyvista.Plotter(shape=(3,3), groups=[(1,[1,2]),(np.s_[:],0)])
    plotter.subplot(0, 0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(0, 1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(0, 2)
    plotter.add_mesh(pyvista.Arrow())
    plotter.subplot(1, 1)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(2, 1)
    plotter.add_mesh(pyvista.Cone())
    plotter.subplot(2, 2)
    plotter.add_mesh(pyvista.Box())
    plotter.show(before_close_callback=verify_cache_image)


def test_subplot_groups_fail():
    # Test group overlap
    with pytest.raises(AssertionError):
        # Partial overlap
        pyvista.Plotter(shape=(3, 3), groups=[([1, 2], [0, 1]), ([0, 1], [1, 2])])
    with pytest.raises(AssertionError):
        # Full overlap (inner)
        pyvista.Plotter(shape=(4, 4), groups=[(np.s_[:], np.s_[:]), ([1, 2], [1, 2])])
    with pytest.raises(AssertionError):
        # Full overlap (outer)
        pyvista.Plotter(shape=(4, 4), groups=[(1, [1, 2]), ([0, 3], np.s_[:])])

 
@skip_no_plotting
def test_link_views(sphere):
    plotter = pyvista.Plotter(shape=(1, 4))
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
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_orthographic_slicer(uniform):
    uniform.set_active_scalars('Spatial Cell Data')
    slices = uniform.slice_orthogonal()

    # Orthographic Slicer
    p = pyvista.Plotter(shape=(2, 2))

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


@skip_no_plotting
def test_remove_actor(uniform):
    plotter = pyvista.Plotter()
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_image_properties():
    mesh = examples.load_uniform()
    p = pyvista.Plotter()
    p.add_mesh(mesh)
    p.show(auto_close=False)  # DO NOT close plotter
    # Get RGB image
    _ = p.image
    # Get the depth image
    _ = p.get_image_depth()
    p.close()
    p = pyvista.Plotter()
    p.add_mesh(mesh)
    p.store_image = True
    assert p.store_image is True
    p.show() # close plotter
    # Get RGB image
    _ = p.image
    # Get the depth image
    _ = p.get_image_depth()
    p.close()

    # gh-920
    rr = np.array(
        [[-0.5, -0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, -0.5, 1]])
    tris = np.array([[3, 0, 2, 1], [3, 2, 0, 3]])
    mesh = pyvista.PolyData(rr, tris)
    p = pyvista.Plotter()
    p.add_mesh(mesh, color=True)
    p.renderer.camera_position = (0., 0., 1.)
    p.renderer.ResetCamera()
    p.enable_parallel_projection()
    assert p.renderer.camera_set
    p.show(interactive=False, auto_close=False)
    img = p.get_image_depth(fill_value=0.)
    rng = np.ptp(img)
    assert 0.3 < rng < 0.4, rng  # 0.3313504 in testing
    p.close()


@skip_no_plotting
def test_volume_rendering():
    # Really just making sure no errors are thrown
    vol = examples.load_uniform()
    vol.plot(volume=True, opacity='linear')

    plotter = pyvista.Plotter()
    plotter.add_volume(vol, opacity='sigmoid', cmap='jet', n_colors=15)
    plotter.show()

    # Now test MultiBlock rendering
    data = pyvista.MultiBlock(dict(a=examples.load_uniform(),
                                   b=examples.load_uniform(),
                                   c=examples.load_uniform(),
                                   d=examples.load_uniform(),))
    data['a'].rename_array('Spatial Point Data', 'a')
    data['b'].rename_array('Spatial Point Data', 'b')
    data['c'].rename_array('Spatial Point Data', 'c')
    data['d'].rename_array('Spatial Point Data', 'd')
    data.plot(off_screen=OFF_SCREEN, volume=True, multi_colors=True, )

    # Check that NumPy arrays work
    arr = vol["Spatial Point Data"].reshape(vol.dimensions)
    pyvista.plot(arr, volume=True, opacity='linear')


@skip_no_plotting
def test_plot_compar_four():
    # Really just making sure no errors are thrown
    mesh = examples.load_uniform()
    data_a = mesh.contour()
    data_b = mesh.threshold_percent(0.5)
    data_c = mesh.decimate_boundary(0.5)
    data_d = mesh.glyph()
    pyvista.plot_compare_four(data_a, data_b, data_c, data_d,
                              disply_kwargs={'color': 'w'},
                              show_kwargs={'before_close_callback': verify_cache_image})


@skip_no_plotting
def test_plot_depth_peeling():
    mesh = examples.load_airplane()
    p = pyvista.Plotter()
    p.add_mesh(mesh)
    p.enable_depth_peeling()
    p.disable_depth_peeling()
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_plot(airplane):
    airplane.plot(eye_dome_lighting=True, before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_plotter(airplane):
    p = pyvista.Plotter()
    p.add_mesh(airplane)
    p.enable_eye_dome_lighting()
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_enable_disable(airplane):
    p = pyvista.Plotter()
    p.add_mesh(airplane)
    p.enable_eye_dome_lighting()
    p.disable_eye_dome_lighting()
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_opacity_by_array(uniform):
    # Test with opacity array
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['opac'] = opac
    p = pyvista.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity='opac')
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_opacity_by_array_uncertainty(uniform):
    # Test with uncertainty array (transparency)
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['unc'] = opac
    p = pyvista.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity='unc',
               use_transparency=True)
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_opacity_by_array_user_transform(uniform):
    uniform['Spatial Point Data'] /= uniform['Spatial Point Data'].max()

    # Test with user defined transfer function
    opacities = [0, 0.2, 0.9, 0.2, 0.1]
    p = pyvista.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity=opacities)
    p.show()  # note: =verify_cache_image does not work between Xvfb


def test_opactity_mismatched_fail(uniform):
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['unc'] = opac

    # Test using mismatched arrays
    p = pyvista.Plotter()
    with pytest.raises(ValueError):
        p.add_mesh(uniform, scalars='Spatial Cell Data', opacity='unc')


def test_opacity_transfer_functions():
    n = 256
    mapping = pyvista.opacity_transfer_function('linear', n)
    assert len(mapping) == n
    mapping = pyvista.opacity_transfer_function('sigmoid_10', n)
    assert len(mapping) == n
    with pytest.raises(KeyError):
        mapping = pyvista.opacity_transfer_function('foo', n)
    with pytest.raises(RuntimeError):
        mapping = pyvista.opacity_transfer_function(np.linspace(0, 1, 2*n), n)
    foo = np.linspace(0, n, n)
    mapping = pyvista.opacity_transfer_function(foo, n)
    assert np.allclose(foo, mapping)
    foo = [0, 0.2, 0.9, 0.2, 0.1]
    mapping = pyvista.opacity_transfer_function(foo, n, interpolate=False)
    assert len(mapping) == n
    foo = [3, 5, 6, 10]
    mapping = pyvista.opacity_transfer_function(foo, n)
    assert len(mapping) == n


@skip_no_plotting
def test_closing_and_mem_cleanup():
    n = 5
    for _ in range(n):
        for _ in range(n):
            p = pyvista.Plotter()
            for k in range(n):
                p.add_mesh(pyvista.Sphere(radius=k))
            p.show()
        pyvista.close_all()


@skip_no_plotting
def test_above_below_scalar_range_annotations():
    p = pyvista.Plotter()
    p.add_mesh(examples.load_uniform(), clim=[100, 500], cmap='viridis',
               below_color='blue', above_color='red')
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_user_annotations_scalar_bar_mesh(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform, annotations={100.: 'yum'})
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_user_annotations_scalar_bar_volume(uniform):
    p = pyvista.Plotter()
    p.add_volume(uniform, annotations={100.: 'yum'})
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_string_array():
    mesh = examples.load_uniform()
    labels = np.empty(mesh.n_cells, dtype='<U10')
    labels[:] = 'High'
    labels[mesh['Spatial Cell Data'] < 300] = 'Medium'
    labels[mesh['Spatial Cell Data'] < 100] = 'Low'
    mesh['labels'] = labels
    p = pyvista.Plotter()
    p.add_mesh(mesh, scalars='labels')
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_fail_plot_table():
    """Make sure tables cannot be plotted"""
    table = pyvista.Table(np.random.rand(50, 3))
    with pytest.raises(TypeError):
        pyvista.plot(table)
    with pytest.raises(TypeError):
        plotter = pyvista.Plotter()
        plotter.add_mesh(table)


@skip_no_plotting
def test_bad_keyword_arguments():
    """Make sure bad keyword arguments raise an error"""
    mesh = examples.load_uniform()
    with pytest.raises(TypeError):
        pyvista.plot(mesh, foo=5)
    with pytest.raises(TypeError):
        pyvista.plot(mesh, scalar=mesh.active_scalars_name)
    with pytest.raises(TypeError):
        plotter = pyvista.Plotter()
        plotter.add_mesh(mesh, scalar=mesh.active_scalars_name)
        plotter.show()
    with pytest.raises(TypeError):
        plotter = pyvista.Plotter()
        plotter.add_mesh(mesh, foo="bad")
        plotter.show()


@skip_no_plotting
def test_cmap_list():
    mesh = sphere.copy()

    n = mesh.n_points
    scalars = np.empty(n)
    scalars[:n//3] = 0
    scalars[n//3:2*n//3] = 1
    scalars[2*n//3:] = 2

    with pytest.raises(TypeError):
        mesh.plot(scalars=scalars, cmap=['red', None, 'blue'])

    mesh.plot(scalars=scalars, cmap=['red', 'green', 'blue'],
              before_close_callback=verify_cache_image)

@skip_no_plotting
def test_default_name_tracking():
    N = 10
    color = "tan"

    p = pyvista.Plotter()
    for i in range(N):
        for j in range(N):
            center = (i, j, 0)
            mesh = pyvista.Sphere(center=center)
            p.add_mesh(mesh, color=color)
    n_made_it = len(p.renderer._actors)
    p.show()
    assert n_made_it == N**2

    # release attached scalars
    mesh.ReleaseData()
    del mesh


@skip_no_plotting
def test_add_background_image_global():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_background_image(examples.mapfile, as_global=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_background_image_not_global():
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_background_image(examples.mapfile, as_global=False)
    plotter.show(before_close_callback=verify_cache_image)



@skip_no_plotting
def test_add_background_image_subplots():
    pl = pyvista.Plotter(shape=(2, 2))
    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.add_mesh(examples.load_airplane())
    pl.subplot(1, 1)
    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.add_mesh(examples.load_airplane())
    pl.remove_background_image()

    # should error out as there's no background
    with pytest.raises(RuntimeError):
        pl.remove_background_image()

    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_remove_floor():
    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.add_floor(color='b', line_width=2, lighting=True)
    pl.add_bounding_box()  # needed for update_bounds_axes
    assert len(pl.renderer._floors) == 1
    pl.add_mesh(sphere_b)
    pl.update_bounds_axes()
    assert len(pl.renderer._floors) == 1
    pl.show()

    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.add_floor(color='b', line_width=2, lighting=True)
    pl.remove_floors()
    assert not pl.renderer._floors
    pl.show(before_close_callback=verify_cache_image)


def test_reset_camera_clipping_range():
    pl = pyvista.Plotter()
    pl.add_mesh(sphere)

    default_clipping_range = pl.camera.clipping_range # get default clipping range
    assert default_clipping_range != (10, 100) # make sure we assign something different than default

    pl.camera.clipping_range = (10,100) # set clipping range to some random numbers
    assert pl.camera.clipping_range == (10, 100) # make sure assignment is successful

    pl.reset_camera_clipping_range()
    assert pl.camera.clipping_range == default_clipping_range
    assert pl.camera.clipping_range != (10, 100)


def test_index_vs_loc():
    # first: 2d grid
    pl = pyvista.Plotter(shape=(2, 3))
    # index_to_loc valid cases
    vals = [0, 2, 4]
    expecteds = [(0, 0), (0, 2), (1, 1)]
    for val,expected in zip(vals, expecteds):
        assert tuple(pl.index_to_loc(val)) == expected
    # loc_to_index valid cases
    vals = [(0, 0), (0, 2), (1, 1)]
    expecteds = [0, 2, 4]
    for val,expected in zip(vals, expecteds):
        assert pl.loc_to_index(val) == expected
        assert pl.loc_to_index(expected) == expected
    # failing cases
    with pytest.raises(TypeError):
        pl.loc_to_index({1, 2})
    with pytest.raises(TypeError):
        pl.index_to_loc(1.5)
    with pytest.raises(TypeError):
        pl.index_to_loc((1, 2))

    # then: "1d" grid
    pl = pyvista.Plotter(shape='2|3')
    # valid cases
    for val in range(5):
        assert pl.index_to_loc(val) == val
        assert pl.index_to_loc(np.int_(val)) == val
        assert pl.loc_to_index(val) == val
        assert pl.loc_to_index(np.int_(val)) == val


@skip_no_plotting
def test_interactive_update():
    # Regression test for #1053
    p = pyvista.Plotter()
    p.show(interactive_update=True)
    assert isinstance(p.iren, vtk.vtkRenderWindowInteractor)
    p.close()

    p = pyvista.Plotter()
    with pytest.warns(UserWarning):
        p.show(auto_close=True, interactive_update=True)
