import os
import sys
from subprocess import PIPE, Popen
from weakref import proxy

import imageio
import numpy as np
import pytest

import vtki
from vtki import examples
from vtki.plotting import running_xserver

NO_PLOTTING = not running_xserver()
try:
    if os.environ['ALLOW_PLOTTING'] == 'True':
        NO_PLOTTING = False
except KeyError:
    pass


if __name__ != '__main__':
    OFF_SCREEN = 'pytest' in sys.modules
else:
    OFF_SCREEN = False


sphere = vtki.Sphere()
sphere_b = vtki.Sphere(1.0)
sphere_c = vtki.Sphere(2.0)

@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot(tmpdir):
    try:
        filename = str(tmpdir.mkdir("tmpdir").join('tmp.png'))
    except:
        filename = '/tmp/tmp.png'

    scalars = np.arange(sphere.n_points)
    cpos, img = vtki.plot(sphere,
                          off_screen=OFF_SCREEN,
                          full_screen=True,
                          text='this is a sphere',
                          show_bounds=True,
                          color='r',
                          style='wireframe',
                          line_width=10,
                          scalars=scalars,
                          flip_scalars=True,
                          cmap='bwr',
                          interpolate_before_map=True,
                          screenshot=filename, return_img=True)
    assert isinstance(cpos, list)
    assert isinstance(img, np.ndarray)
    assert os.path.isfile(filename)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_invalid_style():
    with pytest.raises(Exception):
        vtki.plot(sphere, style='not a style')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_bounds_axes_with_no_data():
    plotter = vtki.Plotter()
    plotter.add_bounds_axes()
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_show_grid():
    plotter = vtki.Plotter()
    plotter.show_grid()
    plotter.add_mesh(sphere)
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_set_camera_position():
    # with pytest.raises(Exception):
    cpos = [(2.085387555594636, 5.259683527170288, 13.092943022481887),
            (0.0, 0.0, 0.0),
            (-0.7611973344707588, -0.5507178512374836, 0.3424740374436883)]

    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.camera_position = 'xy'
    plotter.camera_position = 'xz'
    plotter.camera_position = 'yz'
    plotter.camera_position = 'yx'
    plotter.camera_position = 'zx'
    plotter.camera_position = 'zy'
    plotter.camera_position = cpos
    cpos_out = plotter.plot()
    assert cpos_out == cpos


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_no_active_scalars():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    with pytest.raises(Exception):
        plotter.update_scalars(np.arange(5))
    with pytest.raises(Exception):
        plotter.update_scalars(np.arange(sphere.n_faces))


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_add_bounds_axes():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.add_bounds_axes(show_xaxis=False,
                            show_yaxis=False,
                            show_zaxis=False,
                            show_xlabels=False,
                            show_ylabels=False,
                            show_zlabels=False,
                            use_2d=True)
    plotter.plot()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_label_fmt():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.add_bounds_axes(xlabel='My X', fmt=r'%.3f')
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
@pytest.mark.parametrize('grid', [True, 'both', 'front', 'back'])
@pytest.mark.parametrize('location', ['all', 'origin', 'outer', 'front', 'back'])
def test_plot_add_bounds_axes_params(grid, location):
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.add_bounds_axes(grid=grid, ticks='inside', location=location)
    plotter.add_bounds_axes(grid=grid, ticks='outside', location=location)
    plotter.add_bounds_axes(grid=grid, ticks='both', location=location)
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plotter_scale():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.set_scale(10, 10, 10)
    plotter.set_scale(5.0)
    plotter.set_scale(yscale=6.0)
    plotter.set_scale(zscale=9.0)
    assert plotter.scale == [5.0, 6.0, 9.0]
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_add_scalar_bar():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.add_scalar_bar(label_font_size=10, title_font_size=20, title='woa',
                interactive=True, vertical=True)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_invalid_add_scalar_bar():
    with pytest.raises(Exception):
        plotter = vtki.Plotter()
        plotter.add_scalar_bar()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_list():
    vtki.plot([sphere, sphere_b],
              off_screen=OFF_SCREEN,
              style='points')

    vtki.plot([sphere, sphere_b, sphere_c],
              off_screen=OFF_SCREEN,
              style='wireframe')

@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_add_lines_invalid():
    plotter = vtki.Plotter()
    with pytest.raises(Exception):
        plotter.add_lines(range(10))


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_open_gif_invalid():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    with pytest.raises(Exception):
        plotter.open_gif('file.abs')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_make_movie():
    try:
        filename = str(tmpdir.mkdir("tmpdir").join('tmp.mp4'))
    except:
        filename = '/tmp/tmp.mp4'

    movie_sphere = sphere.copy()
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.open_movie(filename)
    actor = plotter.add_axes_at_origin()
    plotter.remove_actor(actor)
    plotter.add_mesh(movie_sphere,
                     scalars=np.random.random(movie_sphere.n_faces))
    plotter.plot(auto_close=False, window_size=[304, 304])
    plotter.set_focus([0, 0, 0])
    for i in range(10):
        plotter.write_frame()
        random_points = np.random.random(movie_sphere.points.shape)
        movie_sphere.points = random_points*0.01 + movie_sphere.points*0.99
        movie_sphere.points -= movie_sphere.points.mean(0)
        scalars = np.random.random(movie_sphere.n_faces)
        plotter.update_scalars(scalars)

    # checking if plotter closes
    ref = proxy(plotter)
    plotter.close()

    try:
        ref
    except:
        raise Exception('Plotter did not close')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_add_legend():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    with pytest.raises(Exception):
        plotter.add_legend()
    legend_labels = [['sphere', 'r']]
    plotter.add_legend(labels=legend_labels, border=True, bcolor=None,
                       size=[0.1, 0.1])
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_add_axes_twice():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_axes()
    plotter.add_axes(interactive=True)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_add_point_labels():
    n = 10
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    points = np.random.random((n, 3))

    with pytest.raises(Exception):
        plotter.add_point_labels(points, range(n - 1))

    plotter.set_background('k')
    plotter.add_point_labels(points, range(n), show_points=True, point_color='r')
    plotter.add_point_labels(points - 1, range(n), show_points=False, point_color='r')
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_add_points():
    n = 10
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    points = np.random.random((n, 3))
    plotter.add_points(points, scalars=np.arange(10), cmap=None, flip_scalars=True)
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_key_press_event():
    plotter = vtki.Plotter()
    plotter.key_press_event(None, None)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_left_button_down():
    plotter = vtki.Plotter()
    plotter.left_button_down(None, None)
    # assert np.allclose(plotter.pickpoint, [0, 0, 0])


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_update():
    plotter = vtki.Plotter(off_screen=True)
    plotter.update()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_cell_scalars():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(sphere, interpolate_before_map=True, scalars=scalars,
                     n_colors=5, rng=10)
    plotter.plot()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_clim():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(sphere, interpolate_before_map=True, scalars=scalars,
                     n_colors=5, clim=10)
    plotter.plot()
    assert plotter.mapper.GetScalarRange() == (-10, 10)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_invalid_n_scalars():
    with pytest.raises(Exception):
        plotter = vtki.Plotter(off_screen=OFF_SCREEN)
        plotter.add_mesh(sphere, scalars=np.arange(10))
        plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_arrow():
    cent = np.random.random(3)
    direction = np.random.random(3)
    cpos, img = vtki.plot_arrows(cent, direction, off_screen=True, screenshot=True)
    assert np.any(img)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_arrows():
    cent = np.random.random((100, 3))
    direction = np.random.random((100, 3))
    cpos, img = vtki.plot_arrows(cent, direction, off_screen=True, screenshot=True)
    assert np.any(img)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_axes():
    plotter = vtki.Plotter(off_screen=True)
    plotter.add_axes()
    plotter.add_mesh(vtki.Sphere())
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_screenshot():
    plotter = vtki.Plotter(off_screen=True)
    plotter.add_mesh(vtki.Sphere())
    img = plotter.screenshot(transparent_background=True)
    assert np.any(img)
    img_again = plotter.screenshot()
    assert np.any(img_again)

    # checking if plotter closes
    ref = proxy(plotter)
    plotter.close()

    try:
        ref
    except:
        raise Exception('Plotter did not close')


def test_invalid_color():
    with pytest.raises(Exception):
        femorph.plotting.parse_color('not a color')


def test_invalid_font():
    with pytest.raises(Exception):
        femorph.parse_font_family('not a font')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_scalars_by_name():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    data = examples.load_uniform()
    plotter.add_mesh(data, scalars='Spatial Cell Data')
    plotter.plot()


def test_themes():
    vtki.set_plot_theme('paraview')
    vtki.set_plot_theme('document')
    vtki.set_plot_theme('default')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_multi_block_plot():
    multi = vtki.MultiBlock()
    multi.append(examples.load_rectilinear())
    uni = examples.load_uniform()
    arr = np.random.rand(uni.n_cells)
    uni._add_cell_scalar(arr, 'Random Data')
    multi.append(uni)
    # And now add a data set without the desired array and a NULL component
    multi[3] = examples.load_airplane()
    multi.plot(scalars='Random Data', off_screen=OFF_SCREEN, multi_colors=True)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_clear():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.clear()
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_texture():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = examples.load_globe_texture()
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(globe, texture=texture)
    plotter.plot()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_texture_associated():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(globe, texture=True)
    plotter.plot()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_read_texture_from_numpy():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = vtki.numpy_to_texture(imageio.imread(examples.mapfile))
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(globe, texture=texture)
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_rgb():
    """"Test adding a texture to a plot"""
    image = vtki.read(examples.mapfile)
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(image, rgb=True)
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_plot_multi_component_array():
    """"Test adding a texture to a plot"""
    image = vtki.read(examples.mapfile)
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(image)
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_camera():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.view_xy()
    plotter.view_xz()
    plotter.view_yz()
    plotter.add_mesh(examples.load_uniform(), reset_camera=True, backface_culling=True)
    plotter.view_xy(True)
    plotter.view_xz(True)
    plotter.view_yz(True)
    plotter.show()
    plotter.camera_position = None


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_multi_renderers():
    plotter = vtki.Plotter(shape=(2, 2), off_screen=OFF_SCREEN)

    loc = (0, 0)
    plotter.add_text('Render Window 0', loc=loc, font_size=30)
    sphere = vtki.Sphere()
    plotter.add_mesh(sphere, loc=loc, scalars=sphere.points[:, 2])
    plotter.add_scalar_bar('Z', vertical=True)

    loc = (0, 1)
    plotter.add_text('Render Window 1', loc=loc, font_size=30)
    plotter.add_mesh(vtki.Cube(), loc=loc, show_edges=True)

    loc = (1, 0)
    plotter.add_text('Render Window 2', loc=loc, font_size=30)
    plotter.add_mesh(vtki.Arrow(), color='y', loc=loc, show_edges=True)

    plotter.subplot(1, 1)
    plotter.add_text('Render Window 3', loc=loc, font_size=30)
    plotter.add_mesh(vtki.Cone(), color='g', loc=loc, show_edges=True,
                     backface_culling=True)
    plotter.add_bounding_box(render_lines_as_tubes=True, line_width=5)
    plotter.add_bounds_axes(all_edges=True)

    plotter.update_bounds_axes()
    plotter.plot()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_orthographic_slicer():
    data = examples.load_uniform()
    data.set_active_scalar('Spatial Cell Data')

    slices = data.slice_orthogonal()

    # Orthographic Slicer
    p = vtki.Plotter(shape=(2,2), off_screen=OFF_SCREEN)

    p.subplot(1,1)
    p.add_mesh(slices, clim=data.get_data_range())
    p.add_axes()
    p.enable()

    p.subplot(0,0)
    p.add_mesh(slices['XY'])
    p.view_xy()
    p.disable()

    p.subplot(0,1)
    p.add_mesh(slices['XZ'])
    p.view_xz(negative=True)
    p.disable()

    p.subplot(1,0)
    p.add_mesh(slices['YZ'])
    p.view_yz()
    p.disable()

    p.show()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
def test_remove_actor():
    data = examples.load_uniform()
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(data, name='data')
    plotter.add_mesh(data, name='data')
    plotter.add_mesh(data, name='data')
    plotter.show()
