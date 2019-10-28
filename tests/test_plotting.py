import os
import sys
from subprocess import PIPE, Popen
from weakref import proxy

import imageio
import numpy as np
import pytest

import pyvista
from pyvista import examples
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = not system_supports_plotting()

ffmpeg_failed = False
try:
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import imageio
        imageio.plugins.ffmpeg.download()
except:
    ffmpeg_failed = True


if __name__ != '__main__':
    OFF_SCREEN = 'pytest' in sys.modules
else:
    OFF_SCREEN = False

pyvista.OFF_SCREEN = OFF_SCREEN


sphere = pyvista.Sphere()
sphere_b = pyvista.Sphere(1.0)
sphere_c = pyvista.Sphere(2.0)

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot(tmpdir):
    filename = os.path.join(pyvista.USER_DATA_PATH, 'tmp.png')
    scalars = np.arange(sphere.n_points)
    cpos, img = pyvista.plot(sphere,
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
                             screenshot=filename,
                             return_img=True)
    assert isinstance(cpos, list)
    assert isinstance(img, np.ndarray)
    assert os.path.isfile(filename)
    os.remove(filename)
    filename = os.path.join(pyvista.USER_DATA_PATH, 'foo')
    cpos = pyvista.plot(sphere, off_screen=OFF_SCREEN, screenshot=filename)
    filename = filename + ".png" # Ensure it added a PNG extension by default
    assert os.path.isfile(filename)
    # remove file
    os.remove(filename)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_invalid_style():
    with pytest.raises(Exception):
        pyvista.plot(sphere, style='not a style')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_bounds_axes_with_no_data():
    plotter = pyvista.Plotter()
    plotter.show_bounds()
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_show_grid():
    plotter = pyvista.Plotter()
    plotter.show_grid()
    plotter.add_mesh(sphere)
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_set_camera_position():
    # with pytest.raises(Exception):
    cpos = [(2.085387555594636, 5.259683527170288, 13.092943022481887),
            (0.0, 0.0, 0.0),
            (-0.7611973344707588, -0.5507178512374836, 0.3424740374436883)]

    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.camera_position = 'xy'
    plotter.camera_position = 'xz'
    plotter.camera_position = 'yz'
    plotter.camera_position = 'yx'
    plotter.camera_position = 'zx'
    plotter.camera_position = 'zy'
    plotter.camera_position = cpos
    cpos_out = plotter.show()
    assert cpos_out == cpos


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_no_active_scalars():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    with pytest.raises(Exception):
        plotter.update_scalars(np.arange(5))
    with pytest.raises(Exception):
        plotter.update_scalars(np.arange(sphere.n_faces))


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_show_bounds():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.show_bounds(show_xaxis=False,
                        show_yaxis=False,
                        show_zaxis=False,
                        show_xlabels=False,
                        show_ylabels=False,
                        show_zlabels=False,
                        use_2d=True)
    # And test backwards compatibility
    plotter.add_bounds_axes(show_xaxis=False,
                            show_yaxis=False,
                            show_zaxis=False,
                            show_xlabels=False,
                            show_ylabels=False,
                            show_zlabels=False,
                            use_2d=True)
    plotter.show()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_label_fmt():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.show_bounds(xlabel='My X', fmt=r'%.3f')
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.parametrize('grid', [True, 'both', 'front', 'back'])
@pytest.mark.parametrize('location', ['all', 'origin', 'outer', 'front', 'back'])
def test_plot_show_bounds_params(grid, location):
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(pyvista.Cube())
    plotter.show_bounds(grid=grid, ticks='inside', location=location)
    plotter.show_bounds(grid=grid, ticks='outside', location=location)
    plotter.show_bounds(grid=grid, ticks='both', location=location)
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plotter_scale():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.set_scale(10, 10, 10)
    plotter.set_scale(5.0)
    plotter.set_scale(yscale=6.0)
    plotter.set_scale(zscale=9.0)
    assert plotter.scale == [5.0, 6.0, 9.0]
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_add_scalar_bar():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.add_scalar_bar(label_font_size=10, title_font_size=20, title='woa',
                           interactive=True, vertical=True)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_invalid_add_scalar_bar():
    with pytest.raises(Exception):
        plotter = pyvista.Plotter()
        plotter.add_scalar_bar()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_list():
    pyvista.plot([sphere, sphere_b],
                 off_screen=OFF_SCREEN,
                 style='points')

    pyvista.plot([sphere, sphere_b, sphere_c],
                 off_screen=OFF_SCREEN,
                 style='wireframe')

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_add_lines_invalid():
    plotter = pyvista.Plotter()
    with pytest.raises(Exception):
        plotter.add_lines(range(10))


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_open_gif_invalid():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    with pytest.raises(Exception):
        plotter.open_gif('file.abs')


@pytest.mark.skipif(ffmpeg_failed, reason="Requires imageio-ffmpeg")
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_make_movie():
    # Make temporary file
    filename = os.path.join(pyvista.USER_DATA_PATH, 'tmp.mp4')

    movie_sphere = sphere.copy()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.open_movie(filename)
    actor = plotter.add_axes_at_origin()
    plotter.remove_actor(actor)
    plotter.add_mesh(movie_sphere,
                     scalars=np.random.random(movie_sphere.n_faces))
    plotter.show(auto_close=False, window_size=[304, 304])
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

    # remove file
    os.remove(filename)

    try:
        ref
    except:
        raise Exception('Plotter did not close')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_add_legend():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    with pytest.raises(Exception):
        plotter.add_legend()
    legend_labels = [['sphere', 'r']]
    plotter.add_legend(labels=legend_labels, border=True, bcolor=None,
                       size=[0.1, 0.1])
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_add_axes_twice():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_axes()
    plotter.add_axes(interactive=True)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_add_point_labels():
    n = 10
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    points = np.random.random((n, 3))

    with pytest.raises(Exception):
        plotter.add_point_labels(points, range(n - 1))

    plotter.set_background('k')
    plotter.set_background([0, 0, 0], top=[1,1,1]) # Gradient
    plotter.add_point_labels(points, range(n), show_points=True, point_color='r')
    plotter.add_point_labels(points - 1, range(n), show_points=False, point_color='r')
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_add_points():
    n = 10
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    points = np.random.random((n, 3))
    plotter.add_points(points, scalars=np.arange(10), cmap=None, flip_scalars=True)
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_key_press_event():
    plotter = pyvista.Plotter(off_screen=False)
    plotter.key_press_event(None, None)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_left_button_down():
    plotter = pyvista.Plotter(off_screen=False)
    plotter.left_button_down(None, None)
    # assert np.allclose(plotter.pickpoint, [0, 0, 0])


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_update():
    plotter = pyvista.Plotter(off_screen=True)
    plotter.update()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_cell_arrays():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(sphere, interpolate_before_map=True, scalars=scalars,
                     n_colors=5, rng=10)
    plotter.show()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_clim():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(sphere, interpolate_before_map=True, scalars=scalars,
                     n_colors=5, clim=10)
    plotter.show()
    assert plotter.mapper.GetScalarRange() == (-10, 10)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_invalid_n_arrays():
    with pytest.raises(Exception):
        plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
        plotter.add_mesh(sphere, scalars=np.arange(10))
        plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_arrow():
    cent = np.random.random(3)
    direction = np.random.random(3)
    cpos, img = pyvista.plot_arrows(cent, direction, off_screen=True, screenshot=True)
    assert np.any(img)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_arrows():
    cent = np.random.random((100, 3))
    direction = np.random.random((100, 3))
    cpos, img = pyvista.plot_arrows(cent, direction, off_screen=True, screenshot=True)
    assert np.any(img)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_axes():
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_axes()
    plotter.add_mesh(pyvista.Sphere())
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_box_axes():
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_axes(box=True, box_args={'color_box':True})
    plotter.add_mesh(pyvista.Sphere())
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_screenshot(tmpdir):
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(pyvista.Sphere())
    img = plotter.screenshot(transparent_background=True)
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
        raise Exception('Plotter did not close')


def test_invalid_color():
    with pytest.raises(Exception):
        femorph.plotting.parse_color('not a color')


def test_invalid_font():
    with pytest.raises(Exception):
        femorph.parse_font_family('not a font')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_scalars_by_name():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    data = examples.load_uniform()
    plotter.add_mesh(data, scalars='Spatial Cell Data')
    plotter.show()


def test_themes():
    pyvista.set_plot_theme('paraview')
    pyvista.set_plot_theme('document')
    pyvista.set_plot_theme('night')
    pyvista.set_plot_theme('default')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_multi_block_plot():
    multi = pyvista.MultiBlock()
    multi.append(examples.load_rectilinear())
    uni = examples.load_uniform()
    arr = np.random.rand(uni.n_cells)
    uni._add_cell_array(arr, 'Random Data')
    multi.append(uni)
    # And now add a data set without the desired array and a NULL component
    multi[3] = examples.load_airplane()
    multi.plot(scalars='Random Data', off_screen=OFF_SCREEN, multi_colors=True)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_clear():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(sphere)
    plotter.clear()
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_texture():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = examples.load_globe_texture()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(globe, texture=texture)
    plotter.show()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_texture_associated():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(globe, texture=True)
    plotter.show()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_read_texture_from_numpy():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = pyvista.numpy_to_texture(imageio.imread(examples.mapfile))
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(globe, texture=texture)
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
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
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(cube, scalars='face_colors', rgb=True)
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_multi_component_array():
    """"Test adding a texture to a plot"""
    image = pyvista.UniformGrid((3,3,3))
    image['array'] = np.random.randn(*image.dimensions).ravel(order='f')
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(image, scalars='array')
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_camera():
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
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
    plotter.camera_position = None


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_multi_renderers():
    plotter = pyvista.Plotter(shape=(2, 2), off_screen=OFF_SCREEN)

    loc = (0, 0)
    plotter.add_text('Render Window 0', loc=loc, font_size=30)
    sphere = pyvista.Sphere()
    plotter.add_mesh(sphere, loc=loc, scalars=sphere.points[:, 2])
    plotter.add_scalar_bar('Z', vertical=True)

    loc = (0, 1)
    plotter.add_text('Render Window 1', loc=loc, font_size=30)
    plotter.add_mesh(pyvista.Cube(), loc=loc, show_edges=True)

    loc = (1, 0)
    plotter.add_text('Render Window 2', loc=loc, font_size=30)
    plotter.add_mesh(pyvista.Arrow(), color='y', loc=loc, show_edges=True)

    plotter.subplot(1, 1)
    plotter.add_text('Render Window 3', position=(0., 0.),
                     loc=loc, font_size=30, viewport=True)
    plotter.add_mesh(pyvista.Cone(), color='g', loc=loc, show_edges=True,
                     culling=True)
    plotter.add_bounding_box(render_lines_as_tubes=True, line_width=5)
    plotter.show_bounds(all_edges=True)

    plotter.update_bounds_axes()
    plotter.show()

    # Test subplot indices (2 rows by 1 column)
    plotter = pyvista.Plotter(shape=(2, 1), off_screen=OFF_SCREEN)
    # First row
    plotter.subplot(0,0)
    plotter.add_mesh(pyvista.Sphere())
    # Second row
    plotter.subplot(1,0)
    plotter.add_mesh(pyvista.Cube())
    plotter.show()

    # Test subplot indices (1 row by 2 columns)
    plotter = pyvista.Plotter(shape=(1, 2), off_screen=OFF_SCREEN)
    # First column
    plotter.subplot(0,0)
    plotter.add_mesh(pyvista.Sphere())
    # Second column
    plotter.subplot(0,1)
    plotter.add_mesh(pyvista.Cube())
    plotter.show()

    with pytest.raises(IndexError):
        # Test bad indices
        plotter = pyvista.Plotter(shape=(1, 2), off_screen=OFF_SCREEN)
        plotter.subplot(0,0)
        plotter.add_mesh(pyvista.Sphere())
        plotter.subplot(1,0)
        plotter.add_mesh(pyvista.Cube())
        plotter.show()


    # Test subplot 3 on left, 1 on right
    plotter = pyvista.Plotter(shape='3|1', off_screen=OFF_SCREEN)
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pyvista.Cone())
    plotter.show()

    # Test subplot 3 on bottom, 1 on top
    plotter = pyvista.Plotter(shape='1|3', off_screen=OFF_SCREEN)
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pyvista.Cone())
    plotter.show()



@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_link_views():
    plotter = pyvista.Plotter(shape=(1, 4), off_screen=OFF_SCREEN)
    sphere = pyvista.Sphere()
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


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_orthographic_slicer():
    data = examples.load_uniform()
    data.set_active_scalar('Spatial Cell Data')

    slices = data.slice_orthogonal()

    # Orthographic Slicer
    p = pyvista.Plotter(shape=(2,2), off_screen=OFF_SCREEN)

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

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_remove_actor():
    data = examples.load_uniform()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(data, name='data')
    plotter.add_mesh(data, name='data')
    plotter.add_mesh(data, name='data')
    plotter.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_image_properties():
    mesh = examples.load_uniform()
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(mesh)
    p.show(auto_close=False) # DO NOT close plotter
    # Get RGB image
    img = p.image
    # Get the depth image
    img = p.get_image_depth()
    p.close()
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(mesh)
    p.show() # close plotter
    # Get RGB image
    img = p.image
    # Get the depth image
    img = p.get_image_depth()
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_volume_rendering():
    # Really just making sure no errors are thrown
    vol = examples.load_uniform()
    vol.plot(off_screen=OFF_SCREEN, volume=True, opacity='linear')

    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_volume(vol, opacity='sigmoid', cmap='jet', n_colors=15)
    plotter.show()

    # Now test MultiBlock rendering
    data = pyvista.MultiBlock(dict(a=examples.load_uniform(),
                                   b=examples.load_uniform(),
                                   c=examples.load_uniform(),
                                   d=examples.load_uniform(),))
    data['a'].rename_scalar('Spatial Point Data', 'a')
    data['b'].rename_scalar('Spatial Point Data', 'b')
    data['c'].rename_scalar('Spatial Point Data', 'c')
    data['d'].rename_scalar('Spatial Point Data', 'd')
    data.plot(off_screen=OFF_SCREEN, volume=True, multi_colors=True, )



@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_compar_four():
    # Really just making sure no errors are thrown
    mesh = examples.load_uniform()
    data_a = mesh.contour()
    data_b = mesh.threshold_percent(0.5)
    data_c = mesh.decimate_boundary(0.5)
    data_d = mesh.glyph()
    pyvista.plot_compare_four(data_a, data_b, data_c, data_d,
                              disply_kwargs={'color':'w'},
                              plotter_kwargs={'off_screen':OFF_SCREEN},)
    return


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting():
    mesh = examples.load_airplane()
    mesh.plot(off_screen=OFF_SCREEN, eye_dome_lighting=True)
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(mesh)
    p.enable_eye_dome_lighting()
    p.show()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(mesh)
    p.enable_eye_dome_lighting()
    p.disable_eye_dome_lighting()
    p.show()



@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_opacity_by_array():
    mesh = examples.load_uniform()
    # Test with opacity arry
    mesh['opac'] = mesh['Spatial Point Data'] / 100.
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(mesh, scalars='Spatial Point Data', opacity='opac',)
    p.show()
    # Test with uncertainty array (transperancy)
    mesh['unc'] = mesh['Spatial Point Data']
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(mesh, scalars='Spatial Point Data', opacity='unc',
               use_transparency=True)
    p.show()
    # Test using mismatched arrays
    with pytest.raises(RuntimeError):
        p = pyvista.Plotter(off_screen=OFF_SCREEN)
        p.add_mesh(mesh, scalars='Spatial Cell Data', opacity='unc',)
        p.show()
    # Test with user defined transfer function
    opacities = [0,0.2,0.9,0.2,0.1]
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(mesh, scalars='Spatial Point Data', opacity=opacities,)
    p.show()


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
    foo = [0,0.2,0.9,0.2,0.1]
    mapping = pyvista.opacity_transfer_function(foo, n, interpolate=False)
    assert len(mapping) == n
    foo = [3, 5, 6, 10]
    mapping = pyvista.opacity_transfer_function(foo, n)
    assert len(mapping) == n


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_closing_and_mem_cleanup():
    n = 5
    for i in range(n):
        for j in range(n):
            p = pyvista.Plotter(off_screen=OFF_SCREEN)
            for k in range(n):
                p.add_mesh(pyvista.Sphere(radius=k))
            p.show()
        pyvista.close_all()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_above_below_scalar_range_annotations():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(examples.load_uniform(), clim=[100, 500], cmap='viridis',
               below_color='blue', above_color='red')
    p.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_user_annotations_scalar_bar():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(examples.load_uniform(), annotations={100.:'yum'})
    p.show()
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_volume(examples.load_uniform(), annotations={100.:'yum'})
    p.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_plot_string_array():
    mesh = examples.load_uniform()
    labels = np.empty(mesh.n_cells, dtype='<U10')
    labels[:] = 'High'
    labels[mesh['Spatial Cell Data'] < 300] = 'Medium'
    labels[mesh['Spatial Cell Data'] < 100] = 'Low'
    mesh['labels'] = labels
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh(mesh, scalars='labels')
    p.show()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_fail_plot_table():
    """Make sure tables cannot be plotted"""
    table = pyvista.Table(np.random.rand(50, 3))
    with pytest.raises(TypeError):
        pyvista.plot(table)
    with pytest.raises(TypeError):
        plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
        plotter.add_mesh(table)
