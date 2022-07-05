import pytest

import pyvista
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = not system_supports_plotting()
skip_no_vtk9 = pytest.mark.skipif(pyvista.vtk_version_info < (9,), reason="Requires VTK v9+")

# skip all tests if unable to render
if not system_supports_plotting():
    pytestmark = pytest.mark.skip


@skip_no_vtk9
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_cell_picking():
    with pytest.raises(AttributeError, match="mesh"):
        plotter = pyvista.Plotter()
        plotter.enable_cell_picking(mesh=None)

    sphere = pyvista.Sphere()
    for through in (False, True):
        plotter = pyvista.Plotter(
            window_size=(100, 100),
        )

        def callback(*args, **kwargs):
            pass

        plotter.enable_cell_picking(
            mesh=sphere,
            start=True,
            show=True,
            callback=callback,
            through=through,
        )
        plotter.add_mesh(sphere)
        plotter.show(auto_close=False)  # must start renderer first

        # simulate the pick
        renderer = plotter.renderer
        picker = plotter.iren.get_picker()
        picker.Pick(50, 50, 0, renderer)

        # pick nothing
        picker.Pick(0, 0, 0, renderer)

        plotter.get_pick_position()
        plotter.close()

    # multiblock
    plotter = pyvista.Plotter()
    multi = pyvista.MultiBlock([sphere])
    plotter.add_mesh(multi)
    plotter.enable_cell_picking()
    plotter.close()


@pytest.mark.parametrize('left_clicking', [False, True])
def test_enable_mesh_picking(sphere, left_clicking):
    picked = []

    def callback(picked_mesh):
        picked.append(picked_mesh)

    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.enable_mesh_picking(callback=callback, left_clicking=left_clicking)
    pl.show(auto_close=False)

    width, height = pl.window_size

    # clicking is to "activate" the renderer
    pl.iren._mouse_left_button_press(width // 2, height // 2)
    pl.iren._mouse_left_button_release(width, height)
    pl.iren._mouse_move(width // 2, height // 2)
    if not left_clicking:
        pl.iren._simulate_keypress('p')

    assert sphere in picked
    assert pl.picked_mesh == sphere

    # invalid selection
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(0, 0)
    pl.iren._mouse_move(0, 0)
    if not left_clicking:
        pl.iren._simulate_keypress('p')

    assert pl.picked_mesh is None


@pytest.mark.parametrize('left_clicking', [False, True])
def test_enable_surface_picking(sphere, left_clicking):
    picked = []

    def callback(point):
        picked.append(point)

    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.enable_surface_picking(callback=callback, left_clicking=left_clicking)
    pl.show(auto_close=False)

    width, height = pl.window_size

    # clicking is to "activate" the renderer
    pl.iren._mouse_left_button_press(width // 2, height // 2)
    pl.iren._mouse_left_button_release(width, height)
    pl.iren._mouse_move(width // 2, height // 2)
    if not left_clicking:
        pl.iren._simulate_keypress('p')

    assert len(picked)
    assert pl.picked_point is not None

    # invalid selection
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(0, 0)
    pl.iren._mouse_move(0, 0)
    if not left_clicking:
        pl.iren._simulate_keypress('p')

    assert pl.picked_point is None


@pytest.mark.parametrize('left_clicking', [False, True])
def test_disable_picking(sphere, left_clicking):

    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.enable_surface_picking(left_clicking=left_clicking)
    pl.disable_picking()
    pl.show(auto_close=False)

    width, height = pl.window_size

    # clicking is to "activate" the renderer
    pl.iren._mouse_left_button_press(width // 2, height // 2)
    pl.iren._mouse_left_button_release(width, height)
    pl.iren._mouse_move(width // 2, height // 2)
    if not left_clicking:
        pl.iren._simulate_keypress('p')

    assert pl.picked_point is None

    # ensure it can safely be called twice
    pl.disable_picking()
    assert pl._picking_text not in pl.renderer.actors


def test_enable_cell_picking_interactive():

    n_cells = []

    def callback(picked_cells):
        n_cells.append(picked_cells.n_cells)

    pl = pyvista.Plotter()
    pl.add_mesh(pyvista.Sphere())
    pl.enable_cell_picking(callback=callback)
    pl.show(auto_close=False, interactive=False)

    width, height = pl.window_size

    # simulate "r" keypress
    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(width // 2, height // 2)
    pl.iren._mouse_left_button_release(width, height)

    assert n_cells[0]


def test_enable_cell_picking_interactive_two_ren_win():

    n_cells = []

    def callback(picked_cells):
        n_cells.append(picked_cells.n_cells)

    pl = pyvista.Plotter(shape=(1, 2))
    pl.add_mesh(pyvista.Sphere())
    pl.enable_cell_picking(callback=callback)
    pl.show(auto_close=False, interactive=False)

    width, height = pl.window_size

    # simulate "r" keypress
    pl.iren._simulate_keypress('r')

    # select just the left-hand side
    pl.iren._mouse_left_button_press(width // 4, height // 2)
    pl.iren._mouse_left_button_release(width // 2, height)

    assert n_cells[0]


@skip_no_vtk9
@pytest.mark.parametrize('left_clicking', [False, True])
def test_point_picking(left_clicking):
    sphere = pyvista.Sphere()
    for use_mesh in (False, True):
        plotter = pyvista.Plotter(
            window_size=(100, 100),
        )
        plotter.add_mesh(sphere)
        plotter.enable_point_picking(
            show_message=True,
            use_mesh=use_mesh,
            left_clicking=left_clicking,
            callback=lambda: None,
        )
        # must show to activate the interactive renderer (for left_clicking)
        plotter.show(auto_close=False)

        # simulate the pick
        width, height = plotter.window_size
        if left_clicking:
            plotter.iren._mouse_left_button_press(width // 2, height // 2)
            plotter.iren._mouse_left_button_release(width, height)
            plotter.iren._mouse_move(width // 2, height // 2)
        else:
            renderer = plotter.renderer
            picker = plotter.iren.get_picker()
            picker.Pick(width // 2, height // 2, 0, renderer)
        plotter.close()


@skip_no_vtk9
def test_point_picking_window_not_pickable():

    plotter = pyvista.Plotter(
        window_size=(100, 100),
    )

    # bottom left corner, pickable
    sphere = pyvista.Sphere()
    sphere.translate([-100, -100, 0], inplace=True)
    plotter.add_mesh(sphere, pickable=True)

    # top right corner, not pickable
    unpickable_sphere = pyvista.Sphere()
    unpickable_sphere.translate([100, 100, 0], inplace=True)
    plotter.add_mesh(unpickable_sphere, pickable=False)

    plotter.view_xy()
    plotter.enable_point_picking(
        pickable_window=False,
        tolerance=0.2,
    )

    # simulate the pick
    renderer = plotter.renderer
    picker = plotter.iren.get_picker()

    successful_pick = picker.Pick(0, 0, 0, renderer)
    assert successful_pick

    successful_pick = picker.Pick(100, 100, 0, renderer)
    assert not successful_pick

    plotter.close()


@skip_no_vtk9
def test_path_picking():
    sphere = pyvista.Sphere()
    plotter = pyvista.Plotter(
        window_size=(100, 100),
    )
    plotter.add_mesh(sphere)
    plotter.enable_path_picking(
        show_message=True,
        callback=lambda: None,
    )
    # simulate the pick
    renderer = plotter.renderer
    picker = plotter.iren.get_picker()
    picker.Pick(50, 50, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = plotter.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    plotter.close()


@skip_no_vtk9
def test_geodesic_picking():
    sphere = pyvista.Sphere()
    plotter = pyvista.Plotter(
        window_size=(100, 100),
    )
    plotter.add_mesh(sphere)
    plotter.enable_geodesic_picking(
        show_message=True,
        callback=lambda: None,
        show_path=True,
        keep_order=True,
    )
    plotter.show(auto_close=False)

    # simulate the pick
    renderer = plotter.renderer
    picker = plotter.iren.get_picker()
    picker.Pick(50, 50, 0, renderer)
    picker.Pick(45, 45, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = plotter.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    plotter.close()


@skip_no_vtk9
def test_horizon_picking():
    sphere = pyvista.Sphere()
    plotter = pyvista.Plotter(
        window_size=(100, 100),
    )
    plotter.add_mesh(sphere)
    plotter.enable_horizon_picking(
        show_message=True,
        callback=lambda: None,
        show_horizon=True,
    )
    # simulate the pick
    renderer = plotter.renderer
    picker = plotter.iren.get_picker()
    # at least 3 picks
    picker.Pick(50, 50, 0, renderer)
    picker.Pick(49, 50, 0, renderer)
    picker.Pick(48, 50, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = plotter.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    plotter.close()


def test_enable_fly_to_right_click(sphere):

    point = []

    def callback(click_point):
        point.append(click_point)

    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.enable_fly_to_right_click(callback=callback)
    pl.show(auto_close=False)
    width, height = pl.window_size
    cpos_before = pl.camera_position
    pl.iren._mouse_right_button_press(width // 2, height // 2)

    # ensure callback was called and camera position changes due to "fly"
    assert cpos_before != pl.camera_position
    assert point


def test_enable_fly_to_right_click_multi_render(sphere):
    """Same as enable as fly_to_right_click except with two renders for coverage"""
    point = []

    def callback(click_point):
        point.append(click_point)

    pl = pyvista.Plotter(shape=(1, 2))
    pl.add_mesh(sphere)
    pl.enable_fly_to_right_click(callback=callback)
    pl.show(auto_close=False)
    width, height = pl.window_size
    cpos_before = pl.camera_position
    pl.iren._mouse_right_button_press(width // 4, height // 2)
    # ensure callback was called and camera position changes due to "fly"
    assert cpos_before != pl.camera_position
    assert point


def test_block_picking(multiblock_poly):
    pl = pyvista.Plotter()
    pl.add_composite(multiblock_poly)
