import pytest
import pyvista
from pyvista.plotting import system_supports_plotting
import vtk

NO_PLOTTING = not system_supports_plotting()
skip_no_vtk9 = pytest.mark.skipif(not vtk.vtkVersion().GetVTKMajorVersion() >= 9, reason="Requires VTK9+")


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
    pl.iren._mouse_left_button_press(width//2, height//2)
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
    pl.iren._mouse_left_button_press(width//4, height//2)
    pl.iren._mouse_left_button_release(width//2, height)

    assert n_cells[0]


@skip_no_vtk9
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_point_picking():
    sphere = pyvista.Sphere()
    for use_mesh in (False, True):
        plotter = pyvista.Plotter(
            window_size=(100, 100),
        )
        plotter.add_mesh(sphere)
        plotter.enable_point_picking(
            show_message=True,
            use_mesh=use_mesh,
            callback=lambda: None,
        )
        # simulate the pick
        renderer = plotter.renderer
        picker = plotter.iren.get_picker()
        picker.Pick(50, 50, 0, renderer)
        plotter.close()


@skip_no_vtk9
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
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
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
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
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
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
    pl.iren._mouse_right_button_press(width//2, height//2)

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
    pl.iren._mouse_right_button_press(width//4, height//2)
     # ensure callback was called and camera position changes due to "fly"
    assert cpos_before != pl.camera_position
    assert point
