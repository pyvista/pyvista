import pytest
import pyvista
from pyvista.plotting import system_supports_plotting
import vtk

NO_PLOTTING = not system_supports_plotting()
skip_no_vtk9 = pytest.mark.skipif(not vtk.vtkVersion().GetVTKMajorVersion() >= 9, reason="Requires VTK9+")


@skip_no_vtk9
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_cell_picking():
    with pytest.raises(TypeError, match="notebook"):
        plotter = pyvista.Plotter(notebook=True)
        plotter.enable_cell_picking()

    with pytest.raises(AttributeError, match="mesh"):
        plotter = pyvista.Plotter(off_screen=False)
        plotter.enable_cell_picking(mesh=None)

    sphere = pyvista.Sphere()
    for through in (False, True):
        plotter = pyvista.Plotter(
            window_size=(100, 100),
            off_screen=False
        )
        plotter.enable_cell_picking(
            mesh=sphere,
            start=True,
            show=True,
            callback=lambda: None,
            through=through,
        )
        plotter.add_mesh(sphere)

        # simulate the pick
        renderer = plotter.renderer
        picker = plotter.iren.GetPicker()
        picker.Pick(50, 50, 0, renderer)

        # pick nothing
        picker.Pick(0, 0, 0, renderer)

        plotter.get_pick_position()
        plotter.close()

    # multiblock
    plotter = pyvista.Plotter(off_screen=False)
    multi = pyvista.MultiBlock([sphere])
    plotter.add_mesh(multi)
    plotter.enable_cell_picking()
    plotter.close()


@skip_no_vtk9
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_point_picking():
    with pytest.raises(TypeError, match="notebook"):
        plotter = pyvista.Plotter(notebook=True)
        plotter.enable_point_picking()

    sphere = pyvista.Sphere()
    for use_mesh in (False, True):
        plotter = pyvista.Plotter(
            window_size=(100, 100),
            off_screen=False
        )
        plotter.add_mesh(sphere)
        plotter.enable_point_picking(
            show_message=True,
            use_mesh=use_mesh,
            callback=lambda: None,
        )
        # simulate the pick
        renderer = plotter.renderer
        picker = plotter.iren.GetPicker()
        picker.Pick(50, 50, 0, renderer)
        plotter.close()


@skip_no_vtk9
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_path_picking():
    sphere = pyvista.Sphere()
    plotter = pyvista.Plotter(
        window_size=(100, 100),
        off_screen=False
    )
    plotter.add_mesh(sphere)
    plotter.enable_path_picking(
        show_message=True,
        callback=lambda: None,
    )
    # simulate the pick
    renderer = plotter.renderer
    picker = plotter.iren.GetPicker()
    picker.Pick(50, 50, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = plotter._key_press_event_callbacks['c']
    clear_callback[0]()
    plotter.close()


@skip_no_vtk9
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_geodesic_picking():
    sphere = pyvista.Sphere()
    plotter = pyvista.Plotter(
        window_size=(100, 100),
        off_screen=False
    )
    plotter.add_mesh(sphere)
    plotter.enable_geodesic_picking(
        show_message=True,
        callback=lambda: None,
        show_path=True,
    )
    # simulate the pick
    renderer = plotter.renderer
    picker = plotter.iren.GetPicker()
    picker.Pick(50, 50, 0, renderer)
    picker.Pick(45, 45, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = plotter._key_press_event_callbacks['c']
    clear_callback[0]()
    plotter.close()


@skip_no_vtk9
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_horizon_picking():
    sphere = pyvista.Sphere()
    plotter = pyvista.Plotter(
        window_size=(100, 100),
        off_screen=False
    )
    plotter.add_mesh(sphere)
    plotter.enable_horizon_picking(
        show_message=True,
        callback=lambda: None,
        show_horizon=True,
    )
    # simulate the pick
    renderer = plotter.renderer
    picker = plotter.iren.GetPicker()
    # at least 3 picks
    picker.Pick(50, 50, 0, renderer)
    picker.Pick(49, 50, 0, renderer)
    picker.Pick(48, 50, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = plotter._key_press_event_callbacks['c']
    clear_callback[0]()
    plotter.close()
