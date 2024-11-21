from __future__ import annotations

import os

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista.plotting.errors import PyVistaPickingError

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


def test_single_cell_picking():
    sphere = pv.Sphere()
    width, height = 100, 100

    class PickCallback:
        def __init__(self):
            self.called = False

        def __call__(self, *args, **kwargs):
            self.called = True

    plotter = pv.Plotter(
        window_size=(width, height),
    )

    callback = PickCallback()
    plotter.enable_cell_picking(
        start=False,
        show=True,
        callback=callback,
        through=False,  # Single cell visible picking
    )
    plotter.add_mesh(sphere)
    plotter.show(auto_close=False)  # must start renderer first

    width, height = plotter.window_size
    plotter.iren._mouse_move(width // 2, height // 2)
    plotter.iren._simulate_keypress('p')

    plotter.close()

    assert callback.called
    assert isinstance(plotter.picked_cells, pv.UnstructuredGrid)
    assert plotter.picked_cells.n_cells == 1


@pytest.mark.parametrize('through', [False, True])
def test_multi_cell_picking(through):
    cube = pv.Cube()

    # Test with algorithm source to make sure connections work with picking
    src = vtk.vtkSphereSource()
    src.SetCenter((1, 0, 0))
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(src.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(True)

    plotter = pv.Plotter(window_size=(1024, 768))
    plotter.add_mesh(cube, pickable=True)
    plotter.add_actor(actor)
    plotter.enable_cell_picking(
        color='blue',
        through=through,
        start=True,
        show=True,
        show_frustum=True,
    )
    plotter.show(auto_close=False)  # must start renderer first

    # simulate the pick (169, 113, 875, 684)
    plotter.iren._mouse_left_button_press(169, 113)
    plotter.iren._mouse_move(875, 684)
    plotter.iren._mouse_left_button_release()

    assert plotter.get_pick_position() == (169, 113, 875, 684)

    plotter.close()

    assert isinstance(plotter.picked_cells, pv.MultiBlock)
    # Selection should return 2 submeshes
    assert len(plotter.picked_cells) == 2

    merged = plotter.picked_cells.combine()
    n_sphere_cells = pv.wrap(src.GetOutput()).n_cells
    if through:
        # all cells should have been selected
        assert merged.n_cells == cube.n_cells + n_sphere_cells
    else:
        assert merged.n_cells < cube.n_cells + n_sphere_cells


@pytest.mark.parametrize('left_clicking', [False, True])
def test_mesh_picking(sphere, left_clicking):
    picked = []

    def callback(picked_mesh):
        picked.append(picked_mesh)

    pl = pv.Plotter()
    actor = pl.add_mesh(sphere)
    pl.enable_mesh_picking(callback=callback, left_clicking=left_clicking)
    pl.show(auto_close=False)

    width, height = pl.window_size

    if left_clicking:
        pl.iren._mouse_left_button_click(width // 2, height // 2)
    else:
        pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert sphere in picked
    assert pl.picked_mesh == sphere
    assert pl.picked_actor == actor

    # invalid selection
    if left_clicking:
        pl.iren._mouse_left_button_click(0, 0)
    else:
        pl.iren._mouse_right_button_click(0, 0)

    assert pl.picked_mesh is None


def test_actor_picking(sphere):
    picked = []

    def callback(picked_actor):
        picked.append(picked_actor)

    pl = pv.Plotter()
    actor = pl.add_mesh(sphere)
    pl.enable_mesh_picking(callback=callback, use_actor=True)
    pl.show(auto_close=False)

    width, height = pl.window_size

    pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert actor in picked
    assert pl.picked_mesh == sphere

    # invalid selection
    pl.iren._mouse_right_button_click(0, 0)

    assert pl.picked_mesh is None


@pytest.mark.parametrize('left_clicking', [False, True])
def test_surface_point_picking(sphere, left_clicking):
    picked = []

    def callback(point):
        picked.append(point)

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.enable_surface_point_picking(callback=callback, left_clicking=left_clicking)
    pl.show(auto_close=False)

    width, height = pl.window_size

    if left_clicking:
        pl.iren._mouse_left_button_click(width // 2, height // 2)
    else:
        pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert len(picked)
    assert pl.picked_point is not None

    # invalid selection
    if left_clicking:
        pl.iren._mouse_left_button_click(0, 0)
    else:
        pl.iren._mouse_right_button_click(0, 0)

    assert pl.picked_point is None


@pytest.mark.parametrize('left_clicking', [False, True])
def test_disable_picking(sphere, left_clicking):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.enable_surface_point_picking(left_clicking=left_clicking)
    pl.disable_picking()
    pl.show(auto_close=False)

    assert pl._picking_text not in pl.renderer.actors

    width, height = pl.window_size

    # This verifies callbacks are removed from click events
    if left_clicking:
        pl.iren._mouse_left_button_click(width // 2, height // 2)
    else:
        pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert pl.picked_point is None

    pl.disable_picking()  # ensure it can safely be called twice


def test_cell_picking_interactive():
    n_cells = []

    def callback(picked_cells):
        n_cells.append(picked_cells.n_cells)

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.enable_cell_picking(callback=callback)
    pl.show(auto_close=False, interactive=False)

    width, height = pl.window_size

    # simulate "r" keypress
    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(width // 2, height // 2)
    pl.iren._mouse_left_button_release(width, height)

    assert n_cells[0]
    assert pl.picked_cell


def test_cell_picking_interactive_subplot():
    n_cells = []

    def callback(picked_cells):
        n_cells.append(picked_cells.n_cells)

    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(pv.Sphere())  # TRIANGLE cells
    pl.enable_cell_picking(callback=callback)
    pl.subplot(0, 1)
    pl.add_mesh(pv.Box(level=4), show_edges=True)  # QUAD cells

    pl.show(auto_close=False, interactive=False)

    width, height = pl.window_size

    # Activate picking
    pl.iren._simulate_keypress('r')

    # select just the left-hand side
    pl.iren._mouse_left_button_press(width // 4, height // 2)
    pl.iren._mouse_left_button_release(width // 2, height)

    assert n_cells[0]
    assert pl.picked_cells
    assert pl.picked_cells.get_cell(0).type == pv.CellType.TRIANGLE

    # select just the right-hand side
    pl.iren._mouse_left_button_press(width - width // 4, height // 2)
    pl.iren._mouse_left_button_release(width, height)

    assert n_cells[0]
    assert pl.picked_cells
    assert pl.picked_cells.get_cell(0).type == pv.CellType.QUAD


@pytest.mark.parametrize('left_clicking', [False, True])
def test_point_picking(left_clicking):
    picked = []

    def callback(picked_point):
        picked.append(picked_point)

    sphere = pv.Sphere()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(sphere)
    pl.enable_point_picking(
        callback=callback,
        show_message=True,
        left_clicking=left_clicking,
    )
    # must show to activate the interactive renderer (for left_clicking)
    pl.show(auto_close=False)

    # simulate the pick
    width, height = pl.window_size

    if left_clicking:
        pl.iren._mouse_left_button_click(width // 2, height // 2)
    else:
        pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert picked


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 2, 0),
    reason='Hardware picker unavailable for VTK<9.2',
)
@pytest.mark.skipif(os.name == 'nt', reason='Test fails on Windows')
@pytest.mark.parametrize('pickable_window', [False, True])
def test_point_picking_window(pickable_window):
    class Tracker:
        def __init__(self):
            self.last_picked = None

        def __call__(self, picked_point):
            self.last_picked = picked_point

    pl = pv.Plotter(
        window_size=(100, 100),
    )

    # bottom left corner, pickable
    sphere = pv.Sphere()
    sphere.translate([-1, -1, 0], inplace=True)
    pl.add_mesh(sphere, pickable=True)

    pl.camera_position = [(0.0, 0.0, 8.5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    tracker = Tracker()
    pl.enable_point_picking(
        callback=tracker,
        tolerance=0.2,
        pickable_window=pickable_window,
        picker='hardware',  # picker allows picking in the window
        # do not use point picker as it snaps to points
    )

    # simulate the pick
    renderer = pl.renderer
    picker = pl.iren.picker

    successful_pick = picker.Pick(25, 25, 0, renderer)
    assert successful_pick  # not a complete test
    assert tracker.last_picked is not None
    good_point = tracker.last_picked

    successful_pick = picker.Pick(75, 75, 0, renderer)
    assert not successful_pick  # not a complete test
    if pickable_window:
        assert not np.allclose(tracker.last_picked, good_point)  # make sure new point picked
    else:
        assert np.allclose(tracker.last_picked, good_point)  # make sure point did not change

    pl.close()


def test_path_picking():
    sphere = pv.Sphere()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(sphere)
    pl.enable_path_picking(
        show_message=True,
        callback=lambda path: None,
    )
    # simulate the pick
    renderer = pl.renderer
    picker = pl.iren.picker
    picker.Pick(50, 50, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = pl.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    pl.close()


def test_geodesic_picking():
    sphere = pv.Sphere()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(sphere)
    pl.enable_geodesic_picking(
        show_message=True,
        callback=lambda path: None,
        show_path=True,
        keep_order=True,
    )
    pl.show(auto_close=False)

    # simulate the pick
    renderer = pl.renderer
    picker = pl.iren.picker
    picker.Pick(50, 50, 0, renderer)
    picker.Pick(45, 45, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = pl.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    pl.close()


def test_horizon_picking():
    sphere = pv.Sphere()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(sphere)
    pl.enable_horizon_picking(
        show_message=True,
        callback=lambda path: None,
        show_horizon=True,
    )
    # simulate the pick
    renderer = pl.renderer
    picker = pl.iren.picker
    # at least 3 picks
    picker.Pick(50, 50, 0, renderer)
    picker.Pick(49, 50, 0, renderer)
    picker.Pick(48, 50, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = pl.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    pl.close()


def test_fly_to_right_click(verify_image_cache, sphere):
    point = []

    def callback(click_point):
        point.append(click_point)

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.enable_fly_to_right_click(callback=callback)
    pl.show(auto_close=False)
    width, height = pl.window_size
    cpos_before = pl.camera_position
    pl.iren._mouse_right_button_click(width // 4, height // 2)

    # ensure callback was called and camera position changes due to "fly"
    assert cpos_before != pl.camera_position
    assert point
    pl.close()


def test_fly_to_right_click_multi_render(verify_image_cache, sphere):
    """Same as enable as fly_to_right_click except with two renders for coverage"""
    point = []

    def callback(click_point):
        point.append(click_point)

    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(sphere)
    pl.enable_fly_to_right_click(callback=callback)
    pl.show(auto_close=False)
    width, height = pl.window_size
    cpos_before = pl.camera_position
    pl.iren._mouse_right_button_click(width // 8, height // 2)
    # ensure callback was called and camera position changes due to "fly"
    assert cpos_before != pl.camera_position
    assert point
    pl.close()


def test_fly_to_mouse_position(verify_image_cache, sphere):
    """Same as enable as fly_to_right_click except with two renders for coverage"""
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show(auto_close=False)
    width, height = pl.window_size
    cpos_before = pl.camera_position
    pl.iren._mouse_move(width - width // 4, height // 2)
    pl.fly_to_mouse_position()
    assert cpos_before != pl.camera_position
    pl.close()


def test_block_picking(multiblock_poly):
    """Test we can pick a block."""
    pl = pv.Plotter()
    width, height = pl.window_size
    actor, mapper = pl.add_composite(multiblock_poly)

    picked_blocks = []

    def turn_blue(index, dataset):
        mapper.block_attr[index].color = 'blue'
        picked_blocks.append(index)

    pl.enable_block_picking(callback=turn_blue)
    pl.show(auto_close=False)

    # click in the corner
    assert not picked_blocks
    pl.iren._mouse_left_button_click(0, 0)
    assert not picked_blocks

    # click directly in the middle
    pl.iren._mouse_left_button_click(width // 2, height // 2)
    assert mapper.block_attr[2].color

    assert pl.picked_block_index == picked_blocks[0]


@pytest.mark.parametrize('mode', ['mesh', 'cell', 'face', 'edge', 'point'])
def test_element_picking(mode):
    class Tracker:
        def __init__(self):
            self.last_picked = None

        def __call__(self, picked):
            self.last_picked = picked

    tracker = Tracker()

    mesh = pv.Wavelet()
    plotter = pv.Plotter(
        window_size=(100, 100),
    )
    plotter.add_mesh(mesh)
    plotter.enable_element_picking(
        mode=mode,
        show_message=True,
        left_clicking=True,
        callback=tracker,
    )
    # must show to activate the interactive renderer (for left_clicking)
    plotter.show(auto_close=False)

    # simulate the pick
    width, height = plotter.window_size

    plotter.iren._mouse_left_button_click(width // 2, height // 2)

    plotter.close()

    assert tracker.last_picked is not None

    if mode == 'mesh':
        assert tracker.last_picked == mesh
    elif mode == 'cell':
        assert tracker.last_picked.n_points == 8
    elif mode == 'face':
        assert tracker.last_picked.n_points == 4
    elif mode == 'edge':
        assert tracker.last_picked.n_points == 2
    elif mode == 'point':
        assert isinstance(tracker.last_picked, pv.PolyData)
        assert tracker.last_picked.n_points == 1


def test_switch_picking_type():
    pl = pv.Plotter()
    width, height = pl.window_size
    pl.add_mesh(pv.Sphere())

    cells = []

    def callback(picked):
        cells.append(picked)

    pl.enable_cell_picking(callback=callback)
    with pytest.raises(PyVistaPickingError):
        pl.enable_point_picking()

    pl.show(auto_close=False, interactive=False)
    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(width // 4, height // 4)
    pl.iren._mouse_left_button_release(width, height)

    assert cells
    assert isinstance(cells[0], pv.UnstructuredGrid)
    assert pl.picked_cells is not None

    # Now switch to point picking
    pl.disable_picking()

    points = []

    def callback(click_point):
        points.append(click_point)

    pl.enable_point_picking(callback=callback)
    # simulate the pick
    width, height = pl.window_size

    pl.iren._mouse_right_button_click(width // 3, height // 2)

    pl.close()

    assert points
    assert len(points[0]) == 3
