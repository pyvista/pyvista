"""Test render window interactor"""

import pytest

import pyvista
from pyvista import _vtk


def empty_callback():
    return


def test_remove_observer():
    pl = pyvista.Plotter()
    with pytest.raises(TypeError):
        pl.add_key_event('w', 1)

    key = 'w'
    pl.add_key_event(key, empty_callback)
    assert key in pl.iren._key_press_event_callbacks
    pl.clear_events_for_key(key)
    pl.iren.add_observer(_vtk.vtkCommand.MouseMoveEvent, empty_callback)

    assert _vtk.vtkCommand.MouseMoveEvent in pl.iren._observers
    pl.iren.remove_observer(_vtk.vtkCommand.MouseMoveEvent)
    assert _vtk.vtkCommand.MouseMoveEvent not in pl.iren._observers


def test_clear_key_event_callbacks():
    pl = pyvista.Plotter()
    pl.reset_key_events()


def test_track_mouse_position():
    pl = pyvista.Plotter()
    pl.track_mouse_position()
    pl.show(auto_close=False)
    assert pl.mouse_position is None
    x, y = 10, 20
    pl.iren._mouse_move(x, y)
    assert pl.mouse_position == (x, y)

    pl.iren.untrack_mouse_position()
    assert _vtk.vtkCommand.MouseMoveEvent not in pl.iren._observers


def test_track_click_position_multi_render():
    points = []
    def callback(mouse_point):
        points.append(mouse_point)

    pl = pyvista.Plotter()
    with pytest.raises(TypeError):
        pl.track_click_position(side='dark')

    pl.track_click_position(callback=callback, side='left', viewport=True)
    pl.show(auto_close=False)
    x, y = 10, 20
    pl.iren._mouse_left_button_press(x, y)
    assert points[0] == (x, y)

    # disable and ensure that clicking is no longer being tracked
    pl.untrack_click_position()
    pl.iren._mouse_left_button_press(50, 50)
    assert len(points) == 1
