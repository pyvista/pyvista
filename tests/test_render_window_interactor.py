"""Test render window interactor"""

import pytest

import pyvista
from pyvista import _vtk


def empty_callback():
    return


def test_observers():
    pl = pyvista.Plotter()

    # Key events
    with pytest.raises(TypeError):
        pl.add_key_event('w', 1)

    key = 'w'
    pl.add_key_event(key, empty_callback)
    assert key in pl.iren._key_press_event_callbacks
    pl.clear_events_for_key(key)
    assert key not in pl.iren._key_press_event_callbacks

    # Custom events
    assert not pl.iren.interactor.HasObserver("PickEvent"), "Subsequent PickEvent observer tests are wrong if this fails."
    obs_move = pl.iren.add_observer(_vtk.vtkCommand.MouseMoveEvent, empty_callback)
    obs_double1 = pl.iren.add_observer(_vtk.vtkCommand.LeftButtonDoubleClickEvent, empty_callback)
    obs_double2 = pl.iren.add_observer("LeftButtonDoubleClickEvent", empty_callback)
    obs_picks = tuple(pl.iren.add_observer("PickEvent", empty_callback) for _ in range(5))
    obs_select = pl.iren.add_observer("SelectionChangedEvent", empty_callback)
    assert pl.iren._observers[obs_move] == "MouseMoveEvent" and pl.iren.interactor.HasObserver("MouseMoveEvent")
    assert pl.iren._observers[obs_double1] == "LeftButtonDoubleClickEvent" and pl.iren._observers[obs_double2] == "LeftButtonDoubleClickEvent"
    assert pl.iren.interactor.HasObserver("LeftButtonDoubleClickEvent")
    assert all(pl.iren._observers[obs_pick] == "PickEvent" for obs_pick in obs_picks)
    assert pl.iren.interactor.HasObserver("SelectionChangedEvent")
    pl.iren.remove_observer(obs_move)  # Remove specific observer
    assert obs_move not in pl.iren._observers
    pl.iren.remove_observers(_vtk.vtkCommand.LeftButtonDoubleClickEvent)  # Remove all observers of specific event
    assert obs_double1 not in pl.iren._observers and obs_double2 not in pl.iren._observers
    pl.iren.remove_observers()  # Remove all (remaining) observers
    assert all(obs_pick not in pl.iren._observers for obs_pick in obs_picks)
    assert not pl.iren.interactor.HasObserver("PickEvent")
    assert obs_select not in pl.iren._observers


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
    assert "MouseMoveEvent" not in pl.iren._observers.values()


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
    pl.iren._mouse_left_button_click(x, y)
    assert points[0] == (x, y)

    # disable and ensure that clicking is no longer being tracked
    pl.untrack_click_position(side='left')
    pl.iren._mouse_left_button_click(50, 50)
    assert len(points) == 1
