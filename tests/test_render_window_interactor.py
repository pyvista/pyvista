"""Test render window interactor"""

import pytest

import pyvista
from pyvista import _vtk
from pyvista.plotting import system_supports_plotting

skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)

skip_needs_vtk_9 = pytest.mark.skipif(
    pyvista.vtk_version_info < (9, 1, 0), reason="Requires VTK>=9.1.0"
)


def empty_callback():
    return


@skip_needs_vtk_9
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
    assert not pl.iren.interactor.HasObserver(
        "PickEvent"
    ), "Subsequent PickEvent HasObserver tests are wrong if this fails."
    # Add different observers
    obs_move = pl.iren.add_observer(_vtk.vtkCommand.MouseMoveEvent, empty_callback)
    obs_double1 = pl.iren.add_observer(_vtk.vtkCommand.LeftButtonDoubleClickEvent, empty_callback)
    obs_double2 = pl.iren.add_observer("LeftButtonDoubleClickEvent", empty_callback)
    obs_picks = tuple(pl.iren.add_observer("PickEvent", empty_callback) for _ in range(5))
    pl.iren.add_observer("SelectionChangedEvent", empty_callback)
    assert pl.iren._observers[obs_move] == "MouseMoveEvent"
    assert pl.iren.interactor.HasObserver("MouseMoveEvent")
    assert pl.iren._observers[obs_double1] == "LeftButtonDoubleClickEvent"
    assert pl.iren._observers[obs_double2] == "LeftButtonDoubleClickEvent"
    assert pl.iren.interactor.HasObserver("LeftButtonDoubleClickEvent")
    assert all(pl.iren._observers[obs_pick] == "PickEvent" for obs_pick in obs_picks)
    assert pl.iren.interactor.HasObserver("SelectionChangedEvent")
    # Remove a specific observer
    pl.iren.remove_observer(obs_move)
    assert obs_move not in pl.iren._observers
    # Remove all observers of a specific event
    pl.iren.remove_observers(_vtk.vtkCommand.LeftButtonDoubleClickEvent)
    assert obs_double1 not in pl.iren._observers and obs_double2 not in pl.iren._observers
    # Remove all (remaining) observers
    pl.iren.remove_observers()
    assert len(pl.iren._observers) == 0
    assert not pl.iren.interactor.HasObserver("PickEvent")


def test_clear_key_event_callbacks():
    pl = pyvista.Plotter()
    pl.reset_key_events()


@skip_no_plotting
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


@skip_no_plotting
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
    pl.iren._mouse_right_button_click(2 * x, 2 * y)
    pl.iren._mouse_left_button_click(x, y)
    assert points[0] == (x, y)

    # disable and ensure that clicking is no longer being tracked
    pl.untrack_click_position(side='left')
    pl.iren._mouse_left_button_click(50, 50)
    assert len(points) == 1
