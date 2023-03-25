"""Test render window interactor"""

import platform
import time

import pytest

import pyvista
from pyvista import _vtk
from pyvista.plotting import system_supports_plotting

skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)


def empty_callback():
    return


@pytest.mark.needs_vtk_version(9, 1)
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
    # attempting to clear non-existing events doesn't raise by default
    pl.clear_events_for_key(key)
    with pytest.raises(ValueError, match='No events found for key'):
        pl.clear_events_for_key(key, raise_on_missing=True)

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


@skip_no_plotting
def test_track_click_position():
    events = []

    def single_click_callback(mouse_position):
        events.append("single")

    def double_click_callback(mouse_position):
        events.append("double")

    pl = pyvista.Plotter()
    pl.track_click_position(callback=single_click_callback, side='left', double=False)
    pl.track_click_position(callback=double_click_callback, side='left', double=True)
    pl.show(auto_close=False)

    # Test single and double clicks:
    pl.iren._mouse_left_button_click(10, 10)
    assert len(events) == 1 and events.pop(0) == "single"
    pl.iren._mouse_left_button_click(50, 50, count=2)
    assert len(events) == 2 and events.pop(1) == "double" and events.pop(0) == "single"

    # Test triple click behaviour:
    pl.iren._mouse_left_button_click(10, 10, count=3)
    assert len(events) == 3
    assert events.pop(2) == "single" and events.pop(1) == "double" and events.pop(0) == "single"


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason='vtkCocoaRenderWindowInteractor (MacOS) does not invoke TimerEvents during ProcessEvents. ',
)
@pytest.mark.needs_vtk_version(
    (9, 2),
    reason='vtkXRenderWindowInteractor (Linux) does not invoke TimerEvents during ProcessEvents until VTK9.2.',
)
def test_timer():
    # Create a normal interactor from the offscreen plotter (not generic,
    # which is the default for offscreen rendering)
    pl = pyvista.Plotter()
    iren = pyvista.plotting.RenderWindowInteractor(pl)
    iren.set_render_window(pl.render_window)

    duration = 50  # Duration of created timers
    delay = 5 * duration  # Extra time we wait for the timers to fire at least once
    events = []

    def on_timer(obj, event):
        # TimerEvent callback
        events.append(event)

    def process_events(iren, duration):
        # Helper function to call process_events for the given duration (in milliseconds).
        t = 1000 * time.time()
        while 1000 * time.time() - t < duration:
            iren.process_events()

    # Setup interactor
    iren.add_observer("TimerEvent", on_timer)
    iren.initialize()

    # Test one-shot timer (only fired once for the extended duration)
    iren.create_timer(duration, repeating=False)
    process_events(iren, delay)
    assert len(events) == 1

    # Test repeating timer (fired multiple times for extended duration)
    repeating_timer = iren.create_timer(duration, repeating=True)
    process_events(iren, 2 * delay)
    assert len(events) >= 3
    E = len(events)

    # Test timer destruction (no more events fired)
    iren.destroy_timer(repeating_timer)
    process_events(iren, delay)
    assert len(events) == E
