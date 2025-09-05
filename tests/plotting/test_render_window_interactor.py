"""Test render window interactor"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.plotting.render_window_interactor import InteractorStyleImage
from pyvista.plotting.render_window_interactor import InteractorStyleJoystickActor
from pyvista.plotting.render_window_interactor import InteractorStyleJoystickCamera
from pyvista.plotting.render_window_interactor import InteractorStyleRubberBand2D
from pyvista.plotting.render_window_interactor import InteractorStyleRubberBandPick
from pyvista.plotting.render_window_interactor import InteractorStyleTerrain
from pyvista.plotting.render_window_interactor import InteractorStyleTrackballActor
from pyvista.plotting.render_window_interactor import InteractorStyleTrackballCamera
from pyvista.plotting.render_window_interactor import InteractorStyleZoom
from tests.plotting.test_plotting import skip_windows_mesa

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def empty_callback():
    return


@pytest.mark.parametrize('callback', ['foo', 1, object()])
def test_track_click_position_raises(callback):
    pl = pv.Plotter()
    match = re.escape(
        'Invalid callback provided, it should be either ``None`` or a callable.',
    )
    with pytest.raises(TypeError, match=match):
        pl.track_click_position(callback=callback)


def test_simulate_key_press_raises():
    pl = pv.Plotter()
    with pytest.raises(
        ValueError,
        match=re.escape('Only accepts a single key'),
    ):
        pl.iren._simulate_keypress(key=['f', 't'])


def test_process_events_raises(mocker: MockerFixture):
    pl = pv.Plotter()
    m = mocker.patch.object(pl.iren, 'interactor')
    m.GetInitialized.return_value = False

    with pytest.raises(
        RuntimeError,
        match='Render window interactor must be initialized before processing events.',
    ):
        pl.iren.process_events()


@pytest.mark.parametrize('picker', ['foo', 1000])
def test_picker_raises(picker, mocker: MockerFixture):
    pl = pv.Plotter()  # patching need to occur after init

    from pyvista.plotting import render_window_interactor

    m = mocker.patch.object(render_window_interactor.PickerType, 'from_any')
    m.return_value = (v := len(list(render_window_interactor.PickerType)))

    with pytest.raises(KeyError, match=re.escape(f'Picker class `{v}` is unknown.')):
        pl.iren.picker = picker

    m.assert_called_once_with(picker)


def test_observers():
    pl = pv.Plotter()

    # Key events
    with pytest.raises(TypeError):
        pl.add_key_event('w', 1)

    # Callback must not have any  empty arguments.
    def callback(a, b, *, c, d=1.0):
        pass

    with pytest.raises(TypeError):
        pl.add_key_event('w', callback)

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
        'PickEvent',
    ), 'Subsequent PickEvent HasObserver tests are wrong if this fails.'
    # Add different observers
    obs_move = pl.iren.add_observer(_vtk.vtkCommand.MouseMoveEvent, empty_callback)
    obs_double1 = pl.iren.add_observer(_vtk.vtkCommand.LeftButtonDoubleClickEvent, empty_callback)
    obs_double2 = pl.iren.add_observer('LeftButtonDoubleClickEvent', empty_callback)
    obs_picks = tuple(pl.iren.add_observer('PickEvent', empty_callback) for _ in range(5))
    pl.iren.add_observer('SelectionChangedEvent', empty_callback)
    assert pl.iren._observers[obs_move] == 'MouseMoveEvent'
    assert pl.iren.interactor.HasObserver('MouseMoveEvent')
    assert pl.iren._observers[obs_double1] == 'LeftButtonDoubleClickEvent'
    assert pl.iren._observers[obs_double2] == 'LeftButtonDoubleClickEvent'
    assert pl.iren.interactor.HasObserver('LeftButtonDoubleClickEvent')
    assert all(pl.iren._observers[obs_pick] == 'PickEvent' for obs_pick in obs_picks)
    assert pl.iren.interactor.HasObserver('SelectionChangedEvent')
    # Remove a specific observer
    pl.iren.remove_observer(obs_move)
    assert obs_move not in pl.iren._observers
    # Remove all observers of a specific event
    pl.iren.remove_observers(_vtk.vtkCommand.LeftButtonDoubleClickEvent)
    assert obs_double1 not in pl.iren._observers
    assert obs_double2 not in pl.iren._observers
    # Remove all (remaining) observers
    pl.iren.remove_observers()
    assert len(pl.iren._observers) == 0
    assert not pl.iren.interactor.HasObserver('PickEvent')


def test_clear_key_event_callbacks():
    pl = pv.Plotter()
    pl.reset_key_events()


@pytest.mark.skip_plotting
def test_track_mouse_position():
    pl = pv.Plotter()
    pl.track_mouse_position()
    pl.show(auto_close=False)
    assert pl.mouse_position is None
    x, y = 10, 20
    pl.iren._mouse_move(x, y)
    assert pl.mouse_position == (x, y)

    pl.iren.untrack_mouse_position()
    assert 'MouseMoveEvent' not in pl.iren._observers.values()


@pytest.mark.skip_plotting
def test_track_click_position_multi_render():
    points = []

    def callback(mouse_point):
        points.append(mouse_point)

    pl = pv.Plotter()
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


@pytest.mark.skip_plotting
def test_track_click_position():
    events = []

    def single_click_callback(mouse_position):  # noqa: ARG001
        events.append('single')

    def double_click_callback(mouse_position):  # noqa: ARG001
        events.append('double')

    pl = pv.Plotter()
    pl.track_click_position(callback=single_click_callback, side='left', double=False)
    pl.track_click_position(callback=double_click_callback, side='left', double=True)
    pl.show(auto_close=False)

    # Test single and double clicks:
    pl.iren._mouse_left_button_click(10, 10)
    assert len(events) == 1
    assert events.pop(0) == 'single'
    pl.iren._mouse_left_button_click(50, 50, count=2)
    assert len(events) == 2
    assert events.pop(1) == 'double'
    assert events.pop(0) == 'single'

    # Test triple click behaviour:
    pl.iren._mouse_left_button_click(10, 10, count=3)
    assert len(events) == 3
    assert events.pop(2) == 'single'
    assert events.pop(1) == 'double'
    assert events.pop(0) == 'single'


@skip_windows_mesa
@pytest.mark.skipif(
    type(_vtk.vtkRenderWindowInteractor()).__name__
    not in ('vtkWin32RenderWindowInteractor', 'vtkXRenderWindowInteractor'),
    reason='Other RenderWindowInteractors do not invoke TimerEvents during ProcessEvents.',
)
def test_timer():
    # Create a normal interactor from the offscreen plotter (not generic,
    # which is the default for offscreen rendering)
    pl = pv.Plotter()
    iren = pv.plotting.render_window_interactor.RenderWindowInteractor(pl)
    iren.set_render_window(pl.render_window)

    duration = 50  # Duration of created timers
    delay = 5 * duration  # Extra time we wait for the timers to fire at least once
    events = []

    def on_timer(obj, event):  # noqa: ARG001
        # TimerEvent callback
        events.append(event)

    def process_events(iren, duration):
        # Helper function to call process_events for the given duration (in milliseconds).
        t = 1000 * time.time()
        while 1000 * time.time() - t < duration:
            iren.process_events()

    # Setup interactor
    iren.add_observer('TimerEvent', on_timer)
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


def test_add_timer_event():
    sphere = pv.Sphere()

    pl = pv.Plotter()
    actor = pl.add_mesh(sphere)

    def callback(step):
        actor.position = [step / 100.0, step / 100.0, 0]

    pl.add_timer_event(max_steps=200, duration=500, callback=callback)

    cpos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    pl.show(cpos=cpos)


@pytest.mark.skip_plotting
def test_poked_subplot_loc():
    pl = pv.Plotter(shape=(2, 2), window_size=(800, 800))

    pl.iren._mouse_left_button_press(200, 600)
    assert tuple(pl.iren.get_event_subplot_loc()) == (0, 0)

    pl.iren._mouse_left_button_press(200, 200)
    assert tuple(pl.iren.get_event_subplot_loc()) == (1, 0)

    pl.iren._mouse_left_button_press(600, 600)
    assert tuple(pl.iren.get_event_subplot_loc()) == (0, 1)

    pl.iren._mouse_left_button_press(600, 200)
    assert tuple(pl.iren.get_event_subplot_loc()) == (1, 1)

    pl.close()


@pytest.mark.skip_plotting
@pytest.mark.usefixtures('verify_image_cache')
def test_poked_subplot_context():
    pl = pv.Plotter(shape=(2, 2), window_size=(800, 800))

    pl.iren._mouse_left_button_press(200, 600)
    with pl.iren.poked_subplot():
        pl.add_mesh(pv.Cone(), color=True)

    pl.iren._mouse_left_button_press(200, 200)
    with pl.iren.poked_subplot():
        pl.add_mesh(pv.Cube(), color=True)

    pl.iren._mouse_left_button_press(600, 600)
    with pl.iren.poked_subplot():
        pl.add_mesh(pv.Sphere(), color=True)

    pl.iren._mouse_left_button_press(600, 200)
    with pl.iren.poked_subplot():
        pl.add_mesh(pv.Arrow(), color=True)

    pl.show()


@pytest.mark.skip_plotting
def test_add_pick_observer():
    pl = pv.Plotter()
    with pytest.warns(PyVistaDeprecationWarning, match='`add_pick_obeserver` is deprecated'):
        pl.iren.add_pick_obeserver(empty_callback)
    pl = pv.Plotter()
    pl.iren.add_pick_observer(empty_callback)


@pytest.mark.parametrize('event', ['LeftButtonReleaseEvent', 'RightButtonReleaseEvent'])
def test_release_button_observers(event):
    class CallBack:
        def __init__(self):
            self._i = 0

        def __call__(self, *_):
            self._i += 1

    cb = CallBack()
    pl = pv.Plotter()
    pl.iren.add_observer(event, cb)

    pl.iren.interactor.GetInteractorStyle().InvokeEvent(event)
    assert cb._i == 1

    pl.iren.interactor.GetInteractorStyle().InvokeEvent(event)
    assert cb._i == 2


def test_enable_custom_trackball_style():
    pl = pv.Plotter()
    pl.enable_custom_trackball_style()
    pl.close()

    pl = pv.Plotter()
    with pytest.raises(ValueError, match="Action 'not an option' not in the allowed"):
        pl.enable_custom_trackball_style(left='not an option')


def test_enable_2d_style():
    pl = pv.Plotter()
    pl.enable_2d_style()


def test_enable_interactors():
    mapping = {
        'enable_trackball_style': InteractorStyleTrackballCamera,
        'enable_custom_trackball_style': InteractorStyleTrackballCamera,
        'enable_2d_style': InteractorStyleTrackballCamera,
        'enable_trackball_actor_style': InteractorStyleTrackballActor,
        'enable_image_style': InteractorStyleImage,
        'enable_joystick_style': InteractorStyleJoystickCamera,
        'enable_joystick_actor_style': InteractorStyleJoystickActor,
        'enable_zoom_style': InteractorStyleZoom,
        'enable_terrain_style': InteractorStyleTerrain,
        'enable_rubber_band_style': InteractorStyleRubberBandPick,
        'enable_rubber_band_2d_style': InteractorStyleRubberBand2D,
    }

    pl = pv.Plotter()

    # check that all "enable_*_style" methods on plotter are in the mapping and vice versa
    attrs = dir(pl)
    attrs_enable_style = {
        attr for attr in attrs if attr.startswith('enable_') and attr.endswith('_style')
    }

    check_set = set(mapping.keys())
    assert attrs_enable_style == check_set

    # do the same for methods on the RenderWindowInteractor
    attrs = dir(pl.iren)
    attrs_enable_style = {
        attr for attr in attrs if attr.startswith('enable_') and attr.endswith('_style')
    }
    check_set = set(mapping.keys())
    assert attrs_enable_style == check_set

    # check that the method gives the right class
    for attr, class_ in mapping.items():
        print(attr, class_)
        getattr(pl, attr)()
        assert isinstance(pl.iren.style, class_)

    for attr, class_ in mapping.items():
        getattr(pl.iren, attr)()
        assert isinstance(pl.iren.style, class_)


def test_setting_custom_style():
    pl = pv.Plotter()
    pl.iren.style = _vtk.vtkInteractorStyleJoystickActor()
    assert isinstance(pl.iren.style, _vtk.vtkInteractorStyleJoystickActor)
