"""Wrap vtk.vtkRenderWindowInteractor."""
import weakref
import logging
import collections.abc
from functools import partial

from pyvista.utilities import try_callback

from pyvista import _vtk

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')
log.addHandler(logging.StreamHandler())


class RenderWindowInteractor():
    """Wrap vtk.vtkRenderWindowInteractor.

    This class has been added for the purpose of making some methods
    we add to the RenderWindowInteractor more python, like certain
    testing methods.

    """

    def __init__(self, plotter, desired_update_rate=30, light_follow_camera=True,
                 interactor=None):
        """Initialize."""
        if interactor is None:
            interactor = _vtk.vtkRenderWindowInteractor()
        self.interactor = interactor
        self.interactor.SetDesiredUpdateRate(desired_update_rate)
        if not light_follow_camera:
            self.interactor.LightFollowCameraOff()

        # Map of events to observers
        self._observers = {}
        self._key_press_event_callbacks = collections.defaultdict(list)

        # Set default style
        self._style = 'RubberBandPick'
        self._style_class = None
        self._plotter = plotter
        self._click_observer = None

    def add_key_event(self, key, callback):
        """Add a function to callback when the given key is pressed.

        These are non-unique - thus a key could map to many callback
        functions. The callback function must not have any arguments.

        Parameters
        ----------
        key : str
            The key to trigger the event

        callback : callable
            A callable that takes no arguments

        """
        if not hasattr(callback, '__call__'):
            raise TypeError('callback must be callable.')
        self._key_press_event_callbacks[key].append(callback)

    def add_observer(self, event, call):
        """Add an observer."""
        call = partial(try_callback, call)
        self._observers[event] = self.interactor.AddObserver(event, call)

    def remove_observer(self, event):
        """Remove an observer."""
        if event in self._observers:
            self.interactor.RemoveObserver(event)
            del self._observers[event]

    def remove_observers(self):
        """Remove all observers."""
        for obs in list(self._observers.values()):
            self.remove_observer(obs)

    def clear_events_for_key(self, key):
        """Remove the callbacks associated to the key."""
        self._key_press_event_callbacks.pop(key)

    def track_mouse_position(self, callback):
        """Keep track of the mouse position.

        This will potentially slow down the interactor. No callbacks supported
        here - use :func:`pyvista.BasePlotter.track_click_position` instead.

        """
        self.add_observer(_vtk.vtkCommand.MouseMoveEvent, callback)

    def untrack_mouse_position(self):
        """Stop tracking the mouse position."""
        self.remove_observer(_vtk.vtkCommand.MouseMoveEvent)

    def track_click_position(self, callback=None, side="right",
                             viewport=False):
        """Keep track of the click position.

        By default, it only tracks right clicks.

        Parameters
        ----------
        callback : callable, optional
            A callable method that will use the click position. Passes
            the click position as a length two tuple.

        side : str, optional
            The side of the mouse for the button to track (left or
            right).  Default is left. Also accepts ``'r'`` or ``'l'``.

        viewport: bool, optional
            If ``True``, uses the normalized viewport coordinate
            system (values between 0.0 and 1.0 and support for HiDPI)
            when passing the click position to the callback.

        """
        side = str(side).lower()
        if side in ["right", "r"]:
            event = _vtk.vtkCommand.RightButtonPressEvent
        elif side in ["left", "l"]:
            event = _vtk.vtkCommand.LeftButtonPressEvent
        else:
            raise TypeError(f"Side ({side}) not supported. Try `left` or `right`")

        def _click_callback(obj, event):
            self._plotter.store_click_position()
            if hasattr(callback, '__call__'):
                if viewport:
                    callback(self._plotter.click_position)
                else:
                    callback(self._plotter.pick_click_position())

        self._click_observer = event
        self.add_observer(event, _click_callback)

    def untrack_click_position(self):
        """Stop tracking the click position."""
        self.remove_observer(self._click_observer)
        self._click_observer = None

    def clear_key_event_callbacks(self):
        """Clear key event callbacks."""
        self._key_press_event_callbacks.clear()

    def key_press_event(self, obj, event):
        """Listen for key press event."""
        key = self.interactor.GetKeySym()
        log.debug(f'Key {key} pressed')
        self._last_key = key
        if key in self._key_press_event_callbacks.keys():
            # Note that defaultdict's will never throw a key error
            callbacks = self._key_press_event_callbacks[key]
            for func in callbacks:
                func()

    def update_style(self):
        """Update the camera interactor style."""
        if self._style_class is None:
            # We need an actually custom style to handle button up events
            self._style_class = _style_factory(self._style)(self)
        return self.interactor.SetInteractorStyle(self._style_class)

    def enable_trackball_style(self):
        """Set the interactive style to trackball camera.

        The trackball camera is the default interactor style.

        """
        self._style = 'TrackballCamera'
        self._style_class = None
        return self.update_style()

    def enable_trackball_actor_style(self):
        """Set the interactive style to trackball actor.

        This allows to rotate actors around the scene.

        """
        self._style = 'TrackballActor'
        self._style_class = None
        return self.update_style()

    def enable_image_style(self):
        """Set the interactive style to image.

        Controls:
         - Left Mouse button triggers window level events
         - CTRL Left Mouse spins the camera around its view plane normal
         - SHIFT Left Mouse pans the camera
         - CTRL SHIFT Left Mouse dollys (a positional zoom) the camera
         - Middle mouse button pans the camera
         - Right mouse button dollys the camera.
         - SHIFT Right Mouse triggers pick events

        """
        self._style = 'Image'
        self._style_class = None
        return self.update_style()

    def enable_joystick_style(self):
        """Set the interactive style to joystick.

        It allows the user to move (rotate, pan, etc.) the camera, the
        point of view for the scene.  The position of the mouse
        relative to the center of the scene determines the speed at
        which the camera moves, and the speed of the mouse movement
        determines the acceleration of the camera, so the camera
        continues to move even if the mouse if not moving.

        For a 3-button mouse, the left button is for rotation, the
        right button for zooming, the middle button for panning, and
        ctrl + left button for spinning.  (With fewer mouse buttons,
        ctrl + shift + left button is for zooming, and shift + left
        button is for panning.)

        """
        self._style = 'JoystickCamera'
        self._style_class = None
        return self.update_style()

    def enable_zoom_style(self):
        """Set the interactive style to rubber band zoom.

        This interactor style allows the user to draw a rectangle in
        the render window using the left mouse button.  When the mouse
        button is released, the current camera zooms by an amount
        determined from the shorter side of the drawn rectangle.

        """
        self._style = 'RubberBandZoom'
        self._style_class = None
        return self.update_style()

    def enable_terrain_style(self):
        """Set the interactive style to terrain.

        Used to manipulate a camera which is viewing a scene with a natural
        view up, e.g., terrain. The camera in such a scene is manipulated by
        specifying azimuth (angle around the view up vector) and elevation
        (the angle from the horizon).

        """
        self._style = 'Terrain'
        self._style_class = None
        return self.update_style()

    def enable_rubber_band_style(self):
        """Set the interactive style to rubber band picking.

        This interactor style allows the user to draw a rectangle in the render
        window by hitting 'r' and then using the left mouse button.
        When the mouse button is released, the attached picker operates on the
        pixel in the center of the selection rectangle. If the picker happens to
        be a vtkAreaPicker it will operate on the entire selection rectangle.
        When the 'p' key is hit the above pick operation occurs on a 1x1
        rectangle. In other respects it behaves the same as its parent class.

        """
        self._style = 'RubberBandPick'
        self._style_class = None
        return self.update_style()

    def enable_rubber_band_2d_style(self):
        """Set the interactive style to rubber band 2d.

        Camera rotation is not allowed with this interactor
        style. Zooming affects the camera's parallel scale only, and
        assumes that the camera is in parallel projection mode. The
        style also allows draws a rubber band using the left
        button. All camera changes invoke StartInteractionEvent when
        the button is pressed, InteractionEvent when the mouse (or
        wheel) is moved, and EndInteractionEvent when the button is
        released. The bindings are as follows: Left mouse - Select
        (invokes a SelectionChangedEvent). Right mouse - Zoom.  Middle
        mouse - Pan. Scroll wheel - Zoom.

        """
        self._style = 'RubberBand2D'
        self._style_class = None
        return self.update_style()

    def _simulate_keypress(self, key):  # pragma: no cover
        """Simulate a keypress."""
        if len(key) > 1:
            raise ValueError('Only accepts a single key')
        self.interactor.SetKeyCode(key)
        self.interactor.CharEvent()

    def _mouse_left_button_press(self, x=None, y=None):  # pragma: no cover
        """Simulate a left mouse button press.

        If ``x`` and ``y`` are entered then simulates a movement to
        that position.

        """
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.LeftButtonPressEvent()

    def _mouse_left_button_release(self, x=None, y=None):  # pragma: no cover
        """Simulate a left mouse button release."""
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.LeftButtonReleaseEvent()

    def _mouse_right_button_press(self, x=None, y=None):  # pragma: no cover
        """Simulate a right mouse button press.

        If ``x`` and ``y`` are entered then simulates a movement to
        that position.

        """
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.RightButtonPressEvent()

    def _mouse_right_button_release(self, x=None, y=None):  # pragma: no cover
        """Simulate a right mouse button release."""
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.RightButtonReleaseEvent()

    def _mouse_move(self, x, y):  # pragma: no cover
        """Simulate moving the mouse to ``(x, y)`` screen coordinates."""
        self.interactor.SetEventInformation(x, y)
        self.interactor.MouseMoveEvent()

    def get_event_position(self):
        """Get the event position."""
        return self.interactor.GetEventPosition()

    def get_interactor_style(self):
        """Get the interactor style."""
        return self.interactor.GetInteractorStyle()

    def get_desired_update_rate(self):
        """Get the desired update rate."""
        return self.interactor.GetDesiredUpdateRate()

    def create_repeating_timer(self, stime):
        """Create a repeating timer."""
        timer_id = self.interactor.CreateRepeatingTimer(stime)
        self.interactor.Start()
        self.interactor.DestroyTimer(timer_id)
        return timer_id

    def start(self):
        """Start interactions."""
        self.interactor.Start()

    def initialize(self):
        """Initialize the interactor."""
        self.interactor.Initialize()

    def set_render_window(self, ren_win):
        """Set the render window."""
        self.interactor.SetRenderWindow(ren_win)

    def process_events(self):
        """Process events."""
        # Note: This is only available in VTK 9+
        if not self.initialized:
            raise RuntimeError('Render window interactor must be initialized '
                               'before processing events.')
        self.interactor.ProcessEvents()

    @property
    def initialized(self):
        """Return if the interactor has been initialized."""
        return self.interactor.GetInitialized()

    def get_picker(self):
        """Get the piccker."""
        return self.interactor.GetPicker()

    def set_picker(self, picker):
        """Set the picker."""
        self.interactor.SetPicker(picker)

    def fly_to(self, renderer, point):
        """Fly to the given point."""
        self.interactor.FlyTo(renderer, *point)

    def terminate_app(self):
        """Terminate the app."""
        self.interactor.TerminateApp()


def _style_factory(klass):
    """Create a subclass with capturing ability, return it."""
    # We have to use a custom subclass for this because the default ones
    # swallow the release events
    # http://vtk.1045678.n5.nabble.com/Mouse-button-release-event-is-still-broken-in-VTK-6-0-0-td5724762.html  # noqa

    try:
        from vtkmodules import vtkInteractionStyle
    except ImportError:  # pragma: no cover
        import vtk as vtkInteractionStyle

    class CustomStyle(getattr(vtkInteractionStyle, 'vtkInteractorStyle' + klass)):

        def __init__(self, parent):
            super().__init__()
            self._parent = weakref.ref(parent)
            self.AddObserver(
                "LeftButtonPressEvent",
                partial(try_callback, self._press))
            self.AddObserver(
                "LeftButtonReleaseEvent",
                partial(try_callback, self._release))

        def _press(self, obj, event):
            # Figure out which renderer has the event and disable the
            # others
            super().OnLeftButtonDown()
            parent = self._parent()
            if len(parent._plotter.renderers) > 1:
                click_pos = parent.get_event_position()
                for renderer in parent._plotter.renderers:
                    interact = renderer.IsInViewport(*click_pos)
                    renderer.SetInteractive(interact)

        def _release(self, obj, event):
            super().OnLeftButtonUp()
            parent = self._parent()
            if len(parent._plotter.renderers) > 1:
                for renderer in parent._plotter.renderers:
                    renderer.SetInteractive(True)

    return CustomStyle
