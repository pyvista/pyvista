"""Wrap vtk.vtkRenderWindowInteractor."""
import collections.abc
from functools import partial
import logging
import weakref

from pyvista import _vtk
from pyvista.utilities import try_callback

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

        # Toggle interaction style when clicked on a visible chart (to
        # enable interaction with visible charts)
        # Current trigger is right click, alternatively double click triggers could be used:
        # https://kitware.github.io/vtk-examples/site/Cxx/Interaction/DoubleClick/
        self._context_style = _vtk.vtkContextInteractorStyle()
        self.add_observer("RightButtonPressEvent", self._toggle_context_style)

    def add_key_event(self, key, callback):
        """Add a function to callback when the given key is pressed.

        These are non-unique - thus a key could map to many callback
        functions. The callback function must not have any arguments.

        Parameters
        ----------
        key : str
            The key to trigger the event.

        callback : callable
            A callable that takes no arguments.

        """
        if not callable(callback):
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

        viewport : bool, optional
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
            if callable(callback):
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
        self.interactor.SetInteractorStyle(self._style_class)

    def _toggle_context_style(self, obj, event):
        mouse_pos = self.get_event_position()
        scene = None
        for renderer in self._plotter.renderers:
            if scene is None and renderer.IsInViewport(*mouse_pos):
                scene = renderer._charts.toggle_interaction(mouse_pos)
            else:
                # Not in viewport or already an active chart found (in case they overlap), so disable interaction
                renderer.charts.toggle_interaction(False)

        self._context_style.SetScene(scene)  # Set scene to interact with or reset it to stop interaction (otherwise crash)
        if scene is None and self._style == "Context":
            # Switch back to previous interactor style
            self._style = self._prev_style
            self._style_class = self._prev_style_class
            self._prev_style = None
            self._prev_style_class = None
        elif scene is not None and self._style != "Context":
            # Enable context interactor style
            self._prev_style = self._style
            self._prev_style_class = self._style_class
            self._style = "Context"
            self._style_class = self._context_style
        self.update_style()

    def enable_trackball_style(self):
        """Set the interactive style to Trackball Camera.

        The trackball camera is the default interactor style. Moving
        the mouse moves the camera around, leaving the scene intact.

        For a 3-button mouse, the left button is for rotation, the
        right button for zooming, the middle button for panning, and
        ctrl + left button for spinning the view around the vewing
        axis of the camera.  Alternatively, ctrl + shift + left button
        or mouse wheel zooms, and shift + left button pans.

        Examples
        --------
        Create a simple scene with a plotter that has the Trackball
        Camera interactive style (which is also the default):

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_trackball_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'TrackballCamera'
        self._style_class = None
        self.update_style()

    def enable_trackball_actor_style(self):
        """Set the interactive style to Trackball Actor.

        This allows to rotate actors around the scene. The controls
        are similar to the default Trackball Camera style, but
        movements transform specific objects under the mouse cursor.

        For a 3-button mouse, the left button is for rotation, the
        right button for zooming, the middle button for panning, and
        ctrl + left button for spinning objects around the axis
        connecting the camera with the their center.  Alternatively,
        shift + left button pans.

        Examples
        --------
        Create a simple scene with a plotter that has the Trackball
        Actor interactive style:

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_trackball_actor_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'TrackballActor'
        self._style_class = None
        self.update_style()

    def enable_image_style(self):
        """Set the interactive style to Image.

        Controls:
         - Left Mouse button triggers window level events
         - CTRL Left Mouse spins the camera around its view plane normal
         - SHIFT Left Mouse pans the camera
         - CTRL SHIFT Left Mouse dollies (a positional zoom) the camera
         - Middle mouse button pans the camera
         - Right mouse button dollies the camera
         - SHIFT Right Mouse triggers pick events

        Examples
        --------
        Create a simple scene with a plotter that has the Image
        interactive style:

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_image_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'Image'
        self._style_class = None
        self.update_style()

    def enable_joystick_style(self):
        """Set the interactive style to Joystick Camera.

        It allows the user to move (rotate, pan, etc.) the camera, the
        point of view for the scene.  The position of the mouse
        relative to the center of the scene determines the speed at
        which the camera moves, so the camera continues to move even
        if the mouse if not moving.

        For a 3-button mouse, the left button is for rotation, the
        right button for zooming, the middle button for panning, and
        ctrl + left button for spinning.  (With fewer mouse buttons,
        ctrl + shift + left button is for zooming, and shift + left
        button is for panning.)

        Examples
        --------
        Create a simple scene with a plotter that has the Joystick
        Camera interactive style:

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_joystick_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'JoystickCamera'
        self._style_class = None
        self.update_style()

    def enable_joystick_actor_style(self):
        """Set the interactive style to Joystick Actor.

        Similar to the Joystick Camera interaction style, however
        in case of the Joystick Actor style the objects in the scene
        rather than the camera can be moved (rotated, panned, etc.).
        The position of the mouse relative to the center of the object
        determines the speed at which the object moves, so the object
        continues to move even if the mouse is not moving.

        For a 3-button mouse, the left button is for rotation, the
        right button for zooming, the middle button for panning, and
        ctrl + left button for spinning.  (With fewer mouse buttons,
        ctrl + shift + left button is for zooming, and shift + left
        button is for panning.)

        Examples
        --------
        Create a simple scene with a plotter that has the Joystick
        Actor interactive style:

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_joystick_actor_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'JoystickActor'
        self._style_class = None
        self.update_style()

    def enable_zoom_style(self):
        """Set the interactive style to Rubber Band Zoom.

        This interactor style allows the user to draw a rectangle in
        the render window using the left mouse button.  When the mouse
        button is released, the current camera zooms by an amount
        determined from the shorter side of the drawn rectangle.

        Examples
        --------
        Create a simple scene with a plotter that has the Rubber Band
        Zoom interactive style:

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_zoom_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'RubberBandZoom'
        self._style_class = None
        self.update_style()

    def enable_terrain_style(self, mouse_wheel_zooms=False, shift_pans=False):
        """Set the interactive style to Terrain.

        Used to manipulate a camera which is viewing a scene with a
        natural view up, e.g., terrain. The camera in such a scene is
        manipulated by specifying azimuth (angle around the view up
        vector) and elevation (the angle from the horizon). Similar to
        the default Trackball Camera style and in contrast to the
        Joystick Camera style, movements of the mouse translate to
        movements of the camera.

        Left mouse click rotates the camera around the focal point
        using both elevation and azimuth invocations on the camera.
        Left mouse motion in the horizontal direction results in
        azimuth motion; left mouse motion in the vertical direction
        results in elevation motion. Therefore, diagonal motion results
        in a combination of azimuth and elevation. (If the shift key is
        held during motion, then only one of elevation or azimuth is
        invoked, depending on the whether the mouse motion is primarily
        horizontal or vertical.) Middle mouse button pans the camera
        across the scene (again the shift key has a similar effect on
        limiting the motion to the vertical or horizontal direction.
        The right mouse is used to dolly towards or away from the focal
        point (zoom in or out). Panning and zooming behavior can be
        overridden to match the Trackball Camera style.

        The class also supports some keypress events. The ``r`` key
        resets the camera. The ``e`` key invokes the exit callback
        and closes the plotter. The ``f`` key sets a new
        camera focal point and flies towards that point. The ``u``
        key invokes the user event. The ``3`` key toggles between
        stereo and non-stero mode. The ``l`` key toggles on/off
        latitude/longitude markers that can be used to estimate/control
        position.

        Parameters
        ----------
        mouse_wheel_zooms : bool, optional
            Whether to use the mouse wheel for zooming. By default
            zooming can be performed with right click and drag.

        shift_pans : bool, optional
            Whether shift + left mouse button pans the scene. By default
            shift + left mouse button rotates the view restricted to
            only horizontal or vertical movements, and panning is done
            holding down the middle mouse button.

        Examples
        --------
        Create a simple scene with a plotter that has the Terrain
        interactive style:

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_terrain_style()
        >>> plotter.show()  # doctest:+SKIP

        Use controls that are closer to the default style:

        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_terrain_style(mouse_wheel_zooms=True,
        ...                              shift_pans=True)
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'Terrain'
        self._style_class = None
        self.update_style()

        if mouse_wheel_zooms:
            def wheel_zoom_callback(obj, event):  # pragma: no cover
                """Zoom in or out on mouse wheel roll."""
                if event == 'MouseWheelForwardEvent':
                    # zoom in
                    zoom_factor = 1.1
                elif event == 'MouseWheelBackwardEvent':
                    # zoom out
                    zoom_factor = 1/1.1
                self._plotter.camera.zoom(zoom_factor)
                self._plotter.render()
            callback = partial(try_callback, wheel_zoom_callback)

            for event in 'MouseWheelForwardEvent', 'MouseWheelBackwardEvent':
                self._style_class.AddObserver(event, callback)

        if shift_pans:
            def pan_on_shift_callback(obj, event):  # pragma: no cover
                """Trigger left mouse panning if shift is pressed."""
                if event == 'LeftButtonPressEvent':
                    if self.interactor.GetShiftKey():
                        self._style_class.StartPan()
                    self._style_class.OnLeftButtonDown()
                elif event == 'LeftButtonReleaseEvent':
                    # always stop panning on release
                    self._style_class.EndPan()
                    self._style_class.OnLeftButtonUp()
            callback = partial(try_callback, pan_on_shift_callback)

            for event in 'LeftButtonPressEvent', 'LeftButtonReleaseEvent':
                self._style_class.AddObserver(event, callback)

    def enable_rubber_band_style(self):
        """Set the interactive style to Rubber Band Picking.

        This interactor style allows the user to draw a rectangle in
        the render window by hitting ``r`` and then using the left
        mouse button. When the mouse button is released, the attached
        picker operates on the pixel in the center of the selection
        rectangle. If the picker happens to be a ``vtkAreaPicker``
        it will operate on the entire selection rectangle. When the
        ``p`` key is hit the above pick operation occurs on a 1x1
        rectangle. In other respects it behaves the same as the
        Trackball Camera style.

        Examples
        --------
        Create a simple scene with a plotter that has the Rubber Band
        Pick interactive style:

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_rubber_band_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'RubberBandPick'
        self._style_class = None
        self.update_style()

    def enable_rubber_band_2d_style(self):
        """Set the interactive style to Rubber Band 2D.

        Camera rotation is not enabled with this interactor
        style. Zooming affects the camera's parallel scale only, and
        assumes that the camera is in parallel projection mode. The
        style also allows to draw a rubber band using the left mouse
        button. All camera changes invoke ``StartInteractionEvent`` when
        the button is pressed, ``InteractionEvent`` when the mouse (or
        wheel) is moved, and ``EndInteractionEvent`` when the button is
        released. The bindings are as follows:

          * Left mouse: Select (invokes a ``SelectionChangedEvent``).
          * Right mouse: Zoom.
          * Middle mouse: Pan.
          * Scroll wheel: Zoom.

        Examples
        --------
        Create a simple scene with a plotter that has the Rubber Band
        2D interactive style:

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_rubber_band_2d_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self._style = 'RubberBand2D'
        self._style_class = None
        self.update_style()

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
        """Get the event position.

        Returns
        -------
        tuple
            The ``(x, y)`` coordinate position.

        """
        return self.interactor.GetEventPosition()

    def get_interactor_style(self):
        """Get the interactor style.

        Returns
        -------
        vtk.vtkInteractorStyle
            VTK interactor style.
        """
        return self.interactor.GetInteractorStyle()

    def get_desired_update_rate(self):
        """Get the desired update rate.

        Returns
        -------
        float
            Desired update rate.
        """
        return self.interactor.GetDesiredUpdateRate()

    def create_repeating_timer(self, stime):
        """Create a repeating timer.

        Returns
        -------
        int
            Timer ID.
        """
        timer_id = self.interactor.CreateRepeatingTimer(stime)
        if hasattr(self.interactor, 'ProcessEvents'):
            self.process_events()
        else:
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
        """Get the picker.

        Returns
        -------
        vtk.vtkAbstractPicker
            VTK picker.
        """
        return self.interactor.GetPicker()

    def set_picker(self, picker):
        """Set the picker."""
        self.interactor.SetPicker(picker)

    def fly_to(self, renderer, point):
        """Fly to the given point."""
        self.interactor.FlyTo(renderer, *point)

    def terminate_app(self):
        """Terminate the app."""
        if self.initialized:

            # #################################################################
            # 9.0.2+ compatibility:
            # See: https://gitlab.kitware.com/vtk/vtk/-/issues/18242
            if hasattr(self.interactor, 'GetDone'):
                self.interactor.SetDone(True)
            # #################################################################

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
