"""Wrap :vtk:`vtkRenderWindowInteractor`."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from inspect import signature
import logging
import time
from typing import Literal
import warnings
import weakref

import numpy as np

from pyvista import vtk_version_info
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core._vtk_core import DisableVtkSnakeCase
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.misc import try_callback

from . import _vtk
from .opts import PickerType

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')
log.addHandler(logging.StreamHandler())


class Timer(_NoNewAttrMixin):
    """Timer class.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps to allow for the timer before destroying it.

    callback : callable
        A callable that takes one argument. It will be passed `step`,
        which is the number of times the timer event has occurred.

    """

    def __init__(self, max_steps, callback):
        """Initialize."""
        self.step = 0
        self.max_steps = max_steps
        self.id = None
        self.callback = callback

    def execute(self, obj, _event):  # pragma: no cover # numpydoc ignore=PR01,RT01
        """Execute Timer."""
        # https://github.com/pyvista/pyvista/pull/5618
        iren = obj

        if self.step < self.max_steps:
            self.callback(self.step)
            iren.GetRenderWindow().Render()
            self.step += 1
        elif self.id:
            iren.DestroyTimer(self.id)


class RenderWindowInteractor(_NoNewAttrMixin):
    """Wrap :vtk:`vtkRenderWindowInteractor`.

    This class has been added for the purpose of making some methods
    we add to the RenderWindowInteractor more python, like certain
    testing methods.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter object upon which the initialization of
        RenderWindowInteractor is based.

    desired_update_rate : float, default: 30
        The desired update rate of the interactor.

    light_follow_camera : bool, default: True
        If set to ``True``, the light follows the camera.

    interactor : :vtk:`vtkRenderWindowInteractor`, default: None
        The render window interactor. If set to ``None``, a new
        :vtk:`vtkRenderWindowInteractor` instance will be created.

    """

    @_deprecate_positional_args(allowed=['plotter'])
    def __init__(  # noqa: PLR0917
        self,
        plotter,
        desired_update_rate=30,
        light_follow_camera=True,  # noqa: FBT002
        interactor=None,
    ):
        """Initialize."""
        if interactor is None:
            interactor = _vtk.vtkRenderWindowInteractor()
        self.interactor = interactor
        self.interactor.SetDesiredUpdateRate(desired_update_rate)
        if not light_follow_camera:
            self.interactor.LightFollowCameraOff()

        # Map of observers to events
        self._observers = {}
        self._last_key: str | None = None
        self._key_press_event_callbacks = defaultdict(list)
        self._click_event_callbacks = {  # type: ignore[var-annotated]
            event: {(double, v): [] for double in (False, True) for v in (False, True)}
            for event in ('LeftButtonPressEvent', 'RightButtonPressEvent')
        }
        self._timer = None
        self._timer_event = None
        self._click_time = 0
        self._MAX_CLICK_DELAY = 0.8  # seconds
        self._MAX_CLICK_DELTA = 40  # squared => ~6 pixels

        # Set default style
        self._style_class: _vtk.vtkInteractorStyle | None = None
        self._style: Literal['Interactor', 'Context'] | None = 'Interactor'
        self._prev_style_class: _vtk.vtkInteractorStyle | None = self._style_class
        self._prev_style: Literal['Interactor', 'Context'] | None = self._style
        self.style = InteractorStyleRubberBandPick(self)
        self.__plotter = weakref.ref(plotter)

        # Toggle interaction style when clicked on a visible chart (to
        # enable interaction with visible charts)
        self._context_style = _vtk.vtkContextInteractorStyle()
        self.track_click_position(
            self._toggle_chart_interaction,
            side='left',
            double=True,
            viewport=True,
        )

        self.reset_picker()
        self.picker = PickerType.POINT

    @property
    def _plotter(self):
        """Return the plotter."""
        return self.__plotter()

    def add_key_event(self, key, callback):
        """Add a function to callback when the given key is pressed.

        These are non-unique - thus a key could map to many callback
        functions. The callback function must not have any arguments.

        Parameters
        ----------
        key : str
            The key to trigger the event.

        callback : callable
            A callable that takes no arguments (keyword arguments are allowed).

        """
        if not callable(callback):
            msg = 'callback must be callable.'
            raise TypeError(msg)
        for param in signature(callback).parameters.values():
            if param.default is param.empty:
                msg = '`callback` must not have any arguments without default values.'
                raise TypeError(msg)
        self._key_press_event_callbacks[key].append(callback)

    def add_timer_event(self, max_steps, duration, callback):
        """Add a function to callback as timer event.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps for integrating a timer.

        duration : int
            Time (in milliseconds) before the timer emits a TimerEvent and
            ``callback`` is called.

        callback : callable
            A callable that takes one argument. It will be passed
            `step`, which is the number of times the timer event has occurred.

        See Also
        --------
        :ref:`animation_example`

        Examples
        --------
        Add a timer to a Plotter to move a sphere across a scene.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(sphere)
        >>> def callback(step):
        ...     actor.position = [step / 100.0, step / 100.0, 0]
        >>> pl.add_timer_event(max_steps=200, duration=500, callback=callback)

        """
        self._timer = Timer(max_steps, callback)
        self.add_observer('TimerEvent', self._timer.execute)
        self._timer.id = self.create_timer(duration)

    @staticmethod
    def _get_event_str(event):
        if isinstance(event, str):
            # Make sure we pass it at least once through these functions, such that
            # invalid event names are mapped to "NoEvent".
            event = _vtk.vtkCommand.GetEventIdFromString(event)
        return _vtk.vtkCommand.GetStringFromEventId(event)

    @_deprecate_positional_args(allowed=['event', 'call'])
    def add_observer(self, event, call, interactor_style_fallback=True):  # noqa: FBT002
        """Add an observer for the given event.

        Parameters
        ----------
        event : str | int
            The event to observe. Either the name of this event (string) or
            a VTK event identifier (int).

        call : callable
            Callback to be called when the event is invoked.

        interactor_style_fallback : bool
            If ``True``, the observer will be added to the interactor style
            in cases known to be problematic.

        Returns
        -------
        int
            The identifier of the added observer.

        Examples
        --------
        Add a custom observer.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> obs_enter = pl.iren.add_observer('EnterEvent', lambda *_: print('Enter!'))

        """
        call = partial(try_callback, call)
        event = self._get_event_str(event)

        if (
            isinstance(self.style, InteractorStyleCaptureMixin)
            and interactor_style_fallback
            and event
            in [
                'LeftButtonReleaseEvent',
                'RightButtonReleaseEvent',
            ]
        ):
            # Release events are swallowed by the interactor, but registering
            # on the interactor style seems to work.
            # See https://github.com/pyvista/pyvista/issues/4976
            observer = self.style.add_observer(event, call)
        else:
            observer = self.interactor.AddObserver(event, call)
            self._observers[observer] = event
        return observer

    def remove_observer(self, observer):
        """Remove an observer.

        Parameters
        ----------
        observer : int
            The identifier of the observer to remove.

        Examples
        --------
        Add an observer and immediately remove it.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> obs_enter = pl.iren.add_observer('EnterEvent', lambda *_: print('Enter!'))
        >>> pl.iren.remove_observer(obs_enter)

        """
        if observer in self._observers:
            self.interactor.RemoveObserver(observer)
            del self._observers[observer]

    def remove_observers(self, event=None):
        """Remove all observers.

        Parameters
        ----------
        event : str | int, optional
            If provided, only removes observers of the given event. Otherwise,
            if it is ``None``, removes all observers.

        Examples
        --------
        Add two observers and immediately remove them.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> obs_enter = pl.iren.add_observer('EnterEvent', lambda *_: print('Enter!'))
        >>> obs_leave = pl.iren.add_observer('LeaveEvent', lambda *_: print('Leave!'))
        >>> pl.iren.remove_observers()

        """
        if event is None:
            observers = list(self._observers.keys())
        else:
            event = self._get_event_str(event)
            observers = [obs for obs, ev in self._observers.items() if event == ev]
        for observer in observers:
            self.remove_observer(observer)

    @_deprecate_positional_args(allowed=['key'])
    def clear_events_for_key(self, key, raise_on_missing=False):  # noqa: FBT002
        """Remove the callbacks associated to the key.

        Parameters
        ----------
        key : str
            Key to clear events for.

        raise_on_missing : bool, default: False
            Whether to raise a :class:`ValueError` if there are no events
            registered for the given key.

        """
        try:
            self._key_press_event_callbacks.pop(key)
        except KeyError:
            if raise_on_missing:
                msg = f'No events found for key {key!r}.'
                raise ValueError(msg) from None

    def track_mouse_position(self, callback):
        """Keep track of the mouse position.

        This will potentially slow down the interactor. No callbacks supported
        here - use :func:`pyvista.Plotter.track_click_position` instead.

        Parameters
        ----------
        callback : callable
            A function to call back when the mouse moves. This function will be
            passed the current mouse position.

        """
        self.add_observer(_vtk.vtkCommand.MouseMoveEvent, callback)

    def untrack_mouse_position(self):
        """Stop tracking the mouse position."""
        self.remove_observers(_vtk.vtkCommand.MouseMoveEvent)

    @staticmethod
    def _get_click_event(side) -> str:
        side = str(side).lower()
        if side in ['right', 'r']:
            return 'RightButtonPressEvent'
        elif side in ['left', 'l']:
            return 'LeftButtonPressEvent'
        else:
            msg = f'Side ({side}) not supported. Try `left` or `right`.'
            raise TypeError(msg)

    def _click_event(self, _obj, event):
        t = time.time()
        dt = t - self._click_time
        last_pos = self._plotter.click_position or (0, 0)

        self._plotter.store_click_position()
        dp = (self._plotter.click_position[0] - last_pos[0]) ** 2
        dp += (self._plotter.click_position[1] - last_pos[1]) ** 2
        double = dp < self._MAX_CLICK_DELTA and dt < self._MAX_CLICK_DELAY
        # Reset click time in case of a double click, otherwise a subsequent third click
        # is considered to be a double click as well.
        self._click_time = 0 if double else t  # type: ignore[assignment]

        for callback in self._click_event_callbacks[event][double, False]:
            callback(self._plotter.pick_click_position())
        for callback in self._click_event_callbacks[event][double, True]:
            callback(self._plotter.click_position)

    @_deprecate_positional_args(allowed=['callback', 'side'])
    def track_click_position(  # noqa: PLR0917,
        self,
        callback=None,
        side='right',
        double=False,  # noqa: FBT002
        viewport=False,  # noqa: FBT002
    ):
        """Keep track of the click position.

        By default, it only tracks right clicks.

        Parameters
        ----------
        callback : callable, optional
            A callable method that will use the click position. Passes
            the click position as a length two tuple.

        side : str, default: "right"
            The mouse button to track (either ``'left'`` or ``'right'``).
            Also accepts ``'r'`` or ``'l'``.

        double : bool, default: False
            Track single clicks if ``False``, double clicks if ``True``.
            Defaults to single clicks.

        viewport : bool, default: False
            If ``True``, uses the normalized viewport coordinate
            system (values between 0.0 and 1.0 and support for HiDPI)
            when passing the click position to the callback.

        """
        event = self._get_click_event(side)
        add_observer = all(len(cbs) == 0 for cbs in self._click_event_callbacks[event].values())
        if callback is None and add_observer:
            # No observers for this event yet and custom callback not given
            # insert dummy callback
            callback = lambda _, __: None
        if callable(callback):
            self._click_event_callbacks[event][double, viewport].append(callback)
        else:
            msg = 'Invalid callback provided, it should be either ``None`` or a callable.'
            raise TypeError(msg)

        if add_observer:
            self.add_observer(event, self._click_event)

    def untrack_click_position(self, side='right'):
        """Stop tracking the click position.

        Parameters
        ----------
        side : str, optional
            The mouse button to stop tracking (either ``'left'`` or
            ``'right'``). Default is ``'right'``. Also accepts ``'r'``
            or ``'l'``.

        """
        event = self._get_click_event(side)
        self.remove_observers(event)
        for cbs in self._click_event_callbacks[event].values():
            cbs.clear()

    def clear_key_event_callbacks(self):
        """Clear key event callbacks."""
        self._key_press_event_callbacks.clear()

    def key_press_event(self, *args):  # noqa: ARG002
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
        """Update the camera interactor style.

        Called when setting :meth:`style` attribute.
        """
        self.interactor.SetInteractorStyle(self.style)

    @property
    def style(
        self,
    ) -> (
        _vtk.vtkContextInteractorStyle
        | _vtk.vtkInteractorStyle
        | InteractorStyleCaptureMixin
        | None
    ):
        """Get/set the current interactor style.

        .. warning::

            Setting an interactor style needs careful control of events handling.
            See :class:`~plotting.render_window_interactor.InteractorStyleCaptureMixin`
            and its implementation as an example.

        Returns
        -------
        :vtk:`vtkInteractorStyle` | :vtk:`vtkContextInteractorStyle` | None
            The current interactor style.

        Examples
        --------
        Set interactor style with a customized vtk interactor

        >>> import pyvista as pv
        >>> from vtkmodules.vtkInteractionStyle import (
        ...     vtkInteractorStyleTrackballCamera,
        ... )

        >>> class MyCustomInteractorStyle(vtkInteractorStyleTrackballCamera):
        ...     # Implement custom functionality
        ...     def __repr__(self):
        ...         return 'A custom interactor style.'

        >>> pl = pv.Plotter()
        >>> plotter.iren.style = MyCustomInteractorStyle()
        >>> plotter.iren.style
        A custom interactor style.

        """
        return self._style_class

    @style.setter
    def style(self, style: _vtk.vtkInteractorStyle | InteractorStyleCaptureMixin | None):
        self._style = 'Interactor'
        self._style_class = style
        self.update_style()

    def _toggle_chart_interaction(self, mouse_pos):
        """Toggle interaction with indicated charts.

        Parameters
        ----------
        mouse_pos : tuple of float
            Tuple containing the mouse position.

        """
        # Loop over all renderers to see whether any charts need to be made interactive
        interactive_scene = None
        for renderer in self._plotter.renderers:
            if interactive_scene is None and renderer.IsInViewport(*mouse_pos):
                # No interactive charts yet and mouse is within this renderer's viewport,
                # so collect all charts indicated by the mouse (typically only one, except
                # when there are overlapping charts).
                origin = renderer.GetOrigin()  # Correct for viewport origin (see #3278)
                charts = renderer._get_charts_by_pos(
                    (mouse_pos[0] - origin[0], mouse_pos[1] - origin[1]),
                )
                if charts:
                    # Toggle interaction for indicated charts and determine whether
                    # there are any remaining interactive charts.
                    interactive_charts = renderer.set_chart_interaction(charts, toggle=True)
                    if interactive_charts:
                        # Save a reference to this renderer's scene if there are
                        # remaining interactive charts.
                        interactive_scene = renderer._charts._scene
                else:
                    # No indicated charts, so disable interaction with all charts
                    # for this renderer.
                    renderer.set_chart_interaction(False)
            else:
                # Not in viewport or interactive charts were already found in another
                # renderer, so disable interaction with all charts for this renderer.
                renderer.set_chart_interaction(False)
        # Manually set context_style based on found interactive scene (or stop interaction
        # with any scene if there are no interactive charts).
        self._set_context_style(interactive_scene)

    def _set_context_style(self, scene):
        """Set the context style interactor or switch back to previous interactor style.

        Parameters
        ----------
        scene : :vtk:`vtkContextScene`, optional
            The scene to interact with or ``None`` to stop interaction with any scene.

        """
        # Set scene to interact with or reset it to stop interaction (otherwise crash)
        if (
            vtk_version_info < (9, 3, 0) and scene is not None and len(self._plotter.renderers) > 1
        ):  # pragma: no cover
            warnings.warn(
                'Interaction with charts is not possible when using multiple subplots.'
                'Upgrade to VTK 9.3 or newer to enable this feature.',
                stacklevel=2,
            )
            scene = None
        self._context_style.SetScene(scene)
        if scene is None and self._style == 'Context':
            # Switch back to previous interactor style
            self._style = self._prev_style
            self.style = self._prev_style_class
            self._prev_style = None
            self._prev_style_class = None
        elif scene is not None and self._style != 'Context':
            # Enable context interactor style
            self._prev_style = self._style
            self._prev_style_class = self.style
            self._style = 'Context'
            self._style_class = self._context_style
        self.update_style()

    def enable_trackball_style(self):
        """Set the interactive style to Trackball Camera.

        The trackball camera is the default interactor style. Moving
        the mouse moves the camera around, leaving the scene intact.

        For a 3-button mouse, the left button is for rotation, the
        right button for zooming, the middle button for panning, and
        ctrl + left button for spinning the view around the viewing
        axis of the camera.  Alternatively, ctrl + shift + left button
        or mouse wheel zooms, and shift + left button pans.

        See Also
        --------
        pyvista.Plotter.enable_custom_trackball_style
            A style that can be customized for mouse actions.

        Examples
        --------
        Create a simple scene with a plotter that has the Trackball
        Camera interactive style (which is also the default):

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_trackball_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleTrackballCamera(self)

    @_deprecate_positional_args
    def enable_custom_trackball_style(  # noqa: PLR0917
        self,
        left='rotate',
        shift_left='pan',
        control_left='spin',
        middle='pan',
        shift_middle='pan',
        control_middle='pan',
        right='dolly',
        shift_right='environment_rotate',
        control_right='dolly',
    ):
        """Set the interactive style to a custom style based on Trackball Camera.

        For each choice of button, control-button, and shift-button,
        the behavior when the mouse is moved can be chosen by passing the
        following strings:

        * ``"dolly"``
        * ``"environment_rotate"``
        * ``"pan"``
        * ``"rotate"``
        * ``"spin"``

        ``None`` can also be passed, which also results in the default behavior.

        .. versionadded:: 0.44.0

        Parameters
        ----------
        left : str, default: "rotate"
            Action when the left button is clicked and the mouse is moved.

        shift_left : str, default: "pan"
            Action when the left button is clicked with the shift key and the mouse is moved.

        control_left : str, default: "spin"
            Action when the left button is clicked with the control key and mouse moved.

        middle : str, default: "pan"
            Action when the middle button is clicked and the mouse is moved.

        shift_middle : str, default: "pan"
            Action when the middle button is clicked with the shift key and the mouse is moved.

        control_middle : str, default: "pan"
            Action when the middle button is clicked with the control key and mouse moved.

        right : str, default: "dolly"
            Action when the right button is clicked and the mouse is moved.

        shift_right : str, default: "environment_rotate"
            Action when the right button is clicked with the shift key and the mouse is moved.

        control_right : str, default: "dolly"
            Action when the right button is clicked with the control key and the mouse is moved.

        See Also
        --------
        pyvista.Plotter.enable_trackball_style
            Base style.

        Examples
        --------
        Create a simple scene with a plotter that has the left button
        dolly.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_custom_trackball_style(left='dolly')
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleTrackballCamera(self)

        start_action_map = {
            'environment_rotate': self.style.StartEnvRotate,
            'rotate': self.style.StartRotate,
            'pan': self.style.StartPan,
            'spin': self.style.StartSpin,
            'dolly': self.style.StartDolly,
        }

        end_action_map = {
            'environment_rotate': self.style.EndEnvRotate,
            'rotate': self.style.EndRotate,
            'pan': self.style.EndPan,
            'spin': self.style.EndSpin,
            'dolly': self.style.EndDolly,
        }

        for p in [
            left,
            shift_left,
            control_left,
            middle,
            shift_middle,
            control_middle,
            right,
            shift_right,
            control_right,
        ]:
            if p not in start_action_map:
                msg = f"Action '{p}' not in the allowed {list(start_action_map.keys())}"
                raise ValueError(msg)

        button_press_map = {
            'left': self.style.OnLeftButtonDown,
            'middle': self.style.OnMiddleButtonDown,
            'right': self.style.OnRightButtonDown,
        }
        button_release_map = {
            'left': self.style.OnLeftButtonUp,
            'middle': self.style.OnMiddleButtonUp,
            'right': self.style.OnRightButtonUp,
        }

        def _setup_callbacks(*, button, click, control, shift):
            """Return callbacks for press and release events.

            Callbacks are formed for a button and action for a click,
            control-click, and shift-click.

            """
            button_press = button_press_map[button]
            button_release = button_release_map[button]

            click_action = start_action_map[click]
            control_action = start_action_map[control]
            shift_action = start_action_map[shift]

            click_release_action = end_action_map[click]
            control_release_action = end_action_map[control]
            shift_release_action = end_action_map[shift]

            def _press_callback(_obj, _):
                if self.interactor.GetControlKey():
                    control_action()
                elif self.interactor.GetShiftKey():
                    shift_action()
                else:
                    click_action()
                button_press()

            def _release_callback(_obj, _):
                click_release_action()
                control_release_action()
                shift_release_action()
                button_release()

            return partial(try_callback, _press_callback), partial(try_callback, _release_callback)

        _left_button_press_callback, _left_button_release_callback = _setup_callbacks(
            button='left',
            click=left,
            control=control_left,
            shift=shift_left,
        )
        self.style.add_observer('LeftButtonPressEvent', _left_button_press_callback)
        self.style.add_observer('LeftButtonReleaseEvent', _left_button_release_callback)

        _middle_button_press_callback, _middle_button_release_callback = _setup_callbacks(
            button='middle',
            click=middle,
            control=control_middle,
            shift=shift_middle,
        )
        self.style.add_observer('MiddleButtonPressEvent', _middle_button_press_callback)
        self.style.add_observer('MiddleButtonReleaseEvent', _middle_button_release_callback)

        _right_button_press_callback, _right_button_release_callback = _setup_callbacks(
            button='right',
            click=right,
            control=control_right,
            shift=shift_right,
        )
        self.style.add_observer('RightButtonPressEvent', _right_button_press_callback)
        self.style.add_observer('RightButtonReleaseEvent', _right_button_release_callback)

    def enable_2d_style(self):
        """Set the interactive style to 2D.

        For a 3-button mouse, the left button pans, the
        right button dollys, the middle button spins, and the wheel
        dollys.
        ctrl + left button spins, shift + left button dollys,
        ctrl + middle button pans, shift + middle button dollys,
        ctrl + right button rotates in 3D, and shift + right button
        dollys.

        Recommended to use with
        :func:`pyvista.Plotter.enable_parallel_projection`.

        See Also
        --------
        pyvista.Plotter.enable_parallel_projection
            Set parallel projection, which is useful for 2D views.

        pyvista.Plotter.enable_custom_trackball_style
            A style that can be customized for mouse actions.

        Examples
        --------
        Create a simple scene with a plotter that has a
        ParaView-like 2D style:

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_parallel_projection()
        >>> plotter.enable_2d_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.enable_custom_trackball_style(
            left='pan',
            shift_left='dolly',
            control_left='spin',
            middle='spin',
            shift_middle='dolly',
            control_middle='pan',
            right='dolly',
            shift_right='dolly',
            control_right='rotate',
        )

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
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_trackball_actor_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleTrackballActor(self)

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
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_image_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleImage(self)

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
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_joystick_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleJoystickCamera(self)

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
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_joystick_actor_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleJoystickActor(self)

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
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_zoom_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleZoom(self)

    @_deprecate_positional_args
    def enable_terrain_style(
        self,
        mouse_wheel_zooms: bool | float = True,  # noqa: FBT001, FBT002
        shift_pans: bool = True,  # noqa: FBT001, FBT002
    ):
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

        .. versionchanged:: 0.45
            mouse_wheel_zooms and shift_pans parameters are not True by
            default to be more intuitive. We also improved the scroll
            zooming factor to be less jumpy.

        Parameters
        ----------
        mouse_wheel_zooms : bool | float, default: True
            Whether to use the mouse wheel for zooming. If ``False``,
            you can still zoom with right click and drag. Pass a float
            value for to control the zoom factor, default is ``1.05``.

        shift_pans : bool, default: True
            Whether shift + left mouse button pans the scene. If
            ``False``, shift + left mouse button rotates the view
            restricted to only horizontal or vertical movements, and
            panning is done holding down the middle mouse button.

        Examples
        --------
        Create a simple scene with a plotter that has the Terrain
        interactive style:

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_terrain_style()
        >>> plotter.show()  # doctest:+SKIP

        Use controls that are closer to the default style:

        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleTerrain(self)

        if mouse_wheel_zooms:
            factor = 1.05 if isinstance(mouse_wheel_zooms, bool) else mouse_wheel_zooms

            def wheel_zoom_callback(_obj, event):  # pragma: no cover
                """Zoom in or out on mouse wheel roll."""
                if event == 'MouseWheelForwardEvent':
                    # zoom in
                    zoom_factor = 1.0 / factor
                elif event == 'MouseWheelBackwardEvent':
                    # zoom out
                    zoom_factor = factor

                with self.poked_subplot():
                    if self._plotter.camera.parallel_projection:
                        self._plotter.camera.parallel_scale *= zoom_factor
                    else:
                        camera_position = np.array(self._plotter.camera.position)
                        camera_focal_point = np.array(self._plotter.camera.focal_point)
                        camera_vector = camera_position - camera_focal_point
                        self._plotter.camera.position = (
                            camera_focal_point + zoom_factor * camera_vector
                        )

                    self._plotter.reset_camera_clipping_range()
                self._plotter.render()

            callback = partial(try_callback, wheel_zoom_callback)

            for event in 'MouseWheelForwardEvent', 'MouseWheelBackwardEvent':
                self.style.add_observer(event, callback)

        if shift_pans:

            def pan_on_shift_callback(_obj, event):  # pragma: no cover
                """Trigger left mouse panning if shift is pressed."""
                if event == 'LeftButtonPressEvent':
                    if self.interactor.GetShiftKey():
                        self.style.StartPan()  # type: ignore[union-attr]
                    self.style.OnLeftButtonDown()  # type: ignore[union-attr]
                elif event == 'LeftButtonReleaseEvent':
                    # always stop panning on release
                    self.style.EndPan()  # type: ignore[union-attr]
                    self.style.OnLeftButtonUp()  # type: ignore[union-attr]

            callback = partial(try_callback, pan_on_shift_callback)

            for event in 'LeftButtonPressEvent', 'LeftButtonReleaseEvent':
                self.style.add_observer(event, callback)

    def enable_rubber_band_style(self):
        """Set the interactive style to Rubber Band Picking.

        This interactor style allows the user to draw a rectangle in
        the render window by hitting ``r`` and then using the left
        mouse button. When the mouse button is released, the attached
        picker operates on the pixel in the center of the selection
        rectangle. If the picker happens to be a :vtk:`vtkAreaPicker`
        it will operate on the entire selection rectangle. When the
        ``p`` key is hit the above pick operation occurs on a 1x1
        rectangle. In other respects it behaves the same as the
        Trackball Camera style.

        Examples
        --------
        Create a simple scene with a plotter that has the Rubber Band
        Pick interactive style:

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_rubber_band_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleRubberBandPick(self)

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
        >>> pl = pv.Plotter()
        >>> _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
        >>> _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
        >>> plotter.show_axes()
        >>> plotter.enable_rubber_band_2d_style()
        >>> plotter.show()  # doctest:+SKIP

        """
        self.style = InteractorStyleRubberBand2D(self)

    def _simulate_keypress(self, key):
        """Simulate a keypress."""
        if len(key) > 1:
            msg = 'Only accepts a single key'
            raise ValueError(msg)
        self.interactor.SetKeyCode(key)
        self.interactor.SetKeySym(key)
        self.interactor.CharEvent()

    def _control_key_press(self):
        """Simulate a control keypress."""
        self.interactor.SetControlKey(1)

    def _control_key_release(self):
        """Simulate a control keypress."""
        self.interactor.SetControlKey(0)

    def _shift_key_press(self):
        """Simulate a shift keypress."""
        self.interactor.SetShiftKey(1)

    def _shift_key_release(self):
        """Simulate a shift keypress."""
        self.interactor.SetShiftKey(0)

    def _mouse_left_button_press(
        self,
        x=None,
        y=None,
    ):  # pragma: no cover # numpydoc ignore=PR01,RT01
        """Simulate a left mouse button press.

        If ``x`` and ``y`` are entered then simulates a movement to
        that position.

        """
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.LeftButtonPressEvent()

    def _mouse_left_button_release(
        self,
        x=None,
        y=None,
    ):  # pragma: no cover # numpydoc ignore=PR01,RT01
        """Simulate a left mouse button release."""
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.LeftButtonReleaseEvent()

    def _mouse_left_button_click(self, x=None, y=None, count=1):
        for _ in range(count):
            self._mouse_left_button_press(x, y)
            self._mouse_left_button_release()

    def _mouse_middle_button_press(
        self,
        x=None,
        y=None,
    ):  # pragma: no cover # numpydoc ignore=PR01,RT01
        """Simulate a middle mouse button press.

        If ``x`` and ``y`` are entered then simulates a movement to
        that position.

        """
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.MiddleButtonPressEvent()

    def _mouse_middle_button_release(
        self,
        x=None,
        y=None,
    ):  # pragma: no cover # numpydoc ignore=PR01,RT01
        """Simulate a middle mouse button release."""
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.MiddleButtonReleaseEvent()

    def _mouse_middle_button_click(self, x=None, y=None, count=1):
        for _ in range(count):
            self._mouse_middle_button_press(x, y)
            self._mouse_middle_button_release()

    def _mouse_right_button_press(
        self,
        x=None,
        y=None,
    ):  # pragma: no cover # numpydoc ignore=PR01,RT01
        """Simulate a right mouse button press.

        If ``x`` and ``y`` are entered then simulates a movement to
        that position.

        """
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.RightButtonPressEvent()

    def _mouse_right_button_release(
        self,
        x=None,
        y=None,
    ):  # pragma: no cover # numpydoc ignore=PR01,RT01
        """Simulate a right mouse button release."""
        if x is not None and y is not None:
            self._mouse_move(x, y)
        self.interactor.RightButtonReleaseEvent()

    def _mouse_right_button_click(self, x=None, y=None, count=1):
        for _ in range(count):
            self._mouse_right_button_press(x, y)
            self._mouse_right_button_release()

    def _mouse_move(self, x, y):  # pragma:
        """Simulate moving the mouse to ``(x, y)`` screen coordinates."""
        self.interactor.SetEventPosition(x, y)
        self.interactor.MouseMoveEvent()

    def get_event_position(self):
        """Get the event position.

        Returns
        -------
        tuple
            The ``(x, y)`` coordinate position.

        """
        return self.interactor.GetEventPosition()

    def get_poked_renderer(self, x=None, y=None):
        """Get poked renderer for last or specific event position.

        Parameters
        ----------
        x : float, default: None
            The x-coordinate for a user-defined event position.

        y : float, default: None
            The y-coordinate for a user-defined event position.

        Returns
        -------
        :vtk:`vtkRenderer`
            The poked renderer for given or last event position.

        """
        if x is None or y is None:
            x, y = self.get_event_position()
        return self.interactor.FindPokedRenderer(x, y)

    def get_event_subplot_loc(self):
        """Get the subplot location of the last event.

        Returns
        -------
        tuple
            A tuple containing the location of the subplot.

        Raises
        ------
        RuntimeError
            If the poked renderer is not found in the Plotter.

        """
        poked_renderer = self.get_poked_renderer()
        for index in range(len(self._plotter.renderers)):
            renderer = self._plotter.renderers[index]
            if renderer is poked_renderer:
                return self._plotter.renderers.index_to_loc(index)
        msg = 'Poked renderer not found in Plotter.'
        raise RuntimeError(msg)

    @contextmanager
    def poked_subplot(self):
        """Activate the subplot that was last interacted."""
        active_renderer_index = self._plotter.renderers._active_index
        loc = self.get_event_subplot_loc()
        self._plotter.subplot(*loc)
        try:
            yield
        finally:
            # Reset to the active renderer.
            loc = self._plotter.renderers.index_to_loc(active_renderer_index)
            self._plotter.subplot(*loc)

    def get_interactor_style(self):
        """Get the interactor style.

        Returns
        -------
        :vtk:`vtkInteractorStyle`
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

    @_deprecate_positional_args(allowed=['duration'])
    def create_timer(self, duration, repeating=True):  # noqa: FBT002
        """Create a timer.

        Parameters
        ----------
        duration : int
            Time (in milliseconds) before the timer emits a TimerEvent.

        repeating : bool, default: True
            When ``False`` a one-shot timer is created, which only fires
            once. When ``True`` a repeating timer is created, which
            continuously fires (every ``duration`` milliseconds) until
            destruction.

        Returns
        -------
        int
            Timer ID.

        """
        if repeating:
            timer_id = self.interactor.CreateRepeatingTimer(duration)
        else:
            timer_id = self.interactor.CreateOneShotTimer(duration)
        return timer_id

    def destroy_timer(self, timer_id):
        """Destroy the given timer.

        Parameters
        ----------
        timer_id : int
            The ID of the timer to destroy.

        """
        self.interactor.DestroyTimer(timer_id)

    def start(self):
        """Start interactions."""
        self.interactor.Start()

    def initialize(self):
        """Initialize the interactor."""
        self.interactor.Initialize()

    def set_render_window(self, render_window):
        """Set the render window for the interactor.

        Parameters
        ----------
        render_window : :vtk:`vtkRenderWindow`
            Render window to set for the interactor.

        """
        self.interactor.SetRenderWindow(render_window)

    def process_events(self):
        """Process events."""
        if not self.initialized:
            msg = 'Render window interactor must be initialized before processing events.'
            raise RuntimeError(msg)
        self.interactor.ProcessEvents()

    @property
    def initialized(self):  # numpydoc ignore=RT01
        """Return if the interactor has been initialized."""
        return self.interactor.GetInitialized()

    @property
    def picker(self):  # numpydoc ignore=RT01
        """Get/set the picker.

        Returns
        -------
        :vtk:`vtkAbstractPicker`
            VTK picker.

        """
        return self.interactor.GetPicker()

    @picker.setter
    def picker(self, picker):
        pickers = {
            PickerType.AREA: _vtk.vtkAreaPicker,
            PickerType.CELL: _vtk.vtkCellPicker,
            PickerType.POINT: _vtk.vtkPointPicker,
            PickerType.PROP: _vtk.vtkPropPicker,
            PickerType.RENDERED: _vtk.vtkRenderedAreaPicker,
            PickerType.RESLICE: _vtk.vtkResliceCursorPicker,
            PickerType.SCENE: _vtk.vtkScenePicker,
            PickerType.VOLUME: _vtk.vtkVolumePicker,
            PickerType.WORLD: _vtk.vtkWorldPointPicker,
        }
        if _vtk.vtkHardwarePicker is not None:
            # Unavailable on VTK < 9.2
            pickers[PickerType.HARDWARE] = _vtk.vtkHardwarePicker
        if isinstance(picker, (str, int, PickerType)):
            picker = PickerType.from_any(picker)
            try:
                picker = pickers[picker]()
            except KeyError:
                msg = f'Picker class `{picker}` is unknown.'
                raise KeyError(msg)
            # Set default tolerance for internal configurations
            if hasattr(picker, 'SetTolerance'):
                picker.SetTolerance(0.025)
        self.interactor.SetPicker(picker)

    def add_pick_obeserver(self, observer):
        """Add an observer to call back when pick events end.

        .. deprecated:: 0.42.2
            This function is deprecated. Use
            :func:`pyvista.RenderWindowInteractor.add_pick_observer` instead.

        Parameters
        ----------
        observer : callable
            The observer function to call when a pick event ends.

        """
        warnings.warn(
            '`add_pick_obeserver` is deprecated, use `add_pick_observer`',
            PyVistaDeprecationWarning,
            stacklevel=2,
        )
        self.add_pick_observer(observer)

    def add_pick_observer(self, observer):
        """Add an observer to call back when pick events end.

        Parameters
        ----------
        observer : callable
            The observer function to call when a pick event ends.

        """
        self.picker.AddObserver(_vtk.vtkCommand.EndPickEvent, observer)

    def reset_picker(self):
        """Reset the picker."""
        # Remove observers
        self.picker.RemoveObservers(_vtk.vtkCommand.EndPickEvent)
        # Set default picker to vtkWorldPointPicker
        self.picker = 'world'

    def fly_to(self, renderer, point):
        """Fly the interactor to the given point in a renderer.

        Parameters
        ----------
        renderer : :vtk:`vtkRenderer`
            The renderer in which the action will take place.

        point : list or tuple
            The point to fly to.

        """
        self.interactor.FlyTo(renderer, *point)

    def terminate_app(self):
        """Terminate the app."""
        if self.initialized:
            self.interactor.SetDone(True)  # See: https://gitlab.kitware.com/vtk/vtk/-/issues/18242
            self.interactor.TerminateApp()

    def close(self):
        """Close out the render window interactor.

        This will terminate the render window if it is not already closed.
        """
        self.remove_observers()
        if self.style == self._context_style:  # pragma: no cover
            self._set_context_style(None)  # Disable context interactor style first
        if self.style is not None:
            if hasattr(self.style, 'remove_observers'):
                self.style.remove_observers()
            self.style = None

        self.terminate_app()
        self.interactor = None
        self._click_event_callbacks = None  # type: ignore[assignment]
        self._timer_event = None


@abstract_class
class InteractorStyleCaptureMixin(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkInteractorStyle):
    """A mixin for subclasses of vtkInteractorStyle with capturing ability.

    Use a custom capturing events because the default ones
    swallow the release events. See
    https://public.kitware.com/pipermail/vtkusers/2013-December/082315.html.

    """

    def __init__(self, render_window_interactor: RenderWindowInteractor):
        super().__init__()
        self._parent = weakref.ref(render_window_interactor)

        # An unknown problem with AddObserver not typed to include string despite overload.
        # Ignore typing.
        self._observers = []
        self._observers.append(
            self.AddObserver('LeftButtonPressEvent', partial(try_callback, self._press)),  # type: ignore[arg-type]
        )
        self._observers.append(
            self.AddObserver(
                'LeftButtonReleaseEvent',  # type: ignore[arg-type]
                partial(try_callback, self._release),
            ),
        )

    def _press(self, *_):
        # Figure out which renderer has the event and disable the
        # others
        self.OnLeftButtonDown()
        parent = self._parent()
        if len(parent._plotter.renderers) > 1:  # type: ignore[union-attr]
            click_pos = parent.get_event_position()  # type: ignore[union-attr]
            for renderer in parent._plotter.renderers:  # type: ignore[union-attr]
                interact = renderer.IsInViewport(*click_pos)
                renderer.SetInteractive(interact)

    def _release(self, *_):
        self.OnLeftButtonUp()
        parent = self._parent()
        if len(parent._plotter.renderers) > 1:  # type: ignore[union-attr]
            for renderer in parent._plotter.renderers:  # type: ignore[union-attr]
                renderer.SetInteractive(True)

    def add_observer(self, event, callback):
        """Keep track of observers.

        Parameters
        ----------
        event : str
            VTK event string
        callback : callable
            Function to call during callback

        """
        self._observers.append(self.AddObserver(event, callback))

    def remove_observers(self):  # numpydoc ignore=SS06
        """Remove all observers added through
        :func:`~pyvista.plotting.render_window_interactor.InteractorStyleCaptureMixin.add_observer`.
        """  # noqa : D205
        for obs in self._observers:
            self.RemoveObserver(obs)


# All interactor styles here inherit from `InteractorStyleCaptureMixin`, which
# inherits from `DisableVtkSnakeCase`, so don't duplicate again.
class InteractorStyleImage(InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleImage):
    """Image interactor style.

    Wraps :vtk:`vtkInteractorStyleImage`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_image_style`

    """


class InteractorStyleJoystickActor(
    InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleJoystickActor
):
    """Joystick actor interactor style.

    Wraps :vtk:`vtkInteractorStyleJoystickActor`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_joystick_actor_style`

    """


class InteractorStyleJoystickCamera(
    InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleJoystickCamera
):
    """Joystick camera interactor style.

    Wraps :vtk:`vtkInteractorStyleJoystickCamera`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_joystick_style`

    """


class InteractorStyleRubberBand2D(
    InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleRubberBand2D
):
    """Rubber band 2D interactor style.

    Wraps :vtk:`vtkInteractorStyleRubberBand2D`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_rubber_band_2d_style`

    """


class InteractorStyleRubberBandPick(
    InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleRubberBandPick
):
    """Rubber band pick interactor style.

    Wraps :vtk:`vtkInteractorStyleRubberBandPick`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_rubber_band_style`

    """


class InteractorStyleTrackballActor(
    InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleTrackballActor
):
    """Trackball actor interactor style.

    Wraps :vtk:`vtkInteractorStyleTrackballActor`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_trackball_actor_style`

    """


class InteractorStyleTrackballCamera(
    InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleTrackballCamera
):
    """Trackball camera interactor style.

    Wraps :vtk:`vtkInteractorStyleTrackballCamera`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_trackball_style`
    :meth:`pyvista.RenderWindowInteractor.enable_custom_trackball_style`
    :meth:`pyvista.RenderWindowInteractor.enable_2d_style`

    """


class InteractorStyleTerrain(InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleTerrain):
    """Terrain interactor style.

    Wraps :vtk:`vtkInteractorStyleTerrain`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_terrain_style`

    """


class InteractorStyleZoom(InteractorStyleCaptureMixin, _vtk.vtkInteractorStyleRubberBandZoom):
    """Rubber band zoom interactor style.

    Wraps :vtk:`vtkInteractorStyleRubberBandZoom`.

    See Also
    --------
    :meth:`pyvista.RenderWindowInteractor.enable_zoom_style`

    """
