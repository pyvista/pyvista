import logging
import os
import warnings

from vtkmodules.vtkRenderingCore import (
    vtkOpenVRCamera,
    vtkOpenVRRenderer,
    vtkOpenVRRenderWindow,
    vtkOpenVRRenderWindowInteractor,
)

import pyvista
from pyvista import _vtk
from pyvista.plotting.camera import BaseCamera
from pyvista.plotting.plotting import BasePlotter
from pyvista.plotting.render_window_interactor import RenderWindowInteractor
from pyvista.plotting.renderer import BaseRenderer
from pyvista.utilities import assert_empty_kwargs

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')
log.addHandler(logging.StreamHandler())


class OpenVRCamera(vtkOpenVRCamera, BaseCamera):
    pass


class OpenVRRenderer(vtkOpenVRRenderer, BaseRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_floor = True

    @property
    def show_floor(self):
        return self.GetShowFloor()

    @show_floor.setter
    def show_floor(self, value):
        return self.SetShowFloor(value)


class OpenVRPlotter(BasePlotter):
    last_update_time = 0.0
    right_timer_id = -1
    renderer_class = OpenVRRenderer

    def __init__(
        self,
        window_size=None,
        multi_samples=None,
        line_smoothing=False,
        point_smoothing=False,
        polygon_smoothing=False,
        splitting_position=None,
        title=None,
        lighting='light kit',
        theme=None,
    ):
        """Initialize a vtk plotting object."""
        super().__init__(
            shape=(1, 1),
            border=None,
            groups=None,
            row_weights=None,
            col_weights=None,
            splitting_position=splitting_position,
            title=title,
            lighting=lighting,
            theme=theme,
        )

        log.debug('Plotter init start')

        # check if a plotting backend is enabled
        # _warn_xserver()

        def on_timer(iren, event_id):
            """Exit application if interactive renderer stops."""
            if event_id == 'TimerEvent' and self.iren._style != "Context":
                self.iren.terminate_app()

        self._window_size_unset = False
        if window_size is None:
            self._window_size_unset = True
            window_size = self._theme.window_size
        self.__prior_window_size = window_size

        if multi_samples is None:
            multi_samples = self._theme.multi_samples

        # initialize render window
        self.ren_win = vtkOpenVRRenderWindow()
        self.ren_win.SetMultiSamples(multi_samples)
        self.ren_win.SetBorders(True)
        if line_smoothing:
            self.ren_win.LineSmoothingOn()
        if point_smoothing:
            self.ren_win.PointSmoothingOn()
        if polygon_smoothing:
            self.ren_win.PolygonSmoothingOn()

        for renderer in self.renderers:
            self.ren_win.AddRenderer(renderer)

        # Add the shadow renderer to allow us to capture interactions within
        # a given viewport
        # https://vtk.org/pipermail/vtkusers/2018-June/102030.html
        number_or_layers = self.ren_win.GetNumberOfLayers()
        current_layer = self.renderer.GetLayer()
        self.ren_win.SetNumberOfLayers(number_or_layers + 1)
        self.ren_win.AddRenderer(self.renderers.shadow_renderer)
        self.renderers.shadow_renderer.SetLayer(current_layer + 1)
        self.renderers.shadow_renderer.SetInteractive(False)  # never needs to capture

        # Add ren win and interactor
        self.iren = RenderWindowInteractor(
            self, light_follow_camera=False, interactor=vtkOpenVRRenderWindowInteractor()
        )
        self.iren.set_render_window(self.ren_win)
        self.enable_trackball_style()  # internally calls update_style()
        self.iren.add_observer("KeyPressEvent", self.key_press_event)

        # Set camera widget based on theme. This requires that an
        # interactor be present.
        if self.theme._enable_camera_orientation_widget:
            self.add_camera_orientation_widget()

        # Set background
        self.set_background(self._theme.background)

        # Set window size
        self.window_size = window_size

        # add timer event if interactive render exists
        self.iren.add_observer(_vtk.vtkCommand.TimerEvent, on_timer)

        if self._theme.depth_peeling.enabled:
            if self.enable_depth_peeling():
                for renderer in self.renderers:
                    renderer.enable_depth_peeling()

        # crazy frame rate requirement
        # need to look into that at some point
        self.ren_win.SetDesiredUpdateRate(350.0)
        self.iren.SetDesiredUpdateRate(350.0)
        self.iren.SetStillUpdateRate(350.0)

        self.renderer.RemoveCuller(self.renderer.GetCullers().GetLastItem())

        log.debug('Plotter init stop')

    def show(
        self,
        title=None,
        window_size=None,
        interactive=True,
        auto_close=None,
        interactive_update=False,
        full_screen=None,
        screenshot=False,
        return_img=False,
        cpos=None,
        use_ipyvtk=None,
        jupyter_backend=None,
        return_viewer=False,
        return_cpos=None,
        **kwargs,
    ):
        self._before_close_callback = kwargs.pop('before_close_callback', None)
        assert_empty_kwargs(**kwargs)

        if interactive_update and auto_close is None:
            auto_close = False
        elif auto_close is None:
            auto_close = self._theme.auto_close

        if not hasattr(self, "ren_win"):
            raise RuntimeError("This plotter has been closed and cannot be shown.")

        if full_screen is None:
            full_screen = self._theme.full_screen

        if full_screen:
            self.ren_win.SetFullScreen(True)
            self.ren_win.BordersOn()  # super buggy when disabled
        else:
            if window_size is None:
                window_size = self.window_size
            else:
                self._window_size_unset = False
            self.ren_win.SetSize(window_size[0], window_size[1])

        # reset unless camera for the first render unless camera is set
        self._on_first_render_request(cpos)

        self.render()

        # This has to be after the first render for some reason
        if title is None:
            title = self.title
        if title:
            self.ren_win.SetWindowName(title)
            self.title = title

        # Keep track of image for sphinx-gallery
        if pyvista.BUILDING_GALLERY or screenshot:
            # always save screenshots for sphinx_gallery
            self.last_image = self.screenshot(screenshot, return_img=True)
            self.last_image_depth = self.get_image_depth()

        # See: https://github.com/pyvista/pyvista/issues/186#issuecomment-550993270
        if interactive:
            try:  # interrupts will be caught here
                log.debug('Starting iren')
                self.iren.update_style()
                if not interactive_update:

                    # Resolves #1260
                    if os.name == 'nt':
                        if _vtk.VTK9:
                            self.iren.process_events()
                        else:
                            global VERY_FIRST_RENDER
                            if not VERY_FIRST_RENDER:
                                self.iren.start()
                            VERY_FIRST_RENDER = False

                    self.iren.start()
                self.iren.initialize()
            except KeyboardInterrupt:
                log.debug('KeyboardInterrupt')
                self.close()
                raise KeyboardInterrupt
        # In the event that the user hits the exit-button on the GUI  (on
        # Windows OS) then it must be finalized and deleted as accessing it
        # will kill the kernel.
        # Here we check for that and clean it up before moving on to any of
        # the closing routines that might try to still access that
        # render window.
        if not self.ren_win.IsCurrent():
            self._clear_ren_win()  # The ren_win is deleted
            # proper screenshots cannot be saved if this happens
            if not auto_close:
                warnings.warn(
                    "`auto_close` ignored: by clicking the exit button, "
                    "you have destroyed the render window and we have to "
                    "close it out."
                )
                auto_close = True
        # NOTE: after this point, nothing from the render window can be accessed
        #       as if a user presed the close button, then it destroys the
        #       the render view and a stream of errors will kill the Python
        #       kernel if code here tries to access that renderer.
        #       See issues #135 and #186 for insight before editing the
        #       remainder of this function.

        # Close the render window if requested
        if auto_close:
            self.close()

        # If user asked for screenshot, return as numpy array after camera
        # position
        if return_img or screenshot is True:
            if return_cpos:
                return self.camera_position, self.last_image

        if return_cpos:
            return self.camera_position
