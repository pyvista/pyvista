"""Module to wrap vtkOpenGLRenderer."""

import weakref

import pyvista

from .renderers import Renderers


class RenderWindow:
    """Wrap vtkOpenGLRenderer."""

    def __init__(
        self,
        shape=(1, 1),
        splitting_position=None,
        row_weights=None,
        col_weights=None,
        groups=None,
        border=None,
        border_color='k',
        border_width=2.0,
        plotter=None,
    ):
        """Initialize the render window."""
        super().__init__()

        self._plotter = None
        if plotter is not None:
            if not isinstance(plotter, (weakref.ref, type(None))):
                raise TypeError('`plotter` must be a weak reference.')
            self._plotter = plotter

        self._interactor_ref = None
        self._ren_win = None
        self._camera_setup = False
        self._rendered = False
        self._renderers = Renderers(
            self,
            shape,
            splitting_position,
            row_weights,
            col_weights,
            groups,
            border,
            border_color,
            border_width,
        )

    @property
    def plotter(self):
        if self._plotter is not None:
            return self._plotter()

    @property
    def theme(self):
        if self._plotter is None or self._plotter() is None:
            return pyvista.global_theme
        else:
            return self._plotter()._theme

    def attach_render_window(self, ren_win=None):
        """Attach an existing or new render window."""
        if self._ren_win is not None:
            raise RuntimeError('Render window already attached')
        if ren_win is None:
            ren_win = pyvista._vtk.vtkRenderWindow()
        else:
            if not isinstance(ren_win, pyvista._vtk.vtkRenderWindow):
                raise TypeError(
                    '`ren_win` Must be a vtk.vtkRenderWindow or None, not ' '{type(ren_win)}'
                )
        self._ren_win = ren_win

        for renderer in self._renderers:
            self.add_renderer(renderer)

    def __getattribute__(self, name):
        """Allow getting attributes of the underlying VTK object."""
        try:
            return super().__getattribute__(name)
        except:
            return getattr(self.__dict__['_ren_win'], name)

    @property
    def borders(self):
        """Return or set borders."""
        return bool(self._ren_win.GetBorders())

    @borders.setter
    def borders(self, value):
        """Return or set borders."""
        self._ren_win.SetBorders(value)

    @property
    def off_screen(self):
        """Return or set off_screen rendering."""
        return bool(self._ren_win.GetOffScreenRendering())

    @off_screen.setter
    def off_screen(self, value):
        """Return or set off_screen."""
        self._ren_win.SetOffScreenRendering(value)

    def add_renderer(self, renderer):
        """Add a renderer to the render window."""
        self._ren_win.AddRenderer(renderer)

    @property
    def n_layers(self):
        """Return or set the number of layers."""
        return self._ren_win.GetNumberOfLayers()

    @n_layers.setter
    def n_layers(self, value):
        return self._ren_win.SetNumberOfLayers(value)

    @property
    def polygon_smoothing(self):
        """Return or set polygon smoothing."""
        return self._ren_win.GetPolygonSmoothing()

    @polygon_smoothing.setter
    def polygon_smoothing(self, value):
        return self._ren_win.SetPolygonSmoothing(value)

    @property
    def line_smoothing(self):
        """Return or set line smoothing."""
        return self._ren_win.GetLineSmoothing()

    @line_smoothing.setter
    def line_smoothing(self, value):
        return self._ren_win.SetLineSmoothing(value)

    @property
    def point_smoothing(self):
        """Return or set point smoothing."""
        return self._ren_win.GetPointSmoothing()

    @point_smoothing.setter
    def point_smoothing(self, value):
        return self._ren_win.SetPointSmoothing(value)

    @property
    def multi_samples(self):
        """Return or set the number of multi-samples."""
        return self._ren_win.GetMultiSamples()

    @multi_samples.setter
    def multi_samples(self, value):
        return self._ren_win.SetMultiSamples(value)

    def render(self):
        """Render this render window.

        This does nothing until the camera(s) have been set-up.
        """
        if self._ren_win is None:
            return

        # do not render until the camera has been setup
        if not self._camera_setup:
            return

        self._ren_win.Render()
        self._rendered = True

    @property
    def rendered(self):
        """Return if this render window has ever been rendered."""
        return self._rendered

    @property
    def renderers(self):
        """Return the renderers of this render window."""
        return self._renderers

    def finalize(self):
        """Finalize and clear out the render window."""
        self._renderers = None
        if self._ren_win is not None:
            self._ren_win.Finalize()
            self._ren_win = None

    def _set_up_camera(self, cpos=None):
        """Set up the camera on the very first render request.

        For example on the show call or any screenshot producing code.
        """
        # reset unless camera for the first render unless camera is set
        if not self._camera_setup:
            for renderer in self.renderers:
                if not isinstance(renderer, pyvista.Renderer):
                    continue
                if not renderer.camera_set and cpos is None:
                    renderer.camera_position = renderer.get_default_cam_pos()
                    renderer.ResetCamera()
                elif cpos is not None:
                    renderer.camera_position = cpos
            self._camera_setup = True

    def show(self):
        """Set up the cameras if needed and render."""
        self._set_up_camera()
        self.render()

    def _check_has_ren_win(self):
        """Check if render window attribute exists and raise an exception if not."""
        if self._ren_win is None:
            raise AttributeError(
                '\n\nTo retrieve an image after the render window '
                'has been closed, set:\n\n'
                ' ``plotter.store_image = True``\n\n'
                'before closing the plotter.'
            )

    @property
    def vtk_obj(self):
        """Return the vtk render window."""
        return self._ren_win

    @property
    def size(self):
        """Return the render window size in ``(width, height)``.

        Examples
        --------
        Change the window size from ``200 x 200`` to ``400 x 400``.

        >>> import pyvista
        >>> pl = pyvista.Plotter(window_size=[200, 200])
        >>> pl.window_size
        [200, 200]
        >>> pl.window_size = [400, 400]
        >>> pl.window_size
        [400, 400]

        """
        return self._ren_win.GetSize()

    @size.setter
    def size(self, value):
        self._ren_win.SetSize(value[0], value[1])

    @property
    def interactor(self):
        if self._ren_win is not None:
            return self._ren_win.GetInteractor()
        elif self._interactor_ref is not None:
            return self._interactor_ref()

    @interactor.setter
    def interactor(self, obj):
        self._ren_win.SetInteractor(obj)
        self._interactor_ref = weakref.ref(obj)
