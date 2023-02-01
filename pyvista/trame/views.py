"""Trame view interface for PyVista."""
import weakref

from trame.widgets.vtk import VtkLocalView, VtkRemoteLocalView, VtkRemoteView

CLOSED_PLOTTER_ERROR = "The render window for this plotter has been destroyed. Do not call `show()` for the plotter before passing to trame."


class _BasePyVistaView:
    def __init__(self, plotter):
        self._plotter = weakref.ref(plotter)
        self.pyvista_initialize()

    def pyvista_initialize(self):
        if self._plotter().render_window is None:
            raise RuntimeError(CLOSED_PLOTTER_ERROR)
        for renderer in self._plotter().renderers:
            if not renderer.camera.is_set:
                renderer.camera_position = renderer.get_default_cam_pos()
                renderer.ResetCamera()


class PyVistaRemoteView(VtkRemoteView, _BasePyVistaView):
    """PyVista wrapping of trame ``VtkRemoteView`` for server rendering.

    This will connect to a PyVista plotter and stream the server-side
    renderings.

    Parameters
    ----------
    plotter : pyvista.BasePlotter
        The PyVista Plotter to display in the output view.

    interactive_ratio : int, optional
        Image size scale factor while interacting. Increasing this
        value will give higher resulotuion images during interaction
        events at the cost of performance. Use lower values (e.g.,
        ``0.5``) to increase performance while interacting.
        Defaults to 1.

    still_ratio : int, optional
        Image size scale factor while not interacting (still).
        Increasing this value will give higher resulotuion images
        when not interacting with the scene. Defaults to 1.

    namespace : str, optional
        The namespace for this view component. A default value is
        chosen based on the ``_id_name`` of the plotter.

    **kwargs : dict, optional
        Any additional keyword arguments to pass to
        ``trame.widgets.vtk.VtkRemoteView``.

    Notes
    -----
    For optimal rendering results, you may want to have the same
    value for ``interactive_ratio`` and ``still_ratio`` so that
    the entire rendering is not re-scaled between interaction events.

    """

    def __init__(self, plotter, interactive_ratio=None, still_ratio=None, namespace=None, **kwargs):
        """Create a trame remote view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        if namespace is None:
            namespace = f'{plotter._id_name}'
        if interactive_ratio is None:
            interactive_ratio = plotter._theme.trame.interactive_ratio
        if still_ratio is None:
            still_ratio = plotter._theme.trame.still_ratio
        VtkRemoteView.__init__(
            self,
            self._plotter().render_window,
            interactive_ratio=interactive_ratio,
            still_ratio=still_ratio,
            __properties=[('still_ratio', 'stillRatio')],
            ref=f'view_{plotter._id_name}',
            namespace=namespace,
            **kwargs,
        )
        # Sometimes there is a lag
        self._server.controller.on_server_ready.add(self.update)

        plotter.add_on_render_callback(lambda *args: self.update(), render_event=True)

    def push_camera(self, *args, **kwargs):
        """No-op implementation to match local viewers."""
        pass  # pragma: no cover


class PyVistaLocalView(VtkLocalView, _BasePyVistaView):
    """PyVista wrapping of trame VtkLocalView for in-browser rendering.

    This will connect to and synchronize with a PyVista plotter to
    perform client-side rendering with VTK.js in the browser.

    Parameters
    ----------
    plotter : pyvista.BasePlotter
        The PyVista Plotter to represent in the output view.

    namespace : str, optional
        The namespace for this view component. A default value is
        chosen based on the ``_id_name`` of the plotter.

    **kwargs : dict, optional
        Any additional keyword arguments to pass to
        ``trame.widgets.vtk.VtkLocalView``.

    """

    def __init__(self, plotter, namespace=None, **kwargs):
        """Create a trame local view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        if namespace is None:
            namespace = f'{plotter._id_name}'
        VtkLocalView.__init__(
            self,
            self._plotter().render_window,
            ref=f'view_{plotter._id_name}',
            namespace=namespace,
            **kwargs,
        )
        if self._server.running:
            self.update()
        else:
            self._server.controller.on_server_ready.add(self.update)

        # Callback to sync view on PyVista's render call when renders are suppressed
        plotter.add_on_render_callback(lambda *args: self.update(), render_event=False)


class PyVistaRemoteLocalView(VtkRemoteLocalView, _BasePyVistaView):
    """PyVista wrapping of trame ``VtkRemoteLocalView``.

    Dynamically switch between client and server rendering.

    Parameters
    ----------
    plotter : pyvista.BasePlotter
        The PyVista Plotter to display in the output view.

    interactive_ratio : int, optional
        Image size scale factor while interacting. Increasing this
        value will give higher resulotuion images during interaction
        events at the cost of performance. Use lower values (e.g.,
        ``0.5``) to increase performance while interacting.
        Defaults to 1. This is only valid in the ``'remote'`` mode.

    still_ratio : int, optional
        Image size scale factor while not interacting (still).
        Increasing this value will give higher resulotuion images
        when not interacting with the scene. Defaults to 1.
        This is only valid in the ``'remote'`` mode.

    namespace : str, optional
        The namespace for this view component. A default value is
        chosen based on the ``_id_name`` of the plotter.

    **kwargs : dict, optional
        Any additional keyword arguments to pass to
        ``trame.widgets.vtk.VtkRemoteLocalView``.

    """

    def __init__(self, plotter, interactive_ratio=None, still_ratio=None, namespace=None, **kwargs):
        """Create a trame remote/local view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        if namespace is None:
            namespace = f'{plotter._id_name}'
        if interactive_ratio is None:
            interactive_ratio = plotter._theme.trame.interactive_ratio
        if still_ratio is None:
            still_ratio = plotter._theme.trame.still_ratio
        VtkRemoteLocalView.__init__(
            self,
            self._plotter().render_window,
            interactive_ratio=interactive_ratio,
            still_ratio=still_ratio,
            __properties=[('still_ratio', 'stillRatio')],
            ref=f'view_{plotter._id_name}',
            namespace=namespace,
            **kwargs,
        )
        # Track namespace for our use since trame attributes are name mangled
        self._namespace = namespace

        if self._server.running:
            self.update()
        else:
            self._server.controller.on_server_ready.add(self.update)

        # Callback to sync view on PyVista's render call when using local view
        plotter.add_on_render_callback(lambda *args: self.update(), render_event=False)
