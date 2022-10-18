"""Trame interface for PyVista.

This requires trame v2.3.0

"""
from trame.widgets.vtk import VtkLocalView, VtkRemoteView

CLOSED_PLOTTER_ERROR = "The render window for this plotter has been destroyed. Do not call `show()` for the plotter before passing to trame."


class _BasePyVistaView:
    def __init__(self, plotter):
        self._plotter = plotter
        self.pyvista_initialize()

    def pyvista_initialize(self):
        if self._plotter.render_window is None:
            raise RuntimeError(CLOSED_PLOTTER_ERROR)
        if not self._plotter.camera_set:
            self._plotter.view_isometric()


class PyVistaRemoteView(VtkRemoteView, _BasePyVistaView):
    """PyVista wrapping of trame VtkRemoteView."""

    def __init__(self, plotter, interactive_ratio=1, ref='view', **kwargs):
        """Create a trame remote view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        VtkRemoteView.__init__(
            self,
            self._plotter.render_window,
            interactive_ratio=interactive_ratio,
            ref=ref,
            **kwargs,
        )


class PyVistaLocalView(VtkLocalView, _BasePyVistaView):
    """PyVista wrapping of trame VtkLocalView."""

    def __init__(self, plotter, ref='view', **kwargs):
        """Create a trame local view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        VtkLocalView.__init__(self, self._plotter.render_window, ref=ref, **kwargs)
        # CRITICAL to initialize the client render window
        self._server.controller.on_server_ready.add(self.update)
