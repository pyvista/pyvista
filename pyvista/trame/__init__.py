"""Trame interface for PyVista."""
from trame.widgets.vtk import VtkLocalView, VtkRemoteView


class _BasePyVistaView:
    def __init__(self, plotter):
        self._plotter = plotter
        self.pyvista_initialize()

    def pyvista_initialize(self):
        if self._plotter.render_window is None:
            raise RuntimeError(
                'The render window for this plotter has been destroyed. Do not call `show()` for the plotter before passing to trame.'
            )
        if not self._plotter.camera_set:
            self._plotter.view_isometric()


class PyVistaRemoteView(VtkRemoteView, _BasePyVistaView):
    """PyVista wrapping of trame VtkRemoteView."""

    def __init__(self, plotter, interactive_ratio=1, ref='view', **kwargs):
        """Create a trame remote view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        VtkRemoteView.__init__(
            self, self._plotter.ren_win, interactive_ratio=interactive_ratio, ref=ref, **kwargs
        )


class PyVistaLocalView(VtkLocalView, _BasePyVistaView):
    """PyVista wrapping of trame VtkLocalView."""

    def __init__(self, plotter, ref='view', **kwargs):
        """Create a trame local view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        VtkLocalView.__init__(self, self._plotter.ren_win, ref=ref, **kwargs)
        # CRITICAL to initialize the client render window
        self._server.controller.on_server_ready.add(self.update)
