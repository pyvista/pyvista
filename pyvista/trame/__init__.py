"""Trame interface for PyVista.

This requires trame v2.3.0

"""
from pyvista import _vtk

import weakref
from trame.widgets.vtk import VtkLocalView, VtkRemoteView, VtkRemoteLocalView

CLOSED_PLOTTER_ERROR = "The render window for this plotter has been destroyed. Do not call `show()` for the plotter before passing to trame."


class _BasePyVistaView:
    def __init__(self, plotter):
        self._plotter = weakref.ref(plotter)
        self.pyvista_initialize()

    def pyvista_initialize(self):
        if self._plotter().render_window is None:
            raise RuntimeError(CLOSED_PLOTTER_ERROR)
        if not self._plotter().camera_set:
            self._plotter().view_isometric()


class PyVistaRemoteView(VtkRemoteView, _BasePyVistaView):
    """PyVista wrapping of trame VtkRemoteView."""

    def __init__(self, plotter, interactive_ratio=1, still_ratio=1, ref=None, **kwargs):
        """Create a trame remote view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        if ref is None:
            ref = f'view_{plotter._id_name}'
        VtkRemoteView.__init__(
            self,
            self._plotter().render_window,
            interactive_ratio=interactive_ratio,
            still_ratio=still_ratio,
            __properties=[('still_ratio', 'stillRatio')],
            ref=ref,
            **kwargs,
        )
        # Sometimes there is a lag
        self._server.controller.on_server_ready.add(self.update)

        plotter.add_on_render_callback(lambda *args: self.update, render_event=True)


class PyVistaLocalView(VtkLocalView, _BasePyVistaView):
    """PyVista wrapping of trame VtkLocalView."""

    def __init__(self, plotter, ref=None, **kwargs):
        """Create a trame local view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        if ref is None:
            ref = f'view_{plotter._id_name}'
        VtkLocalView.__init__(self, self._plotter().render_window, ref=ref, **kwargs)
        # CRITICAL to initialize the client render window
        self._server.controller.on_server_ready.add(self.update)


class PyVistaRemoteLocalView(VtkRemoteLocalView, _BasePyVistaView):
    """PyVista wrapping of trame VtkRemoteLocalView.

    Makes it easy to dynamically switch between client and server rendering.
    """

    def __init__(self, plotter, interactive_ratio=1, still_ratio=1, ref=None, **kwargs):
        """Create a trame remote/local view from a PyVista Plotter."""
        _BasePyVistaView.__init__(self, plotter)
        if ref is None:
            ref = f'view_{plotter._id_name}'
        VtkRemoteLocalView.__init__(
            self,
            self._plotter().render_window,
            interactive_ratio=interactive_ratio,
            still_ratio=still_ratio,
            __properties=[('still_ratio', 'stillRatio')],
            ref=ref,
            **kwargs,
        )
        # CRITICAL to initialize the client render window
        self._server.controller.on_server_ready.add(self.update)
