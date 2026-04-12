"""PyVista Trame Base Viewer class.

This base class defines methods to manipulate a PyVista Plotter.
This base class does not define a `ui` method, but its derived classes do.
See `pyvista.trame.ui.vuetify2` and ``pyvista.trame.ui.vuetify3` for its derived classes.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from trame.app import get_server

import pyvista as pv

if TYPE_CHECKING:
    from trame_client.ui.core import AbstractLayout


class BaseViewer:
    """Internal wrapper to sync trame view with Plotter.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Target Plotter instance to view.
    server : trame.Server, optional
        Current Server for Trame Application.
    suppress_rendering : bool, default=False
        Whether to suppress rendering on the Plotter.

    """

    def __init__(self, plotter, server=None, suppress_rendering=False):
        """Initialize Viewer."""
        self._html_views = set()

        if server is None:
            server = get_server()
        self.server = server
        self.plotter = plotter
        self.plotter.suppress_rendering = suppress_rendering

        # State variable names
        self.SHOW_UI = f'{plotter._id_name}_show_ui'
        self.GRID = f'{plotter._id_name}_grid_visibility'
        self.OUTLINE = f'{plotter._id_name}_outline_visibility'
        self.EDGES = f'{plotter._id_name}_edge_visibility'
        self.AXIS = f'{plotter._id_name}_axis_visibility'
        self.PARALLEL = f'{plotter._id_name}_parallel_projection'
        self.SERVER_RENDERING = f'{plotter._id_name}_use_server_rendering'
        self.VALID_UI_MODES = [
            'trame',
            'client',
            'server',
        ]
        server.state[self.SHOW_UI] = True
        server.state[self.GRID] = False
        server.state[self.OUTLINE] = False
        server.state[self.EDGES] = False
        server.state[self.AXIS] = False
        server.state[self.PARALLEL] = False

    @property
    def views(self):  # numpydoc ignore=RT01
        """Get a set of all associate trame views for this viewer."""
        return self._html_views

    def update(self, **kwargs):  # noqa: ARG002
        """Update all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.update()

    def push_camera(self, **kwargs):  # noqa: ARG002
        """Push camera to all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.push_camera()

    def reset_camera(self, **kwargs):  # noqa: ARG002
        """Reset camera for all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.reset_camera()

    def update_image(self, **kwargs):  # noqa: ARG002
        """Update image for all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.update_image()

    def update_camera(self, **kwargs):  # noqa: ARG002
        """Update image and camera for all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.update_camera()

    def view_isometric(self):
        """View isometric."""
        self.plotter.view_isometric(render=False)
        self.update_camera()

    def view_yz(self):
        """View YZ plane."""
        self.plotter.view_yz(render=False)
        self.update_camera()

    def view_xz(self):
        """View XZ plane."""
        self.plotter.view_xz(render=False)
        self.update_camera()

    def view_xy(self):
        """View XY plane."""
        self.plotter.view_xy(render=False)
        self.update_camera()

    def on_parallel_projection_change(self, **kwargs):
        """Toggle parallel projection for all renderers.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        value = kwargs[self.PARALLEL]
        for renderer in self.plotter.renderers:
            if value:
                renderer.enable_parallel_projection()
            else:
                renderer.disable_parallel_projection()
        self.update()

    def on_edge_visibility_change(self, **kwargs):
        """Toggle edge visibility for all actors.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        value = kwargs[self.EDGES]
        for renderer in self.plotter.renderers:
            for actor in renderer.actors.values():
                if isinstance(actor, pv.Actor):
                    actor.prop.show_edges = value
        self.update()

    def on_grid_visibility_change(self, **kwargs):
        """Handle axes grid visibility.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        value = kwargs[self.GRID]
        for renderer in self.plotter.renderers:
            if value:
                renderer.show_grid()
            else:
                renderer.remove_bounds_axes()
        self.update()

    def on_outline_visibility_change(self, **kwargs):
        """Handle outline visibility.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        value = kwargs[self.OUTLINE]
        for renderer in self.plotter.renderers:
            if value:
                renderer.add_bounding_box(reset_camera=False)
            else:
                renderer.remove_bounding_box()
        self.update()

    def on_axis_visibility_change(self, **kwargs):
        """Handle outline visibility.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        value = kwargs[self.AXIS]
        for renderer in self.plotter.renderers:
            if value:
                renderer.show_axes()
            else:
                renderer.hide_axes()
        for view in self._html_views:
            if view.set_widgets:
                # VtkRemoteView does not have set_widgets function, but
                # VtkRemoteLocalView and VtkLocalView do.
                view.set_widgets(
                    [
                        ren.axes_widget
                        for ren in self.plotter.renderers
                        if ren.axes_widget is not None
                    ],
                )
        self.update()

    def on_rendering_mode_change(self, **kwargs):
        """Handle any configurations when the render mode changes between client and server.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        if not kwargs[self.SERVER_RENDERING]:
            self.update_camera()

    @property
    def actors(self):  # numpydoc ignore=RT01
        """Get dataset actors."""
        return {k: v for k, v in self.plotter.actors.items() if isinstance(v, pv.Actor)}

    def screenshot(self):
        """Take screenshot and add attachament.

        Returns
        -------
        memoryview
            Screenshot as a ``memoryview``.

        """
        self.plotter.render()
        self.update()  # makes sure the plotter and views are in sync
        buffer = io.BytesIO()
        self.plotter.screenshot(filename=buffer)
        buffer.seek(0)
        return memoryview(buffer.read())

    def export(self):  # numpydoc ignore=RT01
        """Export the scene as a zip file."""
        for view in self._html_views:
            return memoryview(view.export_html())
        msg = 'This viewer cannot be exported.'
        raise TypeError(msg)

    def ui(self):
        """Implement in derived classes."""
        raise NotImplementedError

    def make_layout(self, *args, **kwargs) -> AbstractLayout:  # pragma: no cover
        """Create an instance of an AbstractLayout which is appropriate for a concrete viewer.

        Parameters
        ----------
        *args : tuple
            Positional arguments.

        **kwargs : dict, optional
            Keyword arguments.

        Returns
        -------
        AbstractLayout
            A layout this viewer can be embedded in.

        """
        raise NotImplementedError
