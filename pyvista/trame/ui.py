# flake8: noqa: D102,D103,D107
"""PyVista Trame User Interface.

This module builds a base UI for manipulating a PyVista Plotter.
The UI generated here is the default for rendering in Jupyter
environments and provides a starting point for custom user-built
applications.
"""
import io

from trame.widgets import html, vuetify

import pyvista
from pyvista.trame.views import PyVistaLocalView, PyVistaRemoteLocalView, PyVistaRemoteView

UI_TITLE = 'PyVista'

VALID_UI_MODES = [
    'trame',
    'client',
    'server',
]

_VIEWERS = {}


def button(click, icon, tooltip):
    """Create a vuetify button."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator='{ on, attrs }'):
            with vuetify.VBtn(icon=True, v_bind='attrs', v_on='on', click=click):
                vuetify.VIcon(icon)
        html.Span(tooltip)


def checkbox(model, icons, tooltip):
    """Create a vuetify checkbox."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator='{ on, attrs }'):
            with html.Div(v_on='on', v_bind='attrs'):
                vuetify.VCheckbox(
                    v_model=model,
                    on_icon=icons[0],
                    off_icon=icons[1],
                    dense=True,
                    hide_details=True,
                    classes='my-0 py-0 ml-1',
                )
        html.Span(tooltip)


class Viewer:
    """Internal wrapper to sync trame view with Plotter."""

    def __init__(self, plotter, suppress_rendering=False):
        """Initialize Viewer."""
        if plotter._id_name in _VIEWERS:
            raise RuntimeError('A viewer instance already exists for this Plotter.')
        _VIEWERS[plotter._id_name] = self
        self._html_views = set()

        self.plotter = plotter
        self.plotter.suppress_rendering = suppress_rendering

        # State variable names
        self.SHOW_UI = f'{plotter._id_name}_show_ui'
        self.GRID = f'{plotter._id_name}_grid_visibility'
        self.OUTLINE = f'{plotter._id_name}_outline_visibility'
        self.EDGES = f'{plotter._id_name}_edge_visibility'
        self.AXIS = f'{plotter._id_name}_axis_visiblity'
        self.SERVER_RENDERING = f'{plotter._id_name}_use_server_rendering'

    @property
    def views(self):
        """Get a set of all associate trame views for this viewer."""
        return self._html_views

    def update(self, **kwargs):
        """Update all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.update()

    def push_camera(self, **kwargs):
        """Push camera to all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.push_camera()

    def reset_camera(self, **kwargs):
        """Reset camera for all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.reset_camera()

    def update_image(self, **kwargs):
        """Update image for all associated views.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        for view in self._html_views:
            view.update_image()

    def update_camera(self, **kwargs):
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

    def on_edge_visiblity_change(self, **kwargs):
        """Toggle edge visibility for all actors.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused keyword arguments.

        """
        value = kwargs[self.EDGES]
        for renderer in self.plotter.renderers:
            for _, actor in renderer.actors.items():
                if isinstance(actor, pyvista.Actor):
                    actor.prop.show_edges = value
        self.update()

    def on_grid_visiblity_change(self, **kwargs):
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

    def on_outline_visiblity_change(self, **kwargs):
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

    def on_axis_visiblity_change(self, **kwargs):
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
    def actors(self):
        """Get dataset actors."""
        return {k: v for k, v in self.plotter.actors.items() if isinstance(v, pyvista.Actor)}

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

    def ui_controls(self, mode=None, default_server_rendering=True, v_show=None):
        """Create a VRow for the UI controls.

        Parameters
        ----------
        mode : str, default: 'trame'
            The UI view mode. Options are:

            * ``'trame'``: Uses a view that can switch between client and server
              rendering modes.
            * ``'server'``: Uses a view that is purely server rendering.
            * ``'client'``: Uses a view that is purely client rendering (generally
              safe without a virtual frame buffer)

        default_server_rendering : bool, default: True
            Whether to use server-side or client-side rendering on-start when
            using the ``'trame'`` mode.

        v_show : bool, optional
            Conditionally show the viewer controls.

        """
        if mode is None:
            mode = self.plotter._theme.trame.default_mode
        if mode not in VALID_UI_MODES:
            raise ValueError(f'`{mode}` is not a valid mode choice. Use one of: {VALID_UI_MODES}')
        if mode != 'trame':
            default_server_rendering = mode == 'server'

        with vuetify.VRow(
            v_show=v_show,
            classes='pa-0 ma-0 align-center',
        ) as row:
            server = row.server
            # Listen to state changes
            server.state.change(self.EDGES)(self.on_edge_visiblity_change)
            server.state.change(self.GRID)(self.on_grid_visiblity_change)
            server.state.change(self.OUTLINE)(self.on_outline_visiblity_change)
            server.state.change(self.AXIS)(self.on_axis_visiblity_change)
            server.state.change(self.SERVER_RENDERING)(self.on_rendering_mode_change)
            vuetify.VDivider(vertical=True, classes='mr-1')
            button(
                click=self.reset_camera,
                icon='mdi-arrow-expand-all',
                tooltip='Reset Camera',
            )
            vuetify.VDivider(vertical=True, classes='mx-1')
            button(
                click=self.view_isometric,
                icon='mdi-axis-arrow',
                tooltip='Perspective view',
            )
            button(
                click=self.view_yz,
                icon='mdi-axis-x-arrow',
                tooltip='Reset Camera X',
            )
            button(
                click=self.view_xz,
                icon='mdi-axis-y-arrow',
                tooltip='Reset Camera Y',
            )
            button(
                click=self.view_xy,
                icon='mdi-axis-z-arrow',
                tooltip='Reset Camera Z',
            )
            vuetify.VDivider(vertical=True, classes='mx-1')
            checkbox(
                model=(self.EDGES, False),
                icons=('mdi-grid', 'mdi-grid-off'),
                tooltip=f"Toggle edge visibility ({{{{ {self.EDGES} ? 'on' : 'off' }}}})",
            )
            checkbox(
                model=(self.OUTLINE, False),
                icons=('mdi-cube', 'mdi-cube-off'),
                tooltip=f"Toggle bounding box ({{{{ {self.OUTLINE} ? 'on' : 'off' }}}})",
            )
            checkbox(
                model=(self.GRID, False),
                icons=('mdi-ruler-square', 'mdi-ruler-square'),
                tooltip=f"Toggle ruler ({{{{ {self.GRID} ? 'on' : 'off' }}}})",
            )
            checkbox(
                model=(self.AXIS, False),
                icons=('mdi-axis-arrow-info', 'mdi-axis-arrow-info'),
                tooltip=f"Toggle axis ({{{{ {self.AXIS} ? 'on' : 'off' }}}})",
            )
            # Server rendering options
            if mode == 'trame':
                vuetify.VDivider(vertical=True, classes='mx-1')
                checkbox(
                    model=(self.SERVER_RENDERING, default_server_rendering),
                    icons=('mdi-dns', 'mdi-open-in-app'),
                    tooltip=f"Toggle rendering mode ({{{{ {self.SERVER_RENDERING} ? 'remote' : 'local' }}}})",
                )
            with vuetify.VRow(
                v_show=(self.SERVER_RENDERING, default_server_rendering),
                classes='pa-0 ma-0 align-center',
            ):

                def attach_screenshot():
                    return server.protocol.addAttachment(self.screenshot())

                button(
                    # Must use single-quote string for JS here
                    click=f"utils.download('screenshot.png', trigger('{server.trigger_name(attach_screenshot)}'), 'image/png')",
                    icon='mdi-file-png-box',
                    tooltip='Save screenshot',
                )

    def ui(
        self,
        mode=None,
        default_server_rendering=True,
        collapse_menu=False,
        add_menu=True,
        **kwargs,
    ):
        """Generate VContainer for PyVista Plotter.

        Parameters
        ----------
        mode : str, default: 'trame'
            The UI view mode. Options are:

            * ``'trame'``: Uses a view that can switch between client and server
              rendering modes.
            * ``'server'``: Uses a view that is purely server rendering.
            * ``'client'``: Uses a view that is purely client rendering (generally
              safe without a virtual frame buffer)

        default_server_rendering : bool, default: True
            Whether to use server-side or client-side rendering on-start when
            using the ``'trame'`` mode.

        collapse_menu : bool, default: False
            Collapse the UI menu (camera controls, etc.) on start.

        add_menu : bool, default: True
            Add a UI controls VCard to the VContainer.

        **kwargs : dict, optional
            Additional keyword arguments are passed to the view being created.

        Returns
        -------
        PyVistaRemoteLocalView, PyVistaRemoteView, or PyVistaLocalView
            Trame view interface for pyvista.

        """
        if mode is None:
            mode = self.plotter._theme.trame.default_mode
        if mode not in VALID_UI_MODES:
            raise ValueError(f'`{mode}` is not a valid mode choice. Use one of: {VALID_UI_MODES}')
        if mode != 'trame':
            default_server_rendering = mode == 'server'

        with vuetify.VContainer(
            fluid=True,
            classes='pa-0 fill-height',
        ) as container:
            server = container.server
            # Initialize state variables
            server.state[self.EDGES] = False
            server.state[self.GRID] = self.plotter.renderer.cube_axes_actor is not None
            server.state[self.OUTLINE] = hasattr(self.plotter.renderer, '_box_object')
            server.state[self.AXIS] = (
                hasattr(self.plotter.renderer, 'axes_widget')
                and self.plotter.renderer.axes_widget.GetEnabled()
            )
            server.state[self.SERVER_RENDERING] = default_server_rendering
            if add_menu:
                server.state[self.SHOW_UI] = not collapse_menu
                with vuetify.VCard(
                    style='position: absolute; top: 20px; left: 20px; z-index: 1; height: 36px;',
                    classes=(f"{{ 'rounded-circle': !{self.SHOW_UI} }}",),
                ):
                    with vuetify.VRow(classes='pa-0 ma-0') as row:
                        button(
                            click=f'{self.SHOW_UI}=!{self.SHOW_UI}',
                            icon='mdi-dots-vertical',
                            tooltip=f"{{{{ {self.SHOW_UI} ? 'Hide' : 'Show' }}}} menu",
                        )
                        self.ui_controls(
                            mode=mode,
                            default_server_rendering=default_server_rendering,
                            v_show=(f'{self.SHOW_UI}',),
                        )
            if mode == 'trame':
                view = PyVistaRemoteLocalView(
                    self.plotter,
                    mode=(
                        # Must use single-quote string for JS here
                        f"{self.SERVER_RENDERING} ? 'remote' : 'local'",
                        'remote' if default_server_rendering else 'local',
                    ),
                    **kwargs,
                )
            elif mode == 'server':
                view = PyVistaRemoteView(self.plotter, **kwargs)
            elif mode == 'client':
                view = PyVistaLocalView(self.plotter, **kwargs)

            self._html_views.add(view)

        return view


def get_or_create_viewer(plotter, suppress_rendering=False):
    """Get or create a Viewer instance for a given Plotter.

    There should be only one Viewer instance per plotter. A Viewer
    can have multiple UI views though.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to return or create the viewer instance for.

    suppress_rendering : bool, default: False
        Suppress rendering on the plotter.

    Returns
    -------
    pyvista.trame.ui.Viewer
        Trame viewer.

    """
    if plotter._id_name in _VIEWERS:
        viewer = _VIEWERS[plotter._id_name]
        if suppress_rendering != plotter.suppress_rendering:
            plotter.suppress_rendering = suppress_rendering
            # TODO: warn user?
        return viewer
    return Viewer(plotter, suppress_rendering=suppress_rendering)


def plotter_ui(
    plotter, mode=None, default_server_rendering=True, collapse_menu=False, add_menu=True, **kwargs
):
    """Create a UI view for the given Plotter.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to create the UI for.

    mode : str, default: 'trame'
        The UI view mode. Options are:

        * ``'trame'``: Uses a view that can switch between client and server
          rendering modes.
        * ``'server'``: Uses a view that is purely server rendering.
        * ``'client'``: Uses a view that is purely client rendering (generally
          safe without a virtual frame buffer)

    default_server_rendering : bool, default: True
        Whether to use server-side or client-side rendering on-start when
        using the ``'trame'`` mode.

    collapse_menu : bool, default: False
        Collapse the UI menu (camera controls, etc.) on start.

    add_menu : bool, default: True
        Add a UI controls VCard to the VContainer.

    **kwargs : dict, optional
        Additional keyword arguments are passed to the viewer being created.

    Returns
    -------
    PyVistaRemoteLocalView, PyVistaRemoteView, or PyVistaLocalView
        Trame view interface for pyvista.

    """
    viewer = get_or_create_viewer(plotter, suppress_rendering=mode == 'client')
    return viewer.ui(
        mode=mode,
        default_server_rendering=default_server_rendering,
        collapse_menu=collapse_menu,
        add_menu=add_menu,
        **kwargs,
    )
