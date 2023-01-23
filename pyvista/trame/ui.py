# flake8: noqa: D102,D103,D107
"""PyVista Trame User Interface.

This module builds a base UI for manipulating a PyVista Plotter.
The UI generated here is the default for rendering in Jupyter
environments and provides a starting point for custom user-built
applications.
"""
import io

from trame.ui.vuetify import VAppLayout
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
        self._html_views = set()  # TODO: weakref

        self.plotter = plotter
        self.plotter.suppress_rendering = suppress_rendering

        # State variable names
        self.SHOW_UI = f'{plotter._id_name}_show_ui'
        self.GRID = f'{plotter._id_name}_grid_visibility'
        self.OUTLINE = f'{plotter._id_name}_outline_visibility'
        self.EDGES = f'{plotter._id_name}_edge_visibility'
        self.AXIS = f'{plotter._id_name}_axis_visiblity'
        self.SERVER_RENDERING = f'{plotter._id_name}_use_server_rendering'

    def update(self, **kwargs):
        """Update all associated views."""
        for view in self._html_views:
            view.update()

    def push_camera(self, **kwargs):
        """Push camera to all associated views."""
        for view in self._html_views:
            view.push_camera()

    def reset_camera(self, **kwargs):
        """Reset camera for all associated views."""
        for view in self._html_views:
            view.reset_camera()

    def update_image(self, **kwargs):
        """Update image for all associated views."""
        for view in self._html_views:
            if hasattr(view, 'update_image'):
                view.update_image()

    def view_isometric(self):
        """View isometric."""
        self.plotter.view_isometric()
        self.push_camera()
        self.update()

    def view_yz(self):
        """View YZ plane."""
        self.plotter.view_yz()
        self.push_camera()
        self.update()

    def view_xz(self):
        """View XZ plane."""
        self.plotter.view_xz()
        self.push_camera()
        self.update()

    def view_xy(self):
        """View XY plane."""
        self.plotter.view_xy()
        self.push_camera()
        self.update()

    def on_edge_visiblity_change(self, **kwargs):
        """Toggle edge visibility for all actors."""
        value = kwargs[self.EDGES]
        for _, actor in self.plotter.actors.items():
            if isinstance(actor, pyvista.Actor):
                actor.prop.show_edges = value
        self.update()

    def on_grid_visiblity_change(self, **kwargs):
        """Handle axes grid visibility."""
        if kwargs[self.GRID]:
            self.plotter.show_grid()
        else:
            self.plotter.remove_bounds_axes()
        self.update()

    def on_outline_visiblity_change(self, **kwargs):
        """Handle outline visibility."""
        if kwargs[self.OUTLINE]:
            self.plotter.add_bounding_box(reset_camera=False)
        else:
            self.plotter.remove_bounding_box()
        self.update()

    def on_axis_visiblity_change(self, **kwargs):
        """Handle outline visibility."""
        if kwargs[self.AXIS]:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()
        self.update()

    def on_rendering_mode_change(self, **kwargs):
        """Handle any configurations when the render mode changes between client and server."""
        if not kwargs[self.SERVER_RENDERING]:
            self.push_camera()
            self.update()

    @property
    def actors(self):
        """Get dataset actors."""
        return {k: v for k, v in self.plotter.actors.items() if isinstance(v, pyvista.Actor)}

    def screenshot(self):
        """Take screenshot and add attachament."""
        self.plotter.render()
        self.update()  # makes sure the plotter and views are in sync
        buffer = io.BytesIO()
        self.plotter.screenshot(filename=buffer)
        buffer.seek(0)
        return memoryview(buffer.read())

    def ui(self, mode=None, default_server_rendering=True, collapse_menu=False, **kwargs):
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

        **kwargs
            Addition keyword arguments are passed to the view being created.

        """
        if mode is None:
            mode = pyvista.global_theme.trame.default_mode
        if mode not in VALID_UI_MODES:
            raise ValueError(f'`{mode}` is not a valid mode choice. Use one of: {VALID_UI_MODES}')
        if mode != 'trame':
            default_server_rendering = mode == 'server'

        with vuetify.VContainer(
            fluid=True,
            classes='pa-0 fill-height',
        ) as container:
            server = container.server
            # Listen to state changes
            server.state.change(self.EDGES)(self.on_edge_visiblity_change)
            server.state.change(self.GRID)(self.on_grid_visiblity_change)
            server.state.change(self.OUTLINE)(self.on_outline_visiblity_change)
            server.state.change(self.AXIS)(self.on_axis_visiblity_change)
            server.state.change(self.SERVER_RENDERING)(self.on_rendering_mode_change)
            with vuetify.VCard(
                style='position: absolute; top: 20px; left: 20px; z-index: 1; height: 36px;',
                classes=(f"{{ 'rounded-circle': !{self.SHOW_UI} }}",),
            ):
                with vuetify.VRow(classes='pa-0 ma-0'):
                    button(
                        click=f'{self.SHOW_UI}=!{self.SHOW_UI}',
                        icon='mdi-dots-vertical',
                        tooltip=f"{{{{ {self.SHOW_UI} ? 'Hide' : 'Show' }}}} menu",
                    )
                    with vuetify.VRow(
                        v_show=(f'{self.SHOW_UI}', not collapse_menu),
                        classes='pa-0 ma-0 align-center',
                    ):
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
                            checkbox(
                                model=(self.AXIS, False),
                                icons=('mdi-axis-arrow-info', 'mdi-axis-arrow-info'),
                                tooltip=f"Toggle axis ({{{{ {self.AXIS} ? 'on' : 'off' }}}})",
                            )

                            def attach_screenshot():
                                return server.protocol.addAttachment(self.screenshot())

                            button(
                                # Must use single-quote string for JS here
                                click=f"utils.download('screenshot.png', trigger('{server.trigger_name(attach_screenshot)}'), 'image/png')",
                                icon='mdi-file-png-box',
                                tooltip='Save screenshot',
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
    """
    if plotter._id_name in _VIEWERS:
        viewer = _VIEWERS[plotter._id_name]
        if suppress_rendering != plotter.suppress_rendering:
            plotter.suppress_rendering = suppress_rendering
            # TODO: warn user?
        return viewer
    return Viewer(plotter, suppress_rendering=suppress_rendering)


def plotter_ui(plotter, mode=None, default_server_rendering=True, collapse_menu=False, **kwargs):
    """Create a UI view for the given Plotter."""
    viewer = Viewer(plotter, suppress_rendering=mode == 'client')  # TODO: get or create
    return viewer.ui(
        mode=mode,
        default_server_rendering=default_server_rendering,
        collapse_menu=collapse_menu,
        **kwargs,
    )


def initialize(server, plotter, mode=None, default_server_rendering=True, collapse_menu=False):
    """Generate the UI for a given plotter."""
    state = server.state
    state.trame__title = UI_TITLE

    viewer = get_or_create_viewer(plotter, suppress_rendering=mode == 'client')

    with VAppLayout(server, template_name=plotter._id_name):
        viewer.ui(
            mode=mode,
            default_server_rendering=default_server_rendering,
            collapse_menu=collapse_menu,
        )

    # Returns the UI identifier (used in `template_name`)
    return plotter._id_name
