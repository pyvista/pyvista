"""PyVista Trame Viewer class for a Vue 3 client.

This class, derived from `pyvista.trame.ui.base_viewer`,
is intended for use with a trame application where the client type is "vue3".
Therefore, the `ui` method implemented by this class utilizes the API of Vuetify 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trame.ui.vuetify3 import VAppLayout
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

from pyvista.trame.views import PyVistaLocalView
from pyvista.trame.views import PyVistaRemoteLocalView
from pyvista.trame.views import PyVistaRemoteView

from .base_viewer import BaseViewer

if TYPE_CHECKING:
    from trame_client.ui.core import AbstractLayout


def button(click, icon, tooltip):  # numpydoc ignore=PR01
    """Create a vuetify button."""
    with vuetify.VTooltip(location='bottom'):
        with vuetify.Template(v_slot_activator=('{ props }',)):
            with vuetify.VBtn(
                icon=True,
                v_bind=('props',),
                variant='text',
                size='default',
                click=click,
                density='comfortable',
            ):
                vuetify.VIcon(icon)
        html.Span(tooltip)


def checkbox(model, icons, tooltip):  # numpydoc ignore=PR01
    """Create a vuetify checkbox."""
    with vuetify.VTooltip(location='bottom'):
        with vuetify.Template(v_slot_activator=('{ props }',)):
            with html.Div(v_bind=('props',)):
                vuetify.VCheckbox(
                    v_model=model,
                    true_icon=icons[0],
                    false_icon=icons[1],
                    density='comfortable',
                    hide_details=True,
                )
        html.Span(tooltip)


def slider(model, tooltip, **kwargs):  # numpydoc ignore=PR01
    """Create a vuetify slider."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator=('{ props }',)):
            with html.Div(v_bind=('props',)):
                vuetify.VSlider(v_model=model, **kwargs)
        html.Span(tooltip)


def text_field(model, tooltip, **kwargs):  # numpydoc ignore=PR01
    """Create a vuetify text field."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator=('{ props }',)):
            with html.Div(v_bind=('props',)):
                vuetify.VTextField(v_model=model, **kwargs)
        html.Span(tooltip)


def select(model, tooltip, **kwargs):  # numpydoc ignore=PR01
    """Create a vuetify select menu."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator=('{ props }',)):
            with html.Div(v_bind=('props',)):
                vuetify.VSelect(v_model=model, **kwargs)
        html.Span(tooltip)


def divider(**kwargs):  # numpydoc ignore=PR01
    """Create a vuetify divider."""
    vuetify.VDivider(**kwargs)


class Viewer(BaseViewer):
    """Viewer implementation compatible with Vue 3 Trame Applications."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_layout(self, *args, **kwargs) -> AbstractLayout:
        """Create instance of an AbstractLayout which is appropriate for this viewer.

        Parameters
        ----------
        *args : tuple
            Positional arguments.

        **kwargs : dict, optional
            Keyword arguments.

        Returns
        -------
        VAppLayout (vue3)
            A layout this viewer can be embedded in.

        """
        return VAppLayout(*args, **kwargs)

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
        with vuetify.VRow(
            v_show=v_show,
            classes='pa-0 ma-0 align-center fill-height',
            style='flex-wrap: nowrap',
        ) as row:
            server = row.server
            # Listen to state changes
            server.state.change(self.EDGES)(self.on_edge_visibility_change)
            server.state.change(self.GRID)(self.on_grid_visibility_change)
            server.state.change(self.OUTLINE)(self.on_outline_visibility_change)
            server.state.change(self.AXIS)(self.on_axis_visibility_change)
            server.state.change(self.SERVER_RENDERING)(self.on_rendering_mode_change)
            server.state.change(self.PARALLEL)(self.on_parallel_projection_change)
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
                    tooltip=f'Toggle rendering mode '
                    f"({{{{ {self.SERVER_RENDERING} ? 'remote' : 'local' }}}})",
                )
            with vuetify.VRow(
                v_show=(self.SERVER_RENDERING, default_server_rendering),
                classes='pa-0 ma-0 align-center fill-height',
                style='flex-wrap: nowrap; flex: unset',
            ):
                checkbox(
                    model=(self.PARALLEL, False),
                    icons=('mdi-camera-off', 'mdi-camera-switch'),
                    tooltip=f'Toggle parallel projection '
                    f"({{{{ {self.PARALLEL} ? 'on' : 'off' }}}})",
                )

                def attach_screenshot():
                    return server.protocol.addAttachment(self.screenshot())

                button(
                    # Must use single-quote string for JS here
                    click="utils.download('screenshot.png', "
                    f"trigger('{server.trigger_name(attach_screenshot)}'), "
                    "'image/png')",
                    icon='mdi-file-png-box',
                    tooltip='Save screenshot',
                )

            def attach_export():
                return server.protocol.addAttachment(self.export())

            button(
                # Must use single-quote string for JS here
                click="utils.download('scene-export.html', "
                f"trigger('{server.trigger_name(attach_export)}'), "
                "'application/octet-stream')",
                icon='mdi-download',
                tooltip='Export scene as HTML',
            )

    def ui(
        self,
        mode=None,
        default_server_rendering=True,
        collapse_menu=False,
        add_menu=True,
        add_menu_items=None,
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

        add_menu_items : callable, default: None
            Append more UI controls to the VCard menu. Should be a function similar to
            `Viewer.ui_controls()`.

        **kwargs : dict, optional
            Additional keyword arguments are passed to the view being created.

        Returns
        -------
        PyVistaRemoteLocalView, PyVistaRemoteView, or PyVistaLocalView
            Trame view interface for pyvista.

        """
        if mode is None:
            mode = self.plotter._theme.trame.default_mode
        if mode not in self.VALID_UI_MODES:
            msg = f'`{mode}` is not a valid mode choice. Use one of: {self.VALID_UI_MODES}'
            raise ValueError(msg)
        if mode != 'trame':
            default_server_rendering = mode == 'server'

        with vuetify.VContainer(
            fluid=True,
            classes='pa-0 fill-height',
            style='position: relative',
            trame_server=self.server,
        ) as container:
            server = container.server
            # Initialize state variables
            server.state[self.EDGES] = False
            server.state[self.GRID] = self.plotter.renderer.cube_axes_actor is not None
            server.state[self.OUTLINE] = self.plotter.renderer._box_object is not None
            server.state[self.AXIS] = (
                self.plotter.renderer.axes_widget is not None
                and self.plotter.renderer.axes_widget.GetEnabled()
            )
            server.state[self.SERVER_RENDERING] = default_server_rendering
            if add_menu:
                server.state[self.SHOW_UI] = not collapse_menu
                with vuetify.VCard(
                    style='position: absolute; top: 20px; left: 20px; z-index: 1; height: 36px;',
                    classes=(f"{{ 'rounded-circle': !{self.SHOW_UI} }}",),
                ) as self.menu:
                    with vuetify.VRow(
                        classes='pa-0 ma-0 align-center fill-height',
                        style='flex-wrap: nowrap',
                    ):
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
                        if callable(add_menu_items):
                            with vuetify.VRow(
                                v_show=(f'{self.SHOW_UI}',),
                                classes='pa-0 ma-0 align-center',
                            ):
                                add_menu_items()
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
            if add_menu:
                view.menu = self.menu

        return view
