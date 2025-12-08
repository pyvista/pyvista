# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
#   "trame>=2.5.2",
# ]
# ///

from __future__ import annotations

from pathlib import Path
import tempfile

from trame.app import get_server
from trame.app.file_upload import ClientFile
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import pyvista as pv
from pyvista.trame.ui import plotter_ui

# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'File Viewer'
ctrl.on_server_ready.add(ctrl.view_update)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

pl = pv.Plotter()


@server.state.change('file_exchange')
def handle(file_exchange, **kwargs):  # noqa: ARG001
    # Vuetify3 File Input always returns list
    if file_exchange and len(file_exchange) > 0:
        file = ClientFile(file_exchange[0])

        if file.content:
            print(file.info)
            bytes_ = file.content
            with tempfile.NamedTemporaryFile(suffix=file.name) as path:
                with Path(path.name).open('wb') as f:
                    f.write(bytes_)
                ds = pv.read(path.name)
            pl.add_mesh(ds, name=file.name)
            pl.reset_camera()
        else:
            pl.clear_actors()
            pl.reset_camera()


# -----------------------------------------------------------------------------
# Web App setup
# -----------------------------------------------------------------------------

state.trame__title = 'File Viewer'

with SinglePageLayout(server) as layout:
    layout.title.set_text('File Viewer')
    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VFileInput(
            show_size=True,
            chips=True,
            truncate_length=25,
            v_model=('file_exchange', None),
            density='compact',
            hide_details=True,
            style='max-width: 300px;',
        )
        vuetify3.VProgressLinear(
            indeterminate=True,
            absolute=True,
            bottom=True,
            active=('trame__busy',),
        )

    with layout.content:
        with vuetify3.VContainer(
            fluid=True,
            classes='pa-0 fill-height',
            style='position: relative;',
        ):
            # Use PyVista UI template for Plotters
            view = plotter_ui(pl)
            ctrl.view_update = view.update

server.start()
