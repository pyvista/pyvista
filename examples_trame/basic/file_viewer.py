import tempfile

from trame.app import get_server
from trame.app.file_upload import ClientFile
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

import pyvista as pv
from pyvista.trame.ui import plotter_ui

# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "File Viewer"
ctrl.on_server_ready.add(ctrl.view_update)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

pl = pv.Plotter()


@server.state.change("file_exchange")
def handle(file_exchange, **kwargs):
    file = ClientFile(file_exchange)

    if file.content:
        print(file.info)
        bytes = file.content
        with tempfile.NamedTemporaryFile(suffix=file.name) as path:
            with open(path.name, 'wb') as f:
                f.write(bytes)
            ds = pv.read(path.name)
        pl.add_mesh(ds, name=file.name)
        pl.reset_camera()
    else:
        pl.clear_actors()
        pl.reset_camera()


# -----------------------------------------------------------------------------
# Web App setup
# -----------------------------------------------------------------------------

state.trame__title = "File Viewer"

with SinglePageLayout(server) as layout:
    layout.title.set_text("File Viewer")
    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VFileInput(
            show_size=True,
            small_chips=True,
            truncate_length=25,
            v_model=("file_exchange", None),
            dense=True,
            hide_details=True,
            style="max-width: 300px;",
        )
        vuetify.VProgressLinear(
            indeterminate=True, absolute=True, bottom=True, active=("trame__busy",)
        )

    with layout.content:
        with vuetify.VContainer(
            fluid=True, classes="pa-0 fill-height", style="position: relative;"
        ):
            # Use PyVista UI template for Plotters
            view = plotter_ui(pl)
            ctrl.view_update = view.update

server.start()
