import tempfile

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

import pyvista as pv
from pyvista.trame.ui import ui_container

# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "Contour"
ctrl.on_server_ready.add(ctrl.view_update)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

pl = pv.Plotter()


@state.change("files")
def load_client_files(files, **kwargs):
    if files is None or len(files) == 0:
        pl.clear_actors()
        return

    if files and len(files):
        if not files[0].get("content"):
            return

        for file in files:
            print(f'Load {file.get("name")}')
            bytes = file.get("content")
            with tempfile.NamedTemporaryFile(suffix=file.get("name")) as path:
                with open(path.name, 'wb') as f:
                    f.write(bytes)
                ds = pv.read(path.name)
            pl.add_mesh(ds, name=file.get("name"))
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
            multiple=True,
            show_size=True,
            small_chips=True,
            truncate_length=25,
            v_model=("files", None),
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
            ui_container(server, pl)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
