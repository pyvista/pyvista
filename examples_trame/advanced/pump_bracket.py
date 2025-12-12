import asyncio

import matplotlib.pyplot as plt
import numpy as np
from trame.app import asynchronous, get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

import pyvista as pv
from pyvista import examples
from pyvista.trame.ui import plotter_ui

pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "Pump Bracket"

# -----------------------------------------------------------------------------

dataset = examples.download_pump_bracket()

cpos = [
    (0.744, -0.502, -0.830),
    (0.0520, -0.160, 0.0743),
    (-0.180, -0.958, 0.224),
]

n_frames = 32
phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)


pl = pv.Plotter()
pl.enable_anti_aliasing('fxaa')

# Add the undeformed pump bracket
pl.add_mesh(dataset, color="white", opacity=0.5)

# Add the deformed pump bracket with the mode shape
warped = dataset.copy()
actor = pl.add_mesh(warped, show_scalar_bar=False, ambient=0.2)

pl.camera_position = cpos


@state.change("cmap")
def update_cmap(cmap="viridis", **kwargs):
    actor.mapper.lookup_table.cmap = cmap
    ctrl.view_update()


@state.change("phase_index")
def update_phase(phase_index=0, **kwargs):
    phase = phases[phase_index]
    # feel free to change this to visualize different mode shapes
    mode_shape = 'disp_6'
    # use the original unmodified points
    warped.points = dataset.points + dataset[mode_shape] * np.cos(phase) * 0.05
    ctrl.view_update()


@state.change("play")
@asynchronous.task
async def update_play(**kwargs):
    while state.play:
        with state:
            if state.phase_index >= len(phases):
                state.phase_index = 0
            else:
                state.phase_index += 1
            update_phase(state.phase_index)

        await asyncio.sleep(0.00001)


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(state.trame__title)

    with layout.toolbar:
        vuetify.VSpacer()

        vuetify.VSelect(
            label="Color map",
            v_model=("cmap", "viridis"),
            items=("array_list", plt.colormaps()),
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1 ml-2",
            style="max-width: 250px",
        )

        vuetify.VSlider(
            v_model=("phase_index", 0),
            min=0,
            max=len(phases) - 1,
            hide_details=True,
            dense=True,
            style="max-width: 200px",
        )
        vuetify.VCheckbox(
            v_model=("play", False),
            off_icon="mdi-play",
            on_icon="mdi-stop",
            hide_details=True,
            dense=True,
            classes="mx-2",
        )

    with layout.content:
        # Use PyVista UI template for Plotters
        view = plotter_ui(pl)
        ctrl.view_update = view.update
        ctrl.view_update_image = view.update_image
        ctrl.view_reset_camera = view.reset_camera

    # hide footer
    layout.footer.hide()

server.start()
