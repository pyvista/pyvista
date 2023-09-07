"""
.. _customize_trame_toolbar_example

Customize Trame toolbar
~~~~~~~~~~~~~~~~~~~~~~~~

Bring more of the power of trame to the jupyter view.
"""
import asyncio

from trame.widgets import vuetify
import vtk

import pyvista as pv
from pyvista.trame.ui.vuetify2 import button, select, slider, text_field

###############################################################################
# Here is an explanation of the `btn_play`.


def btn_play():
    state.play = not state.play
    state.flush()


###############################################################################
# Here is an explanation of the custom tools.


def custom_tools():
    vuetify.VDivider(vertical=True, classes='mx-1')
    button(
        click=btn_play,
        icon='mdi-play',
        tooltip='Play',
    )

    slider(
        model=("resolution", 0),
        tooltip="Resolution slider",
        min=3,
        max=20,
        step=1,
        dense=True,
        hide_details=True,
        style="width: 300px",
        classes='my-0 py-0 ml-1 mr-1',
    )
    text_field(
        model=("resolution", 0),
        tooltip="Resolution value",
        readonly=True,
        type="number",
        dense=True,
        hide_details=True,
        style="min-width: 40px; width: 60px",
        classes='my-0 py-0 ml-1 mr-1',
    )

    vuetify.VDivider(vertical=True, classes='mx-1')
    select(
        model=("visibility", "Show"),
        tooltip="Toggle visibility",
        items=['Visibility', ["Hide", "Show"]],
        hide_details=True,
        dense=True,
    )


###############################################################################
# Here is an explanation of the Plotting.

pl = pv.Plotter()
algo = vtk.vtkConeSource()
mesh_actor = pl.add_mesh(algo)

viewer = pl.show(jupyter_kwargs=dict(add_menu_items=custom_tools), return_viewer=True)

###############################################################################
# Here is an explanation of the setting.

state, ctrl = viewer.viewer.server.state, viewer.viewer.server.controller
state.play = False
ctrl.view_update = viewer.viewer.update

###############################################################################
# Here is an explanation of trame callbacks.


# trame callbacks
@state.change("play")
async def _play(play, **kwargs):
    while state.play:
        state.resolution += 1
        state.flush()
        if state.resolution >= 20:
            state.play = False
        await asyncio.sleep(0.3)


@state.change("resolution")
def update_resolution(resolution, **kwargs):
    algo.SetResolution(resolution)
    ctrl.view_update()


@state.change("visibility")
def set_visibility(visibility, **kwargs):
    toggle = {"Hide": 0, "Show": 1}
    mesh_actor.SetVisibility(toggle[visibility])
    ctrl.view_update()


viewer
