"""
.. _customize_trame_toolbar_example:

Customize Trame toolbar
~~~~~~~~~~~~~~~~~~~~~~~~

Bring more of the power of trame to the jupyter view.
This example shows how to add custom tools using the
`jupyter_kwargs` option with :meth:`~pyvista.Plotter.show`.
"""

from __future__ import annotations

import asyncio

import pyvista as pv
from pyvista.trame.ui.vuetify3 import button
from pyvista.trame.ui.vuetify3 import divider
from pyvista.trame.ui.vuetify3 import select
from pyvista.trame.ui.vuetify3 import slider
from pyvista.trame.ui.vuetify3 import text_field

# %%
# Let's first create the menu items we want to add to the trame's toolbar.
# Here we want a "play" button that will be later connected to a slider
# through the ``button_play`` function. The slider itself will represent the
# "resolution" of the model we will render, a text field where the value of
# the "resolution" will be displayed.
# We will also add a dropdown menu to toggle the visibility of the model.
# The dividers are the same as already used to divide and organize the toolbar.


def custom_tools():
    divider(vertical=True, classes='mx-1')
    button(
        click=button_play,
        icon='mdi-play',
        tooltip='Play',
    )

    slider(
        model=('resolution', 10),
        tooltip='Resolution slider',
        min=3,
        max=20,
        step=1,
        dense=True,
        hide_details=True,
        style='width: 300px',
        classes='my-0 py-0 ml-1 mr-1',
    )
    text_field(
        model=('resolution', 10),
        tooltip='Resolution value',
        readonly=True,
        type='number',
        dense=True,
        hide_details=True,
        style='min-width: 40px; width: 60px',
        classes='my-0 py-0 ml-1 mr-1',
    )

    divider(vertical=True, classes='mx-1')
    select(
        model=('visibility', 'Show'),
        tooltip='Toggle visibility',
        items=['Visibility', ['Hide', 'Show']],
        hide_details=True,
        dense=True,
    )


# %%
# The button callback function ``button_play`` needs to be created before starting
# the server. This function will toggle the boolean state variable ``play``
# and flush the server, i.e. "force" the server to see the change.
# We will see more on the state variables in a bit, but we need to create the
# function here otherwise the server will complain ``button_play`` does not exist.


def button_play():
    state.play = not state.play
    state.flush()


# %%
# We will do a simple rendering of a Cone using `ConeSouce`.
#
# When using the ``pl.show`` method. The function we created ``custom_tools``
# should be passed as a ``jupyter_kwargs`` argument under the key
# ``add_menu_items``.

pl = pv.Plotter(notebook=True)
algo = pv.ConeSource()
mesh_actor = pl.add_mesh(algo)

widget = pl.show(jupyter_kwargs=dict(add_menu_items=custom_tools), return_viewer=True)

# %%
# To interact with ``trame``'s server we need to get the server's state.
#
# We initialize the ``play`` variable in the shared state and this will be
# controlled by the play button we created. Note that when creating the
# ``slider``, the ``text_field`` and the ``select`` tools, we passed something
# like ``model=("variable", value). This will automatically create the variable
# "variable" with value ``value`` in the server's shared state, so we do not need
# to create ``state.resolution`` or ``state.visibility``.

state, ctrl = widget.viewer.server.state, widget.viewer.server.controller
state.play = False
ctrl.view_update = widget.viewer.update

# %%
# Now we can create the callback functions for our menu items.
#
# The functions are decorated with a ``state.change("variable")``. This means
# they will be called when this specific variable has its value changed in the
# server's shared state. When ``resolution`` changes, we want to update the
# resolution of our cone algorithm. When ``visibility`` changes, we want to toggle the
# visibility of our cone.
#
# The ``play`` variable is a little bit trickier. We want to start something like
# a timer so that an animation can be set to play. To do that with ``trame`` we need
# to have an asynchronous function so we can continue to do stuff while the
# "timer" function is running. The ``_play`` function will be called when the ``play``
# variable is changed (when we click the play button, through the ``button_play``
# callback). While ``state.play`` is ``True`` we want to play the animation. We
# change the ``state.resolution`` value, but to really call the ``update_resolution``
# function we need to ``flush`` the server and force it to see the change in
# the shared variables. When ``state.play`` changes to ``False``, the animation stops.
#
# Note that using ``while play: ...`` would not work here because it is not the
# actual state variable, but only an argument value passed to the callback function.


# trame callbacks
@state.change('play')
async def _play(play, **kwargs):  # noqa: ARG001
    while state.play:
        state.resolution += 1
        state.flush()
        if state.resolution >= 20:
            state.play = False
        await asyncio.sleep(0.3)


@state.change('resolution')
def update_resolution(resolution, **kwargs):  # noqa: ARG001
    algo.resolution = resolution
    ctrl.view_update()


@state.change('visibility')
def set_visibility(visibility, **kwargs):  # noqa: ARG001
    toggle = {'Hide': 0, 'Show': 1}
    mesh_actor.visibility = toggle[visibility]
    ctrl.view_update()


widget
