"""
.. _checkbox_widget_example:

Checkbox Widget
~~~~~~~~~~~~~~~

Use a checkbox to turn on/off the visibility of meshes in a scene.

See :func:`pyvista.Plotter.add_checkbox_button_widget` for more details.

"""

# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_start_ignore
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

# widgets do not work in interactive examples
from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import pyvista as pv

# %%
# Single Checkbox
# +++++++++++++++

mesh = pv.Sphere()

pl = pv.Plotter()
actor = pl.add_mesh(mesh)


def toggle_vis(flag):
    actor.SetVisibility(flag)


pl.add_checkbox_button_widget(toggle_vis, value=True)
pl.show()

# %%
# Multiple Checkboxes
# +++++++++++++++++++
#
# In this example, we will add many meshes to a scene with unique colors and
# create corresponding checkboxes for those meshes of the same color to toggle
# their visibility in the scene.

colors = [
    ['ff0000', '28e5da', '0000ff'],
    ['ffff00', 'c8bebe', 'f79292'],
    ['fffff0', 'f18c1d', '23dcaa'],
    ['d785ec', '9d5b13', 'e4e0b1'],
    ['894509', 'af45f5', 'fff000'],
]


class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, actor):
        self.actor = actor

    def __call__(self, state):
        self.actor.SetVisibility(state)


# %%

# Widget size
size = 50

pl = pv.Plotter()

Startpos = 12
for i, lst in enumerate(colors):
    for j, color in enumerate(lst):
        actor = pl.add_mesh(pv.Sphere(center=(i, j, 0)), color=color)
        # Make a separate callback for each widget
        callback = SetVisibilityCallback(actor)
        pl.add_checkbox_button_widget(
            callback,
            value=True,
            position=(5.0, Startpos),
            size=size,
            border_size=1,
            color_on=color,
            color_off='grey',
            background_color='grey',
        )
        Startpos = Startpos + size + (size // 10)

pl.show()
# %%
# .. tags:: widgets
