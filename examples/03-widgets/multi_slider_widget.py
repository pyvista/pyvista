"""
.. _multi_slider_widget_example:

Multiple Slider Widgets
~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~pyvista.Plotter.add_slider_widget` and a class-based callback
to track multiple slider widgets for updating a single mesh.

In this example we simply change a few parameters for the
:func:`pyvista.Sphere` method, but this could easily be applied to any
mesh-generating/altering code.

"""

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


class MyCustomRoutine:  # noqa: D101
    def __init__(self, mesh):
        self.output = mesh  # Expected PyVista mesh type
        # default parameters
        self.kwargs = {
            'radius': 0.5,
            'theta_resolution': 30,
            'phi_resolution': 30,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        result = pv.Sphere(**self.kwargs)
        self.output.copy_from(result)


# %%

starting_mesh = pv.Sphere()
engine = MyCustomRoutine(starting_mesh)

# %%

pl = pv.Plotter()
pl.add_mesh(starting_mesh, show_edges=True)
pl.add_slider_widget(
    callback=lambda value: engine('phi_resolution', int(value)),
    rng=[3, 60],
    value=30,
    title='Phi Resolution',
    pointa=(0.025, 0.1),
    pointb=(0.31, 0.1),
    style='modern',
)
pl.add_slider_widget(
    callback=lambda value: engine('theta_resolution', int(value)),
    rng=[3, 60],
    value=30,
    title='Theta Resolution',
    pointa=(0.35, 0.1),
    pointb=(0.64, 0.1),
    style='modern',
)
pl.add_slider_widget(
    callback=lambda value: engine('radius', value),
    rng=[0.1, 1.5],
    value=0.5,
    title='Radius',
    pointa=(0.67, 0.1),
    pointb=(0.98, 0.1),
    style='modern',
)
pl.show()
# %%
# .. tags:: widgets
