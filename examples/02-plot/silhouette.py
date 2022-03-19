"""
.. _silhouette_example:

Silhouette Highlight
~~~~~~~~~~~~~~~~~~~~

Extract a subset of the edges of a polygonal mesh to generate an outline
(silhouette) of a mesh.
"""

import pyvista
from pyvista import examples

###############################################################################
# Prepare a triangulated ``PolyData``
bunny = examples.download_bunny()

###############################################################################
# Now we can display the silhouette of the mesh and compare the result:
plotter = pyvista.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(bunny, color='tan', silhouette=True)
plotter.add_text("Silhouette")
plotter.view_xy()
plotter.subplot(0, 1)
plotter.add_mesh(bunny, color='tan')
plotter.add_text("No silhouette")
plotter.view_xy()
plotter.show()


###############################################################################
# Maybe the default parameters are not enough to really notice the silhouette.
# But by using a ``dict``, it is possible to modify the properties of the
# outline. For example, color and width could be specified like so:
plotter = pyvista.Plotter()
silhouette = dict(
    color='red',
    line_width=8.0,
)
plotter.add_mesh(bunny, silhouette=silhouette)
plotter.view_xy()
plotter.show()


###############################################################################
# By default, PyVista uses a pretty aggressive decimation level but we might
# want to disable it. It is also possible to display sharp edges:
cylinder = pyvista.Cylinder(
    center=(0, 0.04, 0), direction=(0, 1, 0), radius=0.15, height=0.03
).triangulate()

plotter = pyvista.Plotter(shape=(1, 3))
plotter.subplot(0, 0)
plotter.add_mesh(
    cylinder,
    color='tan',
    smooth_shading=True,
    silhouette=dict(color='red', line_width=8.0, decimate=None, feature_angle=True),
)
plotter.add_text("Silhouette with sharp edges")
plotter.view_isometric()
plotter.subplot(0, 1)
plotter.add_mesh(
    cylinder,
    color='tan',
    smooth_shading=True,
    silhouette=dict(color='red', line_width=8.0, decimate=None),
)
plotter.add_text("Silhouette without sharp edges")
plotter.view_isometric()
plotter.subplot(0, 2)
plotter.add_mesh(cylinder, color='tan', smooth_shading=True)
plotter.add_text("No silhouette")
plotter.view_isometric()
plotter.show()


###############################################################################
# Here is another example:
dragon = examples.download_dragon()
plotter = pyvista.Plotter()
plotter.set_background('black', 'blue')
plotter.add_mesh(
    dragon,
    color="green",
    specular=1,
    smooth_shading=True,
    silhouette=dict(line_width=8, color='white'),
)

plotter.add_mesh(
    cylinder,
    color='tan',
    smooth_shading=True,
    silhouette=dict(decimate=None, feature_angle=True, line_width=8, color='white'),
)
plotter.camera_position = [
    (-0.2936731887752889, 0.2389060430625446, 0.35138839367034236),
    (-0.005878899246454239, 0.12495124898850918, -0.004603400826454163),
    (0.34348225747312017, 0.8567703221182346, -0.38466160965007384),
]
plotter.show()
