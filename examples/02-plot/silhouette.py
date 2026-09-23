"""
.. _silhouette_example:

Silhouette Highlight
~~~~~~~~~~~~~~~~~~~~

Extract an outline (silhouette) of a polygonal mesh's edges.

The silhouette may be created using the ``silhouette`` keyword with
:meth:`~pyvista.Plotter.add_mesh`, or by using `~pyvista.Plotter.add_silhouette` directly.

"""

import pyvista as pv
from pyvista import examples

# %%
# Prepare a triangulated ``PolyData``
bunny = examples.download_bunny()

# %%
# Now we can display the silhouette of the mesh and compare the result:
pv.plot_compare(
    [bunny] * 2,
    labels=['Silhouette', 'No silhouette'],
    color='lightblue',
    silhouette=[True, False],
    cpos='xy',
    show_axes=False,
)


# %%
# Maybe the default parameters are not enough to really notice the silhouette.
# But by using a ``dict``, it is possible to modify the properties of the
# outline. For example, both color and width could be specified like so:
pl = pv.Plotter()
silhouette = dict(
    color='red',
    line_width=8.0,
)
pl.add_mesh(bunny, silhouette=silhouette)
pl.view_xy()
pl.show()


# %%
# By default, PyVista uses a pretty aggressive decimation level but we might
# want to disable it. It is also possible to display sharp edges:
cylinder = pv.Cylinder(
    center=(0, 0.04, 0),
    direction=(0, 1, 0),
    radius=0.15,
    height=0.03,
).triangulate()

silhouettes = {
    'Silhouette with sharp edges': dict(
        color='red', line_width=8.0, decimate=None, feature_angle=True
    ),
    'Silhouette without sharp edges': dict(color='red', line_width=8.0, decimate=None),
    'No silhouette': False,
}

pv.plot_compare(
    [cylinder] * 3,
    labels=silhouettes.keys(),
    color='lightblue',
    smooth_shading=True,
    silhouette=silhouettes.values(),
    cpos='iso',
    show_axes=False,
)


# %%
# Here is another example:
pl = pv.Plotter()
pl.set_background('black', top='blue')
pl.add_mesh(
    bunny,
    color='green',
    specular=1,
    smooth_shading=True,
    silhouette=dict(line_width=8, color='white'),
)

pl.add_mesh(
    cylinder,
    color='lightblue',
    smooth_shading=True,
    silhouette=dict(decimate=None, feature_angle=True, line_width=8, color='white'),
)
pl.camera_position = pv.CameraPosition(
    position=(-0.2937, 0.2389, 0.3514),
    focal_point=(-0.005879, 0.125, -0.004603),
    viewup=(0.3435, 0.8568, -0.3847),
)
pl.show()
# %%
# .. tags:: plot
