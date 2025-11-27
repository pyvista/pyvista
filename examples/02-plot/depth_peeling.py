"""
.. _depth_peeling_example:

Depth Peeling
~~~~~~~~~~~~~
Depth peeling is a technique to correctly render translucent geometry.  This is
not enabled by default in :attr:`pyvista.global_theme
<pyvista.plotting.themes.Theme>` as some operating systems and versions of VTK
have issues with this routine.

For this example, we will showcase the difference that depth peeling
provides. See :func:`~pyvista.Plotter.enable_depth_peeling`.

"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import pyvista as pv
from pyvista import examples

# %%
centers = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
radii = [1, 0.5, 0.5, 0.5, 0.5]

spheres = pv.MultiBlock()
for i, c in enumerate(centers):
    spheres.append(pv.Sphere(center=c, radius=radii[i]))

# %%
dargs = dict(opacity=0.5, color='red', smooth_shading=True)

pl = pv.Plotter(shape=(1, 2))

pl.add_mesh(spheres, **dargs)
pl.enable_depth_peeling(10)
pl.add_text('Depth Peeling')

pl.subplot(0, 1)
pl.add_text('Standard')
pl.add_mesh(spheres.copy(), **dargs)

pl.link_views()
pl.camera_position = pv.CameraPosition(
    position=(11.7, 4.7, -4.33), focal_point=(0.0, 0.0, 0.0), viewup=(0.3, 0.07, 0.9)
)
pl.show()

# %%
# The following room surfaces example mesh, provided courtesy of
# `Sam Potter <https://github.com/sampotter>`_ has coincident topology and
# depth rendering helps correctly render those geometries when a global
# opacity value is used.

room = examples.download_room_surface_mesh()

pl = pv.Plotter(shape=(1, 2))

pl.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0)
pl.add_mesh(room, opacity=0.5, color='lightblue')
pl.add_text('Depth Peeling')

pl.subplot(0, 1)
pl.add_text('Standard')
pl.add_mesh(room.copy(), opacity=0.5, color='lightblue')

pl.link_views()
pl.camera_position = pv.CameraPosition(
    position=(43.6, 49.5, 19.8), focal_point=(0.0, 2.25, 0.0), viewup=(-0.57, 0.70, -0.42)
)

pl.show()


# %%
# And here is another example when rendering many translucent contour
# surfaces.

mesh = examples.download_brain().resample(0.5, anti_aliasing=True)
contours = mesh.contour(5)
cmap = 'viridis_r'

pl = pv.Plotter(shape=(1, 2))

pl.add_mesh(contours, opacity=0.5, cmap=cmap)
pl.enable_depth_peeling(10)
pl.add_text('Depth Peeling')

pl.subplot(0, 1)
pl.add_text('Standard')
pl.add_mesh(contours.copy(), opacity=0.5, cmap=cmap)

pl.link_views()
pl.camera_position = pv.CameraPosition(
    position=(418.3, 659.0, 53.8), focal_point=(90.2, 111.5, 90.0), viewup=(0.03, 0.05, 1.0)
)
pl.show()
# %%
# .. tags:: plot
