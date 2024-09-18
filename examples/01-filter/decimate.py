"""
.. _decimate_example:

Decimation
~~~~~~~~~~

Decimate a mesh

"""

# sphinx_gallery_thumbnail_number = 4
from __future__ import annotations

import pyvista as pv
from pyvista import examples

mesh = examples.download_face()

# Define a camera position that shows this mesh properly
cpos = [(0.4, -0.07, -0.31), (0.05, -0.13, -0.06), (-0.1, 1, 0.08)]
dargs = dict(show_edges=True, color=True)

# Preview the mesh
mesh.plot(cpos=cpos, **dargs)

# %%
# Now let's define a target reduction and compare the
# :func:`pyvista.PolyDataFilters.decimate` and
# :func:`pyvista.PolyDataFilters.decimate_pro` filters.
target_reduction = 0.7
print(f'Reducing {target_reduction * 100.0} percent out of the original mesh')

# %%
decimated = mesh.decimate(target_reduction)

decimated.plot(cpos=cpos, **dargs)


# %%
pro_decimated = mesh.decimate_pro(target_reduction, preserve_topology=True)

pro_decimated.plot(cpos=cpos, **dargs)


# %%
# Side by side comparison:

# sphinx_gallery_start_ignore
# text missing in interactive
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter(shape=(1, 3))
pl.add_mesh(mesh, **dargs)
pl.add_text('Input mesh', font_size=24)
pl.camera_position = cpos
pl.reset_camera()
pl.subplot(0, 1)
pl.add_mesh(decimated, **dargs)
pl.add_text('Decimated mesh', font_size=24)
pl.camera_position = cpos
pl.reset_camera()
pl.subplot(0, 2)
pl.add_mesh(pro_decimated, **dargs)
pl.add_text('Pro Decimated mesh', font_size=24)
pl.camera_position = cpos
pl.reset_camera()
pl.link_views()
pl.show()
# %%
# .. tags:: filter
