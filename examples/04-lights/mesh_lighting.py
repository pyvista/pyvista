"""
Disabling Mesh Lighting
~~~~~~~~~~~~~~~~~~~~~~~

While plotters have a default set of lights and a seemingly endless option for
customizing lighting conditions in general, meshes have the option to opt out of
lighting altogether. Pass ``lighting=False`` to ``add_mesh`` to disable lighting
for the given mesh.
"""
# sphinx_gallery_thumbnail_number = 1
import pyvista as pv
from pyvista import examples

mesh = examples.download_cow()
mesh.rotate_x(90)
shifted = mesh.copy()
shifted.translate((0, 6, 0))

plotter = pv.Plotter()
plotter.add_mesh(mesh, color='tan')
plotter.add_mesh(shifted, color='tan', show_edges=True, lighting=False)
plotter.show()


###############################################################################
# Due to the obvious lack of depth detail this mostly makes sense for meshes
# with non-trivial colors or textures. If it weren't for the edges being drawn,
# the second mesh would be practically impossible to understand even with the
# option to interactively explore the surface:

shifted.plot(color='tan', lighting=False)

###############################################################################
# For further examples about fine-tuning mesh properties that affect
# light rendering, see the :ref:`ref_lighting_properties_example` example.
