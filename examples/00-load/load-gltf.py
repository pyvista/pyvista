"""
.. _load_gltf:

Working with a glTF Files
~~~~~~~~~~~~~~~~~~~~~~~~~
Import a glTF directly into a PyVista plotting scene.  For more
details regarding the glTF format, see:
https://www.khronos.org/gltf/

Note this feature is only available for ``vtk>=9``.

First, download the examples.

"""

import pyvista
from pyvista import examples
helmet_file = examples.gltf.download_damaged_helmet()
cubemap = examples.download_sky_box_cube_map()


###############################################################################
# Setup the plotter and enable environment textures.  This works well
# for physically based rendering enabled meshes like the damaged
# helmet example.

pl = pyvista.Plotter()
pl.import_gltf(helmet_file)
pl.set_environment_texture(cubemap)
pl.show()


###############################################################################
# You can also directly read in gltf files and extract the underlying
# mesh.

block = pyvista.read(helmet_file)
mesh = block[0][0][0]
mesh.plot(color='tan', show_edges=True, cpos='xy')
