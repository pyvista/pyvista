"""
.. _load_gltf:

Working with a glTF Files
~~~~~~~~~~~~~~~~~~~~~~~~~
Import a glTF directly into a PyVista plotting scene.  For more
details regarding the glTF format, see:
https://www.khronos.org/gltf/

Note this feature is only available for ``vtk>=9``.

First, download the examples.  Note that here we're using a high
dynamic range texture since glTF files generally contain physically
based rendering and VTK v9 supports high dynamic range textures.

"""

import pyvista
from pyvista import examples
helmet_file = examples.gltf.download_damaged_helmet()
texture = examples.hdr.download_dikhololo_night()


###############################################################################
# Setup the plotter and enable environment textures.  This works well
# for physically based rendering enabled meshes like the damaged
# helmet example.

pl = pyvista.Plotter()
pl.import_gltf(helmet_file)
pl.set_environment_texture(texture)
pl.camera.zoom(1.7)
pl.show()


###############################################################################
# You can also directly read in gltf files and extract the underlying
# mesh.

block = pyvista.read(helmet_file)
mesh = block[0][0][0]
mesh.plot(color='tan', show_edges=True, cpos='xy')
