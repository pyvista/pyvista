"""
Voxelize a Surface Mesh
~~~~~~~~~~~~~~~~~~~~~~~

Create a voxel model (like legos) of a closed surface or volumetric mesh.

"""
# sphinx_gallery_thumbnail_number = 2
from pyvista import examples
import pyvista as pv

# Load a surface to voxelize
surface = examples.download_cow()

cpos = [(12.797670941145535, 3.0876291256753663, -11.170410504223877),
 (1.0601386387865293, -0.4925689018344716, 0.26270230082212054),
 (-0.15392114523357817, 0.9769504628511697, 0.14790562594056045)]

p = pv.Plotter()
p.add_mesh(surface, color=True)
p.show(cpos=cpos)


###############################################################################
# Create a voxel model of the bounding surface
voxels = pv.voxelize(surface, density=surface.length/200)

p = pv.Plotter()
p.add_mesh(voxels, color=True, show_edges=True, opacity=0.75)
p.add_mesh(surface, color="lightblue", opacity=0.5)
p.show(cpos=cpos)
