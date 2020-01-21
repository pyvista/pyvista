"""
BackgroundPlotter example.
~~~~~~~~~~~~~~~~~~~~~~~~~~

This simple sphinx example uses BackgroundPlotter.

"""

# sphinx_gallery_thumbnail_number = 4
import pyvista as pv

###############################################################################

p = pv.BackgroundPlotter()
p.add_mesh(pv.Cone())

###############################################################################

p = pv.BackgroundPlotter()
p.add_mesh(pv.Sphere())
p.close()
