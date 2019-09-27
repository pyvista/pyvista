"""
.. _ref_bgplot:

bgplot
~~~~~~

Testing.
"""
###############################################################################

# sphinx_gallery_thumbnail_number = 1
import pyvista as pv

###############################################################################
# Test Screenshot
p = pv.BackgroundPlotter()
p.add_mesh(pv.Cone())
img = p.screenshot()
p.close()

###############################################################################
# Std Plot
p = pv.BackgroundPlotter()
p.add_mesh(pv.Sphere())

###############################################################################
# Changing the size now
p = pv.BackgroundPlotter()
p.add_mesh(pv.Sphere())
p.window_size = (800, 300)
