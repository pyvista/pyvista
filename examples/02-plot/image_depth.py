"""
Render a depth image
~~~~~~~~~~~~~~~~~~~~

Plot a depth image as viewed from a camera overlooking the "hills"
example mesh.
"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import matplotlib.pyplot as plt
from pyvista import examples

# Load an interesting example of geometry
mesh = examples.load_random_hills()

# Establish geometry within a pv.Plotter()
p = pv.Plotter()
p.add_mesh(mesh, color=True)
p.store_image = True  # permit image caching after plotter is closed
p.show()

###############################################################################
# Record depth image without and with a custom fill value
zval = p.get_image_depth()
zval_filled_by_42s = p.get_image_depth(fill_value=42.0)

###############################################################################
# Visualize depth images
plt.figure()
plt.imshow(zval)
plt.colorbar(label='Distance to Camera')
plt.title('Depth image')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()

###############################################################################
plt.figure()
plt.imshow(zval_filled_by_42s)
plt.title('Depth image (custom fill_value)')
plt.colorbar(label='Distance to Camera')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()
