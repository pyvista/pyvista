"""
Saving Screenshots
~~~~~~~~~~~~~~~~~~
"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples
import matplotlib.pyplot as plt

# Get a sample file
filename = examples.planefile
mesh = pv.read(filename)


###############################################################################
# You can also take a screenshot without creating an interactive plot window
# using the :class:`pyvista.Plotter`:

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, color="orange")
plotter.show(screenshot='airplane.png')

###############################################################################
# The ``img`` array can be used to plot the screenshot in ``matplotlib``:

plt.imshow(plotter.image)
plt.show()
