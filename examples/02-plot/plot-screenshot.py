"""
Saving Screenshots
~~~~~~~~~~~~~~~~~~
"""
# sphinx_gallery_thumbnail_number = 2
import vtki
from vtki import examples
import numpy as np
import matplotlib.pyplot as plt

# Get a sample file
filename = examples.planefile
mesh = vtki.read(filename)


################################################################################
# You can also take a screenshot without creating an interactive plot window using
# the ``Plotter``:

plotter = vtki.Plotter(off_screen=True)
plotter.add_mesh(mesh, color='orange')
plotter.show(auto_close=False)
# plotter.screenshot('airplane.png')
plotter.close()

################################################################################
# The ``img`` array can be used to plot the screenshot in ``matplotlib``:

plt.imshow(plotter.image)
plt.show()
