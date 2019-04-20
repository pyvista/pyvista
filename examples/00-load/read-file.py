"""
Load and Plot from a File
~~~~~~~~~~~~~~~~~~~~~~~~~

Read a dataset from a known file type.

"""
################################################################################
# Loading a mesh is trivial.  The following code block uses a built-in example
# file and displays an airplane mesh.

import vtki
from vtki import examples
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Loading a mesh is trivial.  The following code block uses a built-in example
# file, displays an airplane mesh, saves a screenshot, and returns the camera's
# position:

# Get a sample file
filename = examples.planefile
mesh = vtki.read(filename)
cpos = mesh.plot()

################################################################################
# You can also take a screenshot without creating an interactive plot window using
# the ``Plotter``:

plotter = vtki.Plotter(off_screen=True)
plotter.add_mesh(mesh)
plotter.show(auto_close=False)
# plotter.screenshot('airplane.png')
plotter.close()

################################################################################
# The ``img`` array can be used to plot the screenshot in ``matplotlib``:

plt.imshow(plotter.image)
plt.show()

################################################################################
# If you need to setup the camera you can do this by plotting first and getting
# the camera after running the ``plot`` function:

plotter = vtki.Plotter()
plotter.add_mesh(mesh)
cpos = plotter.show()

################################################################################
# You can then use this cached camera for additional plotting without having to
# manually interact with the ``vtk`` plot window:

plotter = vtki.Plotter(off_screen=True)
plotter.add_mesh(mesh)
plotter.camera_position = cpos
plotter.show(auto_close=False)
# plotter.screenshot('airplane.png')
plotter.close()

################################################################################
# The points from the mesh are directly accessible as a NumPy array:

print(mesh.points)

################################################################################
# The faces from the mesh are also directly accessible as a NumPy array:

print(mesh.faces.reshape(-1, 4)[:, 1:])


################################################################################
#
