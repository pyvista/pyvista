"""
Load and Plot from a File
~~~~~~~~~~~~~~~~~~~~~~~~~

Read a dataset from a known file type.

"""
################################################################################
# Loading a mesh is trivial - if your data is in one of the many supported
# file formats, simply use :func:`vtki.read` to load your spatially referneced
# dataset into a ``vtki`` mesh object.
#
# The following code block uses a built-in example file and displays an airplane
# mesh.

# sphinx_gallery_thumbnail_number = 5
import vtki
from vtki import examples
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# The following code block uses a built-in example
# file, displays an airplane mesh and returns the camera's position:

# Get a sample file
filename = examples.planefile
print(filename)

################################################################################
# Note the above filename, it's a ``.ply`` file - one of the many supported
# formats in ``vtki``.

mesh = vtki.read(filename)
cpos = mesh.plot()

################################################################################
# You can also take a screenshot without creating an interactive plot window
# using the ``Plotter``:

plotter = vtki.Plotter(off_screen=True)
plotter.add_mesh(mesh)
plotter.show(screenshot='myscreenshot.png')


################################################################################
# The points from the mesh are directly accessible as a NumPy array:

print(mesh.points)

################################################################################
# The faces from the mesh are also directly accessible as a NumPy array:

print(mesh.faces.reshape(-1, 4)[:, 1:])


################################################################################
# Loading other files types is just as easy! Simply pass your file path to the
# :func:`vtki.read` function and that's it!
#
# Here are a few other examples - siply replace ``examples.download_*`` in the
# examples below with ``vtki.read('path/to/you/file.ext')``

################################################################################
# Example STL file:
mesh = examples.download_cad_model()
cpos = [(107., 68.5, 204.),
        (128., 86.5, 223.5),
        (0.45, 0.36, -0.8)]
mesh.plot(cpos=cpos)

################################################################################
# Example OBJ file
mesh = examples.download_doorman()
mesh.plot(cpos='xy')


################################################################################
# Example BYU file
mesh = examples.download_teapot()
mesh.plot(cpos=[-1, 2, -5], show_edges=True)


################################################################################
# Example VTK file
mesh = examples.download_bunny_coarse()
cpos = [(0.2, 0.3, 0.9),
        (0,0,0),
        (0,1,0)]
mesh.plot(cpos=cpos, show_edges=True, color=True)
