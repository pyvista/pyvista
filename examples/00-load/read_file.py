"""
.. _read_file_example:

Load and Plot from a File
~~~~~~~~~~~~~~~~~~~~~~~~~

Read a dataset from a known file type.

"""

# %%
# Loading a mesh is trivial - if your data is in one of the many supported
# file formats, simply use :func:`pyvista.read` to load your spatially
# referenced dataset into a PyVista mesh object.
#
# The following code block uses a built-in example file and displays an
# airplane mesh.

# sphinx_gallery_thumbnail_number = 5
from __future__ import annotations

import pyvista as pv
from pyvista import examples

# %%
# The following code block uses a built-in example
# file, displays an airplane mesh and returns the camera's position:

# Get a sample file
filename = examples.planefile
filename

# %%
# Note the above filename, it's a ``.ply`` file - one of the many supported
# formats in PyVista.

mesh = pv.read(filename)
cpos = mesh.plot()

# %%
# You can also take a screenshot without creating an interactive plot window
# using the ``Plotter``:

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh)
plotter.show(screenshot='myscreenshot.png')


# %%
# The points from the mesh are directly accessible as a NumPy array:

mesh.points

# %%
# The faces from the mesh are also directly accessible as a NumPy array:

mesh.faces.reshape(-1, 4)[:, 1:]  # triangular faces


# %%
# Loading other files types is just as easy. Simply pass your file path to the
# :func:`pyvista.read` function and that's it.
#
# Here are a few other examples - simply replace ``examples.download_*`` in the
# examples below with ``pyvista.read('path/to/you/file.ext')``

# %%
# Example STL file:
mesh = examples.download_cad_model()
cpos = pv.CameraPosition(
    position=(107.0, 68.5, 204.0), focal_point=(128.0, 86.5, 223.5), viewup=(0.45, 0.36, -0.8)
)
mesh.plot(cpos=cpos)

# %%
# Example OBJ file
mesh = examples.download_doorman()
mesh.plot(cpos='xy')


# %%
# Example BYU file
mesh = examples.download_teapot()
mesh.plot(cpos=[-1, 2, -5], show_edges=True)


# %%
# Example VTK file
mesh = examples.download_bunny_coarse()
cpos = pv.CameraPosition(position=(0.2, 0.3, 0.9), focal_point=(0, 0, 0), viewup=(0, 1, 0))
mesh.plot(cpos=cpos, show_edges=True, color=True)
# %%
# .. tags:: load
