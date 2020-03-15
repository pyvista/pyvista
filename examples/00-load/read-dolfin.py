"""
Read FEniCS/Dolfin Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~

PyVista leverages `meshio`_ to read many mesh formats not natively supported
by VTK including the `FEniCS/Dolfin`_ XML format.

.. _meshio: https://github.com/nschloe/meshio
.. _FEniCS/Dolfin: https://fenicsproject.org
"""
import pyvista as pv
from pyvista import examples

###############################################################################
# Let's download an example FEniCS/Dolfin mesh from our example data
# repository. This will download an XML Dolfin mesh and save it to PyVista's
# data directory.
saved_file, _ = examples.downloads._download_file("dolfin_fine.xml")
print(saved_file)

###############################################################################
# As shown, we now have an XML Dolfin mesh save locally. This filename can be
# passed directly to PyVista's :func:`pyvista.read` method to be read into
# a PyVista mesh.
dolfin = pv.read(saved_file)
dolfin


###############################################################################
# Now we can do stuff with that Dolfin mesh!
qual = dolfin.compute_cell_quality()
qual.plot(show_edges=True, cpos="xy")
