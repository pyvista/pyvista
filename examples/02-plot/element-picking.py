"""
.. _element_picking_example:

Picking elements of a mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~
This example demonstrates how to pick different elements on meshes using
:func:`enable_element_picking() <pyvista.Plotter.enable_element_picking>`.

The different elements of a mesh are:

* Mesh: pick the entire mesh (equivalent to :func:`enable_mesh_picking() <pyvista.Plotter.enable_mesh_picking>`.)
* Cell: pick a cell of the mesh (equivalent to :func:`enable_cell_picking() <pyvista.Plotter.enable_cell_picking>`.)
* Face: pick a single face of a cell on the mesh
* Edge: pick a single edge of a cell on the mesh
* Point: pick a single point on the mesh

These types are captured in the :class:`pyvista.plotting.opts.ElementType` enum class.

"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista.plotting.opts import ElementType

###############################################################################
# Pick Face on Voxel Cell
# +++++++++++++++++++++++
#
mesh = pv.Wavelet()

p = pv.Plotter()
p.add_mesh(mesh, show_edges=True, pickable=True)
p.enable_element_picking(mode=ElementType.FACE)
p.show()

###############################################################################
# Pick an Edge of a Cell
# ++++++++++++++++++++++
#
sphere = pv.Sphere()

p = pv.Plotter()
p.add_mesh(sphere, show_edges=True, pickable=True)
p.enable_element_picking(mode=ElementType.EDGE)
p.show()
