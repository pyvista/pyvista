"""
.. _openfoam_example:

Plot OpenFOAM data
~~~~~~~~~~~~~~~~~~

"""

import pyvista
from pyvista import examples

###############################################################################
# This example uses data from a lid-driven cavity flow.  It is recommended to
# use :class:`pyvista.OpenFOAMReader` for reading OpenFOAM files for more
# control over reading data.  The OpenFOAMReader in pyvista must be recreated
# each time a new mesh is read in, otherwise the same mesh is always returned.
#
# This example will only run correctly in versions of vtk>=9.1.0.  The names
# of the patch arrays and resulting keys in the read mesh will be different
# in prior versions.


from pyvista import examples
filename = examples.download_cavity(load=False)
reader = pyvista.OpenFOAMReader(filename)

###############################################################################
# OpenFOAM datasets include multiple sub-datasets including the internal mesh
# and patches, typically boundaries.  This data is represented as a
# :class:`pyvista.MultiBlock` object.  The internal mesh will be located in the
# top-level MultiBlock mesh.  By default it is the only mesh read.  This can
# be inspected by the following
#

print(f"All patch names: {reader.patch_array_names}")
print(f"All patch status: {reader.all_patch_arrays_status}")
mesh = reader.read()
print(f"Mesh patches: {mesh.keys()}")
internal_mesh = mesh["internalMesh"]  # or internal_mesh = mesh[0]

###############################################################################
# In this case the internal mesh is a :class:`pyvista.UnstructuredGrid`

print(internal_mesh)

###############################################################################
# The default in OpenFOAMReader is to translate the existing cell data to point
# data.  Therefore, the cell data arrays are duplicated in point data

print("Cell Data:")
print(internal_mesh.cell_data)
print("\nPoint Data:")
print(internal_mesh.point_data)

###############################################################################
# This behavior can be turned off.

reader = pyvista.OpenFOAMReader(filename)
reader.cell_to_point_creation = False
internal_mesh = reader.read()["internalMesh"]
print("Cell Data:")
print(internal_mesh.cell_data)
print("\nPoint Data:")
print(internal_mesh.point_data)

###############################################################################
# The movingWall patch is additionally read into the mesh.  Patches are
# inserted into a sub-MultiBlock mesh, in this case `boundary`.

reader = pyvista.OpenFOAMReader(filename)
reader.enable_patch_array("patch/movingWall")
mesh = reader.read()
print(f"Mesh patches: {mesh.keys()}")

###############################################################################
# The movingWall mesh nested inside is a :class:`pyvista.PolyData` mesh.
boundaries = mesh["boundary"]
print(f"Boundaries patches: {boundaries.keys()}")
print(boundaries["movingWall"])

###############################################################################
# Now we will read in all the data at the last time point

reader = pyvista.OpenFOAMReader(filename)
print(f"Available Time Values: {reader.time_values}")
reader.set_active_time_value(2.5)
reader.enable_all_patch_arrays()
#reader.cell_to_point_creation = False
mesh = reader.read()
internal_mesh = mesh["internalMesh"]
boundaries = mesh["boundary"]

###############################################################################
# Plot the streamlines on a 2D plane.  This OpenFOAM simulation is in 3D with
# only 1 cell in the z-direction.  First, the solution is sliced in the center
# of the z-direction.
# :func:`pyvista.DataSetFilters.streamlines_evenly_spaced_2D` requires the data
# to lie in the z=0 plane.  So, after the domain sliced, it is translated to
# ``z=0``.

slice_internal_mesh = internal_mesh.slice(normal='z')
slice_internal_mesh.translate((0, 0, -slice_internal_mesh.center[-1]))

slice_boundaries = pyvista.MultiBlock({key: boundaries[key].slice('z') for key in boundaries.keys()})
for slice_boundary in slice_boundaries:
    slice_boundary.translate((0, 0, -slice_boundary.center[-1]))

streamlines = slice_internal_mesh.streamlines_evenly_spaced_2D(
    vectors='U', start_position=(0.05, 0.05, 0), separating_distance=1, 
    separating_distance_ratio=0.1
)
plotter = pyvista.Plotter()
plotter.add_mesh(slice_boundaries["movingWall"], color='red', line_width=3)
plotter.add_mesh(slice_boundaries["fixedWalls"], color='black', line_width=3)
plotter.add_mesh(streamlines.tube(radius=0.0005), scalars="U")
plotter.view_xy()
plotter.enable_parallel_projection()
plotter.show()




