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

filename = examples.download_cavity(load=False)
reader = pyvista.OpenFOAMReader(filename)

###############################################################################
# OpenFOAM datasets include multiple sub-datasets including the internal mesh
# and patches, typically boundaries.  This can be inspected before reading the data.

print(f"All patch names: {reader.patch_array_names}")
print(f"All patch status: {reader.all_patch_arrays_status}")

###############################################################################
# This data is represented as a :class:`pyvista.MultiBlock` object.
# The internal mesh will be located in the top-level MultiBlock mesh.

mesh = reader.read()
print(f"Mesh patches: {mesh.keys()}")
internal_mesh = mesh["internalMesh"]  # or internal_mesh = mesh[0]

###############################################################################
# In this case the internal mesh is a :class:`pyvista.UnstructuredGrid`.

print(internal_mesh)

###############################################################################
# Additional Patch meshes are nested inside another MultiBlock mesh.  The name
# of the sub-level MultiBlock mesh depends on the vtk version.

boundaries = mesh["boundary"]
print(boundaries)
print(f"Boundaries patches: {boundaries.keys()}")
print(boundaries["movingWall"])

###############################################################################
# The default in OpenFOAMReader is to translate the existing cell data to point
# data.  Therefore, the cell data arrays are duplicated in point data.

print("Cell Data:")
print(internal_mesh.cell_data)
print("\nPoint Data:")
print(internal_mesh.point_data)

###############################################################################
# This behavior can be turned off if only cell data is required.

reader = pyvista.OpenFOAMReader(filename)
reader.cell_to_point_creation = False
internal_mesh = reader.read()["internalMesh"]
print("Cell Data:")
print(internal_mesh.cell_data)
print("\nPoint Data:")
print(internal_mesh.point_data)

###############################################################################
# Now we will read in all the data at the last time point.

reader = pyvista.OpenFOAMReader(filename)
print(f"Available Time Values: {reader.time_values}")
reader.set_active_time_value(2.5)
mesh = reader.read()
internal_mesh = mesh["internalMesh"]
boundaries = mesh["boundary"]

###############################################################################
# This OpenFOAM simulation is in 3D with
# only 1 cell in the z-direction.  First, the solution is sliced in the center
# of the z-direction.
# :func:`pyvista.DataSetFilters.streamlines_evenly_spaced_2D` requires the data
# to lie in the z=0 plane.  So, after the domain sliced, it is translated to
# ``z=0``.


def slice_z_center(mesh):
    """Slice mesh through center in z normal direction, move to z=0."""
    slice_mesh = mesh.slice(normal='z')
    slice_mesh.translate((0, 0, -slice_mesh.center[-1]), inplace=True)
    return slice_mesh


slice_internal_mesh = slice_z_center(internal_mesh)
slice_boundaries = pyvista.MultiBlock(
    {key: slice_z_center(boundaries[key]) for key in boundaries.keys()}
)

###############################################################################
# Streamlines are generated using the point data "U".

streamlines = slice_internal_mesh.streamlines_evenly_spaced_2D(
    vectors='U',
    start_position=(0.05, 0.05, 0),
    separating_distance=1,
    separating_distance_ratio=0.1,
)

###############################################################################
# Plot streamlines colored by velocity magnitude.  Additionally, the moving
# and fixed wall boundaries are plotted.

plotter = pyvista.Plotter()
plotter.add_mesh(slice_boundaries["movingWall"], color='red', line_width=3)
plotter.add_mesh(slice_boundaries["fixedWalls"], color='black', line_width=3)
plotter.add_mesh(streamlines.tube(radius=0.0005), scalars="U")
plotter.view_xy()
plotter.enable_parallel_projection()
plotter.show()
