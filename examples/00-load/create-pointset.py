"""
.. _create_pointset_example:

Create a PointSet
~~~~~~~~~~~~~~~~~

A :class:`pyvista.PointSet` is a concrete class representing a set of points
that specifies the interface for datasets that explicitly use "point" arrays to
represent geometry. This class is useful for improving the performance of
filters on point clouds.

This example shows the performacne improvement when cliping using a
:class:`pyvista.PointSet` using the :func:`pyvista.DataSet.clip` filter.

"""
import time

from pyvista import examples
import pyvista as pv

lidar = examples.download_lidar()

tstart = time.time()
clipped = lidar.clip(origin=(0, 0, 1.76E3), normal=(0, 0, 1))
t_elapsed = time.time() - tstart
print(f"Time to clip with a PolyData {t_elapsed:.2f} seconds.")

###############################################################################
# Plot the Clipped polydata
clipped.plot(show_scalar_bar=False)

###############################################################################
# Show the performacne improvement when using a PointSet
#
# This is only available with VTK >= 9.1.0

# pset = lidar.cast_to_pointset(deep=False)

if pv.vtk_version_info >= (9, 1):
    lidar_pset = lidar.cast_to_pointset()
    tstart = time.time()
    clipped_pset = lidar_pset.clip(origin=(0, 0, 1.76E3), normal=(0, 0, 1))
    t_elapsed = time.time() - tstart
    print(f"Time to clip with a PointSet {t_elapsed:.2f} seconds.")

###############################################################################
# Plot the same dataset
#
# .. note::
#    PyVista must still create an intermediate PolyData to be able to plot, so
#    there is no performance improvement when using a :class:`pyvista.PointSet`

if pv.vtk_version_info >= (9, 1):
    clipped_pset.plot()


# pv.PointSet(lidar.GetPoints())

# import numpy as np
# arr = np.random.random((10, 3))
# pset = pv.PolyData(arr, deep=False)
# arr[:] = 0 

# import pyvista
# self = lidar
# pset = pyvista.PointSet()
# # pset.SetPointData(self.GetPointData)
# pset.SetPoints(self.GetPoints())


# lidar_pset.point_data = lidar.point_data
