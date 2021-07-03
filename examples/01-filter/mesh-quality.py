"""
.. _mesh_quality_example:

Computing Mesh Quality
~~~~~~~~~~~~~~~~~~~~~~

Leverage powerful VTK algorithms for computing mesh quality.

Here we will use the :func:`pyvista.DataSetFilters.compute_cell_quality` filter
to compute the cell qualities. For a full list of the various quality metrics
available, please refer to the documentation for that filter.
"""
import pyvista as pv
from pyvista import examples
import numpy as np

mesh = examples.download_cow().triangulate().decimate(0.7)

cpos = [(10.10963531890468, 4.61130688407898, -4.503884867626516),
 (1.2896420468715433, -0.055387528972708225, 1.1228250502811408),
 (-0.2970769821136617, 0.9100381451936025, 0.2890948650371137)]

###############################################################################
# Compute the cell quality. Note that there are many different quality measures
qual = mesh.compute_cell_quality(quality_measure='scaled_jacobian')
qual

###############################################################################
qual.plot(cpos=cpos, scalars='CellQuality')
