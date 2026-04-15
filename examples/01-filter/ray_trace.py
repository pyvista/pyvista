"""
.. _ray_trace_example:

Ray Tracing
~~~~~~~~~~~

Single line segment ray tracing for :class:`~pyvista.PolyData` objects
using :meth:`~pyvista.PolyDataFilters.ray_trace`.
"""

from __future__ import annotations

import pyvista as pv

# Create source to ray trace
sphere = pv.Sphere(radius=0.85)

# Define line segment
start = [0, 0, 0]
stop = [0.25, 1, 0.5]

# Perform ray trace
points, ind = sphere.ray_trace(start, stop)

# Create geometry to represent ray trace
ray = pv.Line(start, stop)
intersection = pv.PolyData(points)

# Render the result
pl = pv.Plotter()
pl.add_mesh(
    sphere, show_edges=True, opacity=0.5, color='w', lighting=False, label='Test Mesh'
)
pl.add_mesh(ray, color='blue', line_width=5, label='Ray Segment')
pl.add_mesh(intersection, color='maroon', point_size=25, label='Intersection Points')
pl.add_legend()
pl.show()
# %%
# .. tags:: filter
