"""
Fit Polygons to a Height Map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit polygonal data to a height map image using
:func:`fit_to_height_map <pyvista.PolyDataFilters.fit_to_height_map>`.
"""

from __future__ import annotations

import numpy as np

import pyvista as pv

# Create a height map (ImageData with elevation values)
height_map = pv.ImageData(dimensions=(50, 50, 1))
height_map.origin = (0.0, 0.0, 0.0)
height_map.spacing = (1.0, 1.0, 1.0)
xx, yy = np.meshgrid(
    np.linspace(0, 2 * np.pi, 50),
    np.linspace(0, 2 * np.pi, 50),
)
elevation = np.sin(xx) * np.cos(yy) * 10 + 20
height_map.point_data["elevation"] = elevation.flatten(order="C")

# Create polygons to drape over the height map
polygon1 = pv.Rectangle(bounds=(10, 30, 10, 30, 0, 0))
polygon2 = pv.Rectangle(bounds=(5, 15, 35, 45, 0, 0))

# Fit using point projection strategy
result1 = polygon1.fit_to_height_map(
    height_map,
    fitting_strategy="point_projection",
    use_height_map_offset=True,
)
result2 = polygon2.fit_to_height_map(
    height_map,
    fitting_strategy="point_projection",
    use_height_map_offset=True,
)

# Create terrain surface for visualization
terrain = height_map.warp_by_scalar(scalars="elevation")

# Plot
plotter = pv.Plotter(shape=(1, 2))

plotter.add_mesh(terrain, scalars="elevation", opacity=0.7, cmap="terrain")
plotter.add_mesh(result1, color="red", opacity=0.9)
plotter.add_mesh(result2, color="blue", opacity=0.9)
plotter.add_text("Point Projection", position="upper_left", font_size=12)
plotter.view_isometric()

plotter.subplot(1, 1)
result_cell = polygon1.fit_to_height_map(
    height_map,
    fitting_strategy="cell_average_height",
    use_height_map_offset=True,
)
plotter.add_mesh(terrain, scalars="elevation", opacity=0.7, cmap="terrain")
plotter.add_mesh(result_cell, color="red", opacity=0.9)
plotter.add_text("Cell Average Height", position="upper_left", font_size=12)
plotter.view_isometric()

plotter.show()
