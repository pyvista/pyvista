"""
.. _color_lines_example:

Color Several Lines
~~~~~~~~~~~~~~~~~~~

Render multiple polylines and color them by a line-wise scalar value.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Build a small family of curves
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each curve gets its own scalar value so the lines can be colored as a group.

lines = []
for index, phase in enumerate(np.linspace(0, np.pi, 4)):
    t = np.linspace(0, 1, 80)
    points = np.column_stack(
        (
            4 * t - 2,
            np.sin(2 * np.pi * t + phase),
            0.4 * np.cos(np.pi * t + phase),
        ),
    )
    line = pv.MultipleLines(points)
    line.cell_data['line_id'] = np.array([index])
    lines.append(line)

curves = pv.merge(lines)
curves


# %%
# Tube and color the lines
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Coloring by `line_id` gives each line its own consistent color.

pl = pv.Plotter()
pl.add_mesh(
    curves.tube(radius=0.06),
    scalars='line_id',
    cmap='viridis',
    show_scalar_bar=False,
)
pl.show()
# %%
# .. tags:: plot
