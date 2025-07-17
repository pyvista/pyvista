"""Example script demonstrating the new axis-aligned reflection filter."""

from __future__ import annotations

import pyvista as pv

# Create a simple mesh
mesh = pv.Cube(center=(2, 0, 0))
mesh['example_data'] = mesh.points[:, 0]

print('Original mesh bounds:', mesh.bounds)

# Reflect across YZ plane (x-normal) at x=1.0
reflected = mesh.reflect_axis_aligned(plane='x', value=1.0)

print(f'Result contains {reflected.n_blocks} blocks')
print('Block 0 (original) bounds:', reflected[0].bounds)
print('Block 1 (reflection) bounds:', reflected[1].bounds)

# Visualize if running interactively
if __name__ == '__main__':
    pl = pv.Plotter()
    pl.add_mesh(reflected[0], color='blue', opacity=0.7, label='Original')
    pl.add_mesh(reflected[1], color='red', opacity=0.7, label='Reflection')
    pl.add_legend()
    pl.show_axes()
    pl.show()
