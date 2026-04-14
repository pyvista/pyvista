"""
.. _composite_picking_example:

Composite Picking
~~~~~~~~~~~~~~~~~

Demonstrate how to pick individual blocks of a :class:`pyvista.MultiBlock`
using :func:`pyvista.Plotter.enable_block_picking`.

"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Create a MultiBlock Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create 100 superellipsoids using :func:`pyvista.ParametricSuperEllipsoid`

# Seed rng for reproducibility
rng = np.random.default_rng(seed=0)


def make_poly():
    """Create a superellipsoid in a random location."""
    poly = pv.ParametricSuperEllipsoid(
        n1=rng.random(),
        n2=rng.random() * 2,
        u_res=50,
        v_res=50,
    )
    poly.points += rng.random(3) * 20
    return poly


# Assemble the multiblock and plot it using the default plotting settings
blocks = pv.MultiBlock([make_poly() for _ in range(100)])
blocks.plot()

# %%
# Enable Block Picking
# ~~~~~~~~~~~~~~~~~~~~
# Add ``blocks`` to a :class:`pyvista.Plotter` and enable block picking.  For
# fun, let's also enable physically based rendering and set the callback to set
# the block color to red when the block is clicked and unset the color if the
# color has already been set for the block.

# sphinx_gallery_start_ignore
# physically based rendering does not work in interactive mode
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter()
actor, mapper = pl.add_composite(blocks, color='w', pbr=True, metallic=True)


def callback(index, *args):  # noqa: ARG001
    """Change a block to red if color is unset, and back to the actor color if set."""
    if mapper.block_attr[index].color is None:
        mapper.block_attr[index].color = 'r'
    else:
        mapper.block_attr[index].color = None


pl.enable_block_picking(callback, side='left')
pl.background_color = 'w'
pl.show()
# %%
# .. tags:: plot
