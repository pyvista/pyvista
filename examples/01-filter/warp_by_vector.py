"""
.. _warp_by_vector_example:

Warping by Vectors
~~~~~~~~~~~~~~~~~~

This example applies the :meth:`~pyvista.DataSetFilters.warp_by_vector`
filter to a sphere mesh that has 3D displacement vectors defined at each node.
"""

# %%
# We first compare the unwarped sphere to the warped sphere. Use
# :func:`~pyvista.plot_compare` to show the meshes side-by-side. The keys of the
# dict are used as labels.
import pyvista as pv
from pyvista import examples

sphere = examples.load_sphere_vectors()
warped = sphere.warp_by_vector()

pv.plot_compare(
    {'Before warp': sphere, 'After warp': warped},
    display_kwargs={'color': 'white'},
    link=False,
)

# %%
# We then use several values for the scale factor applied to the warp
# operation. Applying a warping factor that is too high can often lead to
# unrealistic results.

warp_factors = [0, 1.5, 3.5, 5.5]
pv.plot_compare(
    {f'factor={factor}': sphere.warp_by_vector(factor=factor) for factor in warp_factors},
    link=False,
)
# %%
# .. tags:: filter
