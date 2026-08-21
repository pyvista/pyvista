"""
.. _resampling_example:

Detailed Resampling
~~~~~~~~~~~~~~~~~~~

This example uses :func:`pyvista.DataObjectFilters.sample`.

:func:`pyvista.DataSetFilters.interpolate` is similar, and the two
methods are compared in :ref:`interpolate_sample_example`.

Resample one mesh's point/cell arrays onto another mesh's nodes.

"""

# %%
# This example will resample a volumetric mesh's scalar data onto the surface
# of a sphere contained in that volume.

# sphinx_gallery_thumbnail_number = 3
import pyvista as pv
from pyvista import examples

# %%
# Simple Resample
# +++++++++++++++
# Query a grid's points onto a sphere
mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
data_to_probe = examples.load_uniform()

# %%
# Plot the two datasets
pl = pv.Plotter()
pl.add_mesh(mesh, color=True)
pl.add_mesh(data_to_probe, opacity=0.5)
pl.show()

# %%
# Run the algorithm and plot the result
result = mesh.sample(data_to_probe)

# Plot result
name = 'Spatial Point Data'
result.plot(scalars=name, clim=data_to_probe.get_data_range(name))

# %%
# Complex Resample
# ++++++++++++++++
# Take a volume of data and create a grid of lower resolution to resample on
data_to_probe = examples.download_embryo()
mesh = pv.create_grid(data_to_probe, dimensions=(75, 75, 75))

result = mesh.sample(data_to_probe)

# %%
threshold = lambda m: m.threshold(75.0, scalars='SLCImage')
cpos = pv.CameraPosition(
    position=(468.9, -152.8, 152.1),
    focal_point=(121.7, 140.3, 112.3),
    viewup=(-0.1088, 0.006229, 0.994),
)
dargs = dict(clim=[0, 200], cmap='rainbow')

pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(threshold(data_to_probe), **dargs)
pl.subplot(0, 1)
pl.add_mesh(threshold(result), **dargs)
pl.link_views()
pl.view_isometric()
pl.show(cpos=cpos)
# %%
# .. tags:: filter
