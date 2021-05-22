"""
Volumetric Analysis
~~~~~~~~~~~~~~~~~~~


Calculate mass properties such as the volume or area of datasets
"""

# sphinx_gallery_thumbnail_number = 4
import numpy as np
from pyvista import examples

###############################################################################
# Computing mass properties such as the volume or area of datasets in PyVista
# is quite easy using the :func:`pyvista.DataSetFilters.compute_cell_sizes`
# filter and the :attr:`pyvista.DataSet.volume` property on all PyVista meshes.
#
# Let's get started with a simple gridded mesh:

# Load a simple example mesh
dataset = examples.load_uniform()
dataset.set_active_scalars("Spatial Cell Data")

###############################################################################
# We can then calculate the volume of every cell in the array using the
# ``.compute_cell_sizes`` filter which will add arrays to the cell data of the
# mesh core the volume and area by default.

# Compute volumes and areas
sized = dataset.compute_cell_sizes()

# Grab volumes for all cells in the mesh
cell_volumes = sized.cell_arrays["Volume"]

###############################################################################
# We can also compute the total volume of the mesh using the ``.volume``
# property:

# Compute the total volume of the mesh
volume = dataset.volume

###############################################################################
# Okay awesome! But what if we have have a dataset that we threshold with two
# volumetric bodies left over in one dataset? Take this for example:


threshed = dataset.threshold_percent([0.15, 0.50], invert=True)
threshed.plot(show_grid=True, cpos=[-2, 5, 3])

###############################################################################
# We could then assign a classification array for the two bodies, compute the
# cell sizes, then extract the volumes of each body. Note that there is a
# simpler implementation of this below in :ref:`split_vol_ref`.

# Create a classifying array to ID each body
rng = dataset.get_data_range()
cval = ((rng[1] - rng[0]) * 0.20) + rng[0]
classifier = threshed.cell_arrays["Spatial Cell Data"] > cval

# Compute cell volumes
sizes = threshed.compute_cell_sizes()
volumes = sizes.cell_arrays["Volume"]

# Split volumes based on classifier and get volumes!
idx = np.argwhere(classifier)
hvol = np.sum(volumes[idx])
idx = np.argwhere(~classifier)
lvol = np.sum(volumes[idx])

print(f"Low grade volume: {lvol}")
print(f"High grade volume: {hvol}")
print(f"Original volume: {dataset.volume}")

###############################################################################
# Or better yet, you could simply extract the largest volume from your
# thresholded dataset by passing ``largest=True`` to the ``connectivity``
# filter or by using ``extract_largest`` filter (both are equivalent).

# Grab the largest connected volume present
largest = threshed.connectivity(largest=True)
# or: largest = threshed.extract_largest()

# Get volume as numeric value
large_volume = largest.volume

# Display it!
largest.plot(show_grid=True, cpos=[-2, 5, 3])


###############################################################################
# -----
#
# .. _split_vol_ref:
#
# Splitting Volumes
# +++++++++++++++++
#
# What if instead, we wanted to split all the different connected bodies /
# volumes in a dataset like the one above? We could use the
# :func:`pyvista.DataSetFilters.split_bodies` filter to extract all the
# different connected volumes in a dataset into blocks in a
# :class:`pyvista.MultiBlock` dataset. For example, lets split the thresholded
# volume in the example above:

# Load a simple example mesh
dataset = examples.load_uniform()
dataset.set_active_scalars("Spatial Cell Data")
threshed = dataset.threshold_percent([0.15, 0.50], invert=True)

bodies = threshed.split_bodies()

for i, body in enumerate(bodies):
    print(f"Body {i} volume: {body.volume:.3f}")

###############################################################################


bodies.plot(show_grid=True, multi_colors=True, cpos=[-2, 5, 3])


###############################################################################
# -----
#
# A Real Dataset
# ++++++++++++++
#
# Here is a realistic training dataset of fluvial channels in the subsurface.
# This will threshold the channels from the dataset then separate each
# significantly large body and compute the volumes for each!
#
# Load up the data and threshold the channels:

data = examples.load_channels()
channels = data.threshold([0.9, 1.1])

###############################################################################
# Now extract all the different bodies and compute their volumes:

bodies = channels.split_bodies()
# Now remove all bodies with a small volume
for key in bodies.keys():
    b = bodies[key]
    vol = b.volume
    if vol < 1000.0:
        del bodies[key]
        continue
    # Now lets add a volume array to all blocks
    b.cell_arrays["TOTAL VOLUME"] = np.full(b.n_cells, vol)

###############################################################################
# Print out the volumes for each body:

for i, body in enumerate(bodies):
    print(f"Body {i:02d} volume: {body.volume:.3f}")

###############################################################################
# And visualize all the different volumes:

bodies.plot(scalars="TOTAL VOLUME", cmap="viridis", show_grid=True)
