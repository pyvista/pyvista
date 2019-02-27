Compute the Volume of a Mesh
============================


Calculating mass properties such as the volume or area of datasets in ``vtki``
is quite easy using the :func:`vtki.DataSetFilters.compute_cell_sizes` filter
and the :func:`vtki.Common.volume` property on all ``vtki`` meshes.


Let's get started with a simple gridded mesh:

.. testcode:: python

    import numpy as np
    import vtki
    from vtki import examples
    vtki.set_plot_theme('document')

    # Load a simple example mesh
    dataset = examples.load_uniform()
    dataset.set_active_scalar('Spatial Cell Data')


We can then calculate the volume of every cell in the array using the
``.compute_cell_sizes`` filter which will add arrays to the cell data of the
mesh core the volume and area by default.

.. testcode:: python

    # Compute volumes and areas
    sized = dataset.compute_cell_sizes()

    # Grab volumes for all cells in the mesh
    cell_volumes = sized.cell_arrays['Volume']


We can also compute the total volume of the mesh using the ``.volume`` property:

.. testcode:: python

    # Compute the total volume of the mesh
    volume = dataset.volume


Okay awesome! But what if we have have a dataset that we threshold with two
volumetric bodies left over in one dataset? Take this for example:


.. testcode:: python

    threshed = dataset.threshold_percent([0.15, 0.50], invert=True)
    threshed.plot(show_bounds=True)


.. image:: ../../images/two-bodies.png

We could then assign a classification array for the two bodies, compute the
cell sizes, then extract the volumes of each body:

.. testcode:: python

    # Create a classifying array to ID each body
    rng = dataset.get_data_range()
    cval = ((rng[1] -rng[0]) * 0.20) + rng[0]
    classifier = threshed.cell_arrays['Spatial Cell Data'] > cval

    # Compute cell volumes
    sizes = threshed.compute_cell_sizes()
    volumes = sizes.cell_arrays['Volume']

    # Split volumes based on classifier and get volumes!
    idx = np.argwhere(classifier)
    hvol = np.sum(volumes[idx])
    idx = np.argwhere(~classifier)
    lvol = np.sum(volumes[idx])

    print('Low grade volume: {}'.format(lvol))
    print('High grade volume: {}'.format(hvol))
    print('Original volume: {}'.format(dataset.volume))


.. code-block:: text

    Low grade volume: 518.0
    High grade volume: 35.0
    Original volume: 729.0
