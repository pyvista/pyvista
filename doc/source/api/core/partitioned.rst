Partitioned Datasets
====================

The :class:`pyvista.PartitionedDataSet` class is a composite dataset to encapsulates
a dataset consisting of partitions. ``PartitionedDataSet`` behaves mostly like a list.

List-like Features
------------------

Create empty composite dataset

.. jupyter-execute::
   :hide-code:

   # must have this here as our global backend may not be static
   import pyvista
   pyvista.set_plot_theme('document')
   pyvista.set_jupyter_backend('static')
   pyvista.global_theme.window_size = [600, 400]
   pyvista.global_theme.axes.show = False
   pyvista.global_theme.anti_aliasing = 'fxaa'
   pyvista.global_theme.show_scalar_bar = False

.. jupyter-execute::

   import pyvista as pv
   from pyvista import examples
   partitions = pv.PartitionedDataSet()
   partitions

Add some data to the collection.

.. jupyter-execute::

   partitions.append(pv.Sphere())
   partitions.append(pv.Cube(center=(0, 0, -1)))

Plotting the ``PartitionedDataSet`` plots all the meshes contained by it.

.. jupyter-execute::

   partitions.plot(smooth_shading=True)

``PartitionedDataSet`` is List-like, so individual partitions can be accessed via
indices.

.. jupyter-execute::

   partitions[0]  # Sphere

The length of the partition can be accessed through :func:`len`

.. jupyter-execute::

   len(partitions)

or through the ``n_partitions`` attribute

.. jupyter-execute::

   partitions.n_partitions

More specifically, ``PartitionedDataSet`` is a :class:`collections.abc.MutableSequence`
and supports operations such as append, pop, insert, etc. Some of these operations
allow optional names to be provided for the dictionary like usage.

.. jupyter-execute::

   partitions.append(pv.Cone(), name="cone")
   cone = partitions.pop(-1)  # Pops Cone
   partitions.reverse()

``PartitionedDataSet`` also supports slicing for getting or setting partitions.

.. jupyter-execute::

   partitions[0:2]  # The Sphere and Cube objects in a new ``PartitionedDataSet``


Dictionary-like Features
------------------------


``PartitionedDataSet`` also has some dictionary features. We can set the name
of the partitions, and then access them

.. jupyter-execute::

   partitions = pv.PartitionedDataSet([pv.Sphere(), pv.Cube()])
   partitions.set_partition_name(0, "sphere")
   partitions.set_partition_name(1, "cube")
   partitions["sphere"]  # Sphere

It is important to note that ``PartitionedDataSet`` is not a dictionary and does
not enforce unique keys. Keys can also be ``None``. Extra care must be
taken to avoid problems using the Dictionary-like features.

PyVista tries to keep the keys ordered correctly when doing list operations.

.. jupyter-execute::

   partitions.reverse()
   partitions.keys()

The dictionary like features are useful when reading in data from a file. The
keys are often more understandable to access the data than the index.
:func:`pyvista.examples.download_cavity()
<pyvista.examples.downloads.download_cavity>` is an OpenFoam dataset with a nested
``PartitionedDataSet`` structure. There are two entries in the top-level object

.. jupyter-execute::

   data = examples.download_cavity()
   data.keys()

``"internalMesh"`` is a :class:`pyvista.UnstructuredGrid`.

.. jupyter-execute::

   data["internalMesh"]

``"boundary"`` is another :class:`pyvista.PartitionedDataSet`.

.. jupyter-execute::

   data["boundary"]

Using the dictionary like features of :class:`pyvista.PartitionedDataSet` allow for easier
inspection and use of the data coming from an outside source. The names of each key
correspond to human understandable portions of the dataset.

.. jupyter-execute::

   data["boundary"].keys()

Examples using this class:

* :ref:`slice_example`
* :ref:`volumetric_example`
* :ref:`depth_peeling_example`


PartitionedDataSet API Reference
------------------------
The :class:`pyvista.PartitionedDataSet` class holds attributes that
are *common* to all spatially referenced datasets in PyVista. This
base class is analogous to VTK's `vtk.vtkPartitionedDataSetDataSet`_ class.

.. autosummary::
   :toctree: _autosummary

   pyvista.PartitionedDataSet

.. _vtk.vtkPartitionedDataSetDataSet: https://vtk.org/doc/nightly/html/classvtkPartitionedDataSetDataSet.html
