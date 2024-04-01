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

PartitionedDataSet API Reference
--------------------------------

The :class:`pyvista.PartitionedDataSet` class holds attributes that
are *common* to all spatially referenced datasets in PyVista. This
base class is analogous to VTK's `vtk.vtkPartitionedDataSetDataSet`_ class.

.. autosummary::
   :toctree: _autosummary

   pyvista.PartitionedDataSet

.. _vtk.vtkPartitionedDataSetDataSet: https://vtk.org/doc/nightly/html/classvtkPartitionedDataSetDataSet.html
