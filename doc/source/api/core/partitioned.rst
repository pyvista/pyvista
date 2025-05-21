Partitioned Datasets
====================

The :class:`pyvista.PartitionedDataSet` class is a partitioned dataset that encapsulates
a dataset consisting of partitions. ``PartitionedDataSet`` behaves mostly like a list.

List-like Features
------------------

Create an empty partitioned dataset

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

``PartitionedDataSet`` is List-like so that individual partitions can be accessed via
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
and supports operations such as append, insert, etc.

.. jupyter-execute::

   partitions.append(pv.Cone())
   partitions.reverse()

.. warning::

   pop is not supported in ``PartitionedDataSet`` class.

``PartitionedDataSet`` also supports slicing to get or set partitions.

.. jupyter-execute::

   partitions[0:2]  # The Sphere and Cube objects in a new ``PartitionedDataSet``

PartitionedDataSet API Reference
--------------------------------

The :class:`pyvista.PartitionedDataSet` class holds attributes that
are *common* to all spatially referenced datasets in PyVista. This
base class is analogous to VTK's :vtk:`vtkPartitionedDataSet` class.

.. autosummary::
   :toctree: _autosummary

   pyvista.PartitionedDataSet
