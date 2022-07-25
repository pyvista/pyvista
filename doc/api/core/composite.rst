Composite Datasets
==================

The :class:`pyvista.MultiBlock` class is a composite class to hold many
data sets which can be iterated over.

``MultiBlock`` behaves like a list, but also allows some dictionary
like features.  We can iterate over this data structure by index, and we
can also access blocks by their string name.

.. pyvista-plot::

   Create empty composite dataset

   >>> import pyvista as pv
   >>> blocks = pv.MultiBlock()

   Add some data to the collection.

   >>> blocks.append(pv.Sphere())
   >>> blocks.append(pv.Cube(center=(0, 0, -1)))

   Plotting the ``MultiBlock`` plots all the meshes contained by it.

   >>> blocks.plot(smooth_shading=True)

   ``MultiBlock`` is list-like, so individual blocks can be accessed via
   indices.

   >>> blocks[0]  # Sphere

   ``MultiBlock`` also has some dictionary features.  We can set the name
   of the blocks, and then access them 

   >>> blocks.set_block_name(0, "sphere")
   >>> blocks.set_block_name(1, "cube")
   >>> blocks["sphere"]  # Sphere again

   To append data, it is preferred to use :function:`pyvista.MultiBlock.append`.
   A name can be set for the block. It is also possible to append to the MultiBlock using a
   nonexistent key.

   >>> blocks.append(pv.Sphere(center=(-1, 0, 0)), "sphere2")
   >>> blocks["cone"] = pv.Cone()

   Duplicate and ``None`` keys are possible in MultiBlock, so the use of
   dictionary-like features must be used with care. 

   We can use slicing to retrieve or set multiple blocks.

   >>> blocks[0:1]

Examples using this class:

* :ref:`slice_example`
* :ref:`volumetric_example`
* :ref:`depth_peeling_example`


MultiBlock API Reference
------------------------
The :class:`pyvista.MultiBlock` class holds attributes that
are *common* to all spatially referenced datasets in PyVista.  This
base class is analogous to VTK's `vtk.vtkMultiBlockDataSet`_ class.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   pyvista.MultiBlock

.. _vtk.vtkMultiBlockDataSet: https://vtk.org/doc/nightly/html/classvtkMultiBlockDataSet.html
