Composite Datasets
==================

The :class:`pyvista.MultiBlock` class is a composite class to hold many
data sets which can be iterated over.

MultiBlock behaves like a list, but also allows some dictionary
like features.  We can iterate over this data structure by index, and we
can also access blocks by their string name.

.. pyvista-plot::

   Create empty composite dataset

   >>> import pyvista as pv
   >>> blocks = pv.MultiBlock()

   Add a dataset to the collection

   >>> blocks.append(pv.Sphere())

   Add a named block and access it by name like a dict

   >>> blocks.append(pv.Cube(center=(0, 0, -1)), "cube")
   >>> blocks["cube"].bounds  # same as blocks[1].bounds

   Plotting the MultiBlock plots all the meshes contained by it.

   >>> blocks.plot(smooth_shading=True)

   It is also possible to append to the MultiBlock using a
   nonexistent key

   >>> blocks["cone"] = pv.Cone()

   Duplicate and ``None`` keys are possible in MultiBlock, so the use of
   dictionary features must be used with care. 

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
