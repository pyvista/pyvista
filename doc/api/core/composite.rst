Composite Datasets
==================

The :class:`pyvista.MultiBlock` class is a composite class to hold many
data sets which can be iterated over.

You can think of MultiBlock like lists or dictionaries as we can
iterate over this data structure by index and we can also access
blocks by their string name.

.. pyvista-plot::

   Create empty composite dataset

   >>> import pyvista
   >>> blocks = pyvista.MultiBlock()

   Add a dataset to the collection

   >>> blocks.append(pyvista.Sphere())

   Or add a named block

   >>> blocks["cube"] = pyvista.Cube(center=(0, 0, -1))

   Plotting the MultiBlock plots all the meshes contained by it.

   >>> blocks.plot(smooth_shading=True)

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
