Composite Datasets
==================

The :class:`pyvista.MultiBlock` class is a composite class to hold many
data sets which can be iterated over. ``MultiBlock`` behaves mostly like
a list, but has some dictionary-like features.

List-like Features
------------------

.. pyvista-plot::

   Create empty composite dataset

   >>> import pyvista as pv
   >>> from pyvista import examples
   >>> blocks = pv.MultiBlock()

   Add some data to the collection.

   >>> blocks.append(pv.Sphere())
   >>> blocks.append(pv.Cube(center=(0, 0, -1)))

   Plotting the ``MultiBlock`` plots all the meshes contained by it.

   >>> blocks.plot(smooth_shading=True)

   ``MultiBlock`` is list-like, so individual blocks can be accessed via
   indices.

   >>> blocks[0]  # Sphere

   The length of the block can be accessed through :func:`len`

   >>> len(blocks)

   or through the ``n_blocks`` attribute

   >>> blocks.n_blocks

   More specifically, ``MultiBlock`` is a :class:`collections.abc.MutableSequence`
   and supports operations such as append, pop, insert, etc. Some of these operations
   allow optional names to be provided for the dictionary like usage.  TODO: link.

   >>> blocks.append(pv.Cone(), name="cone")
   >>> cone = blocks.pop(-1)  # Pops Cone
   >>> blocks.reverse()

   ``MultiBlock`` also supports slicing for getting or setting blocks.

   >>> blocks[0:2]  # The Sphere and Cube objects in a new ``MultiBlock``


Dictionary-like Features
------------------------
.. pyvista-plot::

   ``MultiBlock`` also has some dictionary features.  We can set the name
   of the blocks, and then access them 
   >>> blocks = pv.MultiBlock([pv.Sphere(), pv.Cube()])
   >>> blocks.set_block_name(0, "sphere")
   >>> blocks.set_block_name(1, "cube")
   >>> blocks["sphere"]  # Sphere

   It is important to note that ``MultiBlock`` is not a dictionary and does
   not enforce unique keys.  Keys can also be ``None``.  Extra care must be
   taken to avoid problems using the dictionary-like features.

   PyVista tries to keep the keys ordered correctly when doing list operations.

   >>> block.reverse()
   >>> block.keys()

   The dictionary like features are useful when reading in data from a file.  The
   keys are often more understandable to access the data than the index.
   :func:`pyvista.examples.download_cavity` is an OpenFoam dataset with a nested
   ``MultiBlock`` structure.  There are two entries in the top-level object

   >>> data = examples.download_cavity()
   >>> data.keys()

   ``"internalMesh"`` is a :class:`pyvista.UnstructuredGrid`.

   >>> data["internalMesh"]

   ``"boundary"`` is another :class:`pyvista.MultiBlock`.

   >>> data["boundary"]

   Using the dictionary like features of :class:`pyvista.MultiBlock` allow for easier
   inspection and use of the data coming from an outside source.  The names of each key
   correspond to human understable portions of the dataset.

   >>> data["boundary"].keys()

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
