Examples
========
PyVista contains a variety of built-in demos and downloadable example
datasets. For example:

.. pyvista-plot::

   Plot the built-in hills dataset

   >>> from pyvista import examples
   >>> hills = examples.load_random_hills()
   >>> hills.plot()

Many datasets are too large to be included with PyVista, but can be
downloaded and cached locally. These datasets can be downloaded and
used with:

.. pyvista-plot::

   Plot the turbine blade mesh.

   >>> from pyvista import examples
   >>> blade_mesh = examples.download_turbine_blade()
   >>> blade_mesh.plot()

Finally, PyVista contains some demos which can be used to quickly
demonstrate features.

.. pyvista-plot::

   Create and show the orientation cube plotter

   >>> from pyvista import demos
   >>> plotter = demos.orientation_plotter()
   >>> plotter.show()

Dataset Gallery
---------------
Many of PyVista's datasets are showcased in the dataset gallery.
You can browse the gallery to find a particular kind of dataset and
view file and instance metadata for all datasets.

.. toctree::
   :maxdepth: 3

   /api/examples/dataset_gallery

API Reference
-------------

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   examples.examples
   examples.downloads
   examples.planets
   examples.vrml
   examples.download_3ds
   demos.demos

Usage Considerations
--------------------
.. warning::
   As you browse this repository and think about how you might use our 3D
   models and range datasets, please remember that several of these artifacts
   have religious or cultural significance.
   Examples include the Buddha, a religious symbol revered by hundreds of
   millions of people; the dragon, a symbol of Chinese culture, the Thai
   statue, which contains elements of religious significance to Hindus; and Lucy, a
   Christian angel commonly seen as statues in Italian churches.
   Keep your renderings and other uses of these particular models in good
   taste. Don't animate or morph them, don't apply Boolean operators to them,
   and don't simulate nasty things happening to them (like breaking, exploding,
   melting, etc.).
   Choose another model for these sorts of experiments.
   (You can do anything you want to the Stanford bunny or the armadillo.)


Downloads Cache and Data Sources
--------------------------------
If you have an internet connection and a normal user account, PyVista should be
able to download and cache examples without an issue. The following two
sections deal with those who wish to customize how PyVista downloads examples.

Cache
~~~~~

PyVista uses `pooch <https://github.com/fatiando/pooch>`_ to download and store
the example files in a local cache. You can determine the location of this cache
at runtime with:

.. code-block:: python

   >>> from pyvista import examples
   >>> # Get the local examples path on Linux
   >>> examples.PATH
   '/home/user/.cache/pyvista_3'


You can clear out the local cache with :func:`examples.delete_downloads()
<pyvista.examples.downloads.delete_downloads>` if needed.

If you want to override this local cache path, set the
``PYVISTA_USERDATA_PATH`` environment variable. This path must be writable.


Data Sources
~~~~~~~~~~~~
PyVista uses `PyVista/vtk-data <https://github.com/pyvista/vtk-data.git>`_ as
the main source for example data. If you do not have internet access or you
prefer using a local or network directory instead, you can override this
source with the ``VTK_DATA_PATH`` environment variable.

The following example first clones the git repository and then exports that
directory to PyVista via ``VTK_DATA_PATH``. Note how the path ends in
``'Data'`` since we need to specify the exact directory of the Data for
``pooch``.

.. code-block:: bash

   git clone https://github.com/pyvista/vtk-data.git
   export VTK_DATA_PATH=/home/alex/python/vtk-data/Data

.. _examples_cells:

Cells
-----
PyVista contains several functions that create single cell
:class:`pyvista.UnstructuredGrid` objects that can be used to learn about VTK
cell types.

.. currentmodule:: pyvista.examples.cells

.. autosummary::
   :toctree: _autosummary

   plot_cell
   Empty
   Vertex
   PolyVertex
   Line
   PolyLine
   Triangle
   TriangleStrip
   Polygon
   Pixel
   Quadrilateral
   Tetrahedron
   Voxel
   Hexahedron
   Wedge
   Pyramid
   PentagonalPrism
   HexagonalPrism
   QuadraticEdge
   QuadraticTriangle
   QuadraticQuadrilateral
   QuadraticPolygon
   QuadraticTetrahedron
   QuadraticHexahedron
   QuadraticWedge
   QuadraticPyramid
   BiQuadraticQuadrilateral
   TriQuadraticHexahedron
   TriQuadraticPyramid
   QuadraticLinearQuadrilateral
   QuadraticLinearWedge
   BiQuadraticQuadraticWedge
   BiQuadraticQuadraticHexahedron
   BiQuadraticTriangle
   CubicLine
