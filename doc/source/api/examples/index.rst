Examples
========

.. automodule:: pyvista.examples
   :noindex:

.. currentmodule:: pyvista

PyVista contains a variety of built-in demos and downloadable example
datasets.

Built-In
--------
Several built-in datasets are included and are available for offline use.
For example, load the built-in :func:`~pyvista.examples.examples.load_random_hills`
dataset:

.. pyvista-plot::

   >>> from pyvista import examples
   >>> hills = examples.load_random_hills()
   >>> hills.plot()

See the API reference for more examples:

.. autosummary::
   :toctree: _autosummary

   examples.examples

Downloads
---------
Many datasets are too large to be included with PyVista, but can be
downloaded and cached locally. For example, we can download the
:func:`~pyvista.examples.downloads.download_turbine_blade` dataset:

.. pyvista-plot::

   >>> from pyvista import examples
   >>> blade_mesh = examples.download_turbine_blade()
   >>> blade_mesh.plot()

See the API reference for more downloads:

.. autosummary::
   :toctree: _autosummary

   examples.downloads

Demos
-----
PyVista also contains some demos which can be used to quickly
demonstrate features. For example, we can create and show the
orientation cube plotter demo:

.. pyvista-plot::

   >>> from pyvista import demos
   >>> plotter = demos.orientation_plotter()
   >>> plotter.show()

See the API reference for more demos:

.. autosummary::
   :toctree: _autosummary

   demos.demos

Planets
-------
Examples of planets and celestial bodies are also included. See the
API reference for details:

.. autosummary::
   :toctree: _autosummary

   examples.planets

3D Scene Datasets
-----------------
Some file formats are imported directly by the :class:`pyvista.Plotter`
instead of using :func:`pyvista.read`. These formats represent 3D geometry,
materials, and scene structure.

Examples of file formats supported by PyVista include ``VRML``
(VirtualReality Modeling Language), ``3DS`` (3D Studio), and
``glTF`` (Graphics Library Transmission Format).
See the API reference for details:

.. autosummary::
   :toctree: _autosummary

   examples.vrml
   examples.download_3ds
   examples.gltf

Cells
-----
Many examples of VTK :class:`cell types <pyvista.CellType>` are
available. These functions create single-cell :class:`pyvista.UnstructuredGrid`
objects which can be useful for learning about the different cells.

See the API reference for details:

.. autosummary::
   :toctree: _autosummary

   examples.cells

Dataset Gallery
---------------
Most of PyVista's datasets are showcased in the dataset gallery.
You can browse the gallery to find a particular kind of dataset and
view file and instance metadata for all datasets.

.. toctree::
   :maxdepth: 3

   /api/examples/dataset_gallery

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
