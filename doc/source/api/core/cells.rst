Cells
=====

The cell :class:`pyvista.Cell` class is the PyVista representation of the
:vtk:`vtkGenericCell` and can be used to inspect a :class:`pyvista.DataSet`'s cells, faces, and edges.

.. note::
   While methods and classes are quite effective at inspecting and plotting
   parts of a dataset, they are inefficient and should be used only for
   interactive exploration and debugging. When working with larger datasets or
   working with multiple cells it is generally more efficient to use bulk methods
   like :func:`pyvista.DataSetFilters.extract_cells`.

Here's a quick example to demonstrate the usage of :func:`pyvista.DataSet.get_cell` by extracting a hexahedral cell from an example :class:`pyvista.UnstructuredGrid`.

.. jupyter-execute::
   :hide-code:

   # jupyterlab boiler plate setup
   import pyvista as pv
   pv.set_plot_theme('document')
   pv.set_jupyter_backend('static')

.. jupyter-execute::

   from pyvista import examples
   mesh = examples.load_hexbeam()
   cell = mesh.get_cell(0)
   cell

| You can then extract a single face of that cell.

.. jupyter-execute::

   face = cell.get_face(0)
   face

| Afterwards, you can extract an edge or edges from the face.

.. jupyter-execute::

   edge = face.get_edge(0)
   edge

| Each :class:`pyvista.Cell` can be individually plotted for convenience.

.. jupyter-execute::

   cell.plot(show_edges=True, line_width=3)


Class Definition
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   pyvista.Cell
   pyvista.CellArray
