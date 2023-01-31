.. _trame_api:

Trame
-----
The PyVista :mod:`pyvista.trame` module allows users to access the `Trame
<https://kitware.github.io/trame/index.html>`_ widget from PyVista to create
web-based 3D visualizations. This allows you to access the VTK pipeline using
the PyVista API so you can pair PyVista and Trame so that PyVista plotters can
be used in a web context with both server and client-side rendering.

For the full user guide, see :ref:`trame_jupyter`.

.. currentmodule:: pyvista.trame

.. autosummary::
   :toctree: _autosummary

   jupyter.launch_server
   jupyter.show_trame
   jupyter.elegantly_launch
   ui.get_or_create_viewer
   ui.plotter_ui
