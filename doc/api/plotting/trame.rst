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
   :no-inherited-members:

   launch_server
   show_trame
   elegantly_launch
   get_or_create_viewer
   plotter_ui
   PyVistaLocalView
   PyVistaRemoteLocalView
   PyVistaRemoteView
