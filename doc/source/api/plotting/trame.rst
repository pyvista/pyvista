.. _trame_api:

Trame
-----
The Trame integration for PyVista has moved to the standalone
`trame-pyvista <https://github.com/pyvista/trame-pyvista>`_ package.
Install it directly::

    pip install trame-pyvista

This pulls in `Trame <https://kitware.github.io/trame/index.html>`_ and
registers the Jupyter ``trame``/``server``/``client``/``html`` backends
plus the ``plotter.trame`` plotter component
(for example, ``plotter.trame.export_html(...)``,
``plotter.trame.export_vtksz(...)``).

For documentation of the views, viewers, and exporters, see the
``trame-pyvista`` project. For the user guide on using Trame with
PyVista in Jupyter, see :ref:`trame_jupyter`.
