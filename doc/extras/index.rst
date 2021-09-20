Extras
******
This section contains resources to expand the usage of PyVista beyond
just running it from a Python console or IDE.  For example, you can
package PyVista in a docker image and build VTK with EGL to enable
rich headless plotting on remote servers.  You can also package
PyVista using `pyinstaller`_ to be used within a standalone
application.  You could even make a basic web application using
`flask`_.

See the :ref:`ref_developer_notes` section for details on contributing
and how you can help develop PyVista.

.. toctree::

   building_vtk
   docker
   pyinstaller
   flask
   developer_notes
   plot_directive
   vtk_data
   extending_pyvista

.. _pyinstaller: https://www.pyinstaller.org/
.. _flask: https://flask.palletsprojects.com/
