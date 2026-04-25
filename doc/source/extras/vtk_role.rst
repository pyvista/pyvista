.. _vtk_role_docs:

Sphinx VTK Role
===============

.. versionadded:: 0.46

PyVista's documentation uses the Sphinx extension https://github.com/pyvista/vtk-xref
to link directly to `VTK's documentation <https://vtk.org/doc/nightly/html/index.html>`_.

This extension adds the ``:vtk:`` role to allow writing, for example,
``:vtk:`vtkImageData``` inside docstrings to link directly to the ``vtkImageData``
documentation. This will render as :vtk:`vtkImageData`.

See `vtk-xref's repository <https://github.com/pyvista/vtk-xref>`_ for installation and usage details
for adding the ``:vtk:`` role to your project.
