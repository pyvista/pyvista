.. _vtk_role_docs:

Sphinx PyVista VTK Role
=======================

You can link directly to VTK's documentation with the ``:vtk:`` role
by adding the following to your ``conf.py`` when building your
documentation using Sphinx.

.. code-block:: python

    extensions = [
        ...,
        'pyvista.ext.vtk_role',
    ]

With the extension enable, you can write, for example, ``:vtk:`vtkPolyData```
in docstrings to link directly to the ``vtkPolyData`` documentation. This
will render as :vtk:`vtkPolyData`.

The role also works for linking to class members such as methods or enums,
e.g. write ``:vtk:`vtkPolyData.GetVerts``` to link directly to the ``GetVerts``
method. This will render as :vtk:`vtkPolyData.GetVerts`.
