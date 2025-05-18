.. _vtk_role_docs:

Sphinx VTK Role
===============

You can link directly to VTK's documentation with the ``:vtk:`` role
by adding the following to your ``conf.py`` when building your
documentation using Sphinx.

.. code-block:: python

    extensions = [
        ...,
        'pyvista.ext.vtk_role',
    ]

With the extension enable, you can write, for example, ``:vtk:`vtkImageData```
in docstrings to link directly to the ``vtkImageData`` documentation. This
will render as :vtk:`vtkImageData`.

The role also works for linking to class members such as methods or enums.
For example write ``:vtk:`vtkImageData.GetDimensions``` to link directly to the
``GetDimensions`` method. This will render as :vtk:`vtkImageData.GetDimensions`.

Just like with standard Sphinx roles, you can use ``~`` to shorten the title
for the link. For example, ``:vtk:`~vtkImageData.GetDimensions``` will render
as :vtk:`~vtkImageData.GetDimensions`.

.. note::

    The directive currently does not support linking to nested members. For example,
    linking to an enum member with ``:vtk:`vtkCommand.EventIds``` is valid,
    but linking to a specific enum value with ``:vtk:`vtkCommand.EventIds.PickEvent```
    is not.
