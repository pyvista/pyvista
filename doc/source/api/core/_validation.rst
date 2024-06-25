Validation
==========

.. warning::

    This package is under active development and is unstable. **The API may
    change without warning**. It is currently intended for internal use by
    ``PyVista`` methods only but will be made public in a future version.

Input validation methods for checking and/or validating a variable has the
correct type and/or for use by an algorithm. These methods can be useful when
writing custom ``Python`` methods, ``VTK`` wrappers, and/or when `Contributing
to PyVista <https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst>`_.

Some common use cases for input validation are:

Validate a 3-element vector:
    * Use :func:`~pyvista.core._validation.validate_array3`
Validate an Nx3 point or vector array:
    * Use :func:`~pyvista.core._validation.validate_arrayNx3`
Validate point or cell IDs:
    * Use :func:`~pyvista.core._validation.validate_arrayN_unsigned`
Validate a transformation matrix:
    * Use :func:`~pyvista.core._validation.validate_transform4x4`

API Reference
-------------
.. currentmodule:: pyvista.core._validation
.. autosummary::
   :toctree: _autosummary

   check
   validate
