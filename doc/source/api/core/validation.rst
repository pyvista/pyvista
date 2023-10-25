Input Validation
================
Input validation methods for checking and/or validating a variable has the
correct type and/or for use by an algorithm. These methods can be useful when
writing custom ``Python`` methods, ``VTK`` wrappers, and/or when `Contributing
to PyVista <https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst>`_.

Some common use cases for input validation are:

Validate an Nx3 point or vector array:
    * Use :func:`~pyvista.core.input_validation.validate.validate_arrayNx3`
Validate a single 3-element vector:
    * Use :func:`~pyvista.core.input_validation.validate.validate_array3`
Validate point or cell IDs (or other unsigned integer IDs):
    * Use :func:`~pyvista.core.input_validation.validate.validate_arrayN_uintlike`
Validate a transformation matrix:
    * Use :func:`~pyvista.core.input_validation.validate.validate_transform4x4`


Input Validation API Reference
------------------------------
.. currentmodule:: pyvista.core.input_validation
.. autosummary::
   :toctree: _autosummary

   validate
   check
