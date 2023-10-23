Input Validation
================
Input validation methods for checking and/or validating a variable has the
correct type and/or for use by an algorithm. These methods can be useful when
writing custom ``Python`` methods, ``VTK`` wrappers, and/or when `Contributing
to PyVista <https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst>`_.

Some common use cases for input validation are:

* Validate an Nx3 point or vector array:
    Use :func:`~validate_arrayNx3`
* Validate a single 3-element vector:
    Use :func:`~validate_array3`
* Validate a data range ``[lower_bound, upper_bound]``:
    Use :func:`~validate_data_range`:
* Validate point or cell IDs (or other unsigned integer IDs):
    Use :func:`~validate_arrayN_uintlike`
* Validate a transformation matrix:
    Use :func:`~validate_transform4x4`


Validate API Reference
----------------------
.. autosummary::
   :toctree: _autosummary

   pyvista.core.input_validation.validate


Check API Reference
----------------------
.. autosummary::
   :toctree: _autosummary

   pyvista.core.input_validation.check