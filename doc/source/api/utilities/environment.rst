.. _environment_api:

Environment
-----------
.. currentmodule:: pyvista

.. seealso::

   :ref:`configuration`
      Central reference for all global configuration, including
      :class:`pyvista.core.config.Config`, module-level flags, and
      environment variables.

PyVista Version Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The PyVista library provides a way of getting the version installed in your
environment.

>>> # Output the version of PyVista.
>>> import pyvista as pv
>>> pv.version_info
(0, 44, 0)

VTK Version and Backend
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   vtk_backend

The PyVista library is heavily dependent on VTK and provides an easy
way of getting the version of VTK in your environment.

>>> # Output the version of VTK.
>>> import pyvista as pv
>>> pv.vtk_version_info
VTKVersionInfo(major=9, minor=1, micro=0)

>>> # Get the major version of VTK
>>> pv.vtk_version_info.major
9

Environment Report
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   Report
   GPUInfo
   get_gpu_info

Runtime Controls
~~~~~~~~~~~~~~~~
These settings apply to the whole process. The state managers also work as
context managers, applying only within the ``with`` block.

.. autosummary::
   :toctree: _autosummary

   vtk_verbosity
   vtk_snake_case
   enable_smp_tools
   allow_new_attributes
   set_new_attribute
   set_pickle_format
