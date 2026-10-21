.. _errors_api:

Errors and Warnings
-------------------
.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   core.errors.AmbiguousDataError
   core.errors.DeprecationError
   plotting.errors.InvalidCameraError
   core.errors.InvalidMeshError
   core.errors.InvalidMeshWarning
   core.errors.MissingDataError
   core.errors.NotAllTrianglesError
   core.errors.PointSetCellOperationError
   core.errors.PointSetDimensionReductionError
   core.errors.PointSetNotSupported
   core.errors.PrecisionWarning
   core.errors.PyVistaAttributeError
   core.errors.PyVistaDeprecationWarning
   core.errors.PyVistaEfficiencyWarning
   core.errors.PyVistaFutureWarning
   core.errors.PyVistaPipelineError
   plotting.errors.RenderWindowUnavailable
   core.errors.VTKExecutionError
   core.errors.VTKExecutionWarning
   core.errors.VTKVersionError

VTK Messages
~~~~~~~~~~~~
VTK reports errors and warnings as messages. By default each one becomes a
Python warning. :attr:`~pyvista.core.config.Config.vtk_errors` and
:attr:`~pyvista.core.config.Config.vtk_warnings` on ``pv.global_config`` set
what happens globally, and :class:`~pyvista.vtk_messages` overrides them for
a block of code and records the messages reported inside it. VTK's own
stderr logger is off by default; :func:`~pyvista.vtk_verbosity` turns it on.

.. autosummary::
   :toctree: _autosummary

   vtk_messages
   vtk_verbosity

VTK Observers
~~~~~~~~~~~~~
These classes observe events on VTK objects, such as errors, warnings, and
the progress of an algorithm.

.. autosummary::
   :toctree: _autosummary

   Observer
   ProgressMonitor

Deprecated
~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   core.utilities.VtkErrorCatcher
   send_errors_to_logging
   core.utilities.set_error_output_file
