.. _errors_api:

Errors and Warnings
-------------------
.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   core.errors.AmbiguousDataError
   core.errors.DeprecationError
   core.errors.InvalidMeshError
   core.errors.InvalidMeshWarning
   core.errors.MissingDataError
   core.errors.NotAllTrianglesError
   core.errors.PointSetCellOperationError
   core.errors.PointSetDimensionReductionError
   core.errors.PointSetNotSupported
   core.errors.PyVistaAttributeError
   core.errors.PyVistaDeprecationWarning
   core.errors.PyVistaEfficiencyWarning
   core.errors.PyVistaFutureWarning
   core.errors.PyVistaPipelineError
   core.errors.VTKExecutionError
   core.errors.VTKExecutionWarning
   core.errors.VTKVersionError
   plotting.errors.InvalidCameraError
   plotting.errors.RenderWindowUnavailable

VTK Observers
~~~~~~~~~~~~~
These classes observe events on VTK objects, such as errors, warnings, and
the progress of an algorithm.

.. autosummary::
   :toctree: _autosummary

   Observer
   ProgressMonitor
   core.utilities.VtkErrorCatcher

VTK Error Output
~~~~~~~~~~~~~~~~
These functions redirect the errors and warnings that VTK emits. To
control how much VTK logs in the first place, see
:func:`~pyvista.vtk_verbosity`.

.. autosummary::
   :toctree: _autosummary

   send_errors_to_logging
   core.utilities.set_error_output_file
