"""PyVista specific errors."""
# flake8: noqa: F401

# Import accessible under `pyvista.errors`
from pyvista.core.errors import (
    AmbiguousDataError,
    DeprecationError,
    MissingDataError,
    NotAllTrianglesError,
    PointSetCellOperationError,
    PointSetDimensionReductionError,
    PointSetNotSupported,
    PyVistaDeprecationWarning,
    PyVistaEfficiencyWarning,
    PyVistaFutureWarning,
    PyVistaPipelineError,
    VTKVersionError,
)
from pyvista.plotting.errors import InvalidCameraError, RenderWindowUnavailable
