"""PyVista specific errors."""

from __future__ import annotations


class NotAllTrianglesError(ValueError):
    """Exception when a mesh does not contain all triangles.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='Mesh must consist of only triangles') -> None:
        """Empty init."""
        ValueError.__init__(self, message)


class DeprecationError(RuntimeError):
    """Used for deprecated methods and functions.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='This feature has been deprecated') -> None:
        """Empty init."""
        RuntimeError.__init__(self, message)


class VTKVersionError(RuntimeError):
    """Requested feature is not supported by the installed VTK version.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self,
        message='The requested feature is not supported by the installed VTK version.',
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Empty init."""
        RuntimeError.__init__(self, message)


class PointSetNotSupported(TypeError):  # noqa: N818
    """Requested filter or property is not supported by the PointSet class.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self,
        message='The requested operation is not supported for PointSets.',
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Empty init."""
        TypeError.__init__(self, message)


class PointSetCellOperationError(PointSetNotSupported):
    """Requested filter or property is not supported by the PointSet class.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self,
        message='Cell operations are not supported. PointSets contain no cells.',
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Empty init."""
        PointSetNotSupported.__init__(self, message)


class PointSetDimensionReductionError(PointSetNotSupported):
    """Requested filter or property is not supported by the PointSet class.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self,
        message='Slice and other dimension reducing filters are not supported on PointSets.',
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Empty init."""
        PointSetNotSupported.__init__(self, message)


class PartitionedDataSetsNotSupported(TypeError):  # noqa: N818
    """Requested filter or property is not supported by the PartitionedDataSets class.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self,
        message='The requested operation is not supported for PartitionedDataSetss.',
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Empty init."""
        TypeError.__init__(self, message)


class MissingDataError(ValueError):
    """Exception when data is missing, for example, no active scalars can be set.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='No data available.') -> None:
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class AmbiguousDataError(ValueError):
    """Exception when data is ambiguous, for example, multiple active scalars can be set.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='Multiple data available.') -> None:
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class CellSizeError(ValueError):
    """Exception when a cell array size is invalid.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='Cell array size is invalid.') -> None:
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class PyVistaPipelineError(RuntimeError):
    """Exception when a VTK pipeline runs into an issue.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self,
        message='VTK pipeline issue detected by PyVista.',
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class PyVistaAttributeError(AttributeError):
    """Exception when accessing an attribute that is not part of the PyVista API.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self,
        message='The attribute is not part of the PyVista API',
    ) -> None:  # numpydoc ignore=PR01,RT01
        super().__init__(message)


class InvalidMeshError(ValueError):
    """Error for invalid mesh properties.

    .. versionadded:: 0.47

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='Invalid mesh.') -> None:
        super().__init__(message)


class VTKExecutionError(RuntimeError):
    """Exception when a VTK output message is detected.

    .. versionadded:: 0.47

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self,
        message='VTK output message was detected by PyVista.',
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class PyVistaDeprecationWarning(Warning):
    """Non-supressed Deprecation Warning."""


class PyVistaFutureWarning(Warning):
    """Non-supressed Future Warning."""


class PyVistaEfficiencyWarning(Warning):
    """Efficiency warning."""


class VTKExecutionWarning(RuntimeWarning):
    """Warning when a VTK output message is detected.

    .. versionadded:: 0.47

    """


class InvalidMeshWarning(Warning):
    """Warning for invalid mesh properties.

    .. versionadded:: 0.47

    """


class PrecisionWarning(Warning):
    """Warning that points could not be generated at the requested precision.

    Raised when :attr:`pyvista.core.config.Config.points_dtype` asks for a wider
    dtype than the VTK algorithm that ran can generate. The output points are cast
    up so that the dtype is the one asked for, but the values they hold have the
    precision the algorithm produced, and casting cannot bring back digits it
    already discarded.

    The message names whatever generated the points: the VTK class for a filter, the
    PyVista class for a source, since a source is its own algorithm, and the library
    for a hull computed outside VTK.

    Being a warning rather than an error is what keeps the choice with the caller.
    Escalate it where the fabricated precision is not acceptable::

        warnings.filterwarnings('error', category=pv.PrecisionWarning)

    or silence it where it is::

        warnings.filterwarnings('ignore', category=pv.PrecisionWarning)

    Either can be scoped to a block with :class:`warnings.catch_warnings`, or set
    for a run from ``-W`` or a test runner's own configuration.

    .. versionadded:: 0.49

    """
