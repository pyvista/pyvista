"""PyVista specific errors."""


class NotAllTrianglesError(ValueError):
    """Exception when a mesh does not contain all triangles.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='Mesh must consist of only triangles'):
        """Empty init."""
        ValueError.__init__(self, message)


class DeprecationError(RuntimeError):
    """Used for deprecated methods and functions.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='This feature has been deprecated'):
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
        self, message='The requested feature is not supported by the installed VTK version.'
    ):  # numpydoc ignore=PR01,RT01
        """Empty init."""
        RuntimeError.__init__(self, message)


class PointSetNotSupported(TypeError):
    """Requested filter or property is not supported by the PointSet class.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(
        self, message='The requested operation is not supported for PointSets.'
    ):  # numpydoc ignore=PR01,RT01
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
        self, message='Cell operations are not supported. PointSets contain no cells.'
    ):  # numpydoc ignore=PR01,RT01
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
        self, message='Slice and other dimension reducing filters are not supported on PointSets.'
    ):  # numpydoc ignore=PR01,RT01
        """Empty init."""
        PointSetNotSupported.__init__(self, message)


class MissingDataError(ValueError):
    """Exception when data is missing, e.g. no active scalars can be set.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message='No data available.'):
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class AmbiguousDataError(ValueError):
    """Exception when data is ambiguous, e.g. multiple active scalars can be set.

    Parameters
    ----------
    message : str
        Error message.

    """

    def __init__(self, message="Multiple data available."):
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
        self, message="VTK pipeline issue detected by PyVista."
    ):  # numpydoc ignore=PR01,RT01
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class PyVistaDeprecationWarning(Warning):
    """Non-supressed Deprecation Warning."""

    pass


class PyVistaFutureWarning(Warning):
    """Non-supressed Future Warning."""

    pass


class PyVistaEfficiencyWarning(Warning):
    """Efficiency warning."""

    pass
