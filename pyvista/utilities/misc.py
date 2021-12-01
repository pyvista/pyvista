"""Miscellaneous pyvista functions."""
import warnings

import numpy as np

from pyvista import _vtk


def _get_vtk_id_type():
    """Return the numpy datatype responding to ``vtk.vtkIdTypeArray``."""
    VTK_ID_TYPE_SIZE = _vtk.vtkIdTypeArray().GetDataTypeSize()
    if VTK_ID_TYPE_SIZE == 4:
        return np.int32
    elif VTK_ID_TYPE_SIZE == 8:
        return np.int64
    return np.int32


class PyvistaDeprecationWarning(Warning):
    """Non-supressed Depreciation Warning."""

    pass


class VTKVersionInfo():
    """Contains the VTK version information.

    Wraps vtk.vtkVersion in a ``sys.version_info`` like manner.

    Notes
    -----
    The micro attribute may be either the build date in YYYYMMDD or
    the patch release.  Use caution when comparing version patch
    release.

    Examples
    --------
    Simply output the vtk version.

    >>> import pyvista
    >>> pyvista.vtk_version_info  # doctest:+SKIP
    VTK Version 9.1.0

    Compare the VTK version.

    >>> pyvista.vtk_version_info > (9, 0)  # doctest:+SKIP
    True

    Return the major version of vtk.

    >>> pyvista.vtk_version_info.major  # doctest:+SKIP
    9
    """

    def __init__(self):
        """Initialize the vtk version from the VTK class."""
        # Get the VTK version. The API here is consistent since at least VTK v4.0.0
        try:
            ver = _vtk.vtkVersion()
            self._major = ver.GetVTKMajorVersion()
            self._minor = ver.GetVTKMinorVersion()
            self._micro = ver.GetVTKBuildVersion()
        except:  # pragma: no cover
            warnings.warn("Unable to detect VTK version. Defaulting to v4.0.0")
            self._major, self._minor, self._micro = (4, 0, 0)

    @property
    def major(self) -> int:
        """Major vtk version.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.vtk_version_info.major  # doctest:+SKIP
        9
        """
        return self._major

    @property
    def minor(self) -> int:
        """Minor vtk version.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.vtk_version_info.minor  # doctest:+SKIP
        1
        """
        return self._minor

    @property
    def micro(self) -> int:
        """Micro vtk version.

        This version may be either the released patch version or an
        integer denoting the build date in YYYYMMDD.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.vtk_version_info.micro  # doctest:+SKIP
        0

        """
        return self._micro

    def _as_tuple(self):
        return (self.major, self.minor, self.micro)

    def __gt__(self, other):
        """Return tuple-like greater than."""
        return self._as_tuple() > other

    def __ge__(self, other):
        """Return tuple-like greater than or equal to."""
        return self._as_tuple() >= other

    def __lt__(self, other):
        """Return tuple-like less than than."""
        return self._as_tuple() < other

    def __le__(self, other):
        """Return tuple-like less than or equal to."""
        return self._as_tuple() <= other

    def __eq__(self, other):
        """Return tuple-like equal to."""
        return self._as_tuple() == other

    def __iter__(self):
        """Yield tuple-like iterator."""
        yield self._as_tuple()

    def __contains__(self, other):
        """Return tuple-like contains."""
        return other in self._as_tuple()

    def __repr__(self):
        """Return representation."""
        return f"VTK Version {self._major}.{self._minor}.{self._micro}"


vtk_version_info = VTKVersionInfo()
