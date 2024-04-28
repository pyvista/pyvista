"""Provide sources for generating images."""

from __future__ import annotations

from typing import Sequence

from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.misc import no_new_attr

from .helpers import wrap


@no_new_attr
class ImageEllipsoidSource(_vtk.vtkImageEllipsoidSource):
    """Create a binary image of an ellipsoid class.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    whole_extent : sequence[int]
        The extent of the whole output image.

    center : sequence[float]
        The center of the ellipsoid.

    radius : tuple
        The radius of the ellipsoid.

    Examples
    --------
    Create an image of an ellipsoid.

    >>> import pyvista as pv
    >>> source = pv.ImageEllipsoidSource(
    ...     whole_extent=(0, 20, 0, 20, 0, 0),
    ...     center=(10, 10, 0),
    ...     radius=(3, 4, 5),
    ... )
    >>> source.output.plot(cpos="xy")
    """

    def __init__(self, whole_extent=None, center=None, radius=None) -> None:
        super().__init__()
        if whole_extent is not None:
            self.whole_extent = whole_extent
        if center is not None:
            self.center = center
        if radius is not None:
            self.radius = radius

    @property
    def whole_extent(self) -> Sequence[int]:
        """Get extent of the whole output image.

        Returns
        -------
        sequence[int]
            The extent of the whole output image.
        """
        return self.GetWholeExtent()

    @whole_extent.setter
    def whole_extent(self, whole_extent: Sequence[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        whole_extent : sequence[int]
            The extent of the whole output image.
        """
        self.SetWholeExtent(whole_extent)

    @property
    def center(self) -> Sequence[float]:
        """Get the center of the ellipsoid.

        Returns
        -------
        sequence[float]
            The center of the ellipsoid.
        """
        return self.GetCenter()

    @center.setter
    def center(self, center: Sequence[float]) -> None:
        """Set the center of the ellipsoid.

        Parameters
        ----------
        center : sequence[float]
            The center of the ellipsoid.
        """
        self.SetCenter(center)

    @property
    def radius(self) -> Sequence[float]:
        """Get the radius of the ellipsoid.

        Returns
        -------
        sequence[float]
            The radius of the ellipsoid.
        """
        return self.GetRadius()

    @radius.setter
    def radius(self, radius: Sequence[float]) -> None:
        """Set the radius of the ellipsoid.

        Parameters
        ----------
        radius : sequence[float]
            The radius of the ellipsoid.
        """
        self.SetRadius(radius)

    @property
    def output(self):
        """Get the output image as a ImageData.

        Returns
        -------
        pyvista.ImageData
            The output image.
        """
        self.Update()
        return wrap(self.GetOutput())
