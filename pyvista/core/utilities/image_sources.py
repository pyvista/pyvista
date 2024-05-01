"""Provide sources for generating images."""

from __future__ import annotations

from typing import ClassVar, List, Sequence

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


@no_new_attr
class ImageMandelbrotSource(_vtk.vtkImageMandelbrotSource):
    """Create an image of the Mandelbrot set.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    whole_extent : sequence[int]
        The extent of the whole output image.

    maxiter : int
        The maximum number of iterations.

    Examples
    --------
    Create an image of the Mandelbrot set.

    >>> import pyvista as pv
    >>> source = pv.ImageMandelbrotSource(
    ...     whole_extent=(0, 200, 0, 200, 0, 0),
    ...     maxiter=100,
    ... )
    >>> source.output.plot(cpos="xy")
    """

    def __init__(self, whole_extent=None, maxiter=None) -> None:
        super().__init__()
        if whole_extent is not None:
            self.whole_extent = whole_extent
        if maxiter is not None:
            self.maxiter = maxiter

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
    def maxiter(self) -> int:
        """Get the maximum number of iterations.

        Returns
        -------
        int
            The maximum number of iterations.
        """
        return self.GetMaximumNumberOfIterations()

    @maxiter.setter
    def maxiter(self, maxiter: int) -> None:
        """Set the maximum number of iterations.

        Parameters
        ----------
        maxiter : int
            The maximum number of iterations.
        """
        self.SetMaximumNumberOfIterations(maxiter)

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


@no_new_attr
class ImageNoiseSource(_vtk.vtkImageNoiseSource):
    """Create a binary image of an ellipsoid class.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    whole_extent : sequence[int]
        The extent of the whole output image.

    minimum : float
        The minimum value for the generated noise.

    maximum : float
        The maximum value for the generated noise.

    Examples
    --------
    Create an image of noise.

    >>> import pyvista as pv
    >>> source = pv.ImageNoiseSource(
    ...     whole_extent=(0, 200, 0, 200, 0, 0),
    ...     minimum=0,
    ...     maximum=255,
    ... )
    >>> source.output.plot(cpos="xy")
    """

    _new_attr_exceptions: ClassVar[List[str]] = ['_whole_extent', 'whole_extent']

    def __init__(self, whole_extent=None, minimum=None, maximum=None) -> None:
        super().__init__()
        if whole_extent is not None:
            self.whole_extent = whole_extent
        if minimum is not None:
            self.minimum = minimum
        if maximum is not None:
            self.maximum = maximum

    @property
    def whole_extent(self) -> Sequence[int]:
        """Get extent of the whole output image.

        Returns
        -------
        sequence[int]
          The extent of the whole output image.
        """
        return self._whole_extent

    @whole_extent.setter
    def whole_extent(self, whole_extent: Sequence[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        whole_extent : sequence[int]
          The extent of the whole output image.
        """
        self._whole_extent = whole_extent
        self.SetWholeExtent(whole_extent)

    @property
    def minimum(self) -> float:
        """Get the minimum value for the generated noise.

        Returns
        -------
        float
          The minimum value for the generated noise.
        """
        return self.GetMinimum()

    @minimum.setter
    def minimum(self, minimum: float) -> None:
        """Set the minimum value for the generated noise.

        Parameters
        ----------
        minimum : float
          The minimum value for the generated noise.
        """
        self.SetMinimum(minimum)

    @property
    def maximum(self) -> float:
        """Get the maximum value for the generated noise.

        Returns
        -------
        float
          The maximum value for the generated noise.
        """
        return self.GetMaximum()

    @maximum.setter
    def maximum(self, maximum: float) -> None:
        """Set the maximum value for the generated noise.

        Parameters
        ----------
        maximum : float
          The maximum value for the generated noise.
        """
        self.SetMaximum(maximum)

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


@no_new_attr
class ImageGaussianSource(_vtk.vtkImageGaussianSource):
    """Create a binary image with Gaussian pixel values.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    center : sequence[float]
        The center of the gaussian.

    whole_extent : sequence[int]
        The extent of the whole output image.

    maximum : float
        The maximum value of the gaussian.

    std : sequence[float]
        The standard deviation of the gaussian.

    Examples
    --------
    Create an image of Gaussian pixel values.

    >>> import pyvista as pv
    >>> source = pv.ImageGaussianSource(
    ...     center=(100, 100, 0),
    ...     whole_extent=(0, 200, 0, 200, 0, 0),
    ...     maximum=255,
    ...     std=0.25,
    ... )
    >>> source.output.plot(cpos="xy")
    """

    _new_attr_exceptions: ClassVar[List[str]] = ['_whole_extent', 'whole_extent']

    def __init__(self, center=None, whole_extent=None, maximum=None, std=None) -> None:
        super().__init__()
        if center is not None:
            self.center = center
        if whole_extent is not None:
            self.whole_extent = whole_extent
        if maximum is not None:
            self.maximum = maximum
        if std is not None:
            self.std = std

    @property
    def center(self) -> Sequence[float]:
        """Get the center of the gaussian.

        Returns
        -------
        sequence[float]
          The center of the gaussian.
        """
        return self.GetCenter()

    @center.setter
    def center(self, center: Sequence[float]) -> None:
        """Set the center of the gaussian.

        Parameters
        ----------
        center : sequence[float]
          The center of the gaussian.
        """
        self.SetCenter(center)

    @property
    def whole_extent(self) -> Sequence[int]:
        """Get extent of the whole output image.

        Returns
        -------
        sequence[int]
          The extent of the whole output image.
        """
        return self._whole_extent

    @whole_extent.setter
    def whole_extent(self, whole_extent: Sequence[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        whole_extent : sequence[int]
          The extent of the whole output image.
        """
        self._whole_extent = whole_extent
        self.SetWholeExtent(
            whole_extent[0],
            whole_extent[1],
            whole_extent[2],
            whole_extent[3],
            whole_extent[4],
            whole_extent[5],
        )

    @property
    def maximum(self) -> float:
        """Get the maximum value of the gaussian.

        Returns
        -------
        float
          The maximum value of the gaussian.
        """
        return self.GetMaximum()

    @maximum.setter
    def maximum(self, maximum: float) -> None:
        """Set the maximum value of the gaussian.

        Parameters
        ----------
        maximum : float
          The maximum value of the gaussian.
        """
        self.SetMaximum(maximum)

    @property
    def std(self) -> float:
        """Get the standard deviation of the gaussian.

        Returns
        -------
        float
          The standard deviation of the gaussian.
        """
        return self.GetStandardDeviation()

    @std.setter
    def std(self, std: float) -> None:
        """Set the standard deviation of the gaussian.

        Parameters
        ----------
        std : float
          The standard deviation of the gaussian.
        """
        self.SetStandardDeviation(std)

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
