"""Provide sources for generating images."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from pyvista import _vtk
from pyvista.core._vtk_utilities import DisableVtkSnakeCase
from pyvista.core.utilities.misc import _NoNewAttrMixin

from .helpers import wrap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista import ImageData
    from pyvista.core._typing_core import VectorLike


class ImageEllipsoidSource(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkImageEllipsoidSource):
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
    >>> source.output.plot(cpos='xy')

    """

    def __init__(
        self,
        whole_extent: VectorLike[int] | None = None,
        center: VectorLike[float] | None = None,
        radius: VectorLike[float] | None = None,
    ) -> None:
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
    def whole_extent(self, whole_extent: VectorLike[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        whole_extent : sequence[int]
            The extent of the whole output image.

        """
        self.SetWholeExtent(whole_extent)  # type: ignore[call-overload]

    @property
    def center(self) -> tuple[float, float, float]:
        """Get the center of the ellipsoid.

        Returns
        -------
        tuple[float, float, float]
            The center of the ellipsoid.

        """
        return self.GetCenter()

    @center.setter
    def center(self, center: VectorLike[float]) -> None:
        """Set the center of the ellipsoid.

        Parameters
        ----------
        center : sequence[float]
            The center of the ellipsoid.

        """
        self.SetCenter(center)  # type: ignore[arg-type]

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
    def radius(self, radius: VectorLike[float]) -> None:
        """Set the radius of the ellipsoid.

        Parameters
        ----------
        radius : sequence[float]
            The radius of the ellipsoid.

        """
        self.SetRadius(radius)  # type: ignore[arg-type]

    @property
    def output(self) -> ImageData:
        """Get the output image as a ImageData.

        Returns
        -------
        pyvista.ImageData
            The output image.

        """
        self.Update()
        return cast('ImageData', wrap(self.GetOutput()))


class ImageMandelbrotSource(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkImageMandelbrotSource):
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
    >>> source.output.plot(cpos='xy')

    """

    def __init__(
        self, whole_extent: VectorLike[int] | None = None, maxiter: int | None = None
    ) -> None:
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
    def whole_extent(self, whole_extent: VectorLike[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        whole_extent : sequence[int]
            The extent of the whole output image.

        """
        self.SetWholeExtent(whole_extent)  # type: ignore[call-overload]

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
    def output(self) -> ImageData:
        """Get the output image as a ImageData.

        Returns
        -------
        pyvista.ImageData
            The output image.

        """
        self.Update()
        return cast('ImageData', wrap(self.GetOutput()))


class ImageNoiseSource(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkImageNoiseSource):
    """Create an image filled with uniform noise.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    whole_extent : sequence[int]
        The extent of the whole output image.

    minimum : float
        The minimum value for the generated noise.

    maximum : float
        The maximum value for the generated noise.

    seed : int, optional
        Seed the random number generator with a value.

    Examples
    --------
    Create an image of noise.

    >>> import pyvista as pv
    >>> source = pv.ImageNoiseSource(
    ...     whole_extent=(0, 200, 0, 200, 0, 0),
    ...     minimum=0,
    ...     maximum=255,
    ...     seed=0,
    ... )
    >>> source.output.plot(cpos='xy')

    """

    def __init__(
        self,
        *,
        whole_extent: VectorLike[int] | None = (0, 255, 0, 255, 0, 0),
        minimum: float | None = 0.0,
        maximum: float | None = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if whole_extent is not None:
            self.whole_extent = whole_extent
        if minimum is not None:
            self.minimum = minimum
        if maximum is not None:
            self.maximum = maximum
        if seed is not None:
            self.seed(seed)

    @property
    def whole_extent(self) -> VectorLike[int]:
        """Get extent of the whole output image.

        Returns
        -------
        sequence[int]
          The extent of the whole output image.

        """
        self.UpdateInformation()
        return self.GetOutputInformation(0).Get(
            _vtk.vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT()
        )

    @whole_extent.setter
    def whole_extent(self, whole_extent: VectorLike[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        whole_extent : sequence[int]
          The extent of the whole output image.

        """
        self.SetWholeExtent(whole_extent)  # type: ignore[arg-type]

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

    def seed(self, value: int) -> None:
        """Seed the random number generator with a value.

        Parameters
        ----------
        value : int
          The seed value for the random number generator to use.

        """
        _vtk.vtkMath().RandomSeed(value)

    @property
    def output(self) -> ImageData:
        """Get the output image as a ImageData.

        Returns
        -------
        pyvista.ImageData
          The output image.

        """
        self.Update()
        return cast('ImageData', wrap(self.GetOutput()))


class ImageSinusoidSource(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkImageSinusoidSource):
    """Create an image of a sinusoid.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    whole_extent : sequence[int]
        The extent of the whole output image.

    direction : tuple
        The direction vector which determines the sinusoidal orientation.

    period : float
        The period of the sinusoid in pixel.

    phase : float
        The phase of the sinusoid in pixel.

    amplitude : float
        The magnitude of the sinusoid.

    Examples
    --------
    Create an image of a sinusoid.

    >>> import pyvista as pv
    >>> source = pv.ImageSinusoidSource(
    ...     whole_extent=(0, 200, 0, 200, 0, 0),
    ...     period=20.0,
    ...     phase=0.0,
    ...     amplitude=255,
    ...     direction=(1.0, 0.0, 0.0),
    ... )
    >>> source.output.plot(cpos='xy')

    """

    def __init__(
        self,
        *,
        whole_extent: VectorLike[int] | None = None,
        direction: VectorLike[float] | None = None,
        period: float | None = None,
        phase: float | None = None,
        amplitude: float | None = None,
    ) -> None:
        super().__init__()
        if whole_extent is not None:
            self.whole_extent = whole_extent
        if direction is not None:
            self.direction = direction
        if period is not None:
            self.period = period
        if phase is not None:
            self.phase = phase
        if amplitude is not None:
            self.amplitude = amplitude

    @property
    def whole_extent(self) -> VectorLike[int]:
        """Get extent of the whole output image.

        Returns
        -------
        sequence[int]
            The extent of the whole output image.

        """
        self.UpdateInformation()
        return self.GetOutputInformation(0).Get(
            _vtk.vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT()
        )

    @whole_extent.setter
    def whole_extent(self, whole_extent: VectorLike[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        whole_extent : sequence[int]
            The extent of the whole output image.

        """
        self.SetWholeExtent(
            whole_extent[0],  # type: ignore[arg-type]
            whole_extent[1],  # type: ignore[arg-type]
            whole_extent[2],  # type: ignore[arg-type]
            whole_extent[3],  # type: ignore[arg-type]
            whole_extent[4],  # type: ignore[arg-type]
            whole_extent[5],  # type: ignore[arg-type]
        )

    @property
    def direction(self) -> Sequence[float]:
        """Get the direction of the sinusoid.

        Returns
        -------
        sequence[float]
            The direction of the sinusoid.

        """
        return self.GetDirection()

    @direction.setter
    def direction(self, direction: VectorLike[float]) -> None:
        """Set the direction of the sinusoid.

        Parameters
        ----------
        direction : sequence[float]
            The direction of the sinusoid.

        """
        self.SetDirection(direction)  # type: ignore[call-overload]

    @property
    def period(self) -> float:
        """Get the period of the sinusoid.

        Returns
        -------
        float
            The period of the sinusoid in pixel.

        """
        return self.GetPeriod()

    @period.setter
    def period(self, period: float) -> None:
        """Set the period of the sinusoid.

        Parameters
        ----------
        period : float
            The period of the sinusoid in pixel.

        """
        self.SetPeriod(period)

    @property
    def phase(self) -> float:
        """Get the phase of the sinusoid.

        Returns
        -------
        float
            The phase of the sinusoid in pixel.

        """
        return self.GetPhase()

    @phase.setter
    def phase(self, phase: float) -> None:
        """Set the phase of the sinusoid.

        Parameters
        ----------
        phase : float
            The phase of the sinusoid in pixel.

        """
        self.SetPhase(phase)

    @property
    def amplitude(self) -> float:
        """Get the magnitude of the sinusoid.

        Returns
        -------
        float
            The magnitude of the sinusoid.

        """
        return self.GetAmplitude()

    @amplitude.setter
    def amplitude(self, amplitude: float) -> None:
        """Set the magnitude of the sinusoid.

        Parameters
        ----------
        amplitude : float
            The magnitude of the sinusoid.

        """
        self.SetAmplitude(amplitude)

    @property
    def output(self) -> ImageData:
        """Get the output image as a ImageData.

        Returns
        -------
        pyvista.ImageData
            The output image.

        """
        self.Update()
        return cast('ImageData', wrap(self.GetOutput()))


class ImageGaussianSource(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkImageGaussianSource):
    """Create a binary image with Gaussian pixel values.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    center : sequence[float]
        The center of the Gaussian.

    whole_extent : sequence[int]
        The extent of the whole output image.

    maximum : float
        The maximum value of the Gaussian.

    std : float
        The standard deviation of the Gaussian.

    Examples
    --------
    Create an image of Gaussian pixel values.

    >>> import pyvista as pv
    >>> source = pv.ImageGaussianSource(
    ...     center=(100, 100, 0),
    ...     whole_extent=(0, 200, 0, 200, 0, 0),
    ...     maximum=255,
    ...     std=100.0,
    ... )
    >>> source.output.plot(cpos='xy')

    """

    def __init__(
        self,
        *,
        center: VectorLike[float] | None = None,
        whole_extent: VectorLike[int] | None = None,
        maximum: float | None = None,
        std: float | None = None,
    ) -> None:
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
    def center(self) -> tuple[float, float, float]:
        """Get the center of the Gaussian.

        Returns
        -------
        tuple[float, float, float]
          The center of the Gaussian.

        """
        return self.GetCenter()

    @center.setter
    def center(self, center: VectorLike[float]) -> None:
        """Set the center of the Gaussian.

        Parameters
        ----------
        center : sequence[float]
          The center of the Gaussian.

        """
        self.SetCenter(center)  # type: ignore[arg-type]

    @property
    def whole_extent(self) -> VectorLike[int]:
        """Get extent of the whole output image.

        Returns
        -------
        sequence[int]
          The extent of the whole output image.

        """
        self.UpdateInformation()
        return self.GetOutputInformation(0).Get(
            _vtk.vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT()
        )

    @whole_extent.setter
    def whole_extent(self, whole_extent: VectorLike[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        whole_extent : sequence[int]
          The extent of the whole output image.

        """
        self.SetWholeExtent(
            whole_extent[0],  # type: ignore[arg-type]
            whole_extent[1],  # type: ignore[arg-type]
            whole_extent[2],  # type: ignore[arg-type]
            whole_extent[3],  # type: ignore[arg-type]
            whole_extent[4],  # type: ignore[arg-type]
            whole_extent[5],  # type: ignore[arg-type]
        )

    @property
    def maximum(self) -> float:
        """Get the maximum value of the Gaussian.

        Returns
        -------
        float
          The maximum value of the Gaussian.

        """
        return self.GetMaximum()

    @maximum.setter
    def maximum(self, maximum: float) -> None:
        """Set the maximum value of the Gaussian.

        Parameters
        ----------
        maximum : float
          The maximum value of the Gaussian.

        """
        self.SetMaximum(maximum)

    @property
    def std(self) -> float:
        """Get the standard deviation of the Gaussian.

        Returns
        -------
        float
          The standard deviation of the Gaussian.

        """
        return self.GetStandardDeviation()

    @std.setter
    def std(self, std: float) -> None:
        """Set the standard deviation of the Gaussian.

        Parameters
        ----------
        std : float
          The standard deviation of the Gaussian.

        """
        self.SetStandardDeviation(std)

    @property
    def output(self) -> ImageData:
        """Get the output image as a ImageData.

        Returns
        -------
        pyvista.ImageData
          The output image.

        """
        self.Update()
        return cast('ImageData', wrap(self.GetOutput()))


class ImageGridSource(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkImageGridSource):
    """Create an image of a grid.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    origin : sequence[int]
        The origin of the grid.

    extent : sequence[int]
        The extent of the whole output image, Default: (0,255,0,255,0,0).

    spacing : tuple
        The pixel spacing.

    Examples
    --------
    Create an image of a grid.

    >>> import pyvista as pv
    >>> source = pv.ImageGridSource(
    ...     extent=(0, 20, 0, 20, 0, 0),
    ...     spacing=(1, 1, 1),
    ... )
    >>> source.output.plot(cpos='xy')

    """

    def __init__(
        self,
        origin: Sequence[int] | None = None,
        extent: VectorLike[int] | None = None,
        spacing: VectorLike[float] | None = None,
    ) -> None:
        super().__init__()
        if origin is not None:
            self.origin = origin
        if extent is not None:
            self.extent = extent
        if spacing is not None:
            self.spacing = spacing

    @property
    def origin(self) -> Sequence[int]:
        """Get the origin of the data.

        Returns
        -------
        sequence[int]
            The origin of the grid.

        """
        return self.GetGridOrigin()

    @origin.setter
    def origin(self, origin: Sequence[int]) -> None:
        """Set the origin of the data.

        Parameters
        ----------
        origin : sequence[int]
            The origin of the grid.

        """
        self.SetGridOrigin(origin)

    @property
    def extent(self) -> Sequence[int]:
        """Get extent of the whole output image.

        Returns
        -------
        sequence[int]
            The extent of the whole output image.

        """
        return self.GetDataExtent()

    @extent.setter
    def extent(self, extent: VectorLike[int]) -> None:
        """Set extent of the whole output image.

        Parameters
        ----------
        extent : sequence[int]
            The extent of the whole output image.

        """
        self.SetDataExtent(extent)  # type: ignore[arg-type]

    @property
    def spacing(self) -> Sequence[float]:
        """Get the spacing of the grid.

        Returns
        -------
        sequence[float]
            The pixel spacing.

        """
        return self.GetDataSpacing()

    @spacing.setter
    def spacing(self, spacing: VectorLike[float]) -> None:
        """Set the spacing of the grid.

        Parameters
        ----------
        spacing : sequence[float]
            The pixel spacing.

        """
        self.SetDataSpacing(spacing)  # type: ignore[arg-type]

    @property
    def output(self) -> ImageData:
        """Get the output image as a ImageData.

        Returns
        -------
        pyvista.ImageData
            The output image.

        """
        self.Update()
        return cast('ImageData', wrap(self.GetOutput()))
