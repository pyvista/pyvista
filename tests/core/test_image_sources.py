from __future__ import annotations

import numpy as np

import pyvista as pv


def test_image_ellip_soid_source():
    whole_extent = (-10, 10, -10, 10, -10, 10)
    center = (0.0, 0.0, 0.0)
    radius = (5.0, 5.0, 5.0)
    source = pv.ImageEllipsoidSource(whole_extent=whole_extent, center=center, radius=radius)
    assert source.whole_extent == whole_extent
    assert source.center == center
    assert source.radius == radius
    whole_extent = (-5, 5, -5, 5, -5, 5)
    center = (1.0, 1.0, 1.0)
    radius = (3.0, 3.0, 3.0)
    source.whole_extent = whole_extent
    source.center = center
    source.radius = radius
    assert source.whole_extent == whole_extent
    assert source.center == center
    assert source.radius == radius
    assert isinstance(source.output, pv.ImageData)


def test_image_noise_source():
    whole_extent = (-10, 10, -10, 10, -10, 10)
    source = pv.ImageNoiseSource(whole_extent=whole_extent, minimum=0, maximum=255, seed=42)
    assert source.whole_extent == whole_extent
    assert source.minimum == 0
    assert source.maximum == 255
    whole_extent = (-5, 5, -5, 5, -5, 5)
    minimum = 100
    maximum = 200
    source.whole_extent = whole_extent
    source.minimum = minimum
    source.maximum = maximum
    assert source.whole_extent == whole_extent
    assert source.minimum == minimum
    assert source.maximum == maximum
    assert isinstance(source.output, pv.ImageData)

    output = pv.ImageNoiseSource().output
    assert output.bounds == (0.0, 255.0, 0.0, 255.0, 0.0, 0.0)
    assert output.dimensions == (256, 256, 1)
    assert np.allclose(output.get_data_range(), (0.0, 1.0), atol=1e-4)

    output_seed = pv.ImageNoiseSource(seed=0).output
    assert not np.array_equal(output_seed.active_scalars, output.active_scalars)

    output_same_seed = pv.ImageNoiseSource(seed=0).output
    assert np.array_equal(output_same_seed.active_scalars, output_seed.active_scalars)


def test_image_mandelbrot_source():
    whole_extent = (0, 20, 0, 20, 0, 0)
    maxiter = 10
    source = pv.ImageMandelbrotSource(
        whole_extent=whole_extent,
        maxiter=maxiter,
    )
    assert source.whole_extent == whole_extent
    assert source.maxiter == maxiter
    whole_extent = (0, 200, 0, 200, 0, 0)
    maxiter = 100
    source.whole_extent = whole_extent
    source.maxiter = maxiter
    assert source.whole_extent == whole_extent
    assert source.maxiter == maxiter
    assert isinstance(source.output, pv.ImageData)


def test_image_gradient_source():
    center = (0.0, 0.0, 0.0)
    whole_extent = (-10, 10, -10, 10, -10, 10)
    maximum = 255
    std = 10.0
    source = pv.ImageGaussianSource(
        center=center,
        whole_extent=whole_extent,
        maximum=maximum,
        std=std,
    )
    assert source.center == center
    assert source.whole_extent == whole_extent
    assert source.maximum == maximum
    assert source.std == std
    center = (5.0, 0.0, 0.0)
    whole_extent = (-20, 20, -20, 20, -20, 20)
    maximum = 100
    std = 20.0
    source.center = center
    source.whole_extent = whole_extent
    source.maximum = maximum
    source.std = std
    assert source.center == center
    assert source.whole_extent == whole_extent
    assert source.maximum == maximum
    assert source.std == std
    assert isinstance(source.output, pv.ImageData)


def test_image_sinusolid_source():
    whole_extent = (0, 20, 0, 20, 0, 0)
    period = 20.0
    phase = 0.0
    amplitude = 255
    direction = (1.0, 0.0, 0.0)
    source = pv.ImageSinusoidSource(
        whole_extent=whole_extent,
        period=period,
        phase=phase,
        amplitude=amplitude,
        direction=direction,
    )
    assert source.whole_extent == whole_extent
    assert source.period == period
    assert source.phase == phase
    assert source.amplitude == amplitude
    assert source.direction == direction
    whole_extent = (0, 200, 0, 200, 0, 0)
    period = 200.0
    phase = 0.0
    amplitude = 100
    direction = (0.0, 1.0, 0.0)
    source.whole_extent = whole_extent
    source.period = period
    source.phase = phase
    source.amplitude = amplitude
    source.direction = direction
    assert source.whole_extent == whole_extent
    assert source.period == period
    assert source.phase == phase
    assert source.amplitude == amplitude
    assert source.direction == direction
    assert isinstance(source.output, pv.ImageData)


def test_image_grid_source():
    origin = (-10, -10, -10)
    extent = (-10, 10, -10, 10, -10, 10)
    spacing = (1, 1, 1)
    source = pv.ImageGridSource(origin=origin, extent=extent, spacing=spacing)
    assert source.origin == origin
    assert source.extent == extent
    assert source.spacing == spacing
    origin = (-5, -5, -5)
    extent = (-5, 5, -5, 5, -5, 5)
    spacing = (0.5, 0.5, 0.5)
    source.origin = origin
    source.extent = extent
    source.spacing = spacing
    assert source.origin == origin
    assert source.extent == extent
    assert source.spacing == spacing
    assert isinstance(source.output, pv.ImageData)
