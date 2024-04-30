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
