from __future__ import annotations

import pytest

import pyvista as pv


@pytest.mark.parametrize(
    ('class_name', 'kwargs'),
    [
        ('Plotter', None),  # Also covers Renderer
        ('MultiBlock', None),
        ('Actor', None),  # Test Prop3D
        ('AxesAssembly', None),  # Test Prop3DMixin
        ('Label', None),
        ('Cell', None),
        ('CubeAxesActor', dict(camera=pv.Camera())),
        ('BoxSource', None),
        ('CubeSource', None),
        ('FixedPointVolumeRayCastMapper', None),  # Test _BaseMapper
        ('PolyData', None),  # Test DataSet
    ],
)
def test_bounds_tuple(class_name, kwargs):
    kwargs = kwargs if kwargs else {}
    instance = getattr(pv, class_name)(**kwargs)
    bounds = instance.bounds
    assert len(bounds) == 6
    assert isinstance(bounds, pv.BoundsTuple)
    assert all(isinstance(bnd, float) for bnd in bounds)
