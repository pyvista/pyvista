from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting.mapper import PointGaussianMapper


@pytest.fixture
def mapper(sphere):
    sphere['array'] = sphere.points[:, 2]
    pl = pv.Plotter()
    actor = pl.add_points(sphere, style='points_gaussian')
    return actor.mapper


def test_mapper_init(mapper):
    assert isinstance(mapper, PointGaussianMapper)
    assert 'False' in repr(mapper)
    assert 'Emissive' in repr(mapper)


def test_emissive(mapper):
    assert isinstance(mapper.emissive, bool)

    emissive = True
    mapper.emissive = emissive
    assert mapper.emissive == emissive


def test_scale_factor(mapper):
    assert isinstance(mapper.scale_factor, float)

    scale_factor = 2.0
    mapper.scale_factor = scale_factor
    assert mapper.scale_factor == scale_factor


def test_scale_array(mapper):
    assert mapper.scale_array is None

    with pytest.raises(KeyError, match='does not exist'):
        mapper.scale_array = 'foo'

    scale_array = 'array'
    mapper.scale_array = scale_array
    assert mapper.scale_array == scale_array


def test_use_circular_splat(mapper):
    mapper.use_circular_splat()
    assert 'offsetVCVSOutput' in mapper.GetSplatShaderCode()

    mapper.use_default_splat()
    assert mapper.GetSplatShaderCode() is None


@pytest.mark.parametrize('scalars', [None, 'radius'])
def test_scalar_range(scalars):
    """The scalar bar of a points_gaussian actor spans its data, as for other styles."""
    rng = np.random.default_rng(seed=0)
    points = pv.PolyData(rng.random((100, 3)))
    points['radius'] = rng.random(100) * 0.02
    pl = pv.Plotter()
    actor = pl.add_mesh(points, scalars=scalars, style='points_gaussian')
    expected = points.get_data_range('radius')
    assert actor.mapper.scalar_range == pytest.approx(expected)
    assert pl.scalar_bar.GetLookupTable().GetRange() == pytest.approx(expected)


def test_scalar_range_clim():
    """An explicit ``clim`` survives the conversion to RGBA."""
    points = pv.PolyData(np.random.default_rng(seed=0).random((100, 3)))
    points['radius'] = np.linspace(0, 0.02, 100)
    pl = pv.Plotter()
    actor = pl.add_mesh(points, style='points_gaussian', clim=[0, 0.05])
    assert actor.mapper.scalar_range == pytest.approx((0, 0.05))
