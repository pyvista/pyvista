import pytest

import pyvista as pv
from pyvista.plotting.mapper import PointGaussianMapper


@pytest.fixture()
def mapper(sphere):
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


def test_use_circular_splat(mapper):
    mapper.use_circular_splat()
    assert 'offsetVCVSOutput' in mapper.GetSplatShaderCode()

    mapper.use_default_splat()
    assert mapper.GetSplatShaderCode() is None
