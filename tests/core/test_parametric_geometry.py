import numpy as np
import pytest

import pyvista


def test_spline():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    points = np.column_stack((x, y, z))
    spline = pyvista.Spline(points, 1000)
    assert spline.n_points == 1000


def test_kochanek_spline():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    tension = np.random.random(3)
    bias = np.random.random(3)
    continuity = np.random.random(3)

    n_points = 1000
    points = np.column_stack((x, y, z))
    kochanek_spline = pyvista.KochanekSpline(points, tension, bias, continuity, n_points)
    assert kochanek_spline.n_points == n_points

    # test default
    kochanek_spline = pyvista.KochanekSpline(points)
    assert kochanek_spline.n_points == points.shape[0]

    # test invalid
    with pytest.raises(ValueError, match='tension'):
        kochanek_spline = pyvista.KochanekSpline(points, [-2, 0, 0], bias, continuity, n_points)
    with pytest.raises(ValueError, match='bias'):
        kochanek_spline = pyvista.KochanekSpline(points, tension, [-2, 0, 0], continuity, n_points)
    with pytest.raises(ValueError, match='continuity'):
        kochanek_spline = pyvista.KochanekSpline(points, tension, bias, [-2, 0, 0], n_points)


def test_ParametricBohemianDome():
    geom = pyvista.ParametricBohemianDome(direction=[0, 0, 1], a=0.5, b=1.5, c=1.0)
    assert geom.n_points


def test_ParametricBour():
    geom = pyvista.ParametricBour()
    assert geom.n_points


def test_ParametricBoy():
    geom = pyvista.ParametricBoy()
    assert geom.n_points


def test_ParametricCatalanMinimal():
    geom = pyvista.ParametricCatalanMinimal()
    assert geom.n_points


def test_ParametricConicSpiral():
    geom = pyvista.ParametricConicSpiral()
    assert geom.n_points


def test_ParametricCrossCap():
    geom = pyvista.ParametricCrossCap()
    assert geom.n_points


def test_ParametricDini():
    geom = pyvista.ParametricDini()
    assert geom.n_points


def test_ParametricEllipsoid():
    geom = pyvista.ParametricEllipsoid()
    assert geom.n_points


def test_ParametricEnneper():
    geom = pyvista.ParametricEnneper()
    assert geom.n_points


def test_ParametricFigure8Klein():
    geom = pyvista.ParametricFigure8Klein()
    assert geom.n_points


def test_ParametricHenneberg():
    geom = pyvista.ParametricHenneberg()
    assert geom.n_points


def test_ParametricKlein():
    geom = pyvista.ParametricKlein()
    assert geom.n_points


def test_ParametricKuen():
    geom = pyvista.ParametricKuen()
    assert geom.n_points


def test_ParametricMobius():
    geom = pyvista.ParametricMobius()
    assert geom.n_points


def test_ParametricPluckerConoid():
    geom = pyvista.ParametricPluckerConoid()
    assert geom.n_points


def test_ParametricPseudosphere():
    geom = pyvista.ParametricPseudosphere()
    assert geom.n_points


def test_ParametricRandomHills():
    geom = pyvista.ParametricRandomHills()
    assert geom.n_points


def test_ParametricRoman():
    geom = pyvista.ParametricRoman()
    assert geom.n_points


def test_ParametricSuperEllipsoid():
    geom = pyvista.ParametricSuperEllipsoid()
    assert geom.n_points


def test_ParametricSuperToroid():
    geom = pyvista.ParametricSuperToroid()
    assert geom.n_points


def test_ParametricTorus():
    geom = pyvista.ParametricTorus()
    assert geom.n_points
