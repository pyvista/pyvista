import numpy as np

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


def test_ParametricBohemianDome():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricBour():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricBoy():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricCatalanMinimal():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricConicSpiral():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricCrossCap():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricDini():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricEllipsoid():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricEnneper():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricFigure8Klein():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricHenneberg():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricKlein():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricKuen():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricMobius():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricPluckerConoid():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricPseudosphere():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricRandomHills():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricRoman():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricSuperEllipsoid():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricSuperToroid():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points


def test_ParametricTorus():
    geom = pyvista.ParametricBohemianDome()
    assert geom.n_points
