from __future__ import annotations

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning


def test_spline():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    points = np.column_stack((x, y, z))
    spline = pv.Spline(points, 1000)
    assert spline.n_points == 1000


def test_kochanek_spline():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    tension = np.random.default_rng().random(3)
    bias = np.random.default_rng().random(3)
    continuity = np.random.default_rng().random(3)

    n_points = 1000
    points = np.column_stack((x, y, z))
    kochanek_spline = pv.KochanekSpline(points, tension, bias, continuity, n_points)
    assert kochanek_spline.n_points == n_points

    # test default
    kochanek_spline = pv.KochanekSpline(points)
    assert kochanek_spline.n_points == points.shape[0]

    # test invalid
    with pytest.raises(ValueError, match='tension'):
        kochanek_spline = pv.KochanekSpline(points, [-2, 0, 0], bias, continuity, n_points)
    with pytest.raises(ValueError, match='bias'):
        kochanek_spline = pv.KochanekSpline(points, tension, [-2, 0, 0], continuity, n_points)
    with pytest.raises(ValueError, match='continuity'):
        kochanek_spline = pv.KochanekSpline(points, tension, bias, [-2, 0, 0], n_points)


def test_ParametricBohemianDome():
    geom = pv.ParametricBohemianDome(direction=[0, 0, 1], a=0.5, b=1.5, c=1.0)
    assert geom.n_points


def test_ParametricBour():
    geom = pv.ParametricBour()
    assert geom.n_points


def test_ParametricBoy():
    geom = pv.ParametricBoy()
    assert geom.n_points


def test_ParametricCatalanMinimal():
    geom = pv.ParametricCatalanMinimal()
    assert geom.n_points


def test_ParametricConicSpiral():
    geom = pv.ParametricConicSpiral()
    assert geom.n_points


def test_ParametricCrossCap():
    geom = pv.ParametricCrossCap()
    assert geom.n_points


def test_ParametricDini():
    geom = pv.ParametricDini()
    assert geom.n_points


def test_ParametricEllipsoid():
    geom = pv.ParametricEllipsoid()
    assert geom.n_points


def test_ParametricEnneper():
    geom = pv.ParametricEnneper()
    assert geom.n_points


def test_ParametricFigure8Klein():
    geom = pv.ParametricFigure8Klein()
    assert geom.n_points


def test_ParametricHenneberg():
    geom = pv.ParametricHenneberg()
    assert geom.n_points


def test_ParametricKlein():
    geom = pv.ParametricKlein()
    assert geom.n_points


def test_ParametricKuen():
    geom = pv.ParametricKuen()
    assert geom.n_points


def test_ParametricMobius():
    geom = pv.ParametricMobius()
    assert geom.n_points


def test_ParametricPluckerConoid():
    geom = pv.ParametricPluckerConoid()
    assert geom.n_points


def test_ParametricPseudosphere():
    geom = pv.ParametricPseudosphere()
    assert geom.n_points


def test_ParametricRandomHills():
    geom = pv.ParametricRandomHills()
    assert geom.n_points
    with pytest.warns(PyVistaDeprecationWarning):
        geom = pv.ParametricRandomHills(numberofhills=30)
        if pv._version.version_info >= (0, 46):
            raise RuntimeError('Remove this deprecated parameter')
    with pytest.warns(PyVistaDeprecationWarning):
        geom = pv.ParametricRandomHills(hillxvariance=30)
        if pv._version.version_info >= (0, 46):
            raise RuntimeError('Remove this deprecated parameter')
    with pytest.warns(PyVistaDeprecationWarning):
        geom = pv.ParametricRandomHills(hillyvariance=2.5)
        if pv._version.version_info >= (0, 46):
            raise RuntimeError('Remove this deprecated parameter')
    with pytest.warns(PyVistaDeprecationWarning):
        geom = pv.ParametricRandomHills(hillamplitude=2.5)
        if pv._version.version_info >= (0, 46):
            raise RuntimeError('Remove this deprecated parameter')
    with pytest.warns(PyVistaDeprecationWarning):
        geom = pv.ParametricRandomHills(randomseed=1)
        if pv._version.version_info >= (0, 46):
            raise RuntimeError('Remove this deprecated parameter')
    with pytest.warns(PyVistaDeprecationWarning):
        geom = pv.ParametricRandomHills(xvariancescalefactor=13)
        if pv._version.version_info >= (0, 46):
            raise RuntimeError('Remove this deprecated parameter')
    with pytest.warns(PyVistaDeprecationWarning):
        geom = pv.ParametricRandomHills(yvariancescalefactor=13)
        if pv._version.version_info >= (0, 46):
            raise RuntimeError('Remove this deprecated parameter')
    with pytest.warns(PyVistaDeprecationWarning):
        geom = pv.ParametricRandomHills(amplitudescalefactor=13)
        if pv._version.version_info >= (0, 46):
            raise RuntimeError('Remove this deprecated parameter')
    geom = pv.ParametricRandomHills(
        number_of_hills=30,
        hill_x_variance=30,
        hill_y_variance=2.5,
        hill_amplitude=2.5,
        random_seed=1,
        x_variance_scale_factor=13,
        y_variance_scale_factor=13,
        amplitude_scale_factor=13,
    )
    assert geom.n_points


def test_ParametricRoman():
    geom = pv.ParametricRoman()
    assert geom.n_points


def test_ParametricSuperEllipsoid():
    geom = pv.ParametricSuperEllipsoid()
    assert geom.n_points


def test_ParametricSuperToroid():
    geom = pv.ParametricSuperToroid()
    assert geom.n_points


def test_ParametricTorus():
    geom = pv.ParametricTorus()
    assert geom.n_points


def test_direction():
    geom1 = pv.ParametricEllipsoid(300, 100, 10, direction=[1, 0, 0])
    geom2 = pv.ParametricEllipsoid(300, 100, 10, direction=[0, 1, 0])
    geom3 = pv.ParametricEllipsoid(300, 100, 10, direction=[0, -1, 0])
    assert geom1.n_points
    assert geom2.n_points
    assert geom3.n_points
    points1 = geom1.points
    points2 = geom2.points
    points3 = geom3.points

    assert np.allclose(points1[:, 0], points2[:, 1])
    assert np.allclose(points1[:, 1], -points2[:, 0])
    assert np.allclose(points1[:, 2], points2[:, 2])

    assert np.allclose(points1[:, 0], -points3[:, 1])
    assert np.allclose(points1[:, 1], points3[:, 0])
    assert np.allclose(points1[:, 2], points3[:, 2])


def test_surface_from_para():
    parametric_function = vtk.vtkParametricBour()
    geom = pv.surface_from_para(parametric_function, texture_coordinates=False)
    assert geom.active_texture_coordinates is None
    geom = pv.surface_from_para(parametric_function, texture_coordinates=True)
    assert geom.active_texture_coordinates is not None
