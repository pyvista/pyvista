from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.core import _vtk_core as _vtk


@pytest.fixture
def points():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.column_stack((x, y, z))


def test_spline_n_points(points):
    spline = pv.Spline(points)
    assert spline.n_points == points.shape[0]

    n_points = 1000
    spline = pv.Spline(points, n_points)
    assert spline.n_points == n_points


def test_spline_parametrize_by(points):
    n_points = 1000
    spline_by_length = pv.Spline(points, n_points, parametrize_by='length')
    spline_by_index = pv.Spline(points, n_points, parametrize_by='index')
    assert spline_by_length.n_points == n_points
    assert spline_by_index.n_points == n_points

    # Test that it produces different points
    assert not np.allclose(spline_by_index.points, spline_by_length.points)


def test_spline_closed(points):
    n_points = 1000
    spline = pv.Spline(points, n_points)
    spline_closed = pv.Spline(points, n_points, closed=True)
    assert not np.allclose(spline.bounds, spline_closed.bounds)


@pytest.mark.parametrize(
    ('boundary_constraint', 'boundary_value'),
    [('finite_difference', None), ('clamped', 0.0), ('second', 0.0), ('scaled_second', 0.0)],
)
def test_boundary_constraints(points, boundary_constraint, boundary_value):
    default_constraint = 'clamped'
    default_value = 0.0
    n_points = 1000
    spline = pv.Spline(points, n_points)

    # Test that different splines are produced
    spline_boundary_left = pv.Spline(
        points,
        n_points,
        boundary_constraints=(boundary_constraint, default_constraint),
        boundary_values=(boundary_value, default_value),
    )
    spline_boundary_right = pv.Spline(
        points,
        n_points,
        boundary_constraints=(default_constraint, boundary_constraint),
        boundary_values=(default_value, boundary_value),
    )

    is_default = boundary_constraint == default_constraint
    left_points = spline_boundary_left.points
    right_points = spline_boundary_right.points
    assert np.allclose(spline.points, left_points) == is_default
    assert np.allclose(spline.points, right_points) == is_default
    assert np.allclose(left_points, right_points) == is_default

    if boundary_constraint == 'finite_difference':
        with pytest.raises(ValueError, match='finite difference boundary value must be None'):
            _ = pv.Spline(
                points,
                n_points,
                boundary_constraints=(boundary_constraint, default_constraint),
                boundary_values=(1.0, 0.0),
            )
    else:
        spline_boundary_left_val = pv.Spline(
            points,
            n_points,
            boundary_constraints=(boundary_constraint, default_constraint),
            boundary_values=(1.0, 0.0),
        )
        spline_boundary_right_val = pv.Spline(
            points,
            n_points,
            boundary_constraints=(default_constraint, boundary_constraint),
            boundary_values=(0.0, 1.0),
        )

        assert not np.allclose(spline_boundary_left_val.points, left_points)
        assert not np.allclose(spline_boundary_right_val.points, right_points)
        assert not np.allclose(spline_boundary_left_val.points, spline_boundary_right_val.points)


@pytest.mark.parametrize(
    ('boundary_constraint', 'boundary_value'),
    [('finite_difference', None), ('clamped', 0.0), ('second', 0.0), ('scaled_second', 0.0)],
)
def test_spline_boundary_single_input(points, boundary_constraint, boundary_value):
    n_points = 1000
    spline1 = pv.Spline(
        points, n_points, boundary_constraints=boundary_constraint, boundary_values=boundary_value
    )
    spline2 = pv.Spline(
        points,
        n_points,
        boundary_constraints=[boundary_constraint, boundary_constraint],
        boundary_values=[boundary_value, boundary_value],
    )
    assert spline1 == spline2


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
    kochanek_spline = pv.KochanekSpline(
        points, tension=tension, bias=bias, continuity=continuity, n_points=n_points
    )
    assert kochanek_spline.n_points == n_points

    # test default
    kochanek_spline = pv.KochanekSpline(points)
    assert kochanek_spline.n_points == points.shape[0]

    # test invalid
    with pytest.raises(ValueError, match='tension'):
        kochanek_spline = pv.KochanekSpline(
            points, tension=[-2, 0, 0], bias=bias, continuity=continuity, n_points=n_points
        )
    with pytest.raises(ValueError, match='bias'):
        kochanek_spline = pv.KochanekSpline(
            points, tension=tension, bias=[-2, 0, 0], continuity=continuity, n_points=n_points
        )
    with pytest.raises(ValueError, match='continuity'):
        kochanek_spline = pv.KochanekSpline(
            points, tension=tension, bias=bias, continuity=[-2, 0, 0], n_points=n_points
        )


def test_parametric_bohemian_dome():
    geom = pv.ParametricBohemianDome(direction=[0, 0, 1], a=0.5, b=1.5, c=1.0)
    assert geom.n_points


def test_parametric_bour():
    geom = pv.ParametricBour()
    assert geom.n_points


def test_parametric_boy():
    geom = pv.ParametricBoy()
    assert geom.n_points


def test_parametric_catalan_minimal():
    geom = pv.ParametricCatalanMinimal()
    assert geom.n_points


def test_parametric_conic_spiral():
    geom = pv.ParametricConicSpiral()
    assert geom.n_points


def test_parametric_cross_cap():
    geom = pv.ParametricCrossCap()
    assert geom.n_points


def test_parametric_dini():
    geom = pv.ParametricDini()
    assert geom.n_points


def test_parametric_ellipsoid():
    geom = pv.ParametricEllipsoid()
    assert geom.n_points


def test_parametric_enneper():
    geom = pv.ParametricEnneper()
    assert geom.n_points


def test_parametric_figure8_klein():
    geom = pv.ParametricFigure8Klein()
    assert geom.n_points


def test_parametric_henneberg():
    geom = pv.ParametricHenneberg()
    assert geom.n_points


def test_parametric_klein():
    geom = pv.ParametricKlein()
    assert geom.n_points


def test_parametric_kuen():
    geom = pv.ParametricKuen()
    assert geom.n_points


def test_parametric_mobius():
    geom = pv.ParametricMobius()
    assert geom.n_points


def test_parametric_plucker_conoid():
    geom = pv.ParametricPluckerConoid()
    assert geom.n_points


def test_parametric_pseudosphere():
    geom = pv.ParametricPseudosphere()
    assert geom.n_points


def test_parametric_random_hills():
    geom = pv.ParametricRandomHills()
    assert geom.n_points
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


def test_parametric_roman():
    geom = pv.ParametricRoman()
    assert geom.n_points


def test_parametric_super_ellipsoid():
    geom = pv.ParametricSuperEllipsoid()
    assert geom.n_points


def test_parametric_super_toroid():
    geom = pv.ParametricSuperToroid()
    assert geom.n_points


def test_parametric_torus():
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
    parametric_function = _vtk.vtkParametricBour()
    geom = pv.surface_from_para(parametric_function, texture_coordinates=False)
    assert geom.active_texture_coordinates is None
    geom = pv.surface_from_para(parametric_function, texture_coordinates=True)
    assert geom.active_texture_coordinates is not None
