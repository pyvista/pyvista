from __future__ import annotations

import numpy as np
import pytest
import vtk

import pyvista as pv


def test_spline():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack((x, y, z))
    # test default (copied from Kochanek spline)
    spline = pv.Spline(points)
    assert spline.n_points == points.shape[0]
    # test number of points
    points = np.column_stack((x, y, z))
    spline = pv.Spline(points, 1000)
    assert spline.n_points == 1000
    # test parametrize by length. Check that it produces different points
    # i.e. absolute distance max > 0
    # vtk default is to parametrize by length
    spline_by_length = pv.Spline(points, 1000, parameterize_by_length=True)
    spline_by_index = pv.Spline(points, 1000, parameterize_by_length=False)
    assert spline_by_length.n_points == spline_by_index.n_points == 1000
    points_abs_diff = np.abs(spline_by_index.points - spline_by_length.points)
    assert points_abs_diff.max() > 0
    # test closing, check if bounds are equal before and after closing
    # which is untrue on test case
    spline_closed = pv.Spline(points, 1000, closed=True)
    assert spline.bounds != spline_closed.bounds
    # test that boundary type / value change has an effect on produced splines
    for boundary_type in range(4):
        spline_boundary_left = pv.Spline(points, 1000, left_constraint_type=boundary_type)
        spline_boundary_right = pv.Spline(points, 1000, right_constraint_type=boundary_type)
        points_abs_diff_left = np.abs(spline.points - spline_boundary_left.points)
        points_abs_diff_right = np.abs(spline.points - spline_boundary_right.points)
        # based on output, it looks like vtk default is 1
        # because it produces an absolute difference in points array of 0.0.
        if boundary_type != 1:
            assert points_abs_diff_left.max() > 0
            assert points_abs_diff_right.max() > 0
        # vtk doc says: "The value is used only if the left(right) constraint is type 1-3."
        if boundary_type != 0:
            spline_boundary_left_val = pv.Spline(
                points, 1000, left_constraint_type=boundary_type, left_derivative_value=1.0
            )
            spline_boundary_right_val = pv.Spline(
                points, 1000, right_constraint_type=boundary_type, right_derivative_value=1.0
            )

            points_abs_diff_left_val = np.abs(
                spline_boundary_left_val.points - spline_boundary_left.points
            )
            points_abs_diff_right_val = np.abs(
                spline_boundary_right_val.points - spline_boundary_right.points
            )
            assert points_abs_diff_left_val.max() > 0
            assert points_abs_diff_right_val.max() > 0


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
    parametric_function = vtk.vtkParametricBour()
    geom = pv.surface_from_para(parametric_function, texture_coordinates=False)
    assert geom.active_texture_coordinates is None
    geom = pv.surface_from_para(parametric_function, texture_coordinates=True)
    assert geom.active_texture_coordinates is not None
