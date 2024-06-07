from __future__ import annotations

import pytest

import pyvista as pv
from pyvista.core.errors import MissingDataError


def test_contour_banded_raise(sphere):
    sphere.clear_data()

    with pytest.raises(MissingDataError):
        sphere.contour_banded(5)

    sphere['data'] = sphere.points[:, 2]
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.contour_banded(5, scalar_mode='foo')

    sphere.clear_data()
    sphere['data'] = range(sphere.n_cells)
    with pytest.raises(MissingDataError):
        _ = sphere.contour_banded(10)


def test_contour_banded_points(sphere):
    sphere.clear_data()
    sphere['data'] = sphere.points[:, 2]
    out, edges = sphere.contour_banded(10)
    assert out.n_cells
    assert edges.n_cells
    assert 'data' in out.point_data

    out = sphere.contour_banded(10, generate_contour_edges=False)
    assert out.n_cells

    rng = [-100, 100]
    out = sphere.contour_banded(
        10,
        rng=rng,
        generate_contour_edges=False,
        scalar_mode='index',
        clipping=True,
    )
    assert out['data'].min() <= rng[0]
    assert out['data'].max() >= rng[1]


def test_boolean_intersect_edge_case():
    a = pv.Cube(x_length=2, y_length=2, z_length=2).triangulate()
    b = pv.Cube().triangulate()  # smaller cube (x_length=1)

    with pytest.warns(UserWarning, match='contained within another'):
        a.boolean_intersection(b)


def test_identical_boolean(sphere):
    with pytest.raises(ValueError, match='identical points'):
        sphere.boolean_intersection(sphere.copy())


@pytest.fixture()
def poly_circle():
    circle = pv.Circle(resolution=30)
    return pv.PolyData(circle.points, lines=[31, *list(range(30)), 0])


def test_decimate_polylines(poly_circle):
    assert poly_circle.n_points == 30
    decimated = poly_circle.decimate_polylines(0.5)
    # Allow some leeway for approximtely 50%
    assert decimated.n_points >= 14
    assert decimated.n_points <= 16

def test_decimate_polylines_maximum_error(poly_circle):
    assert poly_circle.n_points == 30
    # low maximum error will prevent decimation.
    # Since this is a regular shape, no decimation occurs at all with suitable choice
    decimated = poly_circle.decimate_polylines(0.5, maximum_error=0.0001)
    assert decimated.n_points == 30


def test_decimate_polylines_inplace(poly_circle):
    poly_circle.decimate_polylines(0.5, inplace=True)
    # Allow some leeway for approximtely 50%
    assert poly_circle.n_points >= 14
    assert poly_circle.n_points <= 16


def test_triangulate_contours():
    poly = pv.Polygon(n_sides=4, fill=False)
    filled = poly.triangulate_contours()
    for cell in filled.cell:
        assert cell.type == pv.CellType.TRIANGLE
