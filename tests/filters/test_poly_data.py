import numpy as np
import pytest

import pyvista as pv
from pyvista.errors import MissingDataError


def test_contour_banded_raise(sphere):
    sphere.clear_data()

    with pytest.raises(MissingDataError):
        sphere.contour_banded(5)

    sphere['data'] = sphere.points[:, 2]
    with pytest.raises(ValueError):
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
        10, rng=rng, generate_contour_edges=False, scalar_mode='index', clipping=True
    )
    assert out['data'].min() <= rng[0]
    assert out['data'].max() >= rng[1]


def test_boolean_intersect_edge_case():
    a = pv.Cube(x_length=2, y_length=2, z_length=2).triangulate()
    b = pv.Cube().triangulate()  # smaller cube (x_length=1)

    with pytest.warns(UserWarning, match='contained within another'):
        a.boolean_intersection(b)


def test_iterative_closest_point():
    # Create a simple mesh
    source = pv.Cylinder(resolution=30).triangulate().subdivide(1)
    transformed = source.rotate_y(20).rotate_z(25).translate([-0.75, -0.5, 0.5])

    # Perform ICP registration
    aligned = transformed.iterative_closest_point(source)

    # Check if the number of points in the aligned mesh is the same as the original mesh
    assert source.n_points == aligned.n_points

    _, closest_points = aligned.find_closest_cell(source.points, return_closest_point=True)
    dist = np.linalg.norm(source.points - closest_points, axis=1)
    assert np.abs(dist).mean() < 1e-3
