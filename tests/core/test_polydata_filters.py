from __future__ import annotations

import pytest

import pyvista as pv
from pyvista import examples
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


def test_triangulate_contours():
    poly = pv.Polygon(n_sides=4, fill=False)
    filled = poly.triangulate_contours()
    for cell in filled.cell:
        assert cell.type == pv.CellType.TRIANGLE


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 1, 0),
    reason="Requires VTK>=9.1.0 for a vtkIOChemistry.vtkCMLMoleculeReader",
)
def test_protein_ribbon():
    tgqp = examples.download_3gqp()
    ribbon = tgqp.protein_ribbon()
    assert ribbon.n_cells


def test_fit_to_height_map():
    height_map = pv.ImageData(dimensions=(10, 10, 1))
    height_map.origin = (0.0, 0.0, 0.0)
    height_map.spacing = (1.0, 1.0, 1.0)
    height_map.point_data["elevation"] = range(height_map.n_points)

    polygon = pv.Polygon(n_sides=4, radius=2, center=(4.5, 4.5, 0))

    result = polygon.fit_to_height_map(height_map)
    assert result.n_points == polygon.n_points
    assert isinstance(result, pv.PolyData)

    result_offset = polygon.fit_to_height_map(height_map, use_height_map_offset=True)
    result_no_offset = polygon.fit_to_height_map(height_map, use_height_map_offset=False)
    assert result_offset.n_points == result_no_offset.n_points

    for strategy in [
        "point_projection",
        "point_minimum_height",
        "point_maximum_height",
        "point_average_height",
        "cell_minimum_height",
        "cell_maximum_height",
        "cell_average_height",
    ]:
        result = polygon.fit_to_height_map(height_map, fitting_strategy=strategy)
        assert result.n_points == polygon.n_points


def test_fit_to_height_map_invalid_strategy():
    polygon = pv.Polygon(n_sides=4)
    height_map = pv.ImageData(dimensions=(10, 10, 1))
    with pytest.raises(ValueError, match="Invalid fitting strategy"):
        polygon.fit_to_height_map(height_map, fitting_strategy="invalid")

    with pytest.raises(TypeError, match="Invalid type"):
        polygon.fit_to_height_map(height_map, fitting_strategy=0)
