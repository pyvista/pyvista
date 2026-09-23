from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import MissingDataError

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


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


@pytest.mark.parametrize(
    'other_mesh',
    [
        pv.UnstructuredGrid(),
        pv.ImageData(),
        pv.StructuredGrid(),
    ],
    ids=['ugrid', 'image', 'structured'],
)
def test_boolean_raises(other_mesh):
    with pytest.raises(TypeError, match=r'Input mesh must be an instance of'):
        pv.Sphere()._boolean('union', other_mesh=other_mesh, tolerance=0.0, progress_bar=False)


def test_boolean_btype_raises(sphere):
    with pytest.raises(ValueError, match=r"btype 'foo' is not valid"):
        sphere._boolean('foo', other_mesh=pv.Sphere(center=(5, 0, 0)), tolerance=0.0)


def test_clean_raises(mocker: MockerFixture):
    from pyvista.core.filters import poly_data

    m = mocker.patch.object(poly_data, '_get_output')
    m.return_value = pv.PolyData()

    sp = pv.Sphere()
    with pytest.raises(ValueError, match=r'Clean tolerance is too high. Empty mesh returned.'):
        sp.clean()


def test_flip_normals_removed():
    plane = pv.Plane()
    with pytest.raises(
        pv.core.errors.DeprecationError,
        match=r'`flip_normals` is deprecated\. Use `flip_faces` instead',
    ):
        plane.flip_normals()


def test_contour_banded_raises(mocker: MockerFixture):
    from pyvista.core.filters import poly_data

    m = mocker.patch.object(poly_data, 'get_array')
    m.return_value = None

    sp = pv.Sphere()

    with pytest.raises(ValueError, match=r'No arrays present to contour.'):
        sp.contour_banded(1)

    m.return_value = 'foo'
    m = mocker.patch.object(poly_data, 'get_array_association')
    m.return_value = 'foo'

    with pytest.raises(ValueError, match=r'Only point data can be contoured.'):
        sp.contour_banded(1)


def test_boolean_intersect_edge_case():
    a = pv.Cube(x_length=2, y_length=2, z_length=2).triangulate()
    b = pv.Cube().triangulate()  # smaller cube (x_length=1)

    with pytest.warns(UserWarning, match='contained within another'):
        a.boolean_intersection(b)


def test_identical_boolean(sphere):
    with pytest.raises(ValueError, match='identical points'):
        sphere.boolean_intersection(sphere.copy())


@pytest.fixture
def poly_circle():
    circle = pv.Circle(resolution=30)
    return pv.PolyData(circle.points, lines=[31, *list(range(30)), 0])


def test_decimate_polyline(poly_circle):
    assert poly_circle.n_points == 30
    decimated = poly_circle.decimate_polyline(0.5)
    # Allow some leeway for approximtely 50%
    assert decimated.n_points >= 14
    assert decimated.n_points <= 16


def test_decimate_polyline_maximum_error(poly_circle):
    assert poly_circle.n_points == 30
    # low maximum error will prevent decimation.
    # Since this is a regular shape, no decimation occurs at all with suitable choice
    decimated = poly_circle.decimate_polyline(0.5, maximum_error=0.0001)
    assert decimated.n_points == 30


def test_decimate_polyline_inplace(poly_circle):
    poly_circle.decimate_polyline(0.5, inplace=True)
    # Allow some leeway for approximtely 50%
    assert poly_circle.n_points >= 14
    assert poly_circle.n_points <= 16


def test_triangulate_contours():
    poly = pv.Polygon(n_sides=4, fill=False)
    filled = poly.triangulate_contours()
    for cell in filled.cell:
        assert cell.type == pv.CellType.TRIANGLE

    poly.lines = None
    with pytest.raises(RuntimeError, match='input PolyData to have lines'):
        poly.triangulate_contours()


def test_protein_ribbon():
    tgqp = examples.download_3gqp()
    ribbon = tgqp.protein_ribbon()
    assert ribbon.n_cells


def test_ruled_surface():
    poly = pv.PolyData(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
        lines=[[2, 0, 1], [2, 2, 3]],
        force_float=False,
    )
    ruled = poly.ruled_surface(resolution=(21, 21))
    assert ruled.n_cells


def test_dash_lines_solid_returns_the_lines_whole():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=10)
    line.point_data['ids'] = np.arange(line.n_points)

    solid = line.dash_lines('-')
    assert solid is not line
    assert solid.n_cells == line.n_cells
    assert np.array_equal(solid.points, line.points)
    assert np.array_equal(solid.point_data['ids'], line.point_data['ids'])


@pytest.mark.parametrize('style', ['--', ':', '-.', '-..'])
def test_dash_lines_splits_into_multiple_cells(style):
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=100)
    dashed = line.dash_lines(style, scale=0.01)
    drawn = dashed.compute_cell_sizes(length=True, area=False, volume=False)
    assert dashed.n_cells > 1
    assert drawn.cell_data['Length'].sum() < 1.0


def test_dash_lines_total_length_matches_duty_cycle():
    line = pv.Line((0, 0, 0), (1, 0, 0))
    dashed = line.dash_lines('--', scale=1 / 32)
    drawn = dashed.compute_cell_sizes(length=True, area=False, volume=False)
    assert np.isclose(drawn.cell_data['Length'].sum(), 0.5, atol=0.02)


@pytest.mark.parametrize(
    ('style', 'pattern'),
    [
        ('--', [8, 8]),
        (':', [1, 7, 1, 7]),
        ('-.', [4, 6, 2, 4]),
        ('-..', [3, 3, 1, 3, 3, 3]),
    ],
)
def test_dash_lines_pattern_spells_out_a_style(style, pattern):
    # the equivalences the style docstring promises
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=400)
    named = line.dash_lines(style, scale=0.005)
    spelled = line.dash_lines(pattern=pattern, scale=0.005)
    assert named.n_cells == spelled.n_cells
    assert np.array_equal(named.points, spelled.points)


def test_dash_lines_style_and_pattern_are_exclusive():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=50)
    for style in ['-', '--', '']:
        with pytest.raises(ValueError, match='Cannot set both'):
            line.dash_lines(style, pattern=[1, 7])


def test_dash_lines_join_makes_the_pattern_continuous():
    points = np.zeros((51, 3))
    points[:, 0] = np.linspace(0, 1, 51)
    segments = np.column_stack([np.full(50, 2), np.arange(50), np.arange(1, 51)])
    edges = pv.PolyData(points, lines=segments.ravel())
    assert edges.n_cells == 50
    assert edges.dash_lines('--', scale=0.05, join=False).n_cells == 50
    assert edges.dash_lines('--', scale=0.05, join=True).n_cells < 50


def test_dash_lines_interpolates_point_data():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=100)
    line['x'] = line.points[:, 0].copy()
    dashed = line.dash_lines('--', scale=0.01)
    assert np.allclose(dashed['x'], dashed.points[:, 0], atol=1e-6)


def test_dash_lines_copies_cell_data():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=10)
    line.cell_data['tag'] = np.array([7])
    dashed = line.dash_lines('--', scale=0.01, join=False)
    assert np.all(dashed.cell_data['tag'] == 7)
    assert 'tag' not in line.dash_lines('--', scale=0.01).cell_data


def test_dash_lines_removes_non_line_cells():
    assert pv.Sphere().dash_lines().n_cells == 0
    assert pv.PolyData().dash_lines().n_cells == 0


@pytest.mark.parametrize('style', ['--', '-', ''])
def test_dash_lines_removes_the_other_cells_of_a_mixed_mesh(style):
    mesh = pv.Plane(i_resolution=2, j_resolution=2)
    mesh.lines = np.array([2, 0, 8])
    mesh.verts = np.array([1, 3])
    mesh.cell_data['tag'] = np.arange(mesh.n_cells)
    assert mesh.n_verts
    assert mesh.n_faces

    dashed = mesh.dash_lines(style, scale=0.2, join=False)
    assert dashed.n_verts == 0
    assert dashed.n_faces == 0
    assert dashed.n_strips == 0
    assert dashed.n_cells == dashed.n_lines
    if style == '':
        assert dashed.n_cells == 0
    else:
        # the tags all come from the single line cell, which follows the vert
        assert np.all(dashed.cell_data['tag'] == mesh.cell_data['tag'][mesh.n_verts])


def test_dash_lines_default_scale_follows_length():
    small = pv.Line((0, 0, 0), (1, 0, 0), resolution=100)
    large = pv.Line((0, 0, 0), (10, 0, 0), resolution=100)
    assert small.dash_lines().n_cells == large.dash_lines().n_cells


def test_dash_lines_inplace():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=100)
    returned = line.dash_lines('--', scale=0.01, inplace=True)
    assert returned is line
    assert line.n_cells > 1


def test_dash_lines_raises():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=10)
    with pytest.raises(ValueError, match='is not valid'):
        line.dash_lines('wrong')
    with pytest.raises(ValueError, match='even number of lengths'):
        line.dash_lines(pattern=[4, 2, 4])
    with pytest.raises(ValueError, match='greater than 0'):
        line.dash_lines(pattern=[4, 0])
    with pytest.raises(ValueError, match='greater than 0'):
        line.dash_lines(scale=0.0)
    with pytest.raises(ValueError, match='minimum length of 2'):
        line.dash_lines(pattern=[])
    with pytest.raises(ValueError, match='finite'):
        line.dash_lines(scale=float('inf'))


def test_dash_lines_degenerate_cells():
    mesh = pv.PolyData()
    mesh.points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mesh.lines = np.array([1, 0, 2, 1, 2])
    assert mesh.n_lines == 2

    dashed = mesh.dash_lines(join=False)
    assert dashed.n_points == 0
    assert dashed.n_cells == 0


def test_dash_lines_snaps_integer_point_data():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=10)
    line.point_data['ids'] = np.arange(line.n_points)

    dashed = line.dash_lines()
    assert dashed.point_data['ids'].dtype == line.point_data['ids'].dtype
    assert np.isin(dashed.point_data['ids'], line.point_data['ids']).all()


def test_dash_lines_hidden_style():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=10)
    hidden = line.dash_lines('')
    assert hidden.n_points == 0
    assert hidden.n_cells == 0


def test_dash_lines_keeps_active_scalars_and_field_data():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=10)
    line.point_data['vals'] = np.arange(line.n_points, dtype=float)
    line.set_active_scalars('vals')
    line.field_data['meta'] = ['x']

    dashed = line.dash_lines()
    assert dashed.active_scalars_name == 'vals'
    assert list(dashed.field_data['meta']) == ['x']


def test_dash_lines_keeps_inactive_scalars_inactive():
    line = pv.Line((0, 0, 0), (1, 0, 0), resolution=10)
    line.point_data['vals'] = np.arange(line.n_points, dtype=float)
    line.set_active_scalars(None)
    assert line.dash_lines().active_scalars_name is None


def test_dash_lines_keeps_active_normals():
    sphere = pv.Sphere()
    sphere.lines = np.array([2, 0, 1])
    dashed = sphere.dash_lines()
    assert dashed.point_data.active_normals_name == 'Normals'
    assert dashed.active_scalars_name is None
