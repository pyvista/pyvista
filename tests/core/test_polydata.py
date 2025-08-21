from __future__ import annotations

import functools
from math import pi
import pathlib
from pathlib import Path
import re
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import CellSizeError
from pyvista.core.errors import NotAllTrianglesError
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import PyVistaFutureWarning

radius = 0.5


@pytest.fixture
def sphere():
    # this shadows the main sphere fixture from conftest!
    return pv.Sphere(radius=radius, theta_resolution=10, phi_resolution=10)


@pytest.fixture
def sphere_shifted():
    return pv.Sphere(center=[0.5, 0.5, 0.5], theta_resolution=10, phi_resolution=10)


@pytest.fixture
def sphere_dense():
    return pv.Sphere(radius=radius, theta_resolution=100, phi_resolution=100)


@pytest.fixture
def cube_dense():
    return pv.Cube()


test_path = str(Path(__file__).resolve().parent)


def is_binary(filename):
    """Return ``True`` when a file is binary."""
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
    with Path(filename).open('rb') as f:
        data = f.read(1024)
    return bool(data.translate(None, textchars))


def test_init():
    mesh = pv.PolyData()
    assert not mesh.n_points
    assert not mesh.n_cells


def test_init_from_pdata(sphere):
    mesh = pv.PolyData(sphere, deep=True)
    assert mesh.n_points
    assert mesh.n_cells
    mesh.points[0] += 1
    assert not np.allclose(sphere.points[0], mesh.points[0])


@pytest.mark.parametrize('faces_is_cell_array', [False, True])
def test_init_from_arrays(faces_is_cell_array):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

    # mesh faces
    faces = np.hstack([[4, 0, 1, 2, 3], [3, 0, 1, 4], [3, 1, 2, 4]]).astype(np.int8)

    mesh = pv.PolyData(vertices, pv.CellArray(faces) if faces_is_cell_array else faces)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3

    mesh = pv.PolyData(vertices, pv.CellArray(faces) if faces_is_cell_array else faces, deep=True)
    vertices[0] += 1
    assert not np.allclose(vertices[0], mesh.points[0])

    # ensure that polydata raises a warning when inputting non-float dtype
    with pytest.warns(Warning, match=r'Points is not a float type\. This can cause issues'):
        mesh = pv.PolyData(vertices.astype(np.int32), faces)

    # array must be immutable
    with pytest.raises(ValueError):  # noqa: PT011
        mesh.faces[0] += 1

    # attribute is mutable
    faces = [4, 0, 1, 2, 3]
    mesh.faces = pv.CellArray(faces) if faces_is_cell_array else faces
    assert np.allclose(faces, mesh.faces)


@pytest.mark.parametrize('faces_is_cell_array', [False, True])
def test_init_from_arrays_with_vert(faces_is_cell_array):
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0.5, 0.5, -1],
            [0, 1.5, 1.5],
        ]
    )

    # mesh faces
    faces = np.hstack(
        [
            [4, 0, 1, 2, 3],
            [3, 0, 1, 4],
            [3, 1, 2, 4],
            [1, 5],
        ],  # [quad, triangle, triangle, vertex]
    ).astype(np.int8)
    if faces_is_cell_array:
        faces = pv.CellArray(faces)

    mesh = pv.PolyData(vertices, faces)
    assert mesh.n_points == 6
    assert mesh.n_cells == 4


@pytest.mark.parametrize('faces_is_cell_array', [False, True])
def test_init_from_arrays_triangular(faces_is_cell_array):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

    # mesh faces
    faces = np.vstack([[3, 0, 1, 2], [3, 0, 1, 4], [3, 1, 2, 4]])
    if faces_is_cell_array:
        faces = pv.CellArray(faces)

    mesh = pv.PolyData(vertices, faces)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3

    mesh = pv.PolyData(vertices, faces, deep=True)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3


def test_init_as_points():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

    mesh = pv.PolyData(vertices)
    assert mesh.n_points == vertices.shape[0]
    assert mesh.n_cells == vertices.shape[0]
    assert len(mesh.verts) == vertices.shape[0] * 2

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    cells = np.array([1, 0, 1, 1, 1, 2], np.int16)
    to_check = pv.PolyData._make_vertex_cells(len(vertices)).ravel()
    assert np.allclose(to_check, cells)

    # from list
    mesh.verts = [[1, 0], [1, 1], [1, 2]]
    to_check = pv.PolyData._make_vertex_cells(len(vertices)).ravel()
    assert np.allclose(to_check, cells)

    mesh = pv.PolyData()
    mesh.points = vertices
    mesh.verts = cells
    assert mesh.n_points == vertices.shape[0]
    assert mesh.n_cells == vertices.shape[0]
    assert np.allclose(mesh.verts, cells)


def test_init_as_points_from_list():
    points = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    mesh = pv.PolyData(points)
    assert np.allclose(mesh.points, points)


def test_invalid_init():
    with pytest.raises(ValueError):  # noqa: PT011
        pv.PolyData(np.array([1.0]))

    with pytest.raises(TypeError):
        pv.PolyData([1.0, 2.0, 3.0], 'woa')

    with pytest.raises(ValueError):  # noqa: PT011
        pv.PolyData('woa', 'woa')

    poly = pv.PolyData()
    with pytest.raises(ValueError):  # noqa: PT011
        pv.PolyData(poly, 'woa')

    with pytest.raises(TypeError):
        pv.PolyData({'woa'})


def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        pv.PolyData('file.bad')

    filename = str(Path(test_path) / 'test_polydata.py')
    with pytest.raises(IOError):  # noqa: PT011
        pv.PolyData(filename)


@pytest.mark.parametrize(
    ('arr', 'value'),
    [
        ('faces', [3, 1, 2, 3, 3, 0, 1]),
        ('strips', np.array([5, 4, 3, 2, 0])),
        ('lines', [4, 0, 1, 2, 2, 3, 4]),
        ('verts', [1, 0, 1]),
        ('faces', [[3, 0, 1], [3, 2, 1], [4, 0, 1]]),
        ('faces', [[2, 0, 1], [2, 2, 1], [1, 0, 1]]),
    ],
)
def test_invalid_connectivity_arrays(arr, value):
    generator = np.random.default_rng(seed=None)
    points = generator.random((10, 3))
    mesh = pv.PolyData(points)
    with pytest.raises(CellSizeError, match='Cell array size is invalid'):
        setattr(mesh, arr, value)

    with pytest.raises(CellSizeError, match=f'`{arr}` cell array size is invalid'):
        _ = pv.PolyData(points, **{arr: value})


@pytest.mark.parametrize('lines_is_cell_array', [False, True])
def test_lines_on_init(lines_is_cell_array):
    points = np.random.default_rng().random((5, 3))
    lines = [2, 0, 1, 3, 2, 3, 4]
    pd = pv.PolyData(points, lines=pv.CellArray(lines) if lines_is_cell_array else lines)
    assert not pd.faces.size
    assert np.array_equal(pd.lines, lines)
    assert np.array_equal(pd.points, points)


def _assert_verts_equal(
    mesh: pv.PolyData,
    verts: list[int],
    n_verts: int,
    cell_types: dict[int, pv.CellType],
):
    assert np.array_equal(mesh.verts, verts)
    assert mesh.n_verts == n_verts
    for i, expected_typ in cell_types.items():
        assert mesh.get_cell(i).type == expected_typ


@pytest.mark.parametrize('verts_is_cell_array', [False, True])
def test_verts(verts_is_cell_array):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])
    verts = [1, 0, 1, 1, 1, 2, 1, 3, 1, 4]

    if not verts_is_cell_array:
        mesh = pv.PolyData(vertices)
        _assert_verts_equal(mesh, verts, n_verts=5, cell_types={0: pv.CellType.VERTEX})

    mesh = pv.PolyData(vertices, verts=pv.CellArray(verts) if verts_is_cell_array else verts)
    _assert_verts_equal(mesh, verts, n_verts=5, cell_types={0: pv.CellType.VERTEX})

    verts = [1, 0]
    mesh = pv.PolyData(vertices, verts=pv.CellArray(verts) if verts_is_cell_array else verts)
    _assert_verts_equal(mesh, verts, n_verts=1, cell_types={0: pv.CellType.VERTEX})

    verts = [2, 0, 1, 1, 2]
    mesh = pv.PolyData(vertices, verts=pv.CellArray(verts) if verts_is_cell_array else verts)
    _assert_verts_equal(
        mesh,
        verts,
        n_verts=2,
        cell_types={0: pv.CellType.POLY_VERTEX, 1: pv.CellType.VERTEX},
    )


@pytest.mark.parametrize('verts', [([1, 0]), (pv.CellArray([1, 0]))])
@pytest.mark.parametrize('lines', [([2, 1, 2]), (pv.CellArray([2, 1, 2]))])
@pytest.mark.parametrize('faces', [([3, 3, 4, 5]), (pv.CellArray([3, 3, 4, 5]))])
@pytest.mark.parametrize('strips', [([4, 6, 7, 8, 9]), (pv.CellArray([4, 6, 7, 8, 9]))])
def test_mixed_cell_polydata(verts, lines, faces, strips):
    points = np.zeros((10, 3))
    points[:, 0] = np.linspace(0, 9, 10)
    a = pv.PolyData(points, verts=verts, lines=lines, faces=faces, strips=strips)
    assert np.array_equal(a.verts, [1, 0])
    assert np.array_equal(a.lines, [2, 1, 2])
    assert np.array_equal(a.faces, [3, 3, 4, 5])
    assert np.array_equal(a.strips, [4, 6, 7, 8, 9])


def test_polydata_repr_str():
    pd = pv.PolyData()
    assert repr(pd) == str(pd)
    assert 'N Cells' in str(pd)
    assert 'N Points' in str(pd)
    assert 'X Bounds' in str(pd)
    assert 'N Arrays' in str(pd)


def test_geodesic(sphere):
    start, end = 0, sphere.n_points - 1
    geodesic = sphere.geodesic(start, end)
    assert isinstance(geodesic, pv.PolyData)
    assert 'vtkOriginalPointIds' in geodesic.array_names
    ids = geodesic.point_data['vtkOriginalPointIds']
    assert np.allclose(geodesic.points, sphere.points[ids])

    # check keep_order
    geodesic_legacy = sphere.geodesic(start, end, keep_order=False)
    assert geodesic_legacy['vtkOriginalPointIds'][0] == end
    geodesic_ordered = sphere.geodesic(start, end, keep_order=True)
    assert geodesic_ordered['vtkOriginalPointIds'][0] == start

    # finally, inplace
    geodesic_inplace = sphere.geodesic(start, end, inplace=True)
    assert geodesic_inplace is sphere
    assert np.allclose(geodesic.points, sphere.points)


def test_geodesic_fail(sphere, plane):
    with pytest.raises(IndexError):
        sphere.geodesic(-1, -1)
    with pytest.raises(IndexError):
        sphere.geodesic(sphere.n_points, 0)

    with pytest.raises(NotAllTrianglesError):
        plane.geodesic(0, 10)


def test_geodesic_distance(sphere):
    distance = sphere.geodesic_distance(0, sphere.n_points - 1)
    assert isinstance(distance, float)

    # Use scalar weights
    distance_use_scalar_weights = sphere.geodesic_distance(
        0,
        sphere.n_points - 1,
        use_scalar_weights=True,
    )
    assert isinstance(distance_use_scalar_weights, float)


def test_ray_trace(sphere):
    points, ind = sphere.ray_trace([0, 0, 0], [1, 1, 1])
    assert np.any(points)
    assert np.any(ind)


def test_ray_trace_origin():
    # https://github.com/pyvista/pyvista/issues/5372
    plane = pv.Plane(i_resolution=1, j_resolution=1)
    pts, cells = plane.ray_trace([0, 0, 1], [0, 0, -1])
    assert len(cells) == 1
    assert cells[0] == 0


def test_multi_ray_trace(sphere):
    trimesh = pytest.importorskip('trimesh')
    if not trimesh.ray.has_embree:
        pytest.skip('Requires Embree')
    origins = [[1, 0, 1], [0.5, 0, 1], [0.25, 0, 1], [0, 0, 5]]
    directions = [[0, 0, -1]] * 4
    points, ind_r, ind_t = sphere.multi_ray_trace(origins, directions)
    assert np.any(points)
    assert np.any(ind_r)
    assert np.any(ind_t)

    # patch embree to test retry
    with patch.object(
        trimesh.ray.ray_pyembree.RayMeshIntersector,
        'intersects_location',
        return_value=[np.array([])] * 3,
    ):
        points, ind_r, ind_t = sphere.multi_ray_trace(origins, directions, retry=True)
        known_points = np.array(
            [[0.25, 0, 0.42424145], [0.25, 0, -0.42424145], [0, 0, 0.5], [0, 0, -0.5]],
        )
        known_ind_r = np.array([2, 2, 3, 3])
        np.testing.assert_allclose(points, known_points)
        np.testing.assert_allclose(ind_r, known_ind_r)
        assert len(ind_t) == 4

    # check non-triangulated
    mesh = pv.Cylinder()
    with pytest.raises(NotAllTrianglesError):
        mesh.multi_ray_trace(origins, directions)


def test_edge_mask(sphere):
    _ = sphere.edge_mask(10, progress_bar=True)


def test_boolean_union_intersection(sphere, sphere_shifted):
    union = sphere.boolean_union(sphere_shifted, progress_bar=True)
    intersection = sphere.boolean_intersection(sphere_shifted, progress_bar=True)

    # union is volume of sphere + sphere_shifted minus the part intersecting
    expected_volume = sphere.volume + sphere_shifted.volume - intersection.volume
    assert np.isclose(union.volume, expected_volume, atol=1e-3)

    # intersection volume is the volume of both isolated meshes minus the union
    expected_volume = sphere.volume + sphere_shifted.volume - union.volume
    assert np.isclose(intersection.volume, expected_volume, atol=1e-3)


def test_bitwise_and_or(sphere, sphere_shifted):
    union = sphere | sphere_shifted
    intersection = sphere & sphere_shifted

    # union is volume of sphere + sphere_shifted minus the part intersecting
    expected_volume = sphere.volume + sphere_shifted.volume - intersection.volume
    assert np.isclose(union.volume, expected_volume, atol=1e-3)

    # intersection volume is the volume of both isolated meshes minus the union
    expected_volume = sphere.volume + sphere_shifted.volume - union.volume
    assert np.isclose(intersection.volume, expected_volume, atol=1e-3)


def test_boolean_difference(sphere, sphere_shifted):
    difference = sphere.boolean_difference(sphere_shifted, progress_bar=True)
    intersection = sphere.boolean_intersection(sphere_shifted, progress_bar=True)

    expected_volume = sphere.volume - intersection.volume
    assert np.isclose(difference.volume, expected_volume, atol=1e-3)


def test_boolean_difference_fail(plane, sphere):
    with pytest.raises(NotAllTrianglesError):
        plane - sphere


def test_subtract(sphere, sphere_shifted):
    sub_mesh = sphere - sphere_shifted
    assert sub_mesh.n_points == sphere.boolean_difference(sphere_shifted).n_points


def test_isubtract(sphere, sphere_shifted):
    sub_mesh = sphere.copy()
    sub_mesh -= sphere_shifted
    assert sub_mesh.n_points == sphere.boolean_difference(sphere_shifted).n_points


def test_append(
    sphere: pv.PolyData,
    sphere_shifted: pv.PolyData,
    sphere_dense: pv.PolyData,
):
    # 1/ Single argument
    merged = sphere.append_polydata(sphere_shifted)
    assert merged.n_points == (sphere.n_points + sphere_shifted.n_points)
    assert isinstance(merged, pv.PolyData)
    # test point order is kept
    np.testing.assert_array_equal(merged.points[: sphere.n_points], sphere.points)
    np.testing.assert_array_equal(merged.points[sphere.n_points :], sphere_shifted.points)

    # 2/ Multiple arguments
    merged = sphere.append_polydata(sphere_shifted, sphere_dense)
    assert merged.n_points == (sphere.n_points + sphere_shifted.n_points + sphere_dense.n_points)
    assert isinstance(merged, pv.PolyData)
    # test point order is kept
    np.testing.assert_array_equal(merged.points[: sphere.n_points], sphere.points)
    mid = sphere.n_points + sphere_shifted.n_points
    np.testing.assert_array_equal(merged.points[sphere.n_points : mid], sphere_shifted.points)
    np.testing.assert_array_equal(merged.points[mid:], sphere_dense.points)

    # 3/ test in-place merge
    mesh = sphere.copy()
    merged = mesh.append_polydata(sphere_shifted, inplace=True)
    assert merged is mesh


def test_append_raises(sphere: pv.PolyData):
    with pytest.raises(TypeError, match='All meshes need to be of PolyData type'):
        sphere.append_polydata(sphere.cast_to_unstructured_grid())


def test_merge(sphere, sphere_shifted, hexbeam):
    merged = sphere.merge(hexbeam, merge_points=False, progress_bar=True)
    assert merged.n_points == (sphere.n_points + hexbeam.n_points)
    assert isinstance(merged, pv.UnstructuredGrid)
    assert merged.active_scalars_name is None

    # list with unstructuredgrid case
    merged = sphere.merge([hexbeam, hexbeam], merge_points=False, progress_bar=True)
    assert merged.n_points == (sphere.n_points + hexbeam.n_points * 2)
    assert isinstance(merged, pv.UnstructuredGrid)
    assert merged.active_scalars_name is None

    # with polydata
    merged = sphere.merge(sphere_shifted, progress_bar=True)
    assert isinstance(merged, pv.PolyData)
    assert merged.n_points == sphere.n_points + sphere_shifted.n_points
    assert merged.active_scalars_name is None

    # with polydata list (no merge)
    merged = sphere.merge([sphere_shifted, sphere_shifted], merge_points=False, progress_bar=True)
    assert isinstance(merged, pv.PolyData)
    assert merged.n_points == sphere.n_points + sphere_shifted.n_points * 2
    assert merged.active_scalars_name is None

    # with polydata list (merge)
    merged = sphere.merge([sphere_shifted, sphere_shifted], progress_bar=True)
    assert isinstance(merged, pv.PolyData)
    assert merged.n_points == sphere.n_points + sphere_shifted.n_points
    assert merged.active_scalars_name is None

    # test in-place merge
    mesh = sphere.copy()
    merged = mesh.merge(sphere_shifted, inplace=True)
    assert merged is mesh
    assert merged.active_scalars_name is None

    # test merge with lines
    arc_1 = pv.CircularArc(
        pointa=[0, 0, 0], pointb=[10, 10, 0], center=[10, 0, 0], negative=False, resolution=3
    )
    arc_2 = pv.CircularArc(
        pointa=[10, 10, 0], pointb=[20, 0, 0], center=[10, 0, 0], negative=False, resolution=3
    )
    merged = arc_1 + arc_2
    assert merged.n_lines == 2
    assert merged.active_scalars_name == 'Distance'

    # test merge with lines as iterable
    merged = arc_1.merge((arc_2, arc_2))
    assert merged.n_lines == 3
    assert merged.active_scalars_name == 'Distance'


@pytest.mark.parametrize('input_', [examples.load_hexbeam(), pv.Sphere()])
def test_merge_active_scalars(input_):
    mesh1 = input_.copy()
    mesh1['foo'] = np.arange(mesh1.n_points)
    mesh2 = mesh1.copy()

    a = mesh1.copy()
    b = mesh2.copy()
    a.active_scalars_name = None
    b.active_scalars_name = None
    merged = a.merge(b)
    assert merged.active_scalars_name is None
    merged = b.merge(a)
    assert merged.active_scalars_name is None

    a = mesh1.copy()
    b = mesh2.copy()
    a.active_scalars_name = 'foo'
    b.active_scalars_name = None
    merged = a.merge(b)
    assert merged.active_scalars_name is None
    merged = b.merge(a)
    assert merged.active_scalars_name is None

    a = mesh1.copy()
    b = mesh2.copy()
    a.active_scalars_name = None
    b.active_scalars_name = 'foo'
    merged = a.merge(b)
    assert merged.active_scalars_name is None
    merged = b.merge(a)
    assert merged.active_scalars_name is None

    a = mesh1.copy()
    b = mesh2.copy()
    a.active_scalars_name = 'foo'
    b.active_scalars_name = 'foo'
    merged = a.merge(b)
    assert merged.active_scalars_name == 'foo'
    merged = b.merge(a)
    assert merged.active_scalars_name == 'foo'


@pytest.mark.parametrize(
    'input_', [examples.load_hexbeam(), pv.Plane(i_resolution=1, j_resolution=1)]
)
@pytest.mark.parametrize('main_has_priority', [True, False])
def test_merge_main_has_priority(input_, main_has_priority):
    mesh = input_.copy()
    data_main = np.arange(mesh.n_points, dtype=float)
    mesh.point_data['present_in_both'] = data_main
    mesh.set_active_scalars('present_in_both')

    other = mesh.copy()
    data_other = -data_main
    other.point_data['present_in_both'] = data_other
    other.set_active_scalars('present_in_both')

    # note: order of points can change after point merging
    def matching_point_data(this, that, scalars_name):
        """Return True if scalars on two meshes only differ by point order."""
        return all(
            new_val == this.point_data[scalars_name][j]
            for point, new_val in zip(that.points, that.point_data[scalars_name])
            for j in (this.points == point).all(-1).nonzero()
        )

    if pv.vtk_version_info >= (9, 5, 0):
        merged = mesh.merge(other)
        expected_to_match = mesh
    else:
        with pytest.warns(
            pv.PyVistaDeprecationWarning,
            match="The keyword 'main_has_priority' is deprecated and should not be used",
        ):
            merged = mesh.merge(other, main_has_priority=main_has_priority)
        expected_to_match = mesh if main_has_priority else other
    assert matching_point_data(merged, expected_to_match, 'present_in_both')
    assert merged.active_scalars_name == 'present_in_both'


@pytest.mark.parametrize('main_has_priority', [True, False])
def test_merge_main_has_priority_deprecated(sphere, main_has_priority):
    match = (
        "The keyword 'main_has_priority' is deprecated and should not be used.\n"
        'The main mesh will always have priority in a future version.'
    )
    if main_has_priority is False and pv.vtk_version_info >= (9, 5, 0):
        with pytest.raises(ValueError, match=match):
            sphere.merge(sphere, main_has_priority=main_has_priority)
    else:
        with pytest.warns(pv.PyVistaDeprecationWarning, match=match):
            sphere.merge(sphere, main_has_priority=main_has_priority)


@pytest.mark.parametrize('main_has_priority', [True, False])
@pytest.mark.parametrize('mesh', [pv.UnstructuredGrid(), pv.PolyData()])
def test_merge_field_data(mesh, main_has_priority):
    key = 'data'
    data_main = [1, 2, 3]
    data_other = [4, 5, 6]
    mesh.field_data[key] = data_main
    other = mesh.copy()
    other.field_data[key] = data_other

    match = (
        "The keyword 'main_has_priority' is deprecated and should not be used.\n"
        'The main mesh will always have priority in a future version, and this '
        'keyword will be removed.'
    )
    if main_has_priority is False and pv.vtk_version_info >= (9, 5, 0):
        match += '\nIts value cannot be False for vtk>=9.5.0.'
        with pytest.raises(ValueError, match=re.escape(match)):
            mesh.merge(other, main_has_priority=main_has_priority)
        return
    else:
        with pytest.warns(pv.PyVistaDeprecationWarning, match=match):
            merged = mesh.merge(other, main_has_priority=main_has_priority)

    actual = merged.field_data[key]
    expected = data_main if main_has_priority else data_other
    assert np.array_equal(actual, expected)


def test_add(sphere, sphere_shifted):
    merged = sphere + sphere_shifted
    assert isinstance(merged, pv.PolyData)
    assert merged.n_points == sphere.n_points + sphere_shifted.n_points
    assert merged.n_faces_strict == sphere.n_faces_strict + sphere_shifted.n_faces_strict


def test_intersection(sphere, sphere_shifted):
    intersection, first, second = sphere.intersection(
        sphere_shifted,
        split_first=True,
        split_second=True,
        progress_bar=True,
    )

    assert intersection.n_points
    assert first.n_points > sphere.n_points
    assert second.n_points > sphere_shifted.n_points

    intersection, first, second = sphere.intersection(
        sphere_shifted,
        split_first=False,
        split_second=False,
        progress_bar=True,
    )
    assert intersection.n_points
    assert first.n_points == sphere.n_points
    assert second.n_points == sphere_shifted.n_points


@pytest.mark.needs_vtk_version(9, 1, 0)
@pytest.mark.parametrize('curv_type', ['mean', 'gaussian', 'maximum', 'minimum'])
def test_curvature(sphere, curv_type):
    func = functools.partial(sphere.curvature, curv_type)
    if curv_type in ['maximum', 'minimum']:
        with pytest.warns(pv.VTKOutputMessageWarning, match='large computation error'):
            curv = func()
    else:
        curv = func()
    assert np.any(curv)
    assert curv.size == sphere.n_points


def test_invalid_curvature(sphere):
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.curvature('not valid')


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pv.core.pointset.PolyData._WRITERS)
def test_save(sphere, extension, binary, tmpdir):
    filename = str(tmpdir.mkdir('tmpdir').join(f'tmp{extension}'))

    if extension == '.vtkhdf' and not binary:
        with pytest.raises(ValueError, match='.vtkhdf files can only be written in binary format'):
            sphere.save(filename, binary=binary)
        return

    sphere.save(filename, binary=binary)
    if binary:
        if extension == '.vtp':
            with Path(filename).open() as f:
                assert 'binary' in f.read(1000)
        else:
            is_binary(filename)
    else:
        with Path(filename).open() as f:
            fst = f.read(100).lower()
            assert (
                'ascii' in fst
                or 'xml' in fst
                or 'solid' in fst
                or 'pgeometry' in fst
                or '# generated' in fst
                or '#inventor' in fst
            )

    if extension not in ('.geo', '.iv'):
        mesh = pv.PolyData(filename)
        assert mesh.faces.shape == sphere.faces.shape
        assert mesh.points.shape == sphere.points.shape


def test_pathlib_read_write(tmpdir, sphere):
    path = pathlib.Path(str(tmpdir.mkdir('tmpdir').join('tmp.vtk')))
    sphere.save(path)
    assert path.is_file()

    mesh = pv.PolyData(path)
    assert mesh.faces.shape == sphere.faces.shape
    assert mesh.points.shape == sphere.points.shape

    mesh = pv.read(path)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.faces.shape == sphere.faces.shape
    assert mesh.points.shape == sphere.points.shape


def test_invalid_save(sphere):
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.save('file.abc')


def test_triangulate_filter(plane):
    assert not plane.is_all_triangles
    plane.triangulate(inplace=True)
    assert plane.is_all_triangles
    # Make a point cloud and assert false
    assert not pv.PolyData(plane.points).is_all_triangles
    # Extract lines and make sure false
    assert not plane.extract_all_edges().is_all_triangles


@pytest.mark.parametrize('pass_lines', [True, False])
def test_triangulate_filter_pass_lines(sphere: pv.PolyData, plane: pv.PolyData, pass_lines: bool):
    merge: pv.PolyData = plane + (lines := sphere.extract_all_edges())
    tri: pv.PolyData = merge.triangulate(pass_lines=pass_lines, inplace=False)

    assert tri.n_lines == (lines.n_cells if pass_lines else 0)
    assert tri.is_all_triangles if not pass_lines else (not tri.is_all_triangles)


@pytest.mark.parametrize('pass_verts', [True, False])
def test_triangulate_filter_pass_verts(plane: pv.PolyData, pass_verts: bool):
    merge: pv.PolyData = plane + (verts := pv.PolyData([0.0, 1.0, 2.0]))
    tri: pv.PolyData = merge.triangulate(pass_verts=pass_verts, inplace=False)

    assert tri.n_verts == (verts.n_cells if pass_verts else 0)
    assert tri.is_all_triangles if not pass_verts else (not tri.is_all_triangles)


@pytest.mark.parametrize('subfilter', ['butterfly', 'loop', 'linear'])
def test_subdivision(sphere, subfilter):
    mesh = sphere.subdivide(1, subfilter, progress_bar=True)
    assert mesh.n_points > sphere.n_points
    assert mesh.n_faces_strict > sphere.n_faces_strict

    mesh = sphere.copy()
    mesh.subdivide(1, subfilter, inplace=True)
    assert mesh.n_points > sphere.n_points
    assert mesh.n_faces_strict > sphere.n_faces_strict


def test_invalid_subdivision(sphere):
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.subdivide(1, 'not valid')

    # check non-triangulated
    mesh = pv.Cylinder()
    with pytest.raises(NotAllTrianglesError):
        mesh.subdivide(1)


def test_extract_feature_edges(sphere):
    # Test extraction of NO edges
    edges = sphere.extract_feature_edges(90)
    assert not edges.n_points

    mesh = pv.Cube()  # use a mesh that actually has strongly defined edges
    more_edges = mesh.extract_feature_edges(10)
    assert more_edges.n_points


def test_extract_feature_edges_no_data():
    mesh = pv.Wavelet()
    edges = mesh.extract_feature_edges(90, clear_data=True)
    assert edges is not None
    assert isinstance(edges, pv.PolyData)
    assert edges.n_arrays == 0


def test_decimate(sphere):
    mesh = sphere.decimate(0.5, progress_bar=True)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces_strict < sphere.n_faces_strict

    mesh.decimate(0.5, inplace=True, progress_bar=True)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces_strict < sphere.n_faces_strict

    # check non-triangulated
    mesh = pv.Cylinder()
    with pytest.raises(NotAllTrianglesError):
        mesh.decimate(0.5)


def test_decimate_pro(sphere):
    mesh = sphere.decimate_pro(0.5, progress_bar=True, max_degree=10)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces_strict < sphere.n_faces_strict

    mesh.decimate_pro(0.5, inplace=True, progress_bar=True)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces_strict < sphere.n_faces_strict

    # check non-triangulated
    mesh = pv.Cylinder()
    with pytest.raises(NotAllTrianglesError):
        mesh.decimate_pro(0.5)


def test_compute_normals(sphere):
    sphere_normals = sphere
    sphere_normals.compute_normals(inplace=True)

    point_normals = sphere_normals.point_data['Normals']
    cell_normals = sphere_normals.cell_data['Normals']
    assert point_normals.shape[0] == sphere.n_points
    assert cell_normals.shape[0] == sphere.n_cells


def test_compute_normals_raises(sphere):
    msg = (
        'Normals cannot be computed for PolyData containing only vertex cells (e.g. point clouds)'
        '\nand/or line cells. The PolyData cells must be polygons (e.g. triangle cells).'
    )

    point_cloud = pv.PolyData(sphere.points)
    assert point_cloud.n_verts == point_cloud.n_cells
    with pytest.raises(TypeError, match=re.escape(msg)):
        point_cloud.compute_normals()

    lines = pv.MultipleLines()
    assert lines.n_lines == lines.n_cells
    with pytest.raises(TypeError, match=re.escape(msg)):
        lines.compute_normals()


def test_compute_normals_inplace(sphere):
    sphere.point_data['numbers'] = np.arange(sphere.n_points)
    sphere2 = sphere.copy(deep=False)

    sphere['numbers'] *= -1  # sphere2 'numbers' are also modified

    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])

    sphere.compute_normals(inplace=True)

    sphere[
        'numbers'
    ] *= -1  # sphere2 'numbers' are also modified after adding to Plotter.  (See  issue #2461)

    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])


def test_compute_normals_split_vertices(cube):
    # verify edge splitting occurs and point IDs are tracked
    cube_split_norm = cube.compute_normals(split_vertices=True)
    assert cube_split_norm.n_points == 24
    assert 'pyvistaOriginalPointIds' in cube_split_norm.point_data
    assert len(set(cube_split_norm.point_data['pyvistaOriginalPointIds'])) == 8


@pytest.fixture
def ant_with_normals(ant):
    ant['Scalars'] = range(ant.n_points)
    point_normals = [[0, 0, 1]] * ant.n_points
    ant.point_data['PointNormals'] = point_normals
    ant.point_data.active_normals_name = 'PointNormals'

    cell_normals = [[1, 0, 0]] * ant.n_cells
    ant.cell_data['CellNormals'] = cell_normals
    ant.cell_data.active_normals_name = 'CellNormals'
    return ant


def test_point_normals_returns_active_normals(ant_with_normals):
    ant = ant_with_normals
    expected_point_normals = ant['PointNormals']

    actual_point_normals = ant.point_normals
    assert actual_point_normals.shape[0] == ant.n_points
    assert np.array_equal(actual_point_normals, ant.point_data.active_normals)
    assert np.shares_memory(actual_point_normals, ant.point_data.active_normals)
    assert np.array_equal(actual_point_normals, expected_point_normals)


def test_point_normals_computes_new_normals(ant):
    expected_point_normals = ant.copy().compute_normals().point_data['Normals']
    ant.point_data.clear()
    assert ant.array_names == []
    assert ant.point_data.active_normals is None

    actual_point_normals = ant.point_normals
    assert actual_point_normals.shape[0] == ant.n_points
    assert np.array_equal(actual_point_normals, expected_point_normals)


def test_cell_normals_returns_active_normals(ant_with_normals):
    ant = ant_with_normals
    expected_cell_normals = ant['CellNormals']

    actual_cell_normals = ant.cell_normals
    assert actual_cell_normals.shape[0] == ant.n_cells
    assert np.array_equal(actual_cell_normals, ant.cell_data.active_normals)
    assert np.shares_memory(actual_cell_normals, ant.cell_data.active_normals)
    assert np.array_equal(actual_cell_normals, expected_cell_normals)


def test_cell_normals_computes_new_normals(ant):
    expected_cell_normals = ant.copy().compute_normals().cell_data['Normals']
    ant.cell_data.clear()
    assert ant.array_names == []
    assert ant.cell_data.active_normals is None

    actual_cell_normals = ant.cell_normals
    assert actual_cell_normals.shape[0] == ant.n_cells
    assert np.array_equal(actual_cell_normals, expected_cell_normals)


def test_face_normals(sphere):
    assert sphere.face_normals.shape[0] == sphere.n_faces_strict


def test_clip_plane(sphere):
    clipped_sphere = sphere.clip(
        origin=[0, 0, 0],
        normal=[0, 0, -1],
        invert=False,
        progress_bar=True,
    )
    faces = clipped_sphere.faces.reshape(-1, 4)[:, 1:]
    assert np.all(clipped_sphere.points[faces, 2] <= 0)

    sphere.clip(
        origin=[0, 0, 0],
        normal=[0, 0, -1],
        inplace=True,
        invert=False,
        progress_bar=True,
    )
    faces = clipped_sphere.faces.reshape(-1, 4)[:, 1:]
    assert np.all(clipped_sphere.points[faces, 2] <= 0)


def test_extract_largest(sphere):
    mesh = sphere + pv.Sphere(radius=0.1, theta_resolution=5, phi_resolution=5)
    largest = mesh.extract_largest()
    assert largest.n_faces_strict == sphere.n_faces_strict

    mesh.extract_largest(inplace=True)
    assert mesh.n_faces_strict == sphere.n_faces_strict


def test_clean(sphere):
    mesh = sphere.merge(sphere, merge_points=False).extract_surface()
    assert mesh.n_points > sphere.n_points
    cleaned = mesh.clean(merge_tol=1e-5)
    assert cleaned.n_points == sphere.n_points

    mesh.clean(merge_tol=1e-5, inplace=True)
    assert mesh.n_points == sphere.n_points

    cleaned = mesh.clean(point_merging=False)
    assert cleaned.n_points == mesh.n_points

    # test with points but no cells
    mesh = pv.PolyData()
    mesh.points = (0, 0, 0)
    cleaned = mesh.clean()
    assert cleaned.n_points == 0


def test_area(sphere_dense, cube_dense):
    radius = 0.5
    ideal_area = 4 * pi * radius**2
    assert np.isclose(sphere_dense.area, ideal_area, rtol=1e-3)

    ideal_area = 6 * np.cbrt(cube_dense.volume) ** 2
    assert np.isclose(cube_dense.area, ideal_area, rtol=1e-3)


def test_volume(sphere_dense):
    ideal_volume = (4 / 3.0) * pi * radius**3
    assert np.isclose(sphere_dense.volume, ideal_volume, rtol=1e-3)


def test_remove_points_any(sphere):
    remove_mask = np.zeros(sphere.n_points, np.bool_)
    remove_mask[:3] = True
    sphere_mod, ind = sphere.remove_points(remove_mask, inplace=False, mode='any')
    assert (sphere_mod.n_points + remove_mask.sum()) == sphere.n_points
    assert np.allclose(sphere_mod.points, sphere.points[ind])


def test_remove_points_all(sphere):
    sphere_copy = sphere.copy()
    sphere_copy.cell_data['ind'] = np.arange(sphere_copy.n_faces_strict)
    remove = sphere.faces[1:4]
    sphere_copy.remove_points(remove, inplace=True, mode='all')
    assert sphere_copy.n_points == sphere.n_points
    assert sphere_copy.n_faces_strict == sphere.n_faces_strict - 1


def test_remove_points_fail(sphere, plane):
    # not triangles:
    with pytest.raises(NotAllTrianglesError):
        plane.remove_points([0])

    # invalid bool mask size
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.remove_points(np.ones(10, np.bool_))

    # invalid mask type
    with pytest.raises(TypeError):
        sphere.remove_points([0.0])


def test_vertice_cells_on_read(tmpdir):
    point_cloud = pv.PolyData(np.random.default_rng().random((100, 3)))
    filename = str(tmpdir.mkdir('tmpdir').join('foo.ply'))
    point_cloud.save(filename)
    recovered = pv.read(filename)
    assert recovered.n_cells == 100
    recovered = pv.PolyData(filename)
    assert recovered.n_cells == 100


def test_center_of_mass(sphere):
    assert np.allclose(sphere.center_of_mass(), [0, 0, 0])
    cloud = pv.PolyData(np.random.default_rng().random((100, 3)))
    assert len(cloud.center_of_mass()) == 3
    cloud['weights'] = np.random.default_rng().random(cloud.n_points)
    center = cloud.center_of_mass(scalars_weight=True)
    assert len(center) == 3


def test_project_points_to_plane():
    # Define a simple Gaussian surface
    n = 20
    x = np.linspace(-200, 200, num=n) + np.random.default_rng().uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=n) + np.random.default_rng().uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    poly = pv.StructuredGrid(xx, yy, zz).extract_geometry(progress_bar=True)
    poly['elev'] = zz.ravel(order='f')

    # Wrong normal length
    with pytest.raises(TypeError):
        poly.project_points_to_plane(normal=(0, 0, 1, 1))
    # allow Sequence but not Iterable
    with pytest.raises(TypeError):
        poly.project_points_to_plane(normal={0, 1, 2})

    # Test the filter
    projected = poly.project_points_to_plane(origin=poly.center, normal=(0, 0, 1))
    assert np.allclose(projected.points[:, -1], poly.center[-1])
    projected = poly.project_points_to_plane(normal=(0, 1, 1))
    assert projected.n_points

    # finally, test inplace
    poly.project_points_to_plane(normal=(0, 1, 1), inplace=True)
    assert np.allclose(poly.points, projected.points)


def test_tube(spline):
    # Simple
    line = pv.Line()
    tube = line.tube(n_sides=2, progress_bar=True)
    assert tube.n_points
    assert tube.n_cells

    # inplace
    line.tube(n_sides=2, inplace=True, progress_bar=True)
    assert np.allclose(line.points, tube.points)

    # Complicated
    tube = spline.tube(radius=0.5, scalars='arc_length', progress_bar=True)
    assert tube.n_points
    assert tube.n_cells

    # Complicated with absolute radius
    tube = spline.tube(radius=0.5, scalars='arc_length', absolute=True, progress_bar=True)
    assert tube.n_points
    assert tube.n_cells

    with pytest.raises(TypeError):
        spline.tube(scalars=range(10))


def test_smooth_inplace(sphere):
    orig_pts = sphere.points.copy()
    sphere.smooth(inplace=True, progress_bar=True)
    assert not np.allclose(orig_pts, sphere.points)


def test_delaunay_2d():
    n = 20
    x = np.linspace(-200, 200, num=n) + np.random.default_rng().uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=n) + np.random.default_rng().uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    # Get the points as a 2D NumPy array (N by 3)
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    pdata = pv.PolyData(points)
    surf = pdata.delaunay_2d(progress_bar=True)
    # Make sure we have an all triangle mesh now
    assert np.all(surf.faces.reshape((-1, 4))[:, 0] == 3)

    # test inplace
    pdata.delaunay_2d(inplace=True, progress_bar=True)
    assert np.allclose(pdata.points, surf.points)


def test_lines():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack((x, y, z))
    # Create line segments
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    assert poly.n_points == len(points)
    assert poly.n_cells == len(points) - 1
    # Create a poly line
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    assert poly.n_points == len(points)
    assert poly.n_cells == 1


def test_strips():
    # init with strips test
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0]])
    strips = np.array([4, 0, 1, 3, 2])
    strips_init = pv.PolyData(vertices, strips=strips)
    assert len(strips_init.strips) == len(strips)

    # add strips using the setter
    strips_setter = pv.PolyData(vertices)
    strips_setter.strips = strips
    assert len(strips_setter.strips) == len(strips)

    # test n_strips function
    strips = np.array([[4, 0, 1, 3, 2], [4, 1, 2, 3, 0]])
    strips_stack = np.hstack(strips)
    n_strips_test = pv.PolyData(vertices, strips=strips_stack)
    assert n_strips_test.n_strips == len(strips)


def test_ribbon_filter():
    line = examples.load_spline().compute_arc_length(progress_bar=True)
    ribbon = line.ribbon(width=0.5, scalars='arc_length')
    assert ribbon.n_points

    for tcoords in [True, 'lower', 'normalized', False]:
        ribbon = line.ribbon(width=0.5, tcoords=tcoords)
        assert ribbon.n_points


def test_is_all_triangles():
    # mesh points
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

    # mesh faces
    faces = np.hstack(
        [
            [4, 0, 1, 2, 3],
            [3, 0, 1, 4],
            [3, 1, 2, 4],
        ]
    )  # [square, triangle, triangle]

    mesh = pv.PolyData(vertices, faces)
    assert not mesh.is_all_triangles
    mesh = mesh.triangulate()
    assert mesh.is_all_triangles


def test_extrude():
    arc = pv.CircularArc(pointa=[-1, 0, 0], pointb=[1, 0, 0], center=[0, 0, 0])
    poly = arc.extrude([0, 0, 1], progress_bar=True, capping=True)
    assert poly.n_points
    assert poly.n_cells
    assert np.any(poly.strips)

    n_points_old = arc.n_points
    arc.extrude([0, 0, 1], inplace=True, capping=True)
    assert arc.n_points != n_points_old


def test_extrude_capping_warnings():
    arc = pv.CircularArc(pointa=[-1, 0, 0], pointb=[1, 0, 0], center=[0, 0, 0])
    with pytest.warns(PyVistaFutureWarning, match='default value of the ``capping`` keyword'):
        arc.extrude([0, 0, 1])
    with pytest.warns(PyVistaFutureWarning, match='default value of the ``capping`` keyword'):
        arc.extrude_rotate()


def test_flip_normals(sphere):
    with pytest.warns(
        PyVistaDeprecationWarning, match='`flip_normals` is deprecated. Use `flip_faces` instead'
    ):
        sphere.flip_normals()


@pytest.mark.parametrize('mesh', [pv.Sphere(), pv.Plane()])
def test_flip_normal_vectors(mesh):
    mesh = mesh.compute_normals()
    flipped = mesh.flip_normal_vectors(inplace=True, progress_bar=True)
    assert flipped is mesh

    flipped = mesh.flip_normal_vectors()
    assert flipped is not mesh

    assert np.allclose(flipped.point_data['Normals'], -mesh.point_data['Normals'])
    assert np.allclose(flipped.cell_data['Normals'], -mesh.cell_data['Normals'])

    # Test ordering is unaffected
    assert np.allclose(flipped.faces, mesh.faces)


@pytest.mark.parametrize('mesh', [pv.Sphere(), pv.Plane()])
def test_flip_faces(mesh):
    flipped = mesh.flip_faces(inplace=True, progress_bar=True)
    assert flipped is mesh

    flipped = mesh.flip_faces()
    assert flipped is not mesh

    assert np.allclose(flipped.regular_faces[0], mesh.regular_faces[0][::-1])

    # Test normals are unaffected
    assert np.allclose(flipped.point_data['Normals'], mesh.point_data['Normals'])


def test_n_verts():
    mesh = pv.PolyData([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert mesh.n_verts == 2


def test_n_lines():
    mesh = pv.Line()
    assert mesh.n_lines == 1


def test_n_faces_strict():
    # Mesh with one face and one line
    mesh = pv.PolyData(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[3, 0, 1, 2],
        lines=[2, 0, 1],
    )
    assert mesh.n_cells == 2  # n_faces + n_lines
    assert mesh.n_faces_strict == 1


@pytest.fixture
def default_n_faces():
    pv.PolyData._USE_STRICT_N_FACES = False
    yield
    pv.PolyData._USE_STRICT_N_FACES = False


def test_n_faces():
    if pv._version.version_info[:2] >= (0, 46):
        # At version 0.46, n_faces should raise an error instead of warning
        mesh = pv.PolyData(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[3, 0, 1, 2],
            lines=[2, 0, 1],
        )

        # Should raise an AttributeError
        with pytest.raises(
            AttributeError,
            match='The non-strict behavior of `pv.PolyData.n_faces` has been removed',
        ):
            _ = mesh.n_faces
    else:
        # Pre-0.46 behavior: warning
        mesh = pv.PolyData(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[3, 0, 1, 2],
            lines=[2, 0, 1],
        )

        # Should raise a warning the first time
        with pytest.warns(
            pv.PyVistaDeprecationWarning,
            match='The current behavior of `pv.PolyData.n_faces` has been deprecated',
        ):
            nf = mesh.n_faces

        # Current (deprecated) behavior is that n_faces is aliased to n_cells
        assert nf == mesh.n_cells

        # Shouldn't raise deprecation warning the second time
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            nf1 = mesh.n_faces

        assert nf1 == nf

    if pv._version.version_info[:2] > (0, 49):
        msg = 'Convert default n_faces behavior to strict'
        raise RuntimeError(msg)


def test_opt_in_n_faces_strict():
    pv.PolyData.use_strict_n_faces(True)
    mesh = pv.PolyData(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[3, 0, 1, 2],
        lines=[2, 0, 1],
    )
    assert mesh.n_faces == mesh.n_faces_strict


def test_geodesic_disconnected(sphere, sphere_shifted):
    # the sphere and sphere_shifted are disconnected - no path between them
    combined = sphere + sphere_shifted
    start_vertex = 0
    end_vertex = combined.n_points - 1
    match = f'There is no path between vertices {start_vertex} and {end_vertex}.'

    with pytest.raises(ValueError, match=match):
        combined.geodesic(start_vertex, end_vertex)

    with pytest.raises(ValueError, match=match):
        combined.geodesic_distance(start_vertex, end_vertex)


def test_tetrahedron_regular_faces():
    tetra = pv.Tetrahedron()
    assert np.array_equal(tetra.faces.reshape(-1, 4)[:, 1:], tetra.regular_faces)


@pytest.mark.parametrize('deep', [False, True])
def test_regular_faces(deep):
    points = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2], [0, 2, 3], [0, 3, 1]])
    mesh = pv.PolyData.from_regular_faces(points, faces, deep=deep)
    expected_faces = np.hstack([np.full((len(faces), 1), 3), faces]).astype(pv.ID_TYPE).flatten()
    assert np.array_equal(mesh.faces, expected_faces)
    assert np.array_equal(mesh.regular_faces, faces)


def test_set_regular_faces():
    mesh = pv.Tetrahedron()
    flipped_faces = mesh.regular_faces[:, ::-1]
    mesh.regular_faces = flipped_faces
    assert np.array_equal(mesh.regular_faces, flipped_faces)


def test_empty_regular_faces():
    mesh = pv.PolyData()
    assert np.array_equal(mesh.regular_faces, np.array([], dtype=pv.ID_TYPE))


def test_regular_faces_mutable():
    points = [[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]]
    faces = [[0, 1, 2]]
    mesh = pv.PolyData.from_regular_faces(points, faces)
    mesh.regular_faces[0, 2] = 3
    assert np.array_equal(mesh.faces, [3, 0, 1, 3])


def _assert_irregular_faces_equal(faces, expected):
    assert len(faces) == len(expected)
    assert all(map(np.array_equal, faces, expected))


def test_irregular_faces():
    points = [(1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0), (0, 0, 1.61)]
    faces = [(0, 1, 2, 3), (0, 3, 4), (0, 4, 1), (3, 2, 4), (2, 1, 4)]
    expected_faces = [4, 0, 1, 2, 3, 3, 0, 3, 4, 3, 0, 4, 1, 3, 3, 2, 4, 3, 2, 1, 4]
    mesh = pv.PolyData.from_irregular_faces(points, faces)
    assert np.array_equal(mesh.faces, expected_faces)
    _assert_irregular_faces_equal(mesh.irregular_faces, expected=faces)


def test_set_irregular_faces():
    mesh = pv.Pyramid().extract_surface()
    flipped_faces = tuple(f[::-1] for f in mesh.irregular_faces)
    mesh.irregular_faces = flipped_faces
    _assert_irregular_faces_equal(mesh.irregular_faces, flipped_faces)


def test_empty_irregular_faces():
    mesh = pv.PolyData()
    assert mesh.irregular_faces == ()


def test_irregular_faces_mutable():
    points = [(1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0), (0, 0, 1.61)]
    faces = [(0, 1, 2, 3), (0, 3, 4), (0, 4, 1), (3, 2, 4), (2, 1, 4)]
    mesh = pv.PolyData.from_irregular_faces(points, faces)
    mesh.irregular_faces[0][0] = 4
    expected = [(4, 1, 2, 3), *faces[1:]]
    _assert_irregular_faces_equal(mesh.irregular_faces, expected)


@pytest.mark.parametrize('cells', ['faces', 'lines', 'strips', 'verts'])
def test_n_faces_etc_deprecated(cells: str):
    n_cells = 'n_' + cells
    kwargs = {cells: [3, 0, 1, 2], n_cells: 1}  # e.g. specify faces and n_faces
    with pytest.raises(
        TypeError,
        match=f'PolyData constructor parameter `{n_cells}` is deprecated and no longer used',
    ):
        _ = pv.PolyData(np.zeros((3, 3)), **kwargs)
    if pv._version.version_info[:2] > (0, 48):
        msg = f'Remove `PolyData` `{n_cells} constructor kwarg'
        raise RuntimeError(msg)


@pytest.mark.parametrize('inplace', [True, False])
def test_merge_points(inplace):
    mesh = pv.Cylinder(resolution=4)
    assert mesh.n_points == 8 * 2
    output = mesh.merge_points(inplace=inplace)
    assert output.n_points == 8
    assert isinstance(mesh, pv.PolyData)
    assert (mesh is output) == inplace
