from math import pi
import os
import pathlib

import numpy as np
import pytest

import pyvista
from pyvista import examples
from pyvista.core.errors import NotAllTrianglesError
from pyvista.plotting import system_supports_plotting
from pyvista.utilities.misc import PyVistaFutureWarning

radius = 0.5

skip_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)


@pytest.fixture
def sphere():
    # this shadows the main sphere fixture from conftest!
    return pyvista.Sphere(radius, theta_resolution=10, phi_resolution=10)


@pytest.fixture
def sphere_shifted():
    return pyvista.Sphere(center=[0.5, 0.5, 0.5], theta_resolution=10, phi_resolution=10)


@pytest.fixture
def sphere_dense():
    return pyvista.Sphere(radius, theta_resolution=100, phi_resolution=100)


@pytest.fixture
def cube_dense():
    return pyvista.Cube()


test_path = os.path.dirname(os.path.abspath(__file__))


def is_binary(filename):
    """Return ``True`` when a file is binary."""
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
    with open(filename, 'rb') as f:
        data = f.read(1024)
    return bool(data.translate(None, textchars))


def test_init():
    mesh = pyvista.PolyData()
    assert not mesh.n_points
    assert not mesh.n_cells


def test_init_from_pdata(sphere):
    mesh = pyvista.PolyData(sphere, deep=True)
    assert mesh.n_points
    assert mesh.n_cells
    mesh.points[0] += 1
    assert not np.allclose(sphere.points[0], mesh.points[0])


def test_init_from_arrays():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

    # mesh faces
    faces = np.hstack([[4, 0, 1, 2, 3], [3, 0, 1, 4], [3, 1, 2, 4]]).astype(np.int8)

    mesh = pyvista.PolyData(vertices, faces)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3

    mesh = pyvista.PolyData(vertices, faces, deep=True)
    vertices[0] += 1
    assert not np.allclose(vertices[0], mesh.points[0])

    # ensure that polydata raises a warning when inputting non-float dtype
    with pytest.warns(Warning):
        mesh = pyvista.PolyData(vertices.astype(np.int32), faces)


def test_init_from_arrays_with_vert():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1], [0, 1.5, 1.5]])

    # mesh faces
    faces = np.hstack(
        [[4, 0, 1, 2, 3], [3, 0, 1, 4], [3, 1, 2, 4], [1, 5]]  # [quad, triangle, triangle, vertex]
    ).astype(np.int8)

    mesh = pyvista.PolyData(vertices, faces)
    assert mesh.n_points == 6
    assert mesh.n_cells == 4


def test_init_from_arrays_triangular():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

    # mesh faces
    faces = np.vstack([[3, 0, 1, 2], [3, 0, 1, 4], [3, 1, 2, 4]])

    mesh = pyvista.PolyData(vertices, faces)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3

    mesh = pyvista.PolyData(vertices, faces, deep=True)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3


def test_init_as_points():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

    mesh = pyvista.PolyData(vertices)
    assert mesh.n_points == vertices.shape[0]
    assert mesh.n_cells == vertices.shape[0]
    assert len(mesh.verts) == vertices.shape[0] * 2

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    cells = np.array([1, 0, 1, 1, 1, 2], np.int16)
    to_check = pyvista.PolyData._make_vertex_cells(len(vertices)).ravel()
    assert np.allclose(to_check, cells)

    # from list
    mesh.verts = [[1, 0], [1, 1], [1, 2]]
    to_check = pyvista.PolyData._make_vertex_cells(len(vertices)).ravel()
    assert np.allclose(to_check, cells)

    mesh = pyvista.PolyData()
    mesh.points = vertices
    mesh.verts = cells
    assert mesh.n_points == vertices.shape[0]
    assert mesh.n_cells == vertices.shape[0]
    assert np.allclose(mesh.verts, cells)


def test_init_as_points_from_list():
    points = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    mesh = pyvista.PolyData(points)
    assert np.allclose(mesh.points, points)


def test_invalid_init():
    with pytest.raises(ValueError):
        pyvista.PolyData(np.array([1.0]))

    with pytest.raises(TypeError):
        pyvista.PolyData([1.0, 2.0, 3.0], 'woa')

    with pytest.raises(ValueError):
        pyvista.PolyData('woa', 'woa')

    poly = pyvista.PolyData()
    with pytest.raises(ValueError):
        pyvista.PolyData(poly, 'woa')

    with pytest.raises(TypeError):
        pyvista.PolyData({'woa'})


def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        pyvista.PolyData('file.bad')

    with pytest.raises(IOError):
        filename = os.path.join(test_path, 'test_polydata.py')
        pyvista.PolyData(filename)


def test_lines_on_init():
    lines = [2, 0, 1, 3, 2, 3, 4]
    points = np.random.random((5, 3))
    pd = pyvista.PolyData(points, lines=lines)
    assert not pd.faces.size
    assert np.array_equal(pd.lines, lines)
    assert np.array_equal(pd.points, points)


def test_polydata_repr_str():
    pd = pyvista.PolyData()
    assert repr(pd) == str(pd)
    assert 'N Cells' in str(pd)
    assert 'N Points' in str(pd)
    assert 'X Bounds' in str(pd)
    assert 'N Arrays' in str(pd)


def test_geodesic(sphere):
    start, end = 0, sphere.n_points - 1
    geodesic = sphere.geodesic(start, end)
    assert isinstance(geodesic, pyvista.PolyData)
    assert "vtkOriginalPointIds" in geodesic.array_names
    ids = geodesic.point_data["vtkOriginalPointIds"]
    assert np.allclose(geodesic.points, sphere.points[ids])

    # check keep_order
    geodesic_legacy = sphere.geodesic(start, end, keep_order=False)
    assert geodesic_legacy["vtkOriginalPointIds"][0] == end
    geodesic_ordered = sphere.geodesic(start, end, keep_order=True)
    assert geodesic_ordered["vtkOriginalPointIds"][0] == start

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
        0, sphere.n_points - 1, use_scalar_weights=True
    )
    assert isinstance(distance_use_scalar_weights, float)


def test_ray_trace(sphere):
    points, ind = sphere.ray_trace([0, 0, 0], [1, 1, 1])
    assert np.any(points)
    assert np.any(ind)


@skip_plotting
def test_ray_trace_plot(sphere):
    points, ind = sphere.ray_trace(
        [0, 0, 0], [1, 1, 1], plot=True, first_point=True, off_screen=True
    )
    assert np.any(points)
    assert np.any(ind)


def test_multi_ray_trace(sphere):
    pytest.importorskip('rtree')
    pytest.importorskip('pyembree')
    pytest.importorskip('trimesh')
    origins = [[1, 0, 1], [0.5, 0, 1], [0.25, 0, 1], [0, 0, 1]]
    directions = [[0, 0, -1]] * 4
    points, ind_r, ind_t = sphere.multi_ray_trace(origins, directions, retry=True)
    assert np.any(points)
    assert np.any(ind_r)
    assert np.any(ind_t)

    # check non-triangulated
    mesh = pyvista.Cylinder()
    with pytest.raises(NotAllTrianglesError):
        mesh.multi_ray_trace(origins, directions)


@skip_plotting
def test_plot_curvature(sphere):
    sphere.plot_curvature(off_screen=True)


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


def test_boolean_difference(sphere, sphere_shifted):
    difference = sphere.boolean_difference(sphere_shifted, progress_bar=True)
    intersection = sphere.boolean_intersection(sphere_shifted, progress_bar=True)

    expected_volume = sphere.volume - intersection.volume
    assert np.isclose(difference.volume, expected_volume, atol=1e-3)


def test_boolean_difference_fail(plane):
    with pytest.raises(NotAllTrianglesError):
        plane - plane


def test_subtract(sphere, sphere_shifted):
    sub_mesh = sphere - sphere_shifted
    assert sub_mesh.n_points == sphere.boolean_difference(sphere_shifted).n_points


def test_merge(sphere, sphere_shifted, hexbeam):
    merged = sphere.merge(hexbeam, progress_bar=True)
    assert merged.n_points == (sphere.n_points + hexbeam.n_points)
    assert isinstance(merged, pyvista.UnstructuredGrid)

    # list with unstructuredgrid case
    merged = sphere.merge([hexbeam, hexbeam], merge_points=False, progress_bar=True)
    assert merged.n_points == (sphere.n_points + hexbeam.n_points * 2)
    assert isinstance(merged, pyvista.UnstructuredGrid)

    # with polydata
    merged = sphere.merge(sphere_shifted, progress_bar=True)
    assert isinstance(merged, pyvista.PolyData)
    assert merged.n_points == sphere.n_points + sphere_shifted.n_points

    # with polydata list (no merge)
    merged = sphere.merge([sphere_shifted, sphere_shifted], merge_points=False, progress_bar=True)
    assert isinstance(merged, pyvista.PolyData)
    assert merged.n_points == sphere.n_points + sphere_shifted.n_points * 2

    # with polydata list (merge)
    merged = sphere.merge([sphere_shifted, sphere_shifted], progress_bar=True)
    assert isinstance(merged, pyvista.PolyData)
    assert merged.n_points == sphere.n_points + sphere_shifted.n_points

    # test in-place merge
    mesh = sphere.copy()
    merged = mesh.merge(sphere_shifted, inplace=True)
    assert merged is mesh

    # test main_has_priority
    mesh = sphere.copy()
    data_main = np.arange(mesh.n_points, dtype=float)
    mesh.point_data['present_in_both'] = data_main
    other = mesh.copy()
    data_other = -data_main
    other.point_data['present_in_both'] = data_other
    merged = mesh.merge(other, main_has_priority=True)

    # note: order of points can change after point merging
    def matching_point_data(this, that, scalars_name):
        """Return True if scalars on two meshes only differ by point order."""
        return all(
            new_val == this.point_data[scalars_name][j]
            for point, new_val in zip(that.points, that.point_data[scalars_name])
            for j in (this.points == point).all(-1).nonzero()
        )

    assert matching_point_data(merged, mesh, 'present_in_both')
    merged = mesh.merge(other, main_has_priority=False)
    assert matching_point_data(merged, other, 'present_in_both')


def test_add(sphere, sphere_shifted):
    merged = sphere + sphere_shifted
    assert isinstance(merged, pyvista.PolyData)
    assert merged.n_points == sphere.n_points + sphere_shifted.n_points
    assert merged.n_faces == sphere.n_cells + sphere_shifted.n_cells


def test_intersection(sphere, sphere_shifted):
    intersection, first, second = sphere.intersection(
        sphere_shifted, split_first=True, split_second=True, progress_bar=True
    )

    assert intersection.n_points
    assert first.n_points > sphere.n_points
    assert second.n_points > sphere_shifted.n_points

    intersection, first, second = sphere.intersection(
        sphere_shifted, split_first=False, split_second=False, progress_bar=True
    )
    assert intersection.n_points
    assert first.n_points == sphere.n_points
    assert second.n_points == sphere_shifted.n_points


@pytest.mark.parametrize('curv_type', ['mean', 'gaussian', 'maximum', 'minimum'])
def test_curvature(sphere, curv_type):
    curv = sphere.curvature(curv_type)
    assert np.any(curv)
    assert curv.size == sphere.n_points


def test_invalid_curvature(sphere):
    with pytest.raises(ValueError):
        sphere.curvature('not valid')


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pyvista.core.pointset.PolyData._WRITERS)
def test_save(sphere, extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp{extension}'))
    sphere.save(filename, binary)

    if binary:
        if extension == '.vtp':
            assert 'binary' in open(filename).read(1000)
        else:
            is_binary(filename)
    else:
        with open(filename) as f:
            fst = f.read(100).lower()
            assert 'ascii' in fst or 'xml' in fst or 'solid' in fst

    mesh = pyvista.PolyData(filename)
    assert mesh.faces.shape == sphere.faces.shape
    assert mesh.points.shape == sphere.points.shape


@pytest.mark.parametrize('as_str', [True, False])
@pytest.mark.parametrize('ndim', [3, 4])
def test_save_ply_texture_array(sphere, ndim, as_str, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.ply'))

    texture = np.ones((sphere.n_points, ndim), np.uint8)
    texture[:, 2] = np.arange(sphere.n_points)[::-1]
    if as_str:
        sphere.point_data['texture'] = texture
        sphere.save(filename, texture='texture')
    else:
        sphere.save(filename, texture=texture)

    mesh = pyvista.PolyData(filename)
    color_array_name = 'RGB' if ndim == 3 else 'RGBA'
    assert np.allclose(mesh[color_array_name], texture)


@pytest.mark.parametrize('as_str', [True, False])
def test_save_ply_texture_array_catch(sphere, as_str, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.ply'))

    texture = np.ones((sphere.n_points, 3), np.float32)
    with pytest.raises(ValueError, match='Invalid datatype'):
        if as_str:
            sphere.point_data['texture'] = texture
            sphere.save(filename, texture='texture')
        else:
            sphere.save(filename, texture=texture)

    with pytest.raises(TypeError):
        sphere.save(filename, texture=[1, 2, 3])


def test_pathlib_read_write(tmpdir, sphere):
    path = pathlib.Path(str(tmpdir.mkdir("tmpdir").join('tmp.vtk')))
    sphere.save(path)
    assert path.is_file()

    mesh = pyvista.PolyData(path)
    assert mesh.faces.shape == sphere.faces.shape
    assert mesh.points.shape == sphere.points.shape

    mesh = pyvista.read(path)
    assert isinstance(mesh, pyvista.PolyData)
    assert mesh.faces.shape == sphere.faces.shape
    assert mesh.points.shape == sphere.points.shape


def test_invalid_save(sphere):
    with pytest.raises(ValueError):
        sphere.save('file.abc')


def test_triangulate_filter(plane):
    assert not plane.is_all_triangles
    plane.triangulate(inplace=True)
    assert plane.is_all_triangles
    # Make a point cloud and assert false
    assert not pyvista.PolyData(plane.points).is_all_triangles
    # Extract lines and make sure false
    assert not plane.extract_all_edges().is_all_triangles


@pytest.mark.parametrize('subfilter', ['butterfly', 'loop', 'linear'])
def test_subdivision(sphere, subfilter):
    mesh = sphere.subdivide(1, subfilter, progress_bar=True)
    assert mesh.n_points > sphere.n_points
    assert mesh.n_faces > sphere.n_faces

    mesh = sphere.copy()
    mesh.subdivide(1, subfilter, inplace=True)
    assert mesh.n_points > sphere.n_points
    assert mesh.n_faces > sphere.n_faces


def test_invalid_subdivision(sphere):
    with pytest.raises(ValueError):
        sphere.subdivide(1, 'not valid')

    # check non-triangulated
    mesh = pyvista.Cylinder()
    with pytest.raises(NotAllTrianglesError):
        mesh.subdivide(1)


def test_extract_feature_edges(sphere):
    # Test extraction of NO edges
    edges = sphere.extract_feature_edges(90)
    assert not edges.n_points

    mesh = pyvista.Cube()  # use a mesh that actually has strongly defined edges
    more_edges = mesh.extract_feature_edges(10)
    assert more_edges.n_points


def test_extract_feature_edges_no_data():
    mesh = pyvista.Wavelet()
    edges = mesh.extract_feature_edges(90, clear_data=True)
    assert edges is not None
    assert isinstance(edges, pyvista.PolyData)
    assert edges.n_arrays == 0


def test_decimate(sphere):
    mesh = sphere.decimate(0.5, progress_bar=True)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces < sphere.n_faces

    mesh.decimate(0.5, inplace=True, progress_bar=True)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces < sphere.n_faces

    # check non-triangulated
    mesh = pyvista.Cylinder()
    with pytest.raises(NotAllTrianglesError):
        mesh.decimate(0.5)


def test_decimate_pro(sphere):
    mesh = sphere.decimate_pro(0.5, progress_bar=True, max_degree=10)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces < sphere.n_faces

    mesh.decimate_pro(0.5, inplace=True, progress_bar=True)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces < sphere.n_faces

    # check non-triangulated
    mesh = pyvista.Cylinder()
    with pytest.raises(NotAllTrianglesError):
        mesh.decimate_pro(0.5)


def test_compute_normals(sphere):
    sphere_normals = sphere
    sphere_normals.compute_normals(inplace=True)

    point_normals = sphere_normals.point_data['Normals']
    cell_normals = sphere_normals.cell_data['Normals']
    assert point_normals.shape[0] == sphere.n_points
    assert cell_normals.shape[0] == sphere.n_cells


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


def test_point_normals(sphere):
    sphere = sphere.compute_normals(cell_normals=False, point_normals=True)

    # when `Normals` already exist, make sure they are returned
    normals = sphere.point_normals
    assert normals.shape[0] == sphere.n_points
    assert np.all(normals == sphere.point_data['Normals'])
    assert np.shares_memory(normals, sphere.point_data['Normals'])

    # when they don't, compute them
    sphere.point_data.pop('Normals')
    normals = sphere.point_normals
    assert normals.shape[0] == sphere.n_points


def test_cell_normals(sphere):
    sphere = sphere.compute_normals(cell_normals=True, point_normals=False)

    # when `Normals` already exist, make sure they are returned
    normals = sphere.cell_normals
    assert normals.shape[0] == sphere.n_cells
    assert np.all(normals == sphere.cell_data['Normals'])
    assert np.shares_memory(normals, sphere.cell_data['Normals'])

    # when they don't, compute them
    sphere.cell_data.pop('Normals')
    normals = sphere.cell_normals
    assert normals.shape[0] == sphere.n_cells


def test_face_normals(sphere):
    assert sphere.face_normals.shape[0] == sphere.n_faces


def test_clip_plane(sphere):
    clipped_sphere = sphere.clip(
        origin=[0, 0, 0], normal=[0, 0, -1], invert=False, progress_bar=True
    )
    faces = clipped_sphere.faces.reshape(-1, 4)[:, 1:]
    assert np.all(clipped_sphere.points[faces, 2] <= 0)

    sphere.clip(origin=[0, 0, 0], normal=[0, 0, -1], inplace=True, invert=False, progress_bar=True)
    faces = clipped_sphere.faces.reshape(-1, 4)[:, 1:]
    assert np.all(clipped_sphere.points[faces, 2] <= 0)


def test_extract_largest(sphere):
    mesh = sphere + pyvista.Sphere(0.1, theta_resolution=5, phi_resolution=5)
    largest = mesh.extract_largest()
    assert largest.n_faces == sphere.n_faces

    mesh.extract_largest(inplace=True)
    assert mesh.n_faces == sphere.n_faces


def test_clean(sphere):
    mesh = sphere.merge(sphere, merge_points=False).extract_surface()
    assert mesh.n_points > sphere.n_points
    cleaned = mesh.clean(merge_tol=1e-5)
    assert cleaned.n_points == sphere.n_points

    mesh.clean(merge_tol=1e-5, inplace=True)
    assert mesh.n_points == sphere.n_points

    cleaned = mesh.clean(point_merging=False)
    assert cleaned.n_points == mesh.n_points


def test_area(sphere_dense, cube_dense):
    radius = 0.5
    ideal_area = 4 * pi * radius**2
    assert np.isclose(sphere_dense.area, ideal_area, rtol=1e-3)

    ideal_area = 6 * np.cbrt(cube_dense.volume) ** 2
    assert np.isclose(cube_dense.area, ideal_area, rtol=1e-3)


def test_volume(sphere_dense):
    ideal_volume = (4 / 3.0) * pi * radius**3
    assert np.isclose(sphere_dense.volume, ideal_volume, rtol=1e-3)


@skip_plotting
def test_plot_boundaries():
    # make sure to plot an object that has boundaries
    pyvista.Cube().plot_boundaries(off_screen=True)


@skip_plotting
@pytest.mark.parametrize('flip', [True, False])
@pytest.mark.parametrize('faces', [True, False])
def test_plot_normals(sphere, flip, faces):
    sphere.plot_normals(off_screen=True, flip=flip, faces=faces)


def test_remove_points_any(sphere):
    remove_mask = np.zeros(sphere.n_points, np.bool_)
    remove_mask[:3] = True
    sphere_mod, ind = sphere.remove_points(remove_mask, inplace=False, mode='any')
    assert (sphere_mod.n_points + remove_mask.sum()) == sphere.n_points
    assert np.allclose(sphere_mod.points, sphere.points[ind])


def test_remove_points_all(sphere):
    sphere_copy = sphere.copy()
    sphere_copy.cell_data['ind'] = np.arange(sphere_copy.n_faces)
    remove = sphere.faces[1:4]
    sphere_copy.remove_points(remove, inplace=True, mode='all')
    assert sphere_copy.n_points == sphere.n_points
    assert sphere_copy.n_faces == sphere.n_faces - 1


def test_remove_points_fail(sphere, plane):
    # not triangles:
    with pytest.raises(NotAllTrianglesError):
        plane.remove_points([0])

    # invalid bool mask size
    with pytest.raises(ValueError):
        sphere.remove_points(np.ones(10, np.bool_))

    # invalid mask type
    with pytest.raises(TypeError):
        sphere.remove_points([0.0])


def test_vertice_cells_on_read(tmpdir):
    point_cloud = pyvista.PolyData(np.random.rand(100, 3))
    filename = str(tmpdir.mkdir("tmpdir").join('foo.ply'))
    point_cloud.save(filename)
    recovered = pyvista.read(filename)
    assert recovered.n_cells == 100
    recovered = pyvista.PolyData(filename)
    assert recovered.n_cells == 100


def test_center_of_mass(sphere):
    assert np.allclose(sphere.center_of_mass(), [0, 0, 0])
    cloud = pyvista.PolyData(np.random.rand(100, 3))
    assert len(cloud.center_of_mass()) == 3
    cloud['weights'] = np.random.rand(cloud.n_points)
    center = cloud.center_of_mass(True)
    assert len(center) == 3


def test_project_points_to_plane():
    # Define a simple Gaussian surface
    n = 20
    x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    poly = pyvista.StructuredGrid(xx, yy, zz).extract_geometry(progress_bar=True)
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
    line = pyvista.Line()
    tube = line.tube(n_sides=2, progress_bar=True)
    assert tube.n_points and tube.n_cells

    # inplace
    line.tube(n_sides=2, inplace=True, progress_bar=True)
    assert np.allclose(line.points, tube.points)

    # Complicated
    tube = spline.tube(radius=0.5, scalars='arc_length', progress_bar=True)
    assert tube.n_points and tube.n_cells

    # Complicated with absolute radius
    tube = spline.tube(radius=0.5, scalars='arc_length', absolute=True, progress_bar=True)
    assert tube.n_points and tube.n_cells

    with pytest.raises(TypeError):
        spline.tube(scalars=range(10))


def test_smooth_inplace(sphere):
    orig_pts = sphere.points.copy()
    sphere.smooth(inplace=True, progress_bar=True)
    assert not np.allclose(orig_pts, sphere.points)


def test_delaunay_2d():
    n = 20
    x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    # Get the points as a 2D NumPy array (N by 3)
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    pdata = pyvista.PolyData(points)
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
    poly = pyvista.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    assert poly.n_points == len(points)
    assert poly.n_cells == len(points) - 1
    # Create a poly line
    poly = pyvista.PolyData()
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
    strips_init = pyvista.PolyData(vertices, strips=strips)
    assert len(strips_init.strips) == len(strips)

    # add strips using the setter
    strips_setter = pyvista.PolyData(vertices)
    strips_setter.strips = strips
    assert len(strips_setter.strips) == len(strips)

    # test n_strips function
    strips = np.array([[4, 0, 1, 3, 2], [4, 1, 2, 3, 0]])
    strips_stack = np.hstack(strips)
    n_strips_test = pyvista.PolyData(vertices, strips=strips_stack)
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
    faces = np.hstack([[4, 0, 1, 2, 3], [3, 0, 1, 4], [3, 1, 2, 4]])  # [square, triangle, triangle]

    mesh = pyvista.PolyData(vertices, faces)
    assert not mesh.is_all_triangles
    mesh = mesh.triangulate()
    assert mesh.is_all_triangles
    # for backwards compatibility, check if we can call this
    assert mesh.is_all_triangles()


def test_extrude():
    arc = pyvista.CircularArc([-1, 0, 0], [1, 0, 0], [0, 0, 0])
    poly = arc.extrude([0, 0, 1], progress_bar=True, capping=True)
    assert poly.n_points
    assert poly.n_cells
    assert np.any(poly.strips)

    n_points_old = arc.n_points
    arc.extrude([0, 0, 1], inplace=True, capping=True)
    assert arc.n_points != n_points_old


def test_extrude_capping_warnings():
    arc = pyvista.CircularArc([-1, 0, 0], [1, 0, 0], [0, 0, 0])
    with pytest.warns(PyVistaFutureWarning, match='default value of the ``capping`` keyword'):
        arc.extrude([0, 0, 1])
    with pytest.warns(PyVistaFutureWarning, match='default value of the ``capping`` keyword'):
        arc.extrude_rotate()


def test_flip_normals(sphere, plane):
    sphere_flipped = sphere.copy()
    sphere_flipped.flip_normals()

    sphere.compute_normals(inplace=True)
    sphere_flipped.compute_normals(inplace=True)
    assert np.allclose(sphere_flipped.point_data['Normals'], -sphere.point_data['Normals'])

    # invalid case
    with pytest.raises(NotAllTrianglesError):
        plane.flip_normals()


def test_n_verts():
    mesh = pyvista.PolyData([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert mesh.n_verts == 2


def test_n_lines():
    mesh = pyvista.Line()
    assert mesh.n_lines == 1


def test_geodesic_disconnected(sphere, sphere_shifted):
    # the sphere and sphere_shifted are disconnected - no path between them
    combined = sphere + sphere_shifted
    start_vertex = 0
    end_vertex = combined.n_points - 1
    match = f"There is no path between vertices {start_vertex} and {end_vertex}."

    with pytest.raises(ValueError, match=match):
        combined.geodesic(start_vertex, end_vertex)

    with pytest.raises(ValueError, match=match):
        combined.geodesic_distance(start_vertex, end_vertex)
