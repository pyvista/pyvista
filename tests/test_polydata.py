import pathlib
import os
from math import pi

import numpy as np
import pytest

import pyvista
from pyvista import examples
from pyvista.plotting import system_supports_plotting
from pyvista.core.errors import NotAllTrianglesError

radius = 0.5

@pytest.fixture
def sphere():
    # this shadows the main sphere fixture from conftest!
    return pyvista.Sphere(radius, theta_resolution=10, phi_resolution=10)


@pytest.fixture
def sphere_shifted():
    return pyvista.Sphere(center=[0.5, 0.5, 0.5],
                          theta_resolution=10, phi_resolution=10)


@pytest.fixture
def sphere_dense():
    return pyvista.Sphere(radius, theta_resolution=100, phi_resolution=100)


@pytest.fixture
def cube_dense():
    return pyvista.Cube()


test_path = os.path.dirname(os.path.abspath(__file__))


def is_binary(filename):
    """Return ``True`` when a file is binary."""
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
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
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0.5, 0.5, -1]])

    # mesh faces
    faces = np.hstack([[4, 0, 1, 2, 3],
                       [3, 0, 1, 4],
                       [3, 1, 2, 4]]).astype(np.int8)

    mesh = pyvista.PolyData(vertices, faces)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3

    mesh = pyvista.PolyData(vertices, faces, deep=True)
    vertices[0] += 1
    assert not np.allclose(vertices[0], mesh.points[0])


def test_init_from_arrays_with_vert():
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0.5, 0.5, -1],
                         [0, 1.5, 1.5]])

    # mesh faces
    faces = np.hstack([[4, 0, 1, 2, 3],  # quad
                       [3, 0, 1, 4],     # triangle
                       [3, 1, 2, 4],     # triangle
                       [1, 5]]).astype(np.int8)  # vertex

    mesh = pyvista.PolyData(vertices, faces)
    assert mesh.n_points == 6
    assert mesh.n_cells == 4


def test_init_from_arrays_triangular():
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0.5, 0.5, -1]])

    # mesh faces
    faces = np.vstack([[3, 0, 1, 2],
                       [3, 0, 1, 4],
                       [3, 1, 2, 4]])

    mesh = pyvista.PolyData(vertices, faces)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3

    mesh = pyvista.PolyData(vertices, faces, deep=True)
    assert mesh.n_points == 5
    assert mesh.n_cells == 3


def test_init_as_points():
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0.5, 0.5, -1]])

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
    points = [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]
    mesh = pyvista.PolyData(points)
    assert np.allclose(mesh.points, points)


def test_invalid_init():
    with pytest.raises(ValueError):
        pyvista.PolyData(np.array([1]))

    with pytest.raises(TypeError):
        pyvista.PolyData([1, 2, 3], 'woa')

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
    ids = geodesic.point_arrays["vtkOriginalPointIds"]
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


def test_ray_trace(sphere):
    points, ind = sphere.ray_trace([0, 0, 0], [1, 1, 1])
    assert np.any(points)
    assert np.any(ind)


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_ray_trace_plot(sphere):
    points, ind = sphere.ray_trace([0, 0, 0], [1, 1, 1], plot=True, first_point=True,
                                   off_screen=True)
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


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_plot_curvature(sphere):
    cpos = sphere.plot_curvature(off_screen=True)
    assert isinstance(cpos, pyvista.CameraPosition)


def test_edge_mask(sphere):
    mask = sphere.edge_mask(10)


def test_boolean_cut_inplace(sphere, sphere_shifted):
    sub_mesh = sphere
    sub_mesh.boolean_cut(sphere_shifted, inplace=True)
    assert sub_mesh.n_points
    assert sub_mesh.n_cells


def test_boolean_cut_fail(plane):
    with pytest.raises(NotAllTrianglesError):
        plane - plane


def test_subtract(sphere, sphere_shifted):
    sub_mesh = sphere - sphere_shifted
    assert sub_mesh.n_points
    assert sub_mesh.n_cells


def test_add(sphere, sphere_shifted):
    add_mesh = sphere + sphere_shifted

    npoints = sphere.n_points + sphere_shifted.n_points
    assert add_mesh.n_points == npoints

    nfaces = sphere.n_cells + sphere_shifted.n_cells
    assert add_mesh.n_faces == nfaces


def test_boolean_add_inplace(sphere, sphere_shifted):
    sub_mesh = sphere
    sub_mesh.boolean_add(sphere_shifted, inplace=True)
    assert sub_mesh.n_points
    assert sub_mesh.n_cells


def test_boolean_union_inplace(sphere, sphere_shifted):
    sub_mesh = sphere.boolean_union(sphere_shifted)
    assert sub_mesh.n_points
    assert sub_mesh.n_cells

    sub_mesh = sphere
    sub_mesh.boolean_union(sphere_shifted, inplace=True)
    assert sub_mesh.n_points
    assert sub_mesh.n_cells


def test_boolean_difference(sphere, sphere_shifted):
    sub_mesh = sphere.copy()
    sub_mesh.boolean_difference(sphere_shifted, inplace=True)
    assert sub_mesh.n_points
    assert sub_mesh.n_cells

    sub_mesh = sphere.boolean_difference(sphere_shifted)
    assert sub_mesh.n_points
    assert sub_mesh.n_cells


def test_intersection(sphere, sphere_shifted):
    intersection, first, second = sphere.intersection(sphere_shifted, split_first=True, split_second=True)

    assert intersection.n_points
    assert first.n_points > sphere.n_points
    assert second.n_points > sphere_shifted.n_points

    intersection, first, second = sphere.intersection(sphere_shifted, split_first=False, split_second=False)
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
        curv = sphere.curvature('not valid')


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
    assert not plane.is_all_triangles()
    plane.triangulate(inplace=True)
    assert plane.is_all_triangles()
    # Make a point cloud and assert false
    assert not pyvista.PolyData(plane.points).is_all_triangles()
    # Extract lines and make sure false
    assert not plane.extract_all_edges().is_all_triangles()


@pytest.mark.parametrize('subfilter', ['butterfly', 'loop', 'linear'])
def test_subdivision(sphere, subfilter):
    mesh = sphere.subdivide(1, subfilter)
    assert mesh.n_points > sphere.n_points
    assert mesh.n_faces > sphere.n_faces

    mesh = sphere.copy()
    mesh.subdivide(1, subfilter, inplace=True)
    assert mesh.n_points > sphere.n_points
    assert mesh.n_faces > sphere.n_faces


def test_invalid_subdivision(sphere):
    with pytest.raises(ValueError):
        mesh = sphere.subdivide(1, 'not valid')


def test_extract_feature_edges(sphere):
    # Test extraction of NO edges
    edges = sphere.extract_feature_edges(90)
    assert not edges.n_points

    mesh = pyvista.Cube() # use a mesh that actually has strongly defined edges
    more_edges = mesh.extract_feature_edges(10)
    assert more_edges.n_points

    mesh.extract_feature_edges(10, inplace=True)
    assert mesh.n_points == more_edges.n_points


def test_decimate(sphere):
    mesh = sphere.decimate(0.5)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces < sphere.n_faces

    mesh.decimate(0.5, inplace=True)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces < sphere.n_faces


def test_decimate_pro(sphere):
    mesh = sphere.decimate_pro(0.5)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces < sphere.n_faces

    mesh.decimate_pro(0.5, inplace=True)
    assert mesh.n_points < sphere.n_points
    assert mesh.n_faces < sphere.n_faces


def test_compute_normals(sphere):
    sphere_normals = sphere
    sphere_normals.compute_normals(inplace=True)

    point_normals = sphere_normals.point_arrays['Normals']
    cell_normals = sphere_normals.cell_arrays['Normals']
    assert point_normals.shape[0] == sphere.n_points
    assert cell_normals.shape[0] == sphere.n_cells


def test_point_normals(sphere):
    assert sphere.point_normals.shape[0] == sphere.n_points


def test_cell_normals(sphere):
    assert sphere.cell_normals.shape[0] == sphere.n_cells


def test_face_normals(sphere):
    assert sphere.face_normals.shape[0] == sphere.n_faces


def test_clip_plane(sphere):
    clipped_sphere = sphere.clip(origin=[0, 0, 0], normal=[0, 0, -1], invert=False)
    faces = clipped_sphere.faces.reshape(-1, 4)[:, 1:]
    assert np.all(clipped_sphere.points[faces, 2] <= 0)

    sphere.clip(origin=[0, 0, 0], normal=[0, 0, -1], inplace=True, invert=False)
    faces = clipped_sphere.faces.reshape(-1, 4)[:, 1:]
    assert np.all(clipped_sphere.points[faces, 2] <= 0)


def test_extract_largest(sphere):
    mesh = sphere + pyvista.Sphere(0.1, theta_resolution=5, phi_resolution=5)
    largest = mesh.extract_largest()
    assert largest.n_faces == sphere.n_faces

    mesh.extract_largest(inplace=True)
    assert mesh.n_faces == sphere.n_faces


def test_clean(sphere):
    mesh = sphere + sphere
    assert mesh.n_points > sphere.n_points
    cleaned = mesh.clean(merge_tol=1E-5)
    assert cleaned.n_points == sphere.n_points

    mesh.clean(merge_tol=1E-5, inplace=True)
    assert mesh.n_points == sphere.n_points

    cleaned = mesh.clean(point_merging=False)
    assert cleaned.n_points == mesh.n_points


def test_area(sphere_dense, cube_dense):
    radius = 0.5
    ideal_area = 4*pi*radius**2
    assert np.isclose(sphere_dense.area, ideal_area, rtol=1E-3)

    ideal_area = 6*np.cbrt(cube_dense.volume)**2
    assert np.isclose(cube_dense.area, ideal_area, rtol=1E-3)


def test_volume(sphere_dense):
    ideal_volume = (4/3.0)*pi*radius**3
    assert np.isclose(sphere_dense.volume, ideal_volume, rtol=1E-3)


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_plot_boundaries():
    # make sure to plot an object that has boundaries
    pyvista.Cube().plot_boundaries(off_screen=True)


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
@pytest.mark.parametrize('flip', [True, False])
def test_plot_normals(sphere, flip):
    sphere.plot_normals(off_screen=True, flip=flip)


def test_remove_points_any(sphere):
    remove_mask = np.zeros(sphere.n_points, np.bool_)
    remove_mask[:3] = True
    sphere_mod, ind = sphere.remove_points(remove_mask, inplace=False, mode='any')
    assert (sphere_mod.n_points + remove_mask.sum()) == sphere.n_points
    assert np.allclose(sphere_mod.points, sphere.points[ind])


def test_remove_points_all(sphere):
    sphere_copy = sphere.copy()
    sphere_copy.cell_arrays['ind'] = np.arange(sphere_copy.n_faces)
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
    x = np.linspace(-200,200, num=n) + np.random.uniform(-5, 5, size=n)
    y = np.linspace(-200,200, num=n) + np.random.uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A*np.exp(-0.5*((xx/b)**2. + (yy/b)**2.))
    poly = pyvista.StructuredGrid(xx, yy, zz).extract_geometry()
    poly['elev'] = zz.ravel(order='f')

    # Wrong normal length
    with pytest.raises(TypeError):
        poly.project_points_to_plane(normal=(0, 0, 1, 1))
    # allow Sequence but not Iterable
    with pytest.raises(TypeError):
        poly.project_points_to_plane(normal={0, 1, 2})

    # Test the filter
    projected = poly.project_points_to_plane(origin=poly.center, normal=(0,0,1))
    assert np.allclose(projected.points[:,-1], poly.center[-1])
    projected = poly.project_points_to_plane(normal=(0,1,1))
    assert projected.n_points

    # finally, test inplace
    poly.project_points_to_plane(normal=(0,1,1), inplace=True)
    assert np.allclose(poly.points, projected.points)


def test_tube(spline):
    # Simple
    line = pyvista.Line()
    tube = line.tube(n_sides=2)
    assert tube.n_points, tube.n_cells

    # inplace
    line.tube(n_sides=2, inplace=True)
    assert np.allclose(line.points, tube.points)

    # Complicated
    tube = spline.tube(radius=0.5, scalars='arc_length')
    assert tube.n_points, tube.n_cells

    with pytest.raises(TypeError):
        spline.tube(scalars=range(10))


def test_smooth_inplace(sphere):
    orig_pts = sphere.points.copy()
    sphere.smooth(inplace=True)
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
    surf = pdata.delaunay_2d()
    # Make sure we have an all triangle mesh now
    assert np.all(surf.faces.reshape((-1, 4))[:, 0] == 3)

    # test inplace
    pdata.delaunay_2d(inplace=True)
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
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
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


def test_ribbon_filter():
    line = examples.load_spline().compute_arc_length()
    ribbon = line.ribbon(width=0.5, scalars='arc_length')
    assert ribbon.n_points

    for tcoords in [True, 'lower', 'normalized', False]:
        ribbon = line.ribbon(width=0.5, tcoords=tcoords)
        assert ribbon.n_points


def test_is_all_triangles():
    # mesh points
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0.5, 0.5, -1]])

    # mesh faces
    faces = np.hstack([[4, 0, 1, 2, 3],  # square
                       [3, 0, 1, 4],     # triangle
                       [3, 1, 2, 4]])    # triangle

    mesh = pyvista.PolyData(vertices, faces)
    assert not mesh.is_all_triangles()
    mesh = mesh.triangulate()
    assert mesh.is_all_triangles()


def test_extrude():
    arc = pyvista.CircularArc([-1, 0, 0], [1, 0, 0], [0, 0, 0])
    poly = arc.extrude([0, 0, 1])
    assert poly.n_points
    assert poly.n_cells

    n_points_old = arc.n_points
    arc.extrude([0, 0, 1], inplace=True)
    assert arc.n_points != n_points_old


def test_flip_normals(sphere):
    sphere_flipped = sphere.copy()
    sphere_flipped.flip_normals()

    sphere.compute_normals(inplace=True)
    sphere_flipped.compute_normals(inplace=True)
    assert np.allclose(sphere_flipped.point_arrays['Normals'],
                       -sphere.point_arrays['Normals'])
