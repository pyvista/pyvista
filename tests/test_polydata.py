import os
from math import pi

import pytest
import numpy as np

import vtki
from vtki import examples
from vtki.plotting import running_xserver

radius = 0.5
sphere = vtki.Sphere(radius, theta_resolution=10, phi_resolution=10)

sphere_shifted = vtki.Sphere(center=[0.5, 0.5, 0.5],
                             theta_resolution=10, phi_resolution=10)

dense_sphere = vtki.Sphere(radius, theta_resolution=100, phi_resolution=100)

try:
    test_path = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(test_path, 'test_data')
except:
    test_data_path = '/home/alex/afrl/python/source/vtki/tests/test_data'

stl_test_file = os.path.join(test_data_path, 'sphere.stl')
ply_test_file = os.path.join(test_data_path, 'sphere.ply')
vtk_test_file = os.path.join(test_data_path, 'sphere.vtk')
test_files = [stl_test_file,
              ply_test_file,
              vtk_test_file]


def test_init():
    mesh = vtki.PolyData()
    assert not mesh.number_of_points
    assert not mesh.number_of_cells


def test_init_from_pdata():
    mesh = vtki.PolyData(sphere, deep=True)
    assert mesh.number_of_points
    assert mesh.number_of_cells
    mesh.points[0] += 1
    assert not np.allclose(sphere.points[0], mesh.points[0])


# @pytest.mark.parametrize('filename', test_files)
# def test_init_from_file(filename):
#     mesh = vtki.PolyData(filename)
#     assert mesh.faces.shape == sphere.faces.shape
#     assert mesh.points.shape == sphere.points.shape


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

    mesh = vtki.PolyData(vertices, faces)
    assert mesh.number_of_points == 5
    assert mesh.number_of_cells == 3

    mesh = vtki.PolyData(vertices, faces, deep=True)
    vertices[0] += 1
    assert not np.allclose(vertices[0], mesh.points[0])


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

    mesh = vtki.PolyData(vertices, faces)
    assert mesh.number_of_points == 5
    assert mesh.number_of_cells == 3


    mesh = vtki.PolyData(vertices, faces, deep=True)
    assert mesh.number_of_points == 5
    assert mesh.number_of_cells == 3


def test_init_as_points():
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0.5, 0.5, -1]])

    mesh = vtki.PolyData(vertices)
    assert mesh.number_of_points == vertices.shape[0]
    assert mesh.number_of_cells == vertices.shape[0]


def test_invalid_init():
    with pytest.raises(ValueError):
        mesh = vtki.PolyData(np.array([1]))

    with pytest.raises(TypeError):
        mesh = vtki.PolyData(np.array([1]), 'woa')

    with pytest.raises(TypeError):
        mesh = vtki.PolyData('woa', 'woa')

    with pytest.raises(TypeError):
        mesh = vtki.PolyData('woa', 'woa', 'woa')


def test_invalid_file():
    with pytest.raises(Exception):
        mesh = vtki.PolyData('file.bad')

    with pytest.raises(TypeError):
        filename = os.path.join(test_path, 'test_polydata.py')
        mesh = vtki.PolyData(filename)

    # with pytest.raises(Exception):
        # vtki.PolyData(examples.hexbeamfile)


def test_ray_trace():
    points, ind = sphere.ray_trace([0, 0, 0], [1, 1, 1])
    assert np.any(points)
    assert np.any(ind)


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_ray_trace_plot():
    points, ind = sphere.ray_trace([0, 0, 0], [1, 1, 1], plot=True, first_point=True,
                                   off_screen=True)
    assert np.any(points)
    assert np.any(ind)


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_plot_curvature():
    cpos = sphere.plot_curvature(off_screen=True)
    assert isinstance(cpos, list)


def test_edge_mask():
    mask = sphere.edge_mask(10)


def test_boolean_cut_inplace():
    sub_mesh = sphere.copy()
    sub_mesh.boolean_cut(sphere_shifted, inplace=True)
    assert sub_mesh.number_of_points
    assert sub_mesh.number_of_cells


def test_subtract():
    sub_mesh = sphere - sphere_shifted
    assert sub_mesh.number_of_points
    assert sub_mesh.number_of_cells


def test_add():
    add_mesh = sphere + sphere_shifted

    npoints = sphere.number_of_points + sphere_shifted.number_of_points
    assert add_mesh.number_of_points == npoints

    nfaces = sphere.number_of_cells + sphere_shifted.number_of_cells
    assert add_mesh.number_of_faces == nfaces


def test_boolean_add_inplace():
    sub_mesh = sphere.copy()
    sub_mesh.boolean_add(sphere_shifted, inplace=True)
    assert sub_mesh.number_of_points
    assert sub_mesh.number_of_cells


def test_boolean_union_inplace():
    sub_mesh = sphere.boolean_union(sphere_shifted)
    assert sub_mesh.number_of_points
    assert sub_mesh.number_of_cells

    sub_mesh = sphere.copy()
    sub_mesh.boolean_union(sphere_shifted, inplace=True)
    assert sub_mesh.number_of_points
    assert sub_mesh.number_of_cells


def test_boolean_difference():
    sub_mesh = sphere.copy()
    sub_mesh.boolean_difference(sphere_shifted, inplace=True)
    assert sub_mesh.number_of_points
    assert sub_mesh.number_of_cells

    sub_mesh = sphere.boolean_difference(sphere_shifted)
    assert sub_mesh.number_of_points
    assert sub_mesh.number_of_cells


@pytest.mark.parametrize('curv_type', ['mean', 'gaussian', 'maximum', 'minimum'])
def test_curvature(curv_type):
    curv = sphere.curvature(curv_type)
    assert np.any(curv)
    assert curv.size == sphere.number_of_points


def test_invalid_curvature():
    with pytest.raises(Exception):
        curv = sphere.curvature('not valid')


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['stl', 'vtk', 'ply'])
def test_save(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
    sphere.save(filename, binary)

    mesh = vtki.PolyData(filename)
    assert mesh.faces.shape == sphere.faces.shape
    assert mesh.points.shape == sphere.points.shape

def test_invalid_save():
    with pytest.raises(Exception):
        sphere.save('file.abc')


def test_tri_filter():
    arrow = vtki.Arrow([0, 0, 0], [1, 1, 1])
    assert arrow.faces.size % 4 
    arrow.tri_filter(inplace=True)
    assert not(arrow.faces.size % 4)


@pytest.mark.parametrize('subfilter', ['butterfly', 'loop', 'linear'])
def test_subdivision(subfilter):
    mesh = sphere.subdivide(1, subfilter)
    assert mesh.number_of_points > sphere.number_of_points
    assert mesh.number_of_faces > sphere.number_of_faces

    mesh = sphere.copy()
    mesh.subdivide(1, subfilter, inplace=True)
    assert mesh.number_of_points > sphere.number_of_points
    assert mesh.number_of_faces > sphere.number_of_faces


def test_invalid_subdivision():
    with pytest.raises(Exception):
        mesh = sphere.subdivide(1, 'not valid')


def test_extract_edges():
    edges = sphere.extract_edges(90)
    assert not edges.number_of_points

    more_edges = sphere.extract_edges(10)
    assert more_edges.number_of_points


def test_decimate():
    mesh = sphere.copy()
    mesh.decimate(0.5)
    assert mesh.number_of_points < sphere.number_of_points
    assert mesh.number_of_faces < sphere.number_of_faces

    mesh = sphere.decimate(0.5, inplace=False)
    assert mesh.number_of_points < sphere.number_of_points
    assert mesh.number_of_faces < sphere.number_of_faces


def test_center_of_mass():
    assert np.allclose(sphere.center_of_mass(), [0, 0, 0])


def test_compute_normals():
    sphere_normals = sphere.copy()
    sphere_normals.compute_normals()

    point_normals = sphere_normals.point_arrays['Normals']
    cell_normals = sphere_normals.cell_arrays['Normals']
    assert point_normals.shape[0] == sphere.number_of_points
    assert cell_normals.shape[0] == sphere.number_of_cells


def test_point_normals():
    assert sphere.point_normals.shape[0] == sphere.number_of_points


def test_cell_normals():
    assert sphere.cell_normals.shape[0] == sphere.number_of_cells


def test_face_normals():
    assert sphere.face_normals.shape[0] == sphere.number_of_faces


def test_clip_plane():
    clipped_sphere = sphere.copy()
    clipped_sphere.clip_with_plane([0, 0, 0], [0, 0, -1])
    faces = clipped_sphere.faces.reshape(-1 , 4)[:, 1:]
    assert np.all(clipped_sphere.points[faces, 2] <= 0)

    clipped_sphere = sphere.clip_with_plane([0, 0, 0], [0, 0, -1], inplace=False)
    faces = clipped_sphere.faces.reshape(-1 , 4)[:, 1:]
    assert np.all(clipped_sphere.points[faces, 2] <= 0)


def test_extract_largest():
    mesh = sphere + vtki.Sphere(0.1, theta_resolution=5, phi_resolution=5)
    largest = mesh.extract_largest()
    assert largest.number_of_faces == sphere.number_of_faces

    mesh.extract_largest(inplace=True)
    assert mesh.number_of_faces == sphere.number_of_faces


def test_clean():
    mesh = sphere + sphere
    assert mesh.number_of_points > sphere.number_of_points
    cleaned = mesh.clean(merge_tol=1E-5, inplace=False)
    assert cleaned.number_of_points == sphere.number_of_points

    mesh.clean()
    assert mesh.number_of_points == sphere.number_of_points


def test_area():
    radius = 0.5
    ideal_area = 4*pi*radius**2
    assert np.isclose(dense_sphere.area, ideal_area, rtol=1E-3)


def test_volume():
    ideal_volume = (4/3.0)*pi*radius**3
    assert np.isclose(dense_sphere.volume, ideal_volume, rtol=1E-3)


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_plot_boundaries():
    sphere.plot_boundaries(off_screen=True)


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_plot_normals():
    sphere.plot_normals(off_screen=True)


def test_remove_points_any():
    remove_mask = np.zeros(sphere.number_of_points, np.bool)
    remove_mask[:3] = True
    sphere_mod, ind = sphere.remove_points(remove_mask, inplace=False, mode='any')
    assert sphere_mod.number_of_points + remove_mask.sum() == sphere.number_of_points
    assert np.allclose(sphere_mod.points, sphere.points[ind])


def test_remove_points_all():
    sphere_copy = sphere.copy()
    sphere_copy.cell_arrays['ind'] = np.arange(sphere_copy.number_of_faces)
    remove = sphere.faces[1:4]
    sphere_copy.remove_points(remove, inplace=True, mode='all')
    assert sphere_copy.number_of_points == sphere.number_of_points
    assert sphere_copy.number_of_faces == sphere.number_of_faces - 1


def test_remove_points_fail():
    arrow = vtki.Arrow([0, 0, 0], [1, 0, 0])
    with pytest.raises(Exception):
        arrow.remove_points(range(10))
