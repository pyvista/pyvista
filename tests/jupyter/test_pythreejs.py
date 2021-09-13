"""Test optional pythreejs functionality"""

import pytest
import numpy as np

try:
    import pythreejs
except:
    pytestmark = pytest.mark.skip

import pyvista
from pyvista.jupyter import pv_pythreejs


def test_set_jupyter_backend_ipygany():
    try:
        pyvista.global_theme.jupyter_backend = 'pythreejs'
        assert pyvista.global_theme.jupyter_backend == 'pythreejs'
    finally:
        pyvista.global_theme.jupyter_backend = None


def test_export_to_html(cube, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp.html'))

    pl = pyvista.Plotter()
    pl.add_mesh(cube)
    pl.export_html(filename)

    raw = open(filename).read()
    assert 'jupyter-threejs' in raw

    # expect phong material
    assert '"model_name": "MeshPhongMaterialModel"' in raw

    # at least a single instance of lighting
    assert 'DirectionalLightModel' in raw


def test_segment_poly_cells(spline):
    cells = pv_pythreejs.segment_poly_cells(spline)

    assert cells.shape[0] == spline.lines[0] - 1
    assert (np.diff(cells, axis=0) == 1).all()
    assert (np.diff(cells, axis=1) == 1).all()


def test_buffer_normals(sphere):
    buf_attr = pv_pythreejs.buffer_normals(sphere)
    assert np.allclose(buf_attr.array, sphere.point_normals)

    sphere_w_normals = sphere.compute_normals()
    buf_attr = pv_pythreejs.buffer_normals(sphere_w_normals)
    assert np.allclose(buf_attr.array, sphere.point_normals)


def test_get_coloring(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_points))
    assert pv_pythreejs.get_coloring(pl.mapper, sphere) == 'VertexColors'

    sphere.clear_data()

    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_cells))
    assert pv_pythreejs.get_coloring(pl.mapper, sphere) == 'FaceColors'


def test_output_point_cloud(sphere):
    pl = pyvista.Plotter()
    pl.add_points(sphere.points, scalars=range(sphere.n_points))
    pv_pythreejs.convert_plotter(pl)

    # points but no active point scalars
    pl = pyvista.Plotter()
    pl.add_points(sphere, scalars=range(sphere.n_cells))
    pv_pythreejs.convert_plotter(pl)


def test_output_face_scalars(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_cells))
    pv_pythreejs.convert_plotter(pl)


@pytest.mark.parametrize('max_index', [np.iinfo(np.uint16).max - 1,
                                       np.iinfo(np.uint32).max - 1,
                                       np.iinfo(np.uint32).max + 1])
def test_cast_to_min_size(max_index):

    if max_index < np.iinfo(np.uint16).max:
        buf_attr = pv_pythreejs.cast_to_min_size(np.arange(1000), max_index)
        assert buf_attr.array.dtype == np.uint16
    elif max_index < np.iinfo(np.uint32).max:
        buf_attr = pv_pythreejs.cast_to_min_size(np.arange(1000), max_index)
        assert buf_attr.array.dtype == np.uint32
    else:
        with pytest.raises(ValueError):
            buf_attr = pv_pythreejs.cast_to_min_size(np.arange(1000), max_index)


def test_pbr(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_cells), pbr=True)
    pv_pythreejs.convert_plotter(pl)


def test_output_face_scalars_rgba(sphere):
    colors = np.linspace(0, 255, sphere.n_cells)
    rgba = np.vstack((colors, colors, colors)).T

    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=rgba, rgba=True)
    pv_pythreejs.convert_plotter(pl)


def test_output_point_scalars(sphere):
    scalars = np.linspace(0, 255, sphere.n_points)
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=scalars)
    pv_pythreejs.convert_plotter(pl)


def test_output_point_scalars_rgba(sphere):
    colors = np.linspace(0, 255, sphere.n_points)
    rgba = np.vstack((colors, colors, colors)).T

    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=rgba, rgba=True)
    pv_pythreejs.convert_plotter(pl)


def test_output_texture(globe):
    pl = pyvista.Plotter()
    pl.add_mesh(globe)
    pv_pythreejs.convert_plotter(pl)


def test_no_lighting(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, lighting=False)
    pv_pythreejs.convert_plotter(pl)


def test_show_edges(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, show_edges=True)
    pv_pythreejs.convert_plotter(pl)


def test_not_polydata(hexbeam):
    pl = pyvista.Plotter()
    pl.add_mesh(hexbeam)
    pv_pythreejs.convert_plotter(pl)


@pytest.mark.parametrize('with_scalars', [False, True])
def test_output_wireframe(sphere, with_scalars):
    pl = pyvista.Plotter()
    if with_scalars:
        pl.add_mesh(sphere, scalars=range(sphere.n_points), style='wireframe')
    else:
        pl.add_mesh(sphere, style='wireframe')
    pv_pythreejs.convert_plotter(pl)


@pytest.mark.parametrize('with_scalars', [False, True])
def test_output_lines(spline, with_scalars):
    pl = pyvista.Plotter()
    if with_scalars:
        pl.add_mesh(spline, scalars=range(spline.n_points), style='wireframe')
    else:
        pl.add_mesh(spline, style='wireframe')
    pv_pythreejs.convert_plotter(pl)


def test_add_axes():
    pl = pyvista.Plotter()
    pl.add_axes()
    pv_pythreejs.convert_plotter(pl)


def test_grid_layout():
    pl = pyvista.Plotter(shape=(2, 2))
    pv_pythreejs.convert_plotter(pl)


def test_export_after_show():
    pl = pyvista.Plotter(shape=(2, 2))

    # deleting rather than showing to save time
    del pl.ren_win

    with pytest.raises(AttributeError):
        pv_pythreejs.convert_plotter(pl)


def test_non_standard_shape():
    pl = pyvista.Plotter(shape='2|3')
    with pytest.raises(RuntimeError, match='Unsupported plotter shape'):
        pv_pythreejs.convert_plotter(pl)
