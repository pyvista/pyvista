"""Test optional pythreejs functionality"""

import numpy as np
import pytest

try:
    import pythreejs  # noqa
except:  # noqa: E722
    pytestmark = pytest.mark.skip

import pyvista
from pyvista.jupyter import pv_pythreejs
from pyvista.utilities.misc import PyVistaDeprecationWarning


def test_set_jupyter_backend_threejs():
    try:
        with pytest.warns(PyVistaDeprecationWarning):
            pyvista.global_theme.jupyter_backend = 'pythreejs'
        assert pyvista.global_theme.jupyter_backend == 'pythreejs'
    finally:
        pyvista.global_theme.jupyter_backend = None


def test_export_to_html(sphere, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.html'))

    pl = pyvista.Plotter(shape=(1, 2))
    pl.add_text("Sphere 1\n", font_size=30, color='grey')
    pl.add_mesh(sphere, show_edges=False, color='grey', culling='back')

    pl.subplot(0, 1)
    pl.add_text("Sphere 2\n", font_size=30, color='grey')
    pl.add_mesh(sphere, show_edges=False, color='grey', culling='front')
    pl.link_views()

    with pytest.raises(ValueError, match="Invalid backend"):
        pl.export_html(filename, backend='not-a-valid-backend')

    pl.export_html(filename)

    raw = open(filename).read()
    assert 'jupyter-threejs' in raw

    # expect phong material
    assert '"model_name": "MeshPhongMaterialModel"' in raw

    # at least a single instance of lighting
    assert 'DirectionalLightModel' in raw


def test_export_to_html_composite(tmpdir):
    filename = str(tmpdir.join('tmp.html'))

    blocks = pyvista.MultiBlock()
    blocks.append(pyvista.Sphere())
    blocks.append(pyvista.Cube(center=(0, 0, -1)))

    pl = pyvista.Plotter()
    actor, mapper = pl.add_composite(blocks, show_edges=False, color='red')

    # override the color of the sphere
    mapper.block_attr[1].color = 'b'
    mapper.block_attr[1].opacity = 0.5

    pl.export_html(filename)

    # ensure modified block attributes have been outputted
    raw = open(filename).read()
    assert f'"opacity": {mapper.block_attr[1].opacity}' in raw
    assert f'"color": "{mapper.block_attr[1].color.hex_rgb}"' in raw


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
    assert pv_pythreejs.get_coloring(pl.mapper, pl.mesh) == 'VertexColors'

    sphere.clear_data()

    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_cells))
    assert pv_pythreejs.get_coloring(pl.mapper, pl.mesh) == 'FaceColors'


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


@pytest.mark.parametrize(
    'max_index',
    [np.iinfo(np.uint16).max - 1, np.iinfo(np.uint32).max - 1, np.iinfo(np.uint32).max + 1],
)
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


@pytest.mark.needs_vtk9
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


def test_just_points():
    pdata = pyvista.PolyData(np.random.random((10, 3)))
    pl = pyvista.Plotter()
    pl.add_mesh(pdata)
    output = pv_pythreejs.convert_plotter(pl)

    # ensure points in output
    pos = output.scene.children[0].geometry.attributes['position']
    assert np.allclose(pos.array, pdata.points)


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


def test_labels():
    poly = pyvista.PolyData(np.random.rand(10, 3))
    poly["My Labels"] = [f"Label {i}" for i in range(poly.n_points)]

    pl = pyvista.Plotter()
    pl.add_point_labels(poly, "My Labels", point_size=20, font_size=36)

    # TODO: ensure point labels at least don't raise a warning
    # For now, just make sure it doesn't error
    pv_pythreejs.convert_plotter(pl)


def test_linked_views(sphere):
    n_row, n_col = (2, 3)
    pl = pyvista.Plotter(shape=(n_row, n_col))

    for ii in range(n_row):
        for jj in range(n_col):
            pl.subplot(ii, jj)
            pl.add_mesh(sphere)

    pl.link_views((0, 1, 2))  # link first row together
    pl.link_views((3, 4, 5))  # link second row together

    # validate all cameras are linked
    widget = pv_pythreejs.convert_plotter(pl)

    # check first row is linked
    cameras = [widget[0, col].camera for col in range(n_col)]
    assert all([camera is cameras[0] for camera in cameras])

    # check second row is linked
    cameras = [widget[0, col].camera for col in range(n_col)]
    assert all([camera is cameras[0] for camera in cameras])

    # check first row camera is different than the second row
    cameras = [widget[row, 0].camera for row in range(2)]
    assert widget[0, 0].camera is not widget[1, 0].camera
