"""Test any plotting that does not require rendering.

All other tests requiring rendering should to in
./plotting/test_plotting.py

"""
import numpy as np
import pytest

import pyvista
from pyvista.plotting import _plotting


def test_plotter_image():
    plotter = pyvista.Plotter()
    with pytest.raises(AttributeError, match='not yet been set up'):
        plotter.image


def test_enable_hidden_line_removal():
    plotter = pyvista.Plotter(shape=(1, 2))
    plotter.enable_hidden_line_removal(False)
    assert plotter.renderers[0].GetUseHiddenLineRemoval()
    assert not plotter.renderers[1].GetUseHiddenLineRemoval()

    plotter.enable_hidden_line_removal(True)
    assert plotter.renderers[1].GetUseHiddenLineRemoval()


def test_disable_hidden_line_removal():
    plotter = pyvista.Plotter(shape=(1, 2))
    plotter.enable_hidden_line_removal(True)

    plotter.disable_hidden_line_removal(False)
    assert not plotter.renderers[0].GetUseHiddenLineRemoval()
    assert plotter.renderers[1].GetUseHiddenLineRemoval()

    plotter.disable_hidden_line_removal(True)
    assert not plotter.renderers[1].GetUseHiddenLineRemoval()


def test_pickable_actors():

    plotter = pyvista.Plotter()
    sphere = plotter.add_mesh(pyvista.Sphere(), pickable=True)
    cube = plotter.add_mesh(pyvista.Cube(), pickable=False)

    pickable = plotter.pickable_actors
    assert sphere in pickable
    assert cube not in pickable

    plotter.pickable_actors = cube
    pickable = plotter.pickable_actors
    assert sphere not in pickable
    assert cube in pickable

    plotter.pickable_actors = [sphere, cube]
    pickable = plotter.pickable_actors
    assert sphere in pickable
    assert cube in pickable

    plotter.pickable_actors = None
    pickable = plotter.pickable_actors
    assert sphere not in pickable
    assert cube not in pickable

    with pytest.raises(TypeError, match="Expected a vtkActor instance or "):
        plotter.pickable_actors = [0, 10]


def test_prepare_smooth_shading_texture(globe):
    """Test edge cases for smooth shading"""
    mesh, scalars = _plotting.prepare_smooth_shading(globe, None, True, True, False, None)
    assert scalars is None
    assert 'Normals' in mesh.point_data
    assert 'Texture Coordinates' in mesh.point_data


def test_prepare_smooth_shading_not_poly(hexbeam):
    """Test edge cases for smooth shading"""
    scalars_name = 'sample_point_scalars'
    scalars = hexbeam.point_data[scalars_name]
    mesh, scalars = _plotting.prepare_smooth_shading(hexbeam, scalars, False, True, True, None)

    assert 'Normals' in mesh.point_data

    expected_mesh = hexbeam.extract_surface().compute_normals(
        cell_normals=False,
        split_vertices=True,
    )

    assert np.allclose(mesh[scalars_name], expected_mesh[scalars_name])


def test_get_datasets(sphere, hexbeam):
    """Test pyvista.Plotter._datasets."""
    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.add_mesh(hexbeam)
    datasets = pl._datasets
    assert len(datasets) == 2
    assert sphere in datasets and hexbeam in datasets


def test_remove_scalars_single(sphere, hexbeam):
    """Ensure no scalars are added when plotting datasets."""
    # test single component scalars
    hexbeam.clear_data()
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_points))
    pl.add_mesh(hexbeam, scalars=range(hexbeam.n_cells))

    assert sphere.n_arrays == 1
    assert hexbeam.n_arrays == 1

    assert len(pl._added_scalars) == 2
    for mesh, (name, assoc) in pl._added_scalars:
        assert name == 'Data'
        if mesh is sphere:
            assert assoc == 'point'
        else:
            assert assoc == 'cell'

    pl.close()
    assert pl._added_scalars == []

    assert sphere.n_arrays == 0
    assert hexbeam.n_arrays == 0


def test_remove_scalars_complex(sphere):
    """Test plotting complex data."""
    data = np.arange(sphere.n_points, dtype=np.complex128)
    data += np.linspace(0, 1, sphere.n_points) * -1j
    point_data_name = 'data'
    sphere[point_data_name] = data

    pl = pyvista.Plotter()
    with pytest.warns(np.ComplexWarning):
        pl.add_mesh(sphere, scalars=point_data_name)
    assert sphere.n_arrays == 2
    pl.close()
    assert sphere.n_arrays == 1

    assert sphere.point_data.keys() == [point_data_name]
    assert sphere.point_data.active_scalars_name == point_data_name


def test_remove_scalars_normalized(sphere):
    # test scalars are removed for normalized multi-component
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=np.random.random((sphere.n_points, 3)))
    assert sphere.n_arrays == 1
    pl.close()
    assert sphere.n_arrays == 0

    # test scalars are removed for normalized multi-component
    point_data_name = 'data'
    sphere[point_data_name] = np.random.random((sphere.n_points, 3))
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=point_data_name)
    pl.close()
    assert sphere.point_data.keys() == [point_data_name]
    assert sphere.point_data.active_scalars_name == point_data_name


def test_remove_scalars_component(sphere):
    point_data_name = 'data'
    sphere[point_data_name] = np.random.random((sphere.n_points, 3))
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=point_data_name, component=0)
    assert sphere.n_arrays == 2
    pl.close()
    assert sphere.n_arrays == 1

    # only point_data_name remains, no 'data-0' array should remain
    assert sphere.point_data.keys() == [point_data_name]

    # however, the original active array should remain active
    assert sphere.point_data.active_scalars_name == point_data_name


def test_remove_scalars_rgba(sphere):
    point_data_name = 'data'
    sphere[point_data_name] = np.random.random(sphere.n_points)
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=point_data_name, opacity='data')
    assert sphere.n_arrays == 2
    pl.close()
    assert sphere.n_arrays == 1

    # only point_data_name remains, no '_custom_rgba' array should remain
    assert sphere.point_data.keys() == [point_data_name]

    # TODO: we are not re-enabling the old scalars
    # assert sphere.point_data.active_scalars_name == point_data_name
