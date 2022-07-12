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
    with pytest.raises(AttributeError, match="not yet been set up"):
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
    assert "Normals" in mesh.point_data
    assert "Texture Coordinates" in mesh.point_data


def test_prepare_smooth_shading_not_poly(hexbeam):
    """Test edge cases for smooth shading"""
    scalars_name = "sample_point_scalars"
    scalars = hexbeam.point_data[scalars_name]
    mesh, scalars = _plotting.prepare_smooth_shading(hexbeam, scalars, False, True, True, None)

    assert "Normals" in mesh.point_data

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
        assert name == "Data"
        if mesh is sphere:
            assert assoc == "point"
        else:
            assert assoc == "cell"

    pl.close()
    assert pl._added_scalars == []

    assert sphere.n_arrays == 0
    assert hexbeam.n_arrays == 0


def test_remove_scalars_complex(sphere):
    """Test plotting complex data."""
    data = np.arange(sphere.n_points, dtype=np.complex128)
    data += np.linspace(0, 1, sphere.n_points) * -1j
    point_data_name = "data"
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
    point_data_name = "data"
    sphere[point_data_name] = np.random.random((sphere.n_points, 3))
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=point_data_name)
    pl.close()
    assert sphere.point_data.keys() == [point_data_name]
    assert sphere.point_data.active_scalars_name == point_data_name


def test_remove_scalars_component(sphere):
    point_data_name = "data"
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
    point_data_name = "data"
    sphere[point_data_name] = np.random.random(sphere.n_points)
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=point_data_name, opacity=point_data_name)
    assert sphere.n_arrays == 2
    pl.close()
    assert sphere.n_arrays == 1

    # only point_data_name remains, no '_custom_rgba' array should remain
    assert sphere.point_data.keys() == [point_data_name]

    # TODO: we are not re-enabling the old scalars
    # assert sphere.point_data.active_scalars_name == point_data_name


def test_active_scalars_remain(sphere, hexbeam):
    """Ensure active scalars remain active despite plotting different scalars."""
    point_data_name = "point_data"
    cell_data_name = "cell_data"
    sphere[point_data_name] = np.random.random(sphere.n_points)
    hexbeam[cell_data_name] = np.random.random(hexbeam.n_cells)
    assert sphere.point_data.active_scalars_name == point_data_name
    assert hexbeam.cell_data.active_scalars_name == cell_data_name

    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_points))
    pl.add_mesh(hexbeam, scalars=range(hexbeam.n_cells))
    pl.close()

    assert sphere.point_data.active_scalars_name == point_data_name
    assert hexbeam.cell_data.active_scalars_name == cell_data_name


def test_no_added_with_scalar_bar(sphere):
    point_data_name = "data"
    sphere[point_data_name] = np.random.random(sphere.n_points)
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalar_bar_args={"title": "some_title"})
    assert sphere.n_arrays == 1


def test_plotter_remains_shallow():
    sphere = pyvista.Sphere()
    sphere.point_data['numbers'] = np.arange(sphere.n_points)
    sphere2 = sphere.copy(deep=False)

    sphere['numbers'] *= -1  # sphere2 'numbers' are also modified

    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])

    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere, scalars=None)

    sphere[
        'numbers'
    ] *= -1  # sphere2 'numbers' are also modified after adding to Plotter.  (See  issue #2461)

    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])


def test_add_multiple(sphere):
    point_data_name = 'data'
    sphere[point_data_name] = np.random.random(sphere.n_points)
    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.add_mesh(sphere, scalars=np.arange(sphere.n_points))
    pl.add_mesh(sphere, scalars=np.arange(sphere.n_cells))
    pl.add_mesh(sphere, scalars='data')
    pl.show()
    assert sphere.n_arrays == 1


def test_deep_clean(cube):
    pl = pyvista.Plotter()
    cube_orig = cube.copy()
    pl.add_mesh(cube)
    pl.deep_clean()
    assert pl.mesh is None
    assert pl.mapper is None
    assert pl.volume is None
    assert pl.textActor is None
    assert cube == cube_orig
