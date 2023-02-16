"""Test any plotting that does not require rendering.

All other tests requiring rendering should to in
./plotting/test_plotting.py

"""
import numpy as np
import pytest

import pyvista
from pyvista.core.errors import DeprecationError
from pyvista.errors import MissingDataError
from pyvista.plotting import _plotting


def test_plotter_image_before_show():
    plotter = pyvista.Plotter()
    with pytest.raises(AttributeError, match="not yet been set up"):
        plotter.image


def test_screenshot_fail_suppressed_rendering():
    plotter = pyvista.Plotter()
    plotter.suppress_rendering = True
    with pytest.warns(UserWarning, match='screenshot is unable to be taken'):
        plotter.show(screenshot='tmp.png')


def test_plotter_line_point_smoothing():
    pl = pyvista.Plotter()
    assert bool(pl.render_window.GetLineSmoothing()) is False
    assert bool(pl.render_window.GetPointSmoothing()) is False
    assert bool(pl.render_window.GetPolygonSmoothing()) is False

    pl = pyvista.Plotter(line_smoothing=True, point_smoothing=True, polygon_smoothing=True)
    assert bool(pl.render_window.GetLineSmoothing()) is True
    assert bool(pl.render_window.GetPointSmoothing()) is True
    assert bool(pl.render_window.GetPolygonSmoothing()) is True


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


def test_plotter_image_scale():
    pl = pyvista.Plotter()
    assert isinstance(pl.image_scale, int)
    with pytest.raises(ValueError, match='must be a positive integer'):
        pl.image_scale = 0

    pl.image_scale = 2
    assert pl.image_scale == 2


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


def test_smooth_shading_shallow_copy(sphere):
    """See also ``test_compute_normals_inplace``."""
    sphere.point_data['numbers'] = np.arange(sphere.n_points)
    sphere2 = sphere.copy(deep=False)

    sphere['numbers'] *= -1  # sphere2 'numbers' are also modified
    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])

    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=None, smooth_shading=True)
    # Modify after adding and using compute_normals via smooth_shading
    sphere['numbers'] *= -1
    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])
    pl.close()


def test_get_datasets(sphere, hexbeam):
    """Test pyvista.Plotter._datasets."""
    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.add_mesh(hexbeam)
    datasets = pl._datasets
    assert len(datasets) == 2
    assert sphere in datasets and hexbeam in datasets


def test_remove_scalars_single(sphere, hexbeam):
    """Ensure no scalars are added when plotting datasets if copy_mesh=True."""
    # test single component scalars
    sphere.clear_data()
    hexbeam.clear_data()
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_points), copy_mesh=True)
    pl.add_mesh(hexbeam, scalars=range(hexbeam.n_cells), copy_mesh=True)

    # arrays will be added to the mesh
    pl.mesh.n_arrays == 1

    # but not the original data
    assert sphere.n_arrays == 0
    assert hexbeam.n_arrays == 0

    pl.close()


def test_active_scalars_remain(sphere, hexbeam):
    """Ensure active scalars remain active despite plotting different scalars when copy_mesh=True."""
    sphere.clear_data()
    hexbeam.clear_data()
    point_data_name = "point_data"
    cell_data_name = "cell_data"
    sphere[point_data_name] = np.random.random(sphere.n_points)
    hexbeam[cell_data_name] = np.random.random(hexbeam.n_cells)
    assert sphere.point_data.active_scalars_name == point_data_name
    assert hexbeam.cell_data.active_scalars_name == cell_data_name

    pl = pyvista.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_points), copy_mesh=True)
    pl.add_mesh(hexbeam, scalars=range(hexbeam.n_cells), copy_mesh=True)
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
    pl.add_mesh(sphere, copy_mesh=True)
    pl.add_mesh(sphere, scalars=np.arange(sphere.n_points), copy_mesh=True)
    pl.add_mesh(sphere, scalars=np.arange(sphere.n_cells), copy_mesh=True)
    pl.add_mesh(sphere, scalars='data', copy_mesh=True)
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


def test_disable_depth_of_field(sphere):
    pl = pyvista.Plotter()
    pl.enable_depth_of_field()
    assert pl.renderer.GetPass() is not None
    pl.disable_depth_of_field()
    assert pl.renderer.GetPass() is None


def test_remove_blurring(sphere):
    pl = pyvista.Plotter()
    pl.add_blurring()
    assert pl.renderer.GetPass() is not None
    pl.remove_blurring()
    assert pl.renderer.GetPass() is None


def test_add_points_invalid_style(sphere):
    pl = pyvista.Plotter()
    with pytest.raises(ValueError, match='Should be either "points"'):
        pl.add_points(sphere, style='wireframe')


def test_add_lines():
    pl = pyvista.Plotter()
    points = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]])
    actor = pl.add_lines(points)
    dataset = actor.mapper.dataset
    assert dataset.n_cells == 2


def test_clear_actors(cube, sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(cube)
    pl.add_mesh(sphere)
    assert len(pl.renderer.actors) == 2
    pl.clear_actors()
    assert len(pl.renderer.actors) == 0


def test_anti_aliasing_multiplot(sphere):
    pl = pyvista.Plotter(shape=(1, 2))
    pl.enable_anti_aliasing('fxaa', all_renderers=False)
    assert pl.renderers[0].GetUseFXAA()
    assert not pl.renderers[1].GetUseFXAA()

    pl.enable_anti_aliasing('fxaa', all_renderers=True)
    assert pl.renderers[1].GetUseFXAA()

    pl.disable_anti_aliasing(all_renderers=False)
    assert not pl.renderers[0].GetUseFXAA()
    assert pl.renderers[1].GetUseFXAA()

    pl.disable_anti_aliasing(all_renderers=True)
    assert not pl.renderers[0].GetUseFXAA()
    assert not pl.renderers[1].GetUseFXAA()


def test_anti_aliasing_invalid():
    pl = pyvista.Plotter()
    with pytest.raises(ValueError, match='Should be either "fxaa" or "ssaa"'):
        pl.renderer.enable_anti_aliasing('invalid')


def test_plot_return_img_without_cpos(sphere: pyvista.PolyData):
    img = sphere.plot(return_cpos=False, return_img=True, screenshot=True)
    assert isinstance(img, np.ndarray)


def test_plot_return_img_with_cpos(sphere: pyvista.PolyData):
    cpos, img = sphere.plot(return_cpos=True, return_img=True, screenshot=True)
    assert isinstance(cpos, pyvista.CameraPosition)
    assert isinstance(img, np.ndarray)


def test_plotter_actors(sphere, cube):
    pl = pyvista.Plotter()
    actor_a = pl.add_mesh(sphere)
    actor_b = pl.add_mesh(cube)
    assert len(pl.actors) == 2
    assert actor_a in pl.actors.values()
    assert actor_b in pl.actors.values()


def test_plotter_suppress_rendering():
    pl = pyvista.Plotter()
    assert isinstance(pl.suppress_rendering, bool)
    pl.suppress_rendering = True
    assert pl.suppress_rendering is True
    pl.suppress_rendering = False
    assert pl.suppress_rendering is False


def test_plotter_add_volume_raises(uniform: pyvista.UniformGrid, sphere: pyvista.PolyData):
    """Test edge case where add_volume has no scalars."""
    uniform.clear_data()
    pl = pyvista.Plotter()
    with pytest.raises(MissingDataError):
        pl.add_volume(uniform, cmap="coolwarm", opacity="linear")

    with pytest.raises(TypeError, match='not supported for volume rendering'):
        pl.add_volume(sphere)


def test_deprecated_store_image():
    """Test to make sure store_image is deprecated."""
    pl = pyvista.Plotter()
    with pytest.raises(DeprecationError):
        assert isinstance(pl.store_image, bool)

    with pytest.raises(DeprecationError):
        pl.store_image = True


def test_plotter_add_volume_clim(uniform: pyvista.UniformGrid):
    """Verify clim is set correctly for volume."""
    arr = uniform.x.astype(np.uint8)
    pl = pyvista.Plotter()
    vol = pl.add_volume(uniform, scalars=arr)
    assert vol.mapper.scalar_range == (0, 255)

    clim = [-10, 20]
    pl = pyvista.Plotter()
    vol = pl.add_volume(uniform, clim=clim)
    assert vol.mapper.scalar_range == tuple(clim)

    clim_val = 2.0
    pl = pyvista.Plotter()
    vol = pl.add_volume(uniform, clim=clim_val)
    assert vol.mapper.scalar_range == (-clim_val, clim_val)
