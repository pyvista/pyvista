"""Test any plotting that does not require rendering.

All other tests requiring rendering should to in
./plotting/test_plotting.py

"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista.core.errors import MissingDataError
from pyvista.plotting import _plotting
from pyvista.plotting.errors import RenderWindowUnavailable

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.skip_egl('OSMesa/EGL builds will not fail.')
def test_plotter_image_before_show():
    plotter = pv.Plotter()
    with pytest.raises(AttributeError, match='not yet been set up'):
        _ = plotter.image


def test_has_render_window_fail():
    pl = pv.Plotter()
    pl.close()
    with pytest.raises(RenderWindowUnavailable, match='not available'):
        pl._check_has_ren_win()
    with pytest.raises(RenderWindowUnavailable, match='not available'):
        pl._make_render_window_current()


def test_render_lines_as_tubes_show_edges_warning(sphere):
    pl = pv.Plotter()
    with pytest.warns(UserWarning, match='not supported'):
        actor = pl.add_mesh(sphere, render_lines_as_tubes=True, show_edges=True)
    assert not actor.prop.show_edges
    assert actor.prop.render_lines_as_tubes


@pytest.mark.skip_egl('OSMesa/EGL builds will not fail.')
def test_screenshot_fail_suppressed_rendering():
    plotter = pv.Plotter()
    plotter.suppress_rendering = True
    with pytest.warns(UserWarning, match='screenshot is unable to be taken'):
        plotter.show(screenshot='tmp.png')


def test_plotter_theme_raises():
    with pytest.raises(
        TypeError,
        match=re.escape('Expected ``pyvista.plotting.themes.Theme`` for ``theme``, not int.'),
    ):
        pv.Plotter(theme=1)

    pl = pv.Plotter()
    match = re.escape('Expected a pyvista theme like ``pyvista.plotting.themes.Theme``, not int.')

    with pytest.raises(TypeError, match=match):
        pl.theme = 1


def test_plotter_anti_aliasing_raises():
    pl = pv.Plotter()
    pl.close()
    with pytest.raises(
        AttributeError,
        match='The render window has been closed.',
    ):
        pl.enable_anti_aliasing(aa_type='msaa')

    with pytest.raises(
        ValueError,
        match=re.escape('Invalid `aa_type` "foo". Should be either "fxaa", "ssaa", or "msaa"'),
    ):
        pl.enable_anti_aliasing(aa_type='foo')


def test_plotter_store_mouse_position_raises():
    pl = pv.Plotter()
    pl.iren = None
    with pytest.raises(
        RuntimeError,
        match='This plotting window is not interactive.',
    ):
        pl.store_mouse_position()

    with pytest.raises(
        RuntimeError,
        match='This plotting window is not interactive.',
    ):
        pl.store_click_position()


def test_plotter_add_mesh_multiblock_algo_raises(mocker: MockerFixture):
    from pyvista.plotting import plotter

    m = mocker.patch.object(plotter, 'algorithm_to_mesh_handler')
    m.return_value = pv.MultiBlock(), 'foo'

    pl = pv.Plotter()
    match = re.escape(
        'Algorithms with `MultiBlock` output type are not supported by `add_mesh` at this time.'
    )
    with pytest.raises(TypeError, match=match):
        pl.add_mesh('foo')


def test_plotter_add_mesh_smooth_shading_algo_raises(mocker: MockerFixture):
    from pyvista.plotting import plotter

    m = mocker.patch.object(plotter, 'algorithm_to_mesh_handler')
    m.return_value = pv.PolyData(), 'foo'

    pl = pv.Plotter()
    with pytest.raises(
        TypeError,
        match=re.escape(
            'Smooth shading is not currently supported when a vtkAlgorithm is passed.'
        ),
    ):
        pl.add_mesh('foo', smooth_shading=True)


def test_plotter_add_mesh_scalars_rgb_raises():
    pl = pv.Plotter()
    sp = pv.Sphere()
    with pytest.raises(
        ValueError,
        match=re.escape('RGB array must be n_points/n_cells by 3/4 in shape.'),
    ):
        pl.add_mesh(sp, scalars=np.zeros((sp.n_points, 5)), rgb=True)


def test_plotter_add_mesh_texture_raises(mocker: MockerFixture):
    from pyvista.plotting import plotter

    pl = pv.Plotter()
    with pytest.raises(
        TypeError,
        match=re.escape("Invalid texture type (<class 'str'>)"),
    ):
        pl.add_mesh(pv.Sphere(), texture='foo')

    m = mocker.patch.object(plotter, 'numpy_to_texture')
    m.return_value = vtk.vtkTexture()
    with pytest.raises(
        ValueError,
        match=re.escape('Input mesh does not have texture coordinates to support the texture.'),
    ):
        pl.add_mesh(pv.Sphere(), texture=np.array([]))


def test_plotter_add_volume_scalar_raises():
    pl = pv.Plotter()
    match = re.escape(
        '`scalar` is an invalid keyword argument for `add_mesh`. '
        'Perhaps you mean `scalars` with an s?'
    )
    with pytest.raises(TypeError, match=match):
        pl.add_volume(pv.Sphere(), scalar='foo')


def test_plotter_add_volume_resolution_raises(mocker: MockerFixture):
    from pyvista.plotting import plotter

    pl = pv.Plotter()
    mocker.patch.object(plotter, 'wrap')

    with pytest.raises(
        ValueError,
        match=re.escape('Invalid resolution dimensions.'),
    ):
        pl.add_volume(np.zeros(4), resolution=[1, 2])


def test_plotter_add_volume_mapper_raises():
    pl = pv.Plotter()
    im = pv.ImageData(dimensions=(10, 10, 10))
    im.point_data['foo'] = 1
    match = re.escape(
        'Mapper (foo) unknown. Available volume mappers include: '
        'fixed_point, gpu, open_gl, smart, ugrid'
    )
    with pytest.raises(TypeError, match=match):
        pl.add_volume(im, mapper='foo')


def test_unlink_views_raises():
    pl = pv.Plotter()
    with pytest.raises(
        TypeError,
        match=re.escape("Expected type is None, int, list or tuple: <class 'object'> is given"),
    ):
        pl.unlink_views(object())


def test_add_scalar_bar_raises():
    pl = pv.Plotter(shape='1|2')
    pl.add_mesh(pv.Sphere())
    with pytest.raises(
        ValueError,
        match=re.escape('Interactive scalar bars disabled for multi-renderer plots'),
    ):
        pl.add_scalar_bar(interactive=True)


def test_write_frame_raises():
    pl = pv.Plotter()

    with pytest.raises(
        RuntimeError,
        match=re.escape('This plotter has not opened a movie or GIF file.'),
    ):
        pl.write_frame()


def test_add_lines_raises():
    pl = pv.Plotter()
    with pytest.raises(TypeError, match='Label must be a string'):
        pl.add_lines(
            np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]]),
            label=1,
        )


@pytest.mark.parametrize('points', [1, None, 'foo'])
def test_add_point_labels_raises(points):
    pl = pv.Plotter()
    with pytest.raises(
        TypeError,
        match=re.escape(f'Points type not usable: {type(points)}'),
    ):
        pl.add_point_labels(points=points, labels='foo')


@pytest.mark.no_vtk_error_catcher  # vtk error is expected here
@pytest.mark.needs_vtk_version(9, 1, 0)
def test_add_point_labels_algo_raises(mocker: MockerFixture):
    from pyvista.plotting import plotter

    m = mocker.patch.object(plotter, 'algorithm_to_mesh_handler')
    m.return_value = pv.PolyData(), vtk.vtkAlgorithm()

    pl = pv.Plotter()
    match = re.escape(
        'If using a vtkAlgorithm input, the labels must be a named array on the dataset.'
    )
    with pytest.raises(TypeError, match=match):
        pl.add_point_labels(points=np.array([0.0, 0.0, 0.0]), labels=['foo'])


def test_add_point_labels_label_not_found_raises():
    pl = pv.Plotter()
    with pytest.raises(
        ValueError,
        match=re.escape("Array 'foo' not found in point data."),
    ):
        pl.add_point_labels(points=pv.Sphere().points, labels='foo')


def test_add_point_labels_shape_raises():
    pl = pv.Plotter()
    with pytest.raises(
        ValueError,
        match=re.escape('Shape (foo) not understood'),
    ):
        pl.add_point_labels(
            points=pv.Sphere().points,
            labels=[f'{i}' for i in pv.Sphere().points],
            shape='foo',
        )


def test_save_image_raises(mocker: MockerFixture):
    pl = pv.Plotter()
    im = mocker.MagicMock()
    im.size = False
    with pytest.raises(
        ValueError,
        match=re.escape('Empty image. Have you run plot() first?'),
    ):
        pl._save_image(im, filename='foo', return_img=True)


def test_add_volume_scalar_raises(mocker: MockerFixture):
    from pyvista.plotting import plotter

    m = mocker.patch.object(plotter, 'get_array')
    m().ndim = 1
    m2 = mocker.patch.object(plotter, 'np')
    m2.issubdtype.return_value = False

    pl = pv.Plotter()
    with pytest.raises(
        TypeError,
        match='Non-numeric scalars are currently not supported for volume rendering.',
    ):
        pl.add_volume(pv.ImageData(), scalars='foo')

    mocker.resetall()

    m = mocker.patch.object(plotter, 'get_array')
    m().ndim = 0
    m2.issubdtype.return_value = True

    with pytest.raises(
        ValueError,
        match='`add_volume` only supports scalars with 1 or 2 dimensions',
    ):
        pl.add_volume(pv.ImageData(), scalars='foo')


def test_update_scalar_bar_range_raises():
    pl = pv.Plotter()
    match = re.escape('clim argument must be a length 2 iterable of values: (min, max).')

    with pytest.raises(TypeError, match=match):
        pl.update_scalar_bar_range(clim=[1, 2, 3])

    with pytest.raises(AttributeError, match='This plotter does not have an active mapper.'):
        pl.update_scalar_bar_range(clim=[1, 2], name=None)


def test_save_graphic_raises():
    pl = pv.Plotter()
    pl.close()

    with pytest.raises(
        AttributeError, match='This plotter is closed and unable to save a screenshot.'
    ):
        pl.save_graphic(filename='foo.svg')

    pl = pv.Plotter()
    match = re.escape(
        'Extension (.not_supported) is an invalid choice.\n\nValid options include: '
        '.svg, .eps, .ps, .pdf, .tex'
    )
    with pytest.raises(ValueError, match=match):
        pl.save_graphic(filename='foo.not_supported')


def test_add_background_image_raises():
    pl = pv.Plotter()
    pl.add_background_image(pv.examples.mapfile)

    match = re.escape(
        'A background image already exists.  Remove it with ``remove_background_image`` '
        'before adding one'
    )
    with pytest.raises(RuntimeError, match=match):
        pl.add_background_image(pv.examples.mapfile)


def test_show_after_closed_raises():
    pl = pv.Plotter()
    pl.close()

    with pytest.raises(
        RuntimeError,
        match=re.escape('This plotter has been closed and cannot be shown'),
    ):
        pl.show()


def test_plotter_line_point_smoothing():
    pl = pv.Plotter()
    assert bool(pl.render_window.GetLineSmoothing()) is False
    assert bool(pl.render_window.GetPointSmoothing()) is False
    assert bool(pl.render_window.GetPolygonSmoothing()) is False

    pl = pv.Plotter(line_smoothing=True, point_smoothing=True, polygon_smoothing=True)
    assert bool(pl.render_window.GetLineSmoothing()) is True
    assert bool(pl.render_window.GetPointSmoothing()) is True
    assert bool(pl.render_window.GetPolygonSmoothing()) is True


def test_enable_hidden_line_removal():
    plotter = pv.Plotter(shape=(1, 2))
    plotter.enable_hidden_line_removal(all_renderers=False)
    assert plotter.renderers[0].GetUseHiddenLineRemoval()
    assert not plotter.renderers[1].GetUseHiddenLineRemoval()

    plotter.enable_hidden_line_removal(all_renderers=True)
    assert plotter.renderers[1].GetUseHiddenLineRemoval()


def test_disable_hidden_line_removal():
    plotter = pv.Plotter(shape=(1, 2))
    plotter.enable_hidden_line_removal(all_renderers=True)

    plotter.disable_hidden_line_removal(all_renderers=False)
    assert not plotter.renderers[0].GetUseHiddenLineRemoval()
    assert plotter.renderers[1].GetUseHiddenLineRemoval()

    plotter.disable_hidden_line_removal(all_renderers=True)
    assert not plotter.renderers[1].GetUseHiddenLineRemoval()


def test_pickable_actors():
    plotter = pv.Plotter()
    sphere = plotter.add_mesh(pv.Sphere(), pickable=True)
    cube = plotter.add_mesh(pv.Cube(), pickable=False)

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

    with pytest.raises(TypeError, match='Expected a vtkActor instance or '):
        plotter.pickable_actors = [0, 10]


def test_plotter_image_scale():
    pl = pv.Plotter()
    assert isinstance(pl.image_scale, int)
    with pytest.raises(ValueError, match='must be a positive integer'):
        pl.image_scale = 0

    pl.image_scale = 2
    assert pl.image_scale == 2


def test_prepare_smooth_shading_texture(globe):
    """Test edge cases for smooth shading"""
    mesh, scalars = _plotting.prepare_smooth_shading(
        mesh=globe,
        scalars=None,
        texture=True,
        split_sharp_edges=True,
        feature_angle=False,
        preference=None,
    )
    assert scalars is None
    assert 'Normals' in mesh.point_data
    assert 'Texture Coordinates' in mesh.point_data


def test_prepare_smooth_shading_not_poly(hexbeam):
    """Test edge cases for smooth shading"""
    scalars_name = 'sample_point_scalars'
    scalars = hexbeam.point_data[scalars_name]
    mesh, scalars = _plotting.prepare_smooth_shading(
        mesh=hexbeam,
        scalars=scalars,
        texture=False,
        split_sharp_edges=True,
        feature_angle=True,
        preference=None,
    )

    assert 'Normals' in mesh.point_data

    expected_mesh = hexbeam.extract_surface().compute_normals(
        cell_normals=False,
        split_vertices=True,
    )

    assert np.allclose(mesh[scalars_name], expected_mesh[scalars_name])


@pytest.mark.parametrize('split_sharp_edges', [True, False])
def test_prepare_smooth_shading_point_cloud(split_sharp_edges):
    point_cloud = pv.PolyData([0.0, 0.0, 0.0])
    assert point_cloud.n_verts == point_cloud.n_cells
    mesh, scalars = _plotting.prepare_smooth_shading(
        mesh=point_cloud,
        scalars=None,
        texture=True,
        split_sharp_edges=split_sharp_edges,
        feature_angle=False,
        preference=None,
    )
    assert scalars is None
    assert 'Normals' not in mesh.point_data


def test_smooth_shading_shallow_copy(sphere):
    """See also ``test_compute_normals_inplace``."""
    sphere.point_data['numbers'] = np.arange(sphere.n_points)
    sphere2 = sphere.copy(deep=False)

    sphere['numbers'] *= -1  # sphere2 'numbers' are also modified
    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])

    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars=None, smooth_shading=True)
    # Modify after adding and using compute_normals via smooth_shading
    sphere['numbers'] *= -1
    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])
    pl.close()


def test_get_datasets(sphere, hexbeam):
    """Test pv.Plotter._datasets."""
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_mesh(hexbeam)
    datasets = pl._datasets
    assert len(datasets) == 2
    assert sphere in datasets
    assert hexbeam in datasets


def test_remove_scalars_single(sphere, hexbeam):
    """Ensure no scalars are added when plotting datasets if copy_mesh=True."""
    # test single component scalars
    sphere.clear_data()
    hexbeam.clear_data()
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_points), copy_mesh=True)
    pl.add_mesh(hexbeam, scalars=range(hexbeam.n_cells), copy_mesh=True)

    # arrays will be added to the mesh
    assert pl.mesh.n_arrays == 1

    # but not the original data
    assert sphere.n_arrays == 0
    assert hexbeam.n_arrays == 0

    pl.close()


def test_active_scalars_remain(sphere, hexbeam):
    # Ensure active scalars remain active despite plotting different scalars when copy_mesh=True.
    sphere.clear_data()
    hexbeam.clear_data()
    point_data_name = 'point_data'
    cell_data_name = 'cell_data'
    sphere[point_data_name] = np.random.default_rng().random(sphere.n_points)
    hexbeam[cell_data_name] = np.random.default_rng().random(hexbeam.n_cells)
    assert sphere.point_data.active_scalars_name == point_data_name
    assert hexbeam.cell_data.active_scalars_name == cell_data_name

    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_points), copy_mesh=True)
    pl.add_mesh(hexbeam, scalars=range(hexbeam.n_cells), copy_mesh=True)
    pl.close()

    assert sphere.point_data.active_scalars_name == point_data_name
    assert hexbeam.cell_data.active_scalars_name == cell_data_name


def test_no_added_with_scalar_bar(sphere):
    point_data_name = 'data'
    sphere[point_data_name] = np.random.default_rng().random(sphere.n_points)
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalar_bar_args={'title': 'some_title'})
    assert sphere.n_arrays == 1


def test_plotter_remains_shallow():
    sphere = pv.Sphere()
    sphere.point_data['numbers'] = np.arange(sphere.n_points)
    sphere2 = sphere.copy(deep=False)

    sphere['numbers'] *= -1  # sphere2 'numbers' are also modified

    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])

    plotter = pv.Plotter()
    plotter.add_mesh(sphere, scalars=None)

    sphere[
        'numbers'
    ] *= -1  # sphere2 'numbers' are also modified after adding to Plotter.  (See  issue #2461)

    assert np.array_equal(sphere['numbers'], sphere2['numbers'])
    assert np.shares_memory(sphere['numbers'], sphere2['numbers'])


def test_add_multiple(sphere):
    point_data_name = 'data'
    sphere[point_data_name] = np.random.default_rng().random(sphere.n_points)
    pl = pv.Plotter()
    pl.add_mesh(sphere, copy_mesh=True)
    pl.add_mesh(sphere, scalars=np.arange(sphere.n_points), copy_mesh=True)
    pl.add_mesh(sphere, scalars=np.arange(sphere.n_cells), copy_mesh=True)
    pl.add_mesh(sphere, scalars='data', copy_mesh=True)
    pl.show()
    assert sphere.n_arrays == 1


def test_deep_clean(cube):
    pl = pv.Plotter()
    cube_orig = cube.copy()
    pl.add_mesh(cube)
    pl.deep_clean()
    assert pl.mesh is None
    assert pl.mapper is None
    assert pl.volume is None
    assert pl.text is None
    assert cube == cube_orig


def test_disable_depth_of_field():
    pl = pv.Plotter()
    pl.enable_depth_of_field()
    assert pl.renderer.GetPass() is not None
    pl.disable_depth_of_field()
    assert pl.renderer.GetPass() is None


def test_remove_blurring():
    pl = pv.Plotter()
    pl.add_blurring()
    assert pl.renderer.GetPass() is not None
    pl.remove_blurring()
    assert pl.renderer.GetPass() is None


def test_add_points_invalid_style(sphere):
    pl = pv.Plotter()
    with pytest.raises(ValueError, match='Should be either "points"'):
        pl.add_points(sphere, style='wireframe')


@pytest.mark.parametrize(('connected', 'n_lines'), [(False, 2), (True, 3)])
def test_add_lines(connected, n_lines):
    pl = pv.Plotter()
    points = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]])
    actor = pl.add_lines(points, connected=connected)
    dataset = actor.mapper.dataset
    assert dataset.n_cells == n_lines


def test_clear_actors(cube, sphere):
    pl = pv.Plotter()
    pl.add_mesh(cube)
    pl.add_mesh(sphere)
    assert len(pl.renderer.actors) == 2
    pl.clear_actors()
    assert len(pl.renderer.actors) == 0


def test_anti_aliasing_multiplot():
    pl = pv.Plotter(shape=(1, 2))
    pl.enable_anti_aliasing('ssaa', all_renderers=False)
    assert 'vtkSSAAPass' in pl.renderers[0]._render_passes._passes
    assert 'vtkSSAAPass' not in pl.renderers[1]._render_passes._passes

    pl.enable_anti_aliasing('ssaa', all_renderers=True)
    assert 'vtkSSAAPass' in pl.renderers[1]._render_passes._passes

    pl.disable_anti_aliasing(all_renderers=False)
    assert 'vtkSSAAPass' not in pl.renderers[0]._render_passes._passes
    assert 'vtkSSAAPass' in pl.renderers[1]._render_passes._passes

    pl.disable_anti_aliasing(all_renderers=True)
    assert 'vtkSSAAPass' not in pl.renderers[0]._render_passes._passes
    assert 'vtkSSAAPass' not in pl.renderers[1]._render_passes._passes


def test_anti_aliasing_invalid():
    pl = pv.Plotter()
    with pytest.raises(ValueError, match='Should be either "fxaa" or "ssaa"'):
        pl.renderer.enable_anti_aliasing('invalid')


def test_plot_return_img_without_cpos(sphere: pv.PolyData):
    img = sphere.plot(return_cpos=False, return_img=True, screenshot=True)
    assert isinstance(img, np.ndarray)


def test_plot_return_img_with_cpos(sphere: pv.PolyData):
    cpos, img = sphere.plot(return_cpos=True, return_img=True, screenshot=True)
    assert isinstance(cpos, pv.CameraPosition)
    assert isinstance(img, np.ndarray)


def test_plotter_actors(sphere, cube):
    pl = pv.Plotter()
    actor_a = pl.add_mesh(sphere)
    actor_b = pl.add_mesh(cube)
    assert len(pl.actors) == 2
    assert actor_a in pl.actors.values()
    assert actor_b in pl.actors.values()


def test_plotter_suppress_rendering():
    pl = pv.Plotter()
    assert isinstance(pl.suppress_rendering, bool)
    pl.suppress_rendering = True
    assert pl.suppress_rendering is True
    pl.suppress_rendering = False
    assert pl.suppress_rendering is False


def test_plotter_add_volume_raises(uniform: pv.ImageData, sphere: pv.PolyData):
    """Test edge case where add_volume has no scalars."""
    uniform.clear_data()
    pl = pv.Plotter()
    with pytest.raises(MissingDataError):
        pl.add_volume(uniform, cmap='coolwarm', opacity='linear')

    with pytest.raises(TypeError, match='not supported for volume rendering'):
        pl.add_volume(sphere)

    with pytest.raises(TypeError, match='volume must be an instance of any type'):
        pl.add_volume(pv.Table())


def test_plotter_add_volume_clim(uniform: pv.ImageData):
    """Verify clim is set correctly for volume."""
    arr = uniform.x.astype(np.uint8)
    pl = pv.Plotter()
    vol = pl.add_volume(uniform, scalars=arr)
    assert vol.mapper.scalar_range == (arr.min(), arr.max())

    clim = [-10, 20]
    pl = pv.Plotter()
    vol = pl.add_volume(uniform, clim=clim)
    assert vol.mapper.scalar_range == tuple(clim)

    clim_val = 2.0
    pl = pv.Plotter()
    vol = pl.add_volume(uniform, clim=clim_val)
    assert vol.mapper.scalar_range == (-clim_val, clim_val)


def test_plotter_meshes(sphere, cube):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_mesh(cube)

    assert sphere in pl.meshes
    assert cube in pl.meshes
    assert len(pl.meshes) == 2


def test_multi_block_color_cycler():
    """Test passing a custom color cycler"""
    plotter = pv.Plotter()
    data = {
        'sphere1': pv.Sphere(center=(1, 0, 0)),
        'sphere2': pv.Sphere(center=(2, 0, 0)),
        'sphere3': pv.Sphere(center=(3, 0, 0)),
        'sphere4': pv.Sphere(center=(4, 0, 0)),
    }
    spheres = pv.MultiBlock(data)
    actor, mapper = plotter.add_composite(spheres)

    # pass custom cycler
    mapper.set_unique_colors(['red', 'green', 'blue'])

    assert mapper.block_attr[0].color.name == 'red'
    assert mapper.block_attr[1].color.name == 'green'
    assert mapper.block_attr[2].color.name == 'blue'
    assert mapper.block_attr[3].color.name == 'red'

    # test wrong args
    with pytest.raises(ValueError):  # noqa: PT011
        mapper.set_unique_colors('foo')

    with pytest.raises(TypeError):
        mapper.set_unique_colors(5)


@pytest.mark.parametrize(
    ('face', 'normal'),
    [
        ('-Z', (0, 0, 1)),
        ('-Y', (0, 1, 0)),
        ('-X', (1, 0, 0)),
        ('+Z', (0, 0, -1)),
        ('+Y', (0, -1, 0)),
        ('+X', (-1, 0, 0)),
    ],
)
def test_plotter_add_floor(face, normal):
    pl = pv.Plotter()
    pl.add_floor(face=face)
    assert np.allclose(pl.renderer._floor.face_normals[0], normal)


def test_plotter_add_floor_raise_error():
    pl = pv.Plotter()
    with pytest.raises(NotImplementedError, match='not implemented'):
        pl.add_floor(face='invalid')


def test_plotter_zoom_camera():
    pl = pv.Plotter()
    pl.zoom_camera(1.05)


def test_plotter_reset_key_events():
    pl = pv.Plotter()
    pl.reset_key_events()


@pytest.mark.usefixtures('global_variables_reset')
def test_only_screenshots_flag(sphere, tmpdir):
    pv.FIGURE_PATH = str(tmpdir)
    pv.ON_SCREENSHOT = True

    entries = os.listdir(pv.FIGURE_PATH)  # noqa: PTH208
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show()
    entries_after = os.listdir(pv.FIGURE_PATH)  # noqa: PTH208
    assert len(entries) + 1 == len(entries_after)

    res_file = next(iter(set(entries_after) - set(entries)))
    pv.ON_SCREENSHOT = False
    sphere_screenshot = 'sphere_screenshot.png'
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show(screenshot=sphere_screenshot)
    sphere_path = str(Path(pv.FIGURE_PATH) / sphere_screenshot)
    res_path = str(Path(pv.FIGURE_PATH) / res_file)
    error = pv.compare_images(sphere_path, res_path)
    assert error < 100


def test_legend_font(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    legend_labels = [['sphere', 'r']]
    legend = plotter.add_legend(
        labels=legend_labels,
        border=True,
        bcolor=None,
        size=[0.1, 0.1],
        font_family='times',
    )
    assert legend.GetEntryTextProperty().GetFontFamily() == vtk.VTK_TIMES


@pytest.mark.needs_vtk_version(9, 3, reason='Functions not implemented before 9.3.X')
def test_edge_opacity(sphere):
    edge_opacity = np.random.default_rng().random()
    pl = pv.Plotter()
    actor = pl.add_mesh(sphere, edge_opacity=edge_opacity)
    assert actor.prop.edge_opacity == edge_opacity


def test_add_ruler_scale():
    plotter = pv.Plotter()
    ruler = plotter.add_ruler([-0.6, 0.0, 0], [0.6, 0.0, 0], scale=0.5)
    min_, max_ = ruler.GetRange()
    assert min_ == 0.0
    assert max_ == 0.6

    ruler = plotter.add_ruler([-0.6, 0.0, 0], [0.6, 0.0, 0], scale=0.5, flip_range=True)
    min_, max_ = ruler.GetRange()
    assert min_ == 0.6
    assert max_ == 0.0


def test_plotter_shape():
    pl = pv.Plotter()
    assert isinstance(pl.shape, tuple)
    assert pl.shape == (1, 1)
    assert isinstance(pl.shape[0], int)

    pl = pv.Plotter(shape=(1, 2))
    assert isinstance(pl.shape, tuple)
    assert pl.shape == (1, 2)
    assert isinstance(pl.shape[0], int)


@pytest.mark.parametrize(
    'filename_mtl',
    [
        None,
        Path(pv.examples.download_doorman(load=False)).with_suffix('.mtl'),
    ],
)
def test_import_obj_with_filename_mtl(filename_mtl):
    filename = Path(pv.examples.download_doorman(load=False))
    plotter = pv.Plotter()
    plotter.import_obj(filename, filename_mtl=filename_mtl)
