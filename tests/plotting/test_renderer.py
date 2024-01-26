import numpy as np
import pytest
from pytest import raises
import vtk

import pyvista as pv
from pyvista.plotting.renderer import ACTOR_LOC_MAP


def test_show_bounds_axes_ranges():
    plotter = pv.Plotter()

    # test empty call
    plotter.show_bounds()
    cube_axes_actor = plotter.renderer.cube_axes_actor
    assert cube_axes_actor.GetBounds() == tuple(plotter.bounds)

    # send bounds but no axes ranges
    bounds = (0, 1, 0, 1, 0, 1)
    plotter.show_bounds(bounds=bounds)
    cube_axes_actor = plotter.renderer.cube_axes_actor
    assert cube_axes_actor.bounds == bounds

    # send bounds and axes ranges
    axes_ranges = [0, 1, 0, 2, 0, 3]
    plotter.show_bounds(bounds=bounds, axes_ranges=axes_ranges)
    cube_axes_actor = plotter.renderer.cube_axes_actor
    assert cube_axes_actor.GetBounds() == bounds
    test_ranges = [
        *cube_axes_actor.GetXAxisRange(),
        *cube_axes_actor.GetYAxisRange(),
        *cube_axes_actor.GetZAxisRange(),
    ]
    assert test_ranges == axes_ranges


def test_show_bounds_with_scaling(sphere):
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    actor0 = plotter.show_bounds()
    assert actor0.GetUseTextActor3D()
    plotter.set_scale(0.5, 0.5, 2)
    actor1 = plotter.show_bounds()
    assert not actor1.GetUseTextActor3D()


def test_show_bounds_invalid_axes_ranges():
    plotter = pv.Plotter()

    # send incorrect axes_ranges types
    with raises(TypeError, match='numeric sequence'):
        axes_ranges = 1
        plotter.show_bounds(axes_ranges=axes_ranges)

    with raises(TypeError, match='All of the elements'):
        axes_ranges = [0, 1, 'a', 'b', 2, 3]
        plotter.show_bounds(axes_ranges=axes_ranges)

    with raises(ValueError, match='[xmin, xmax, ymin, max, zmin, zmax]'):
        axes_ranges = [0, 1, 2, 3, 4]
        plotter.show_bounds(axes_ranges=axes_ranges)


@pytest.mark.skip_plotting
def test_camera_position():
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Sphere())
    plotter.show()
    assert isinstance(plotter.camera_position, pv.CameraPosition)


@pytest.mark.skip_plotting
def test_plotter_camera_position():
    plotter = pv.Plotter()
    plotter.set_position([1, 1, 1], render=True)


def test_renderer_set_viewup():
    plotter = pv.Plotter()
    plotter.renderer.set_viewup([1, 1, 1])


def test_reset_camera():
    plotter = pv.Plotter()
    plotter.reset_camera(bounds=(-1, 1, -1, 1, -1, 1))


def test_camera_is_set():
    plotter = pv.Plotter()
    assert not plotter.camera_set
    assert not plotter.renderer.camera_set

    renderer = pv.Renderer(plotter)
    assert not renderer.camera_set


def test_layer():
    plotter = pv.Plotter()
    plotter.renderer.layer = 1
    assert plotter.renderer.layer == 1
    plotter.renderer.layer = 0
    assert plotter.renderer.layer == 0


@pytest.mark.parametrize('has_border', (True, False))
def test_border(has_border):
    border_color = (1.0, 1.0, 1.0)
    border_width = 1
    plotter = pv.Plotter(border=has_border, border_color=border_color, border_width=border_width)
    assert plotter.renderer.has_border is has_border

    if has_border:
        assert plotter.renderer.border_color == border_color
    else:
        assert plotter.renderer.border_color is None

    if has_border:
        assert plotter.renderer.border_width == border_width
    else:
        assert plotter.renderer.border_width == 0


def test_bad_legend_origin_and_size(sphere):
    """Ensure bad parameters to origin/size raise ValueErrors."""
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    legend_labels = [['sphere', 'r']]
    with pytest.raises(ValueError, match='Invalid loc'):
        plotter.add_legend(labels=legend_labels, loc='bar')
    with pytest.raises(ValueError, match='size'):
        plotter.add_legend(labels=legend_labels, size=[])
    # test non-sequences also raise
    with pytest.raises(ValueError, match='size'):
        plotter.add_legend(labels=legend_labels, size=type)


@pytest.mark.parametrize('loc', ACTOR_LOC_MAP)
def test_add_legend_loc(loc):
    pl = pv.Plotter()
    pl.add_mesh(pv.PolyData([0.0, 0.0, 0.0]), label='foo')
    legend = pl.add_legend(loc=loc)

    # note: this is only valid with the defaults:
    # border=0.05 and size=(0.2, 0.2)
    positions = {
        'upper right': (0.75, 0.75),
        'upper left': (0.05, 0.75),
        'lower left': (0.05, 0.05),
        'lower right': (0.75, 0.05),
        'center left': (0.05, 0.4),
        'center right': (0.75, 0.4),
        'lower center': (0.4, 0.05),
        'upper center': (0.4, 0.75),
        'center': (0.4, 0.4),
    }
    assert legend.GetPosition() == positions[loc]


def test_add_legend_no_face(sphere):
    pl = pv.Plotter()
    sphere.point_data["Z"] = sphere.points[:, 2]
    pl.add_mesh(sphere, scalars='Z', label='sphere')
    pl.add_legend(face=None)

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.add_legend(labels=[['sphere', 'k']], face=None)


def test_add_remove_legend(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere, label='sphere')
    pl.add_legend()
    pl.remove_legend()


@pytest.mark.parametrize('face', ['-', '^', 'o', 'r', None, pv.PolyData([0.0, 0.0, 0.0])])
def test_legend_face(sphere, face):
    pl = pv.Plotter()
    pl.add_mesh(sphere, label='sphere')
    pl.add_legend(face=face)


class ActorBounds(vtk.vtkActor):
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds

    def GetBounds(self):
        return self.bounds


@pytest.mark.parametrize('override_GetBounds', [True, False])
@pytest.mark.parametrize('reset_camera', [True, False])
def test_bounds_and_reset_camera(override_GetBounds, reset_camera, verify_image_cache):
    # test the following:
    #  - renderer.bounds behaviour matches renderer.ComputeVisiblePropBounds())
    #  - renderer.reset_camera uses correct bounds when called
    #  - renderer bounds are correct when actor overrides GetBounds()
    #  - related bounds-dependent methods camera.tight and plotter.add_actor
    verify_image_cache.skip = True
    dataset = pv.ParametricEllipsoid(xradius=3, yradius=4, zradius=5)
    bounds_offset = 0.5
    if override_GetBounds:
        # define actor with re-defined bounds, e.g. add offset
        actor = ActorBounds(tuple(map(lambda x: x + bounds_offset, dataset.bounds)))
    else:
        actor = pv.Actor(mapper=pv.DataSetMapper(dataset))

    # check default bounds and camera position
    pl = pv.Plotter()
    initial_cpos = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    # Note: ComputeVisiblePropBounds() returns default uninitialized bounds with min/max reversed
    default_bounds = np.array((-1.0, 1, -1, 1, -1, 1))
    assert np.array_equal(pl.renderer.bounds, default_bounds)
    assert np.array_equal(pl.renderer.ComputeVisiblePropBounds(), -default_bounds)
    assert np.array_equal(pl.renderer.camera_position.to_list(), initial_cpos)
    pl.show()
    default_cpos = [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    assert np.array_equal(pl.renderer.get_default_cam_pos(), default_cpos)
    assert np.array_equal(pl.renderer.camera_position.to_list(), default_cpos)

    # do testing with add_actor
    pl = pv.Plotter()
    pl.add_actor(actor, reset_camera=reset_camera)

    # test renderer bounds
    expected_bounds = actor.GetBounds()
    actual_vtk_bounds = pl.renderer.ComputeVisiblePropBounds()
    actual_pyvista_bounds = pl.renderer.bounds
    assert np.allclose(actual_pyvista_bounds, expected_bounds)
    if override_GetBounds:
        # test that ComputeVisiblePropBounds returns incorrect result when actor overrides GetBounds()
        assert not np.allclose(actual_vtk_bounds, expected_bounds)
    else:
        assert np.allclose(actual_vtk_bounds, expected_bounds)

    # test that camera is only reset if called by add_actor
    cpos = pl.renderer.camera_position.to_list()
    if reset_camera:
        assert not np.array_equal(cpos, initial_cpos)
    else:
        assert np.array_equal(cpos, initial_cpos)

    # test that calling show() has no effect on the renderer's bounds
    pl.show()
    actual_vtk_bounds = pl.renderer.ComputeVisiblePropBounds()
    actual_pyvista_bounds = pl.renderer.bounds
    assert np.array_equal(actual_pyvista_bounds, expected_bounds)
    if override_GetBounds:
        assert not np.allclose(actual_vtk_bounds, expected_bounds)
    else:
        assert np.allclose(actual_vtk_bounds, expected_bounds)

    # test that the camera has been reset correctly
    expected_cpos = np.array(
        [
            (15.771915414303583, 15.771160342663569, 15.771160342663569),
            (0.0007550716400146484, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
    )
    if override_GetBounds:
        expected_cpos[(0, 1), :] += bounds_offset
    actual_cpos = pl.renderer.camera_position.to_list()
    assert np.allclose(actual_cpos, expected_cpos)

    # test that bounds are updated when making actor invisible
    actor.VisibilityOff()
    assert np.array_equal(pl.renderer.bounds, default_bounds)
    assert np.array_equal(pl.renderer.ComputeVisiblePropBounds(), -default_bounds)

    # test that camera has not yet changed
    actual_cpos = pl.renderer.camera_position.to_list()
    assert np.array_equal(actual_cpos, expected_cpos)

    # test that the camera is reset back to default
    pl.reset_camera()
    actual_cpos = pl.renderer.camera_position.to_list()
    assert np.array_equal(actual_cpos, default_cpos)

    # test camera.tight
    actor.VisibilityOn()
    pl.camera.tight()
    expected_cpos = np.array(
        [(0.0007550716400146, 0.0, 1.0), (0.0007550716400146, 0.0, 0.0), (0.0, 1.0, 0.0)]
    )
    if override_GetBounds:
        expected_cpos[(0, 1), :] += bounds_offset
    actual_cpos = pl.renderer.camera_position.to_list()
    assert np.allclose(actual_cpos, expected_cpos)

    # test actor bounds are ignored if UseBoundsOff()
    actor.UseBoundsOff()
    assert np.array_equal(pl.renderer.bounds, default_bounds)
    assert np.array_equal(pl.renderer.ComputeVisiblePropBounds(), -default_bounds)
