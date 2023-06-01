import pytest
from pytest import raises

import pyvista
from pyvista.plotting import system_supports_plotting
from pyvista.plotting.renderer import ACTOR_LOC_MAP


def test_show_bounds_axes_ranges():
    plotter = pyvista.Plotter()

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
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    actor0 = plotter.show_bounds()
    assert actor0.GetUseTextActor3D()
    plotter.set_scale(0.5, 0.5, 2)
    actor1 = plotter.show_bounds()
    assert not actor1.GetUseTextActor3D()


def test_show_bounds_invalid_axes_ranges():
    plotter = pyvista.Plotter()

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


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_camera_position():
    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.Sphere())
    plotter.show()
    assert isinstance(plotter.camera_position, pyvista.CameraPosition)


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_plotter_camera_position():
    plotter = pyvista.Plotter()
    plotter.set_position([1, 1, 1], render=True)


def test_renderer_set_viewup():
    plotter = pyvista.Plotter()
    plotter.renderer.set_viewup([1, 1, 1])


def test_reset_camera():
    plotter = pyvista.Plotter()
    plotter.reset_camera(bounds=(-1, 1, -1, 1, -1, 1))


def test_camera_is_set():
    plotter = pyvista.Plotter()
    assert not plotter.camera_set
    assert not plotter.renderer.camera_set

    renderer = pyvista.Renderer(plotter)
    assert not renderer.camera_set


def test_layer():
    plotter = pyvista.Plotter()
    plotter.renderer.layer = 1
    assert plotter.renderer.layer == 1
    plotter.renderer.layer = 0
    assert plotter.renderer.layer == 0


@pytest.mark.parametrize('has_border', (True, False))
def test_border(has_border):
    border_color = (1.0, 1.0, 1.0)
    border_width = 1
    plotter = pyvista.Plotter(
        border=has_border, border_color=border_color, border_width=border_width
    )
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
    plotter = pyvista.Plotter()
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
    pl = pyvista.Plotter()
    pl.add_mesh(pyvista.PolyData([0.0, 0.0, 0.0]), label='foo')
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
    pl = pyvista.Plotter()
    sphere.point_data["Z"] = sphere.points[:, 2]
    pl.add_mesh(sphere, scalars='Z', label='sphere')
    pl.add_legend(face=None)

    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.add_legend(labels=[['sphere', 'k']], face=None)


def test_add_remove_legend(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, label='sphere')
    pl.add_legend()
    pl.remove_legend()


@pytest.mark.parametrize('face', ['-', '^', 'o', 'r', None, pyvista.PolyData([0.0, 0.0, 0.0])])
def test_legend_face(sphere, face):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere, label='sphere')
    pl.add_legend(face=face)
