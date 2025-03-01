from __future__ import annotations

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista.plotting.prop_collection import _PropCollection
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

    # make sure that the axes labels match the axes ranges
    labels_ranges = []
    for axis in range(3):
        axis_labels = plotter.renderer.cube_axes_actor.GetAxisLabels(axis)
        labels_ranges.append(float(axis_labels.GetValue(0)))
        labels_ranges.append(float(axis_labels.GetValue(axis_labels.GetNumberOfValues() - 1)))
    assert labels_ranges == axes_ranges


def test_show_grid_axes_ranges_with_all_edges():
    plotter = pv.Plotter()

    axes_ranges = [5, 10, 5, 10, 5, 10]
    plotter.show_grid(axes_ranges=axes_ranges, all_edges=True)
    labels_ranges = []
    for axis in range(3):
        axis_labels = plotter.renderer.cube_axes_actor.GetAxisLabels(axis)
        labels_ranges.append(float(axis_labels.GetValue(0)))
        labels_ranges.append(float(axis_labels.GetValue(axis_labels.GetNumberOfValues() - 1)))
    assert labels_ranges == axes_ranges


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
    axes_ranges = 1
    with pytest.raises(TypeError, match='numeric sequence'):
        plotter.show_bounds(axes_ranges=axes_ranges)

    axes_ranges = [0, 1, 'a', 'b', 2, 3]
    with pytest.raises(TypeError, match='All of the elements'):
        plotter.show_bounds(axes_ranges=axes_ranges)

    axes_ranges = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match='[xmin, xmax, ymin, max, zmin, zmax]'):
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


def test_view_isometric():
    plotter = pv.Plotter()
    plotter.view_isometric(bounds=(-1, 1, -1, 1, -1, 1))


def test_view_xy():
    plotter = pv.Plotter()
    plotter.view_xy(bounds=(-1, 1, -1, 1, -1, 1))


def test_view_yx():
    plotter = pv.Plotter()
    plotter.view_yx(bounds=(-1, 1, -1, 1, -1, 1))


def test_view_xz():
    plotter = pv.Plotter()
    plotter.view_xz(bounds=(-1, 1, -1, 1, -1, 1))


def test_view_zx():
    plotter = pv.Plotter()
    plotter.view_zx(bounds=(-1, 1, -1, 1, -1, 1))


def test_view_yz():
    plotter = pv.Plotter()
    plotter.view_yz(bounds=(-1, 1, -1, 1, -1, 1))


def test_view_zy():
    plotter = pv.Plotter()
    plotter.view_zy(bounds=(-1, 1, -1, 1, -1, 1))


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


@pytest.mark.parametrize('has_border', [True, False])
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
    sphere.point_data['Z'] = sphere.points[:, 2]
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
def test_legend_face(sphere, face, verify_image_cache):
    pl = pv.Plotter()
    pl.add_mesh(sphere, label='sphere')
    pl.add_legend(face=face, size=(0.5, 0.5))


def test_legend_from_glyph(sphere, verify_image_cache):
    pl = pv.Plotter()
    x = sphere.face_normals[:, 0] ** 2
    y = sphere.face_normals[:, 1] ** 2
    z = sphere.face_normals[:, 2] ** 2

    sphere['scale'] = (x**2 + y**2 + z**2) ** (1 / 2)
    sphere['normals'] = sphere.face_normals * 0.1

    arrows = sphere.glyph(scale='scale', orient='normals', tolerance=0.05)
    pl.add_mesh(arrows, color='red', label='Magnitude')
    pl.add_mesh(sphere)
    pl.add_legend(size=(0.5, 0.5))


def test_legend_from_multiple_glyph(random_hills, verify_image_cache):
    pl = pv.Plotter()

    random_hills['Normals2'] = -1 * random_hills['Normals'].copy()

    arrows = random_hills.glyph(scale='Normals', orient='Normals', tolerance=0.05)
    pl.add_mesh(arrows, color='black', label='label 1')

    arrows2 = random_hills.glyph(scale='Normals', orient='Normals2', tolerance=0.05)
    pl.add_mesh(arrows2, color='red', label='label 2')

    pl.add_mesh(random_hills, scalars='Elevation', cmap='terrain', show_scalar_bar=False)

    pl.add_legend(size=(0.5, 0.5))
    pl.show()


def test_legend_using_add_legend(random_hills, verify_image_cache):
    pl = pv.Plotter()

    arrows = random_hills.glyph(scale='Normals', orient='Normals', tolerance=0.05)
    pl.add_mesh(arrows, color='black', label='label 1')

    pl.add_mesh(random_hills, scalars='Elevation', cmap='terrain', show_scalar_bar=False)

    legend_entries = []
    legend_entries.append(['my label 1', 'g'])
    legend_entries.append(['my label 2', 'blue'])
    pl.add_legend(legend_entries, size=(0.5, 0.5))
    pl.show()


def test_legend_using_add_legend_with_glyph(random_hills, verify_image_cache):
    pl = pv.Plotter()

    arrows = random_hills.glyph(scale='Normals', orient='Normals', tolerance=0.05)
    pl.add_mesh(arrows, color='black', label='label 1')

    pl.add_mesh(random_hills, scalars='Elevation', cmap='terrain', show_scalar_bar=False)

    legend_entries = []
    legend_entries.append(['my label 1', 'g'])
    legend_entries.append(['my label 2', 'blue', pv.Circle()])
    legend_entries.append({'label': 'my label 3', 'color': (0.0, 1.0, 1.0), 'face': pv.Arrow()})
    legend_entries.append({'label': 'my label 3', 'color': (0.0, 1.0, 1.0), 'face': 'circle'})
    legend_entries.append({'label': 'my label 3', 'color': (0.0, 1.0, 1.0), 'face': None})

    pl.add_legend(legend_entries, size=(0.5, 0.5))
    pl.show()


def test_legend_using_add_legend_only_labels(random_hills, verify_image_cache):
    pl = pv.Plotter()

    arrows = random_hills.glyph(scale='Normals', orient='Normals', tolerance=0.05)
    pl.add_mesh(arrows, color='black', label='label 1')

    pl.add_mesh(random_hills, scalars='Elevation', cmap='terrain', show_scalar_bar=False)

    legend_entries = ['label 1', 'label 2']

    pl.add_legend(legend_entries, size=(0.5, 0.5))
    pl.show()


def test_legend_none_face(verify_image_cache):
    """Verifies that ``face="none"`` does not add a face for each label in legend."""
    pl = pv.Plotter()
    pl.add_mesh(
        pv.Icosphere(center=(3, 0, 0), radius=1),
        color='r',
        label='Sphere',
    )
    pl.add_mesh(pv.Box(), color='w', label='Box')
    # add a large legend to ensure test fails if face="none" not configured right
    pl.add_legend(face='none', bcolor='k', size=(0.6, 0.6))
    pl.show()


def test_legend_add_entry_warning(verify_image_cache):
    pl = pv.Plotter()
    legend_entries = [{'label': 'my label 3', 'color': (0.0, 1.0, 1.0), 'non_used_arg': 'asdf'}]

    with pytest.warns(UserWarning, match='Some of the arguments given to legend are not used'):
        pl.add_legend(legend_entries, size=(0.5, 0.5))
        pl.show()


def test_legend_add_entry_exception():
    pl = pv.Plotter()
    legend_entries = np.array([1, 2])  # Not allowed type

    with pytest.raises(ValueError, match='The object passed to the legend'):
        pl.add_legend(legend_entries)
    pl.show()


def test_add_legend_background_opacity(sphere):
    background_opacity = 0.4
    pl = pv.Plotter()
    pl.add_mesh(sphere, label='sphere')
    actor = pl.add_legend(background_opacity=background_opacity)
    assert actor.GetBackgroundOpacity() == background_opacity


def test_viewport():
    pl = pv.Plotter(shape=(1, 2))
    assert pl.renderers[0].viewport == (0.0, 0.0, 0.5, 1.0)
    pl.renderers[0].viewport = (0.125, 0.25, 0.375, 0.75)
    assert pl.renderers[0].viewport == (0.125, 0.25, 0.375, 0.75)


def test_actors_prop_collection_init():
    pl = pv.Plotter()
    prop_collection = pl.renderer._actors
    assert isinstance(prop_collection, _PropCollection)
    assert prop_collection._prop_collection is pl.renderer.GetViewProps()
    assert len(prop_collection) == 0

    pl.add_mesh(pv.Sphere())

    assert len(prop_collection) == 1
    keys = list(pl.renderer.actors.keys())
    assert len(keys) == 1
    assert keys[0].startswith('PolyData(Addr=0')
    assert pl.renderer._actors is prop_collection


@pytest.fixture
def prop_collection():
    vtk_collection = vtk.vtkPropCollection()
    coll = _PropCollection(vtk_collection)
    yield coll
    del vtk_collection
    del coll._prop_collection


def test_actors_prop_collection():
    pl = pv.Plotter()
    collection = pl.renderer._actors
    assert isinstance(collection, _PropCollection)
    sphere = pv.Sphere()
    pl.add_mesh(sphere, name='sphere')

    # Test items
    items = list(collection.items())
    assert len(items) == 1
    name, actor = items[0]
    assert name == 'sphere'
    assert isinstance(actor, pv.Actor)
    assert actor.mapper.dataset is sphere


def test_prop_collection_append(prop_collection):
    a = pv.Actor(name='a')
    b = pv.Actor(name='b')

    prop_collection.append(a)
    assert prop_collection.keys() == ['a']
    assert prop_collection[0] is a

    prop_collection.append(b)
    assert prop_collection.keys() == ['a', 'b']
    assert prop_collection[1] is b


def test_prop_collection_insert(prop_collection):
    axes = pv.AxesAssembly(name='axes')
    prop_collection.insert(0, axes)
    assert prop_collection.keys() == ['axes']

    # Test negative index
    cube = pv.Actor(name='cube')
    prop_collection.insert(-1, cube)
    assert prop_collection.keys() == ['axes', 'cube']

    # Test large index
    label = pv.Label(name='label')
    prop_collection.insert(42, label)
    assert prop_collection.keys() == ['axes', 'cube', 'label']


def test_prop_collection_delitem(prop_collection):
    prop_collection.append(pv.Actor(name='a'))
    prop_collection.append(pv.Actor(name='b'))
    prop_collection.append(pv.Actor(name='c'))
    prop_collection.append(pv.Actor(name='d'))
    assert prop_collection.keys() == ['a', 'b', 'c', 'd']

    # Test delitem index
    del prop_collection[0]
    assert prop_collection.keys() == ['b', 'c', 'd']

    # Test delitem negative index
    del prop_collection[-1]
    assert prop_collection.keys() == ['b', 'c']

    # Test delitem index out of range
    with pytest.raises(IndexError):
        del prop_collection[2]
    del prop_collection[1]
    assert prop_collection.keys() == ['b']

    # Test delitem name
    del prop_collection['b']
    assert prop_collection.keys() == []

    with pytest.raises(KeyError, match=r'No item found with name \'b\'.'):
        del prop_collection['b']


def test_prop_collection_setitem(prop_collection):
    a = pv.Actor(name='a')
    b = pv.Actor(name='b')
    c = pv.Actor(name='c')
    prop_collection.append(a)
    prop_collection.append(b)
    prop_collection.append(c)
    assert prop_collection.keys() == ['a', 'b', 'c']

    # Test setitem index
    prop_collection[0] = b
    assert prop_collection.keys() == ['b', 'b', 'c']

    # Test setitem negative index
    prop_collection[-1] = a
    assert prop_collection.keys() == ['b', 'b', 'a']

    # Test setitem index error
    with pytest.raises(IndexError, match='Index out of range.'):
        prop_collection[-4] = c

    # Test setitem name
    a2 = pv.Actor(name='a2', mapper=pv.DataSetMapper(pv.Sphere()))
    match = "Name of the new actor 'a2' must match the key name 'a'."
    with pytest.raises(ValueError, match=match):
        prop_collection['a'] = a2
    a2.name = 'a'
    prop_collection['a'] = a2
    assert prop_collection.keys() == ['b', 'b', 'a']
    assert prop_collection['a'] is a2


def test_prop_collection_raises(prop_collection):
    # Test invalid types
    match = 'Key must be an index or a string, got dict.'
    with pytest.raises(TypeError, match=match):
        del prop_collection[{}]
    with pytest.raises(TypeError, match=match):
        _ = prop_collection[{}]
    with pytest.raises(TypeError, match=match):
        prop_collection[{}] = pv.Actor()
