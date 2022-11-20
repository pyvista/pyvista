import pytest

import pyvista as pv
from pyvista import _vtk

KEY = 'Data'


@pytest.fixture()
def scalar_bars(sphere):
    sphere[KEY] = sphere.points[:, 2]

    plotter = pv.Plotter()
    plotter.add_mesh(sphere, show_scalar_bar=False)
    plotter.add_scalar_bar(
        KEY,
        interactive=True,
        vertical=False,
        outline=True,
        fmt='%10.5f',
        nan_annotation=True,
        fill=True,
        background_color='k',
    )
    return plotter.scalar_bars


def test_repr(scalar_bars):
    repr_ = repr(scalar_bars)
    assert f'"{KEY}"' in repr_
    assert 'False' in repr_ or 'True' in repr_, 'missing interactive flag'
    assert 'Scalar Bar Title     Interactive' in repr_


def test_remove_fail(scalar_bars):
    scalar_bars.add_scalar_bar('MOARDATA', mapper=scalar_bars._plotter.mapper)
    with pytest.raises(ValueError, match='Multiple scalar bars'):
        scalar_bars.remove_scalar_bar()


def test_add_fail(scalar_bars):
    with pytest.raises(ValueError, match='Mapper cannot be ``None``'):
        scalar_bars.add_scalar_bar('MOARDATA')


def test_dict(scalar_bars):
    assert KEY in scalar_bars
    assert 'Data' in scalar_bars.keys()
    assert len(scalar_bars) == 1
    assert list(scalar_bars.keys())[0] == KEY
    assert isinstance(list(scalar_bars.values())[0], _vtk.vtkScalarBarActor)

    for key, value in scalar_bars.items():
        assert isinstance(value, _vtk.vtkScalarBarActor)
        assert key == 'Data'

    assert isinstance(scalar_bars[KEY], _vtk.vtkScalarBarActor)


def test_clear(scalar_bars):
    assert len(scalar_bars) == 1
    scalar_bars.clear()
    assert len(scalar_bars) == 0


def test_actor_removal(sphere):
    # verify that when removing an actor we also remove the
    # corresponding scalar bar

    sphere['scalars'] = sphere.points[:, 2]

    pl = pv.Plotter()
    actor = pl.add_mesh(sphere, show_scalar_bar=True)
    assert list(pl.scalar_bars.keys()) == ['scalars']
    pl.remove_actor(actor)
    assert len(pl.scalar_bars) == 0


def test_add_remove_bar(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars=sphere.points[:, 2], show_scalar_bar=False)

    # verify that the number of slots is restored
    init_slots = len(pl._scalar_bar_slots)
    pl.add_scalar_bar(interactive=True)
    pl.remove_scalar_bar()
    assert len(pl._scalar_bar_slots) == init_slots


def test_too_many_scalar_bars(sphere):
    pl = pv.Plotter()
    with pytest.raises(RuntimeError, match='Maximum number of color'):
        for i in range(100):
            mesh = pv.Sphere()
            mesh[str(i)] = range(mesh.n_points)
            pl.add_mesh(mesh)


def test_update_scalar_bar_range(sphere):
    sphere['z'] = sphere.points[:, 2]
    minmax = sphere.bounds[2:4]  # ymin, ymax
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, scalars='z')

    # automatic mapper lookup works
    plotter.update_scalar_bar_range(minmax)
    # named mapper lookup works
    plotter.update_scalar_bar_range(minmax, name='z')
    # missing name raises
    with pytest.raises(ValueError, match='not valid/not found in this plotter'):
        plotter.update_scalar_bar_range(minmax, name='invalid')
