from __future__ import annotations

import pytest

import pyvista as pv
from pyvista import _vtk

KEY = 'Data'


@pytest.fixture
def scalar_bars(sphere):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    pl.add_scalar_bar(
        KEY,
        interactive=True,
        vertical=False,
        outline=True,
        fmt='%10.5f',
        nan_annotation=True,
        fill=True,
        background_color='k',
    )
    return pl.scalar_bars


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
    assert next(iter(scalar_bars.keys())) == KEY
    assert isinstance(next(iter(scalar_bars.values())), _vtk.vtkScalarBarActor)

    for key, value in scalar_bars.items():
        assert isinstance(value, _vtk.vtkScalarBarActor)
        assert key == 'Data'

    assert isinstance(scalar_bars[KEY], _vtk.vtkScalarBarActor)


def test_clear(scalar_bars):
    assert len(scalar_bars) == 1
    scalar_bars.clear()
    assert len(scalar_bars) == 0


def test_update_title(scalar_bars):
    new_title = 'Elevation'
    scalar_bars.update_title(KEY, new_title)

    # Verify internal dicts are re-keyed
    assert KEY not in scalar_bars
    assert new_title in scalar_bars
    assert new_title in scalar_bars._scalar_bar_ranges
    assert new_title in scalar_bars._scalar_bar_mappers
    assert len(scalar_bars) == 1

    # Verify VTK actor title is updated
    assert scalar_bars[new_title].GetTitle() == new_title

    # Verify slot lookup is re-keyed
    assert KEY not in scalar_bars._plotter._scalar_bar_slot_lookup
    assert new_title in scalar_bars._plotter._scalar_bar_slot_lookup


def test_update_title_same(scalar_bars):
    scalar_bars.update_title(KEY, KEY)
    assert KEY in scalar_bars
    assert len(scalar_bars) == 1


def test_update_title_not_found(scalar_bars):
    with pytest.raises(KeyError, match='not found'):
        scalar_bars.update_title('DoesNotExist', 'New')


def test_update_title_conflict(scalar_bars):
    scalar_bars.add_scalar_bar('Other', mapper=scalar_bars._plotter.mapper)
    with pytest.raises(ValueError, match='already exists'):
        scalar_bars.update_title(KEY, 'Other')


def test_update_title_image(sphere, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    sphere['Data'] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars='Data')
    pl.scalar_bars.update_title('Data', 'Elevation')
    pl.show()


def test_too_many_scalar_bars():
    pl = pv.Plotter()
    with pytest.raises(RuntimeError, match='Maximum number of color'):  # noqa: PT012
        for i in range(100):
            mesh = pv.Sphere()
            mesh[str(i)] = range(mesh.n_points)
            pl.add_mesh(mesh)
