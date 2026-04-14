from __future__ import annotations

import pytest

import pyvista as pv
from pyvista.plotting import _vtk


@pytest.fixture
def volume_mapper():
    vol = pv.ImageData(dimensions=(10, 10, 10))
    vol['scalars'] = 255 - vol.z * 25
    pl = pv.Plotter()
    actor = pl.add_volume(vol)
    return actor.mapper


@pytest.mark.skip_check_gc
def test_volume_mapper_dataset(volume_mapper):
    assert isinstance(volume_mapper.dataset, pv.ImageData)


@pytest.mark.skip_check_gc
def test_volume_mapper_blend_mode(volume_mapper):
    assert isinstance(volume_mapper.blend_mode, str)

    volume_mapper.blend_mode = _vtk.vtkVolumeMapper.COMPOSITE_BLEND
    assert volume_mapper.blend_mode == 'composite'

    for mode in ['average', 'minimum', 'maximum', 'composite', 'additive']:
        volume_mapper.blend_mode = mode
        assert volume_mapper.blend_mode == mode

    with pytest.raises(ValueError, match='Please choose either "additive"'):
        volume_mapper.blend_mode = 'not a mode'

    with pytest.raises(TypeError, match='int or str'):
        volume_mapper.blend_mode = 0.5
