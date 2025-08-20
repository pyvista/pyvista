from __future__ import annotations

from pathlib import Path

import pytest

import pyvista as pv
from pyvista import examples

pytestmark = pytest.mark.needs_download


def test_download_gltf_milk_truck():
    filename = examples.gltf.download_milk_truck()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


def test_download_gltf_damaged_helmet():
    filename = examples.gltf.download_damaged_helmet()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


@pytest.mark.needs_vtk_version(
    less_than=(9, 1),
    reason='Skip until glTF extension KHR_texture_transform is supported.',
)
def test_download_gltf_sheen_chair():
    filename = examples.gltf.download_sheen_chair()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


def test_download_gltf_gearbox():
    filename = examples.gltf.download_gearbox()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


def test_download_gltf_avocado():
    filename = examples.gltf.download_avocado()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)
