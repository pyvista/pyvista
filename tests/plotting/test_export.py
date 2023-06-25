import os

import numpy as np
import pytest

import pyvista
from pyvista import examples as ex
from pyvista.core.errors import PyVistaDeprecationWarning


@pytest.mark.skip_plotting
def test_export_single(tmpdir, skip_check_gc):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-single'))
    data = ex.load_airplane()
    # Create the scene
    plotter = pyvista.Plotter()
    plotter.add_mesh(data)
    plotter.export_vtksz(filename)
    # Now make sure the file is there
    assert os.path.isfile(f'{filename}')


@pytest.mark.skip_plotting
def test_export_multi(tmpdir, skip_check_gc):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-multi'))
    multi = pyvista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(ex.load_rectilinear())
    # Create the scene
    plotter = pyvista.Plotter()
    plotter.add_mesh(multi)
    plotter.export_vtksz(filename)
    # Now make sure the file is there
    assert os.path.isfile(f'{filename}')


@pytest.mark.skip_plotting
def test_export_texture(tmpdir, skip_check_gc):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-texture'))
    data = ex.load_globe()
    # Create the scene
    plotter = pyvista.Plotter()
    with pytest.warns(PyVistaDeprecationWarning):
        plotter.add_mesh(data, texture=True)
    plotter.export_vtksz(filename)
    # Now make sure the file is there
    assert os.path.isfile(f'{filename}')


@pytest.mark.skip_plotting
def test_export_verts(tmpdir, skip_check_gc):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-verts'))
    data = pyvista.PolyData(np.random.rand(100, 3))
    # Create the scene
    plotter = pyvista.Plotter()
    plotter.add_mesh(data)
    plotter.export_vtksz(filename)
    # Now make sure the file is there
    assert os.path.isfile(f'{filename}')


@pytest.mark.skip_plotting
def test_export_color(tmpdir, skip_check_gc):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-color'))
    data = ex.load_airplane()
    # Create the scene
    plotter = pyvista.Plotter()
    plotter.add_mesh(data, color='yellow')
    plotter.export_vtksz(filename)
    # Now make sure the file is there
    assert os.path.isfile(f'{filename}')
