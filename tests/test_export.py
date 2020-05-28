import os
import sys

import numpy as np
import pytest

import pyvista
from pyvista import examples as ex
from pyvista.plotting import system_supports_plotting

if __name__ != '__main__':
    OFF_SCREEN = 'pytest' in sys.modules
else:
    OFF_SCREEN = False


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_export_single(tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-single'))
    data = ex.load_airplane()
    # Create the scene
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(data)
    plotter.export_vtkjs(filename)
    cpos_out = plotter.show() # Export must be called before showing!
    plotter.close()
    # Now make sure the file is there
    assert os.path.isfile('{}.vtkjs'.format(filename))


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_export_multi(tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-multi'))
    multi = pyvista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(ex.load_rectilinear())
    # Create the scene
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(multi)
    plotter.export_vtkjs(filename, compress_arrays=True)
    cpos_out = plotter.show() # Export must be called before showing!
    plotter.close()
    # Now make sure the file is there
    assert os.path.isfile('{}.vtkjs'.format(filename))


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_export_texture(tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-texture'))
    data = ex.load_globe()
    # Create the scene
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(data, texture=True)
    plotter.export_vtkjs(filename)
    cpos_out = plotter.show() # Export must be called before showing!
    plotter.close()
    # Now make sure the file is there
    assert os.path.isfile('{}.vtkjs'.format(filename))


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_export_verts(tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-verts'))
    data = pyvista.PolyData(np.random.rand(100, 3))
    # Create the scene
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(data)
    plotter.export_vtkjs(filename)
    cpos_out = plotter.show() # Export must be called before showing!
    plotter.close()
    # Now make sure the file is there
    assert os.path.isfile('{}.vtkjs'.format(filename))


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_export_color(tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('scene-color'))
    data = ex.load_airplane()
    # Create the scene
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(data, color='yellow')
    plotter.export_vtkjs(filename)
    cpos_out = plotter.show() # Export must be called before showing!
    plotter.close()
    # Now make sure the file is there
    assert os.path.isfile('{}.vtkjs'.format(filename))


def test_vtkjs_url():
    file_url = 'https://www.dropbox.com/s/6m5ttdbv5bf4ngj/ripple.vtkjs?dl=0'
    vtkjs_url = 'http://viewer.pyvista.org/?fileURL=https://dl.dropbox.com/s/6m5ttdbv5bf4ngj/ripple.vtkjs?dl=0'
    assert vtkjs_url in pyvista.get_vtkjs_url(file_url)
    assert vtkjs_url in pyvista.get_vtkjs_url('dropbox', file_url)
