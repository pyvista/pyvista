import sys

import pytest


import vtki
from vtki.plotting import system_supports_plotting

if __name__ != '__main__':
    OFF_SCREEN = 'pytest' in sys.modules
else:
    OFF_SCREEN = False


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires X11")
def test_camera_position():
    plotter = vtki.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(vtki.Sphere())
    plotter.show()
    assert isinstance(plotter.camera_position, list)
