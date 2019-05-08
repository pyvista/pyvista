import sys

import pytest


import vista
from vista.plotting import system_supports_plotting

if __name__ != '__main__':
    OFF_SCREEN = 'pytest' in sys.modules
else:
    OFF_SCREEN = False


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_camera_position():
    plotter = vista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(vista.Sphere())
    plotter.show()
    assert isinstance(plotter.camera_position, list)
