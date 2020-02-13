"""Quickly check if VTK off screen plotting works."""
import pyvista
from pyvista.plotting import system_supports_plotting

print('system_supports_plotting', system_supports_plotting())
pyvista.OFF_SCREEN = True
sphere = pyvista.Sphere()
pyvista.plot(sphere)
