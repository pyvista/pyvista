import pyvista as pv
from pyvista import examples

filename = examples.download_tos_O1_2001_2002(load=False)
reader = pv.get_reader(filename)

grid = reader.read()
grid.set_active_scalars("tos")
grid.plot(cpos="xy")
