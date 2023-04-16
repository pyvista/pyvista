"""
.. _goniometer

Goniometer
~~~~~~~~~~

The 3D-ruler axis style, a flag pole and a goniometer.
This example is inspired by [vedo's example](https://vedo.embl.es/docs/vedo/addons.html#Goniometer).

"""

import pyvista as pv

mesh = pv.Cone()
pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True)
pl.show()
