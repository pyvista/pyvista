"""
Vector Component
~~~~~~~~~~~~~~~~

Plot a single component of a vector as a scalar array.

We can plot individual components of multi-component arrays with the
``component`` argument  of the ``add_mesh`` method.
"""

import pyvista as pv
from pyvista import examples

mesh = examples.download_carotid().threshold(145, scalars="scalars")
mesh = mesh.compute_gradient(scalars="vectors")

###############################################################################

dargs = dict(scalars="gradient", cmap="rainbow")

p = pv.Plotter(shape=(1,3))
p.subplot(0,0)
p.add_mesh(mesh, component=0, **dargs)
p.add_text("Component 0")
p.subplot(0,1)
p.add_mesh(mesh.copy(), component=1, **dargs)
p.add_text("Component 1")
p.subplot(0,2)
p.add_mesh(mesh.copy(), component=2, **dargs)
p.add_text("Component 2")
p.link_views()
p.camera_position = [(342.8, 200.06, 89.8),
                     (137.5, 104.0, 23.5),
                     (-0.23, -0.165, 0.955)]
p.show()
