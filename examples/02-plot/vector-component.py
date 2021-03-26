"""
Vector Component
~~~~~~~~~~~~~~~~

Plot a single component of a vector as a scalar array.

We can plot individual components of multi-component arrays with the
``component`` argument  of the ``add_mesh`` method.
"""

import numpy as np
import pyvista as pv

###############################################################################
# Start with a plane containing a vector with value of [1, 2, 3]
# everywhere. The magnitude of this vector is 3.74.
###############################################################################
mesh = pv.Plane()
data = np.ones([mesh.n_cells, 3])
data[:] = [1.0, 2.0, 3.0]
mesh["Velocity Vector"] = data
###############################################################################
# The default behavior with no component specified is to use the vector
# magnitude. We can access each component by specifying the component argument.
###############################################################################
dargs = dict(
    scalars="Velocity Vector",
    cmap="rainbow",
    clim=[1.0, 4.0],
    scalar_bar_args = dict(n_labels=4)
)
p = pv.Plotter(shape=(2,2))
p.subplot(0,0)
p.add_mesh(mesh, **dargs)
p.add_text("Vector Magnitude")
p.subplot(0,1)
p.add_mesh(mesh.copy(), component=0, **dargs)
p.add_text("Component 0")
p.subplot(1,0)
p.add_mesh(mesh.copy(), component=1, **dargs)
p.add_text("Component 1")
p.subplot(1,1)
p.add_mesh(mesh.copy(), component=2, **dargs)
p.add_text("Component 2")
p.link_views()
p.show()
