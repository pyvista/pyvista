"""
Orbiting
~~~~~~~~~

Orbit around a scene
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

mesh = examples.download_st_helens().warp_by_scalar()

################################################################################

p = pv.Plotter()
p.add_mesh(mesh)

p.show(auto_close=False)
path = p.generate_orbital_path(n_points=36)
p.open_gif('orbit.gif')
p.orbit_on_path(path, write_frames=True)
p.close()


################################################################################

mesh = examples.download_dragon()
viewup = [0,1,0]

################################################################################
p = pv.Plotter()
p.add_mesh(mesh)

p.show(auto_close=False)
path = p.generate_orbital_path(factor=2., n_points=36, viewup=viewup, shift=0.2)
p.open_gif('orbit.gif')
p.orbit_on_path(path, write_frames=True, viewup=viewup)
p.close()
