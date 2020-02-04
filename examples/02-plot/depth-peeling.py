"""
Depth Peeling
~~~~~~~~~~~~~

Depth peeling is a technique to correctly render translucent geometry.
This is enabled by default in :any:`pyvista.rcParams`.

For this example, we will showcase the difference that depth peeling provides
as the justification for why we have enabled this by default.

"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

###############################################################################
# Turn off depth peeling so we can correctly control it in this example
pv.rcParams["depth_peeling"]["enabled"] = False


###############################################################################
centers = [(0, 0, 0), (1, 0, 0), (-1, 0, 0),
           (0, 1, 0), (0, -1, 0)]
radii = [1, 0.5, 0.5, 0.5, 0.5]

spheres = pv.MultiBlock()
for i, c in enumerate(centers):
    spheres.append(pv.Sphere(center=c, radius=radii[i]))

###############################################################################
dargs = dict(opacity=0.5, color="red", smooth_shading=True)

p = pv.Plotter(shape=(1,2), multi_samples=8)

p.add_mesh(spheres, **dargs)
p.enable_depth_peeling(10)
p.add_text("Depth Peeling")

p.subplot(0,1)
p.add_text("Standard")
p.add_mesh(spheres.copy(), **dargs)

p.link_views()
p.camera_position = [(11.695377287877744, 4.697473022306675, -4.313491106516902),
 (0.0, 0.0, 0.0),
 (0.3201103754961452, 0.07054027895287238, 0.944750451995112)]
p.show()


###############################################################################

mesh = examples.download_brain().contour(5)
cmap = "viridis_r"

p = pv.Plotter(shape=(1,2), multi_samples=4)

p.add_mesh(mesh, opacity=0.5, cmap=cmap)
p.enable_depth_peeling(10)
p.add_text("Depth Peeling")

p.subplot(0,1)
p.add_text("Standard")
p.add_mesh(mesh.copy(), opacity=0.5, cmap=cmap)

p.link_views()
p.camera_position = [(418.29917315895693, 658.9752095516966, 53.784143976243364),
 (90.19444444775581, 111.46052622795105, 90.0),
 (0.03282296324460818, 0.046369526043831856, 0.9983849558854109)]
p.show()

###############################################################################
# Re-enable depth peeling
pv.rcParams["depth_peeling"]["enabled"] = True
