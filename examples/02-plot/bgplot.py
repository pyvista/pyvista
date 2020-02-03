"""
BackgroundPlotter example.
~~~~~~~~~~~~~~~~~~~~~~~~~~

This simple sphinx example uses BackgroundPlotter.

"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv

###############################################################################

p = pv.BackgroundPlotter(
    shape=(2, 2),
    border=True,
    border_width=10,
    border_color='grey',
)
p.set_background('black', top='blue')

p.subplot(0, 0)
actor = p.add_mesh(pv.Cone())
p.remove_actor(actor)
p.add_text('Actor is removed')
p.subplot(0, 1)
p.add_mesh(pv.Sphere(), smooth_shading=True)
p.subplot(1, 0)
p.add_mesh(pv.Cylinder())
p.show_bounds()
p.subplot(1, 1)
p.add_mesh(pv.Box(), color='green', opacity=0.8)

###############################################################################

p = pv.BackgroundPlotter()
p.add_mesh(pv.Sphere())
p.close()
