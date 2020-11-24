"""
Plane Widget
~~~~~~~~~~~~

The plane widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_plane_widget` and
:func:`pyvista.WidgetHelper.clear_plane_widgets` methods respectively.
As with all widgets, you must provide a custom callback method to utilize that
plane. Considering that planes are most commonly used for clipping and slicing
meshes, we have included two helper methods for doing those tasks!

Let's use a plane to clip a mesh:
"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

vol = examples.download_brain()

p = pv.Plotter()
p.add_mesh_clip_plane(vol)
p.show()

###############################################################################
# After interacting with the scene, the clipped mesh is available as:
p.plane_clipped_meshes

###############################################################################
# And here is a screen capture of a user interacting with this
#
# .. image:: ../../images/gifs/plane-clip.gif

###############################################################################
# Or you could slice a mesh using the plane widget:

p = pv.Plotter()
p.add_mesh_slice(vol)
p.show()
###############################################################################
# After interacting with the scene, the slice is available as:
p.plane_sliced_meshes

###############################################################################
# And here is a screen capture of a user interacting with this
#
# .. image:: ../../images/gifs/plane-slice.gif

###############################################################################
# Or you could leverage the plane widget for some custom task like glyphing a
# vector field along that plane. Note that we have to pass a ``name`` when
# calling ``add_mesh`` to ensure that there is only one set of glyphs plotted
# at a time.

import pyvista as pv
from pyvista import examples

mesh = examples.download_carotid()

p = pv.Plotter()
p.add_mesh(mesh.contour(8).extract_largest(), opacity=0.5)

def my_plane_func(normal, origin):
    slc = mesh.slice(normal=normal, origin=origin)
    arrows = slc.glyph(orient='vectors', scale="scalars", factor=0.01)
    p.add_mesh(arrows, name='arrows')

p.add_plane_widget(my_plane_func)
p.show_grid()
p.add_axes()
p.show()

###############################################################################
# And here is a screen capture of a user interacting with this
#
# .. image:: ../../images/gifs/plane-glyph.gif
