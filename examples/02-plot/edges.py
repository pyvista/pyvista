"""
.. _show_edges_example:

Show Edges
~~~~~~~~~~

Show the edges of all geometries within a mesh
"""
###############################################################################
# Sometimes it can be useful to show all of the edges of a mesh when rendering
# to communicate aspects of the dataset like resolution.
#
# Showing the edges for any rendered dataset is as simple as specifying the
# the ``show_edges`` keyword argument to ``True`` when plotting a dataset.

# sphinx_gallery_thumbnail_number = 1
from pyvista import examples

nefertiti = examples.download_nefertiti()

# Camera position to zoom to face
face_view = [
    (194.57658338658473, -327.5539184202715, 28.106692235139377),
    (-10.46795453395034, -67.33281919301498, -19.938084799559192),
    (-0.05444711191580967, 0.13964269728441056, 0.9887039137674948),
]


nefertiti.plot(cpos=face_view, show_edges=True, color=True)
