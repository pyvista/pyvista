"""
.. _show_edges_example:

Show Edges
~~~~~~~~~~

Show the edges of all geometries within a mesh using the
:attr:`~pyvista.Property.show_edges` property.

"""

# %%
# Sometimes it can be useful to show all of the edges of a mesh when rendering
# to communicate aspects of the dataset like resolution.
#
# Showing the edges for any rendered dataset is as simple as specifying the
# the ``show_edges`` keyword argument to ``True`` when plotting a dataset.

# sphinx_gallery_thumbnail_number = 1
from pyvista import examples

bust = examples.download_washington_bust()

bust.plot(show_edges=True, color=True)
# %%
# .. tags:: plot
