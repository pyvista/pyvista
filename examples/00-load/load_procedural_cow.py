"""
.. _procedural_cow_example:

Procedural Cow
~~~~~~~~~~~~~~

Load a cow whose geometry and color share one file, then read the generating script.

"""

from pyvista import examples

# %%
# :func:`~pyvista.examples.downloads.download_procedural_cow` returns one
# :class:`~pyvista.PolyData` surface of 300,000 triangles. The file is in
# PyVista's native ``.pv`` format, which is read by the ``pyvista-zstd``
# package.

cow = examples.download_procedural_cow()
cow

# %%
# The coat is the active point array, so plotting it takes ``rgb=True`` instead
# of a texture and texture coordinates.

cow.plot(rgb=True, smooth_shading=True)

# %%
# The array holds one ``uint8`` triple per point.

cow['RGB'][:5]

# %%
# Units and provenance travel with the mesh as field data.

dict(cow.field_data)

# %%
# The surface is one closed shell, so it has a well-defined volume.

cow.is_manifold, cow.volume

# %%
# Clipping it therefore exposes a hollow interior rather than a solid
# cross-section.

cow.clip('x').plot(rgb=True, smooth_shading=True)

# %%
# The Generation Script
# ~~~~~~~~~~~~~~~~~~~~~
# Nothing here was scanned or modeled by hand. ``generate_cow.py`` in the
# `PyVista data repository <https://github.com/pyvista/data/tree/master/Data/cow>`_
# sums signed distance fields for the body, head, legs, hooves, horns, and tail,
# contours the result, paints the coat, and writes ``cow.pv``. Run it with
# ``uv run --locked generate_cow.py``.
#
# .. dropdown:: generate_cow.py
#
#    .. example-file:: cow/generate_cow.py

# %%
# .. tags:: load
