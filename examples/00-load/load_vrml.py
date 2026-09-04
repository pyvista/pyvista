"""
.. _load_vrml_example:

Working With VRML Files
~~~~~~~~~~~~~~~~~~~~~~~

Import a VRML file directly into a PyVista plotting scene.

For more details regarding the VRML format, see:
https://en.wikipedia.org/wiki/VRML

"""

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import pyvista as pv
from pyvista import examples

sextant_file = examples.download_sextant(load=False)


# %%
# Set up the plotter and import VRML file.
# Use :func:`pyvista.Plotter.import_vrml` to import file.

pl = pv.Plotter()
pl.import_vrml(sextant_file)
pl.show()
# %%
# .. tags:: load
