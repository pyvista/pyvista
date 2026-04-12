"""
.. _load_vrml_example:

Working with VRML Files
~~~~~~~~~~~~~~~~~~~~~~~
Import a VRML file directly into a PyVista plotting scene.
For more details regarding the VRML format, see:
https://en.wikipedia.org/wiki/VRML

"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

sextant_file = examples.vrml.download_sextant()


# %%
# Set up the plotter and import VRML file.
# Use :func:`pyvista.Plotter.import_vrml` to import file.

pl = pv.Plotter()
pl.import_vrml(sextant_file)
pl.show()
# %%
# .. tags:: load
