"""
.. _load_vrml_example:

Working with VRML Files
~~~~~~~~~~~~~~~~~~~~~~~
Import a VRML file directly into a PyVista plotting scene.
For more details regarding the VRML format, see:
https://en.wikipedia.org/wiki/VRML

"""

import pyvista
from pyvista import examples

sextant_file = examples.vrml.download_sextant()


###############################################################################
# Set up the plotter and import VRML file.

pl = pyvista.Plotter()
pl.import_vrml(sextant_file)
pl.show()
