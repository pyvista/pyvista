# Obligatory set up code
import pyvista
from pyvista import examples
import numpy as np
# Set a document-friendly plotting theme
pyvista.set_plot_theme('document')

# Load an example uniform grid
dataset = examples.load_uniform()
# Apply a threshold over a data range
threshed = dataset.threshold([100, 500]) # Figure 4 A

outline = dataset.outline()
contours = dataset.contour() # Figure 4 B
slices = dataset.slice_orthogonal() # Figure 4 C
glyphs = dataset.glyph(factor=1e-3, geom=pyvista.Sphere()) # Figure 4 D

# Two by two comparison
pyvista.plot_compare_four(threshed, contours, slices, glyphs,
                        {'show_scalar_bar':False},
                        {'border':False},
                        camera_position=[-2,5,3], outline=outline,
                        screenshot='filters.png')

# Apply a filtering chain
result = dataset.threshold([100, 500], invert=True).elevation().clip(normal='z').slice_orthogonal()
