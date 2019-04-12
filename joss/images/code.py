# Obligatory set up code
import vtki
from vtki import examples
import numpy as np
# Set a document-friendly plotting theme
vtki.set_plot_theme('document')

# Example mesh of Queen Nefertiti
mesh = examples.download_nefertiti()
# Render the dataset
mesh.plot(cpos=[-1,-1,0.2], eye_dome_lighting=True,
          screenshot='nefertiti.png')

# Load an example uniform grid
dataset = examples.load_uniform()
# Apply a threshold over a data range
threshed = dataset.threshold([100, 500]) # Figure 4 A

outline = dataset.outline()
contours = dataset.contour() # Figure 4 B
slices = dataset.slice_orthogonal() # Figure 4 C
glyphs = dataset.glyph(factor=1e-3, geom=vtki.Sphere()) # Figure 4 D

# Two by two comparison
vtki.plot_compare_four(threshed, contours, slices, glyphs,
                        {'show_scalar_bar':False},
                        camera_position=[-2,5,3], outline=outline,
                        screenshot='filters.png')

# Apply a filtering chain
result = dataset.threshold([100, 500], invert=True).elevation().clip(normal='z').slice_orthogonal()
