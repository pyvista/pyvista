"""
Volume Rendering
~~~~~~~~~~~~~~~~

Volume render uniform mesh types like :class:`pyvista.UniformGrid` or 3D
NumPy arrays
"""

# sphinx_gallery_thumbnail_number = 3
import pyvista as pv
from pyvista import examples

# Volume rendering is not supported with Panel yet
pv.rcParams["use_panel"] = False

# Download a volumetric dataset
vol = examples.download_knee_full()
vol

###############################################################################
# Simple Volume Render
# ++++++++++++++++++++
#

# A nice camera position
cpos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]

vol.plot(volume=True, cmap="bone", cpos=cpos)


###############################################################################
# Opacity Mappings
# ++++++++++++++++
#
# Or use the :func:`pyvista.BasePlotter.add_volume` method like below.
# Note that here we use a non-default opacity mapping to a sigmoid:

p = pv.Plotter()
p.add_volume(vol, cmap="bone", opacity="sigmoid")
p.camera_position = cpos
p.show()

###############################################################################
# You can also use a custom opacity mapping
opacity = [0, 0, 0, 0.1, 0.3, 0.6, 1]

p = pv.Plotter()
p.add_volume(vol, cmap="viridis", opacity=opacity)
p.camera_position = cpos
p.show()


###############################################################################
# Cool Volume Examples
# ++++++++++++++++++++
#
# Here are a few more cool volume rendering examples

head = examples.download_head()

p = pv.Plotter()
p.add_volume(head, cmap="cool", opacity="sigmoid_6")
p.camera_position = [(-228.0, -418.0, -158.0), (94.0, 122.0, 82.0), (-0.2, -0.3, 0.9)]
p.show()

###############################################################################

bolt_nut = examples.download_bolt_nut()

p = pv.Plotter()
p.add_volume(bolt_nut, cmap="coolwarm", opacity="sigmoid_5")
p.show()


###############################################################################

frog = examples.download_frog()

p = pv.Plotter()
p.add_volume(frog, cmap="viridis", opacity="sigmoid_6")
p.camera_position = [(929., 1067., -278.9),
                     (249.5, 234.5, 101.25),
                     (-0.2048, -0.2632, -0.9427)]
p.show()
