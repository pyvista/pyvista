"""
Volume Rendering
~~~~~~~~~~~~~~~~

Volume render uniform mesh types like :class:`pyvista.UniformGrid` or 3D
NumPy arrays
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

# Volume rednering is not supported with Panel yet
pv.rcParams['use_panel'] = False

# Download a volumetric dataset
vol = examples.download_knee_full()
print(vol)

################################################################################
# Simple Volume Render
# ++++++++++++++++++++
#

# A nice camera position
cpos = [(-381.74, -46.02, 216.54),
        (74.8305, 89.2905, 100.0),
        (0.23, 0.072, 0.97)]

vol.plot(volume=True, cmap='bone', cpos=cpos)


################################################################################
# Opacity Mappings
# ++++++++++++++++
#
# Or use the :func:`pyvista.BasePlotter.add_volume` method like below.
# Note that here we use a non-default opacity mapping to a sigmoid:

p = pv.Plotter()
p.add_volume(vol, cmap='bone', opacity='sigmoid')
p.camera_position = cpos
p.show()


################################################################################
# Cool Volume Examples
# ++++++++++++++++++++
#
# Here are a few more cool colume rendering examples

head = examples.download_head()

p = pv.Plotter()
p.add_volume(head, cmap='cool', opacity='sigmoid_6',)
p.camera_position = [(-228., -418., -158.),
                     (94.0, 122.0, 82.0),
                     (-0.2, -0.3, 0.9)]
p.show()

################################################################################

bolt_nut = examples.download_bolt_nut()

p = pv.Plotter()
p.add_volume(bolt_nut, cmap='coolwarm',
             opacity='sigmoid_5',)
p.show()
