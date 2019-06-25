"""
Volume Rendering
~~~~~~~~~~~~~~~~

Volume render uniform mesh types like :class:`pyvista.UniformGrid` or 3D
NumPy arrays
"""
import pyvista as pv
from pyvista import examples

# Volume rednering is not supported with Panel yet
pv.rcParams['use_panel'] = False

# Download a volumetric dataset
vol = examples.download_knee_full()
print(vol)

################################################################################

# A nice camera position
cpos = [(-381.74, -46.02, 216.54),
        (74.8305, 89.2905, 100.0),
        (0.23, 0.072, 0.97)]

vol.plot(volume=True, cmap='bone', cpos=cpos)


################################################################################
# Or use the :func:`pyvista.BasePlotter.add_volume` method like below.
# Note that here we use a non-default opacity mapping to a sigmoid:

p = pv.Plotter()
p.add_volume(vol, cmap='viridis', opacity='sigmoid')
p.camera_position = cpos
p.show()
