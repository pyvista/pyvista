"""
.. _ref_edl:

Eye Dome Lighting
~~~~~~~~~~~~~~~~~

Eye-Dome Lighting (EDL) is a non-photorealistic, image-based shading technique
designed to improve depth perception in scientific visualization images.
To learn more, please see `this blog post`_.

.. _this blog post: https://blog.kitware.com/eye-dome-lighting-a-non-photorealistic-shading-technique/

"""
###############################################################################

# sphinx_gallery_thumbnail_number = 1
import pyvista as pv
from pyvista import examples

###############################################################################
# Statue
# +++++++++++
#
# Eye-Dome Lighting can dramatically improve depth perception when plotting
# incredibly sophisticated meshes like the creative commons Queen Nefertiti
# statue:

nefertiti = examples.download_nefertiti()
nefertiti.plot(eye_dome_lighting=True, cpos=[-1, -1, 0.2], color=True)

###############################################################################
# Here we will compare a EDL shading side by side with normal shading

p = pv.Plotter(shape=(1, 2), border=False)

# With eye-dome lighting
p.subplot(0, 0)
p.add_mesh(nefertiti, color=True)
p.enable_eye_dome_lighting()
p.add_text("Eye-Dome Lighting", font_size=24)
p.camera_position = [-1, -1, 0.2]

# No eye-dome lighting
p.subplot(0, 1)
p.add_mesh(nefertiti, color=True)
p.add_text("No Eye-Dome Lighting", font_size=24)
p.camera_position = [-1, -1, 0.2]

p.show()

###############################################################################
# Point Cloud
# +++++++++++
#
# When plotting a simple point cloud, it can be difficult to perceive depth.
# Take this Lidar point cloud for example:

point_cloud = examples.download_lidar()


###############################################################################
# And now plot this point cloud as-is:

# Plot a typical point cloud with no EDL
p = pv.Plotter()
p.add_mesh(point_cloud, color="tan", point_size=5)
p.show()


###############################################################################
# We can improve the depth mapping by enabling eye dome lighting on the
# renderer with :func:`pyvista.Renderer.enable_eye_dome_lighting`.

# Plot with EDL
p = pv.Plotter()
p.add_mesh(point_cloud, color="tan", point_size=5)
p.enable_eye_dome_lighting()
p.show()


###############################################################################
# The eye dome lighting mode can also handle plotting scalar arrays:

# Plot with EDL and scalar data
p = pv.Plotter()
p.add_mesh(point_cloud, scalars="Elevation", point_size=5)
p.enable_eye_dome_lighting()
p.show()
