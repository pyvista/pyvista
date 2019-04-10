"""
Plot Eye Dome Lighting
~~~~~~~~~~~~~~~~~~~~~~

Eye-Dome Lighting (EDL) is a non-photorealistic, image-based shading technique
designed to improve depth perception in scientific visualization images.
To learn more, please see `this blog post`_.

.. _this blog post: https://blog.kitware.com/eye-dome-lighting-a-non-photorealistic-shading-technique/


When plotting a simple point cloud, it can be difficult to perceive depth.
Take this Lidar point cloud for example:
"""

# sphinx_gallery_thumbnail_number = 2
import vtki
from vtki import examples

################################################################################
# Load a sample point cloud.

point_cloud = examples.download_lidar()


################################################################################
# And now plot this point cloud as-is:

# Plot a typical point cloud with no EDL
p = vtki.Plotter()
p.add_mesh(point_cloud, color='orange', point_size=5)
p.show()


################################################################################
# We can improve the depth mapping by enabling eye dome lighting on the renderer.
# Reference :func:`vtki.Renderer.enable_eye_dome_lighting`.

# Plot with EDL
p = vtki.Plotter()
p.add_mesh(point_cloud, color='orange', point_size=5)
p.enable_eye_dome_lighting()
p.show()


################################################################################
# The eye dome lighting mode can also handle plotting scalar arrays:

# Plot with EDL and scalar data
p = vtki.Plotter()
p.add_mesh(point_cloud, scalars='Elevation', point_size=5)
p.enable_eye_dome_lighting()
p.show()
