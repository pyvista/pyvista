"""
Streamlines
~~~~~~~~~~~

Integrate a vector field to generate streamlines.
"""
###############################################################################
# This example generates streamlines of blood velocity. An isosurface of speed
# provides context. The starting positions for the streamtubes were determined
# by experimenting with the data.

# sphinx_gallery_thumbnail_number = 3
import pyvista as pv
from pyvista import examples

###############################################################################
# Carotid
# +++++++
# Download a sample dataset containing a vector field that can be integrated.

mesh = examples.download_carotid()

###############################################################################
# Run the stream line filtering algorithm.

streamlines, src = mesh.streamlines(
    return_source=True,
    max_time=100.0,
    initial_step_length=2.0,
    terminal_speed=0.1,
    n_points=25,
    source_radius=2.0,
    source_center=(133.1, 116.3, 5.0),
)

###############################################################################
# Display the results! Please note that because this dataset's velocity field
# was measured with low resolution, many streamlines travel outside the artery.

p = pv.Plotter()
p.add_mesh(mesh.outline(), color="k")
p.add_mesh(streamlines.tube(radius=0.15))
p.add_mesh(src)
p.add_mesh(mesh.contour([160]).wireframe(), color="grey", opacity=0.25)
p.camera_position = [(182.0, 177.0, 50), (139, 105, 19), (-0.2, -0.2, 1)]
p.show()


###############################################################################
# Blood Vessels
# +++++++++++++
# Here is another example of blood flow:

mesh = examples.download_blood_vessels().cell_data_to_point_data()
mesh.set_active_scalar("velocity")
streamlines, src = mesh.streamlines(
    return_source=True, source_radius=10, source_center=(92.46, 74.37, 135.5)
)


###############################################################################
boundary = mesh.decimate_boundary().wireframe()

p = pv.Plotter()
p.add_mesh(streamlines.tube(radius=0.2), lighting=False)
p.add_mesh(src)
p.add_mesh(boundary, color="grey", opacity=0.25)
p.camera_position = [(10, 9.5, -43), (87.0, 73.5, 123.0), (-0.5, -0.7, 0.5)]
p.show()


###############################################################################
# Kitchen
# +++++++
#
kpos = [(-6.68, 11.9, 11.6), (3.5, 2.5, 1.26), (0.45, -0.4, 0.8)]

mesh = examples.download_kitchen()
kitchen = examples.download_kitchen(split=True)

###############################################################################
streamlines = mesh.streamlines(n_points=40, source_center=(0.08, 3, 0.71))

###############################################################################
p = pv.Plotter()
p.add_mesh(mesh.outline(), color="k")
p.add_mesh(kitchen, color=True)
p.add_mesh(streamlines.tube(radius=0.01), scalars="velocity", lighting=False)
p.camera_position = kpos
p.show()
