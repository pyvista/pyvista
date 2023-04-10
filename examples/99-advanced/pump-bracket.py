""".. _modal_analysis_example:

Visualize Modal Analysis of a Pump Bracket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following example demonstrates how to use PyVista to visualize the modal
analysis of a pump bracket based on point arrays representing mode shapes for
different modes of vibration.

This example loads a dataset, extracts the mode shapes arrays for a specific
mode of vibration, and adds an interactive slider to visualize the effect of
each mode.

**Background**
Modal analysis is the study of the dynamic properties of mechanical structures
in the frequency domain. It is a common technique in structural dynamics,
particularly for automotive, aerospace, and civil engineering applications. In
this example, we will explore the modal analysis of a pump bracket.

A mode shape is the deformation pattern that occurs at a specific natural
frequency, or mode, of a structure. When a structure is excited by an external
force, it responds at all its natural frequencies with each mode shape being
independent of the others. In this example, we will visualize the mode shapes
to get an understanding of how the pump bracket responds to different modes of
vibration.

"""

import pyvista as pv
from pyvista import examples

###############################################################################
# Load the dataset
# ~~~~~~~~~~~~~~~~
# Start by loading the dataset using :mod:`pyvista.examples` module. This
# module provides access to a range of datasets.

dataset = examples.download_pump_bracket()
dataset

###############################################################################
# Plot the Dataset
# ~~~~~~~~~~~~~~~~
# Choose a mode shape from the available "disp_X" arrays in the dataset. Each
# "disp_X" array represents an eigen solution or a single mode shape for a
# given mode of vibration.
#
# Plot the 6th mode of the dataset. This is the first torsional mode for the
# bracket.

cpos = [(0.744, -0.502, -0.830), (0.0520, -0.160, 0.0743), (-0.180, -0.958, 0.224)]

dataset.plot(scalars='disp_3', cpos=cpos, show_scalar_bar=False, ambient=0.2, anti_aliasing='fxaa')


###############################################################################
# Visualize Displaced Mode Shape
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We will now visualize the mode shapes of the pump bracket.
#

# Create a plotter
pl = pv.Plotter()

# Add the undeformed pump bracket
pl.add_mesh(dataset, color="white", opacity=0.5)

# Add the deformed pump bracket with the mode shape
warp = dataset.warp_by_vector('disp_2', factor=0.1)
pl.add_mesh(warp, show_scalar_bar=False, ambient=0.2)

# Set camera view
pl.camera_position = cpos

# Show the plot with a title
pl.add_title(f"Pump Bracket Bending Mode #3")
pl.enable_anti_aliasing('fxaa')
pl.show()


###############################################################################
# Animate the Mode Shape Displacement
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a plotter and add the pump bracket to it. Animate the mode shape's
# displacement by updating the vertex positions at each time step. For a more
# realistic animation, we can use a sinusoidal function to vary the
# displacement.

ds = dataset.copy()

pl = pv.Plotter(notebook=False)
pl.add_mesh(dataset, show_edges=True)

# Define the time step for the animation
time_step = 0.02

# Number of time steps in one period
steps_per_period = 100

import numpy as np

mode_shape = 'disp_6'


# Create the animation
def sinusoidal_displacement(phase):
    updated_points = ds.points + ds[mode_shape] * 0.1 * np.sin(2 * np.pi * phase)
    ds.points = updated_points


p.open_gif("pump_bracket_mode_shape.gif")
for phase in np.linspace(0, 1, steps_per_period + 1)[:-1]:
    sinusoidal_displacement(phase)
    pl.write_frame()

p.close()


# This example demonstrates the visualization of the pump bracket's mode shape,
# the representation of its magnitude, and an animation of its displacement. The
# dataset used in this example contains 10 mode shapes (disp_0 to disp_9). You
# can replace the example's `mode_shape` variable with any of the available mode
# shapes to visualize the natural frequency and displacement associated with
# that mode. Understanding the modal behavior of structures is essential for
# design improvement, failure investigation, and optimization in engineering
# applications.
