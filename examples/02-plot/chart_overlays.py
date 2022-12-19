"""
.. _chart_overlays_example:

Chart Overlays
~~~~~~~~~~~~~~

This example shows how you can combine multiple charts as overlays in
the same renderer. For an overview of the different chart types you
can use, please refer to :ref:`chart_basics_example`. Interaction with
a chart can be enabled by a double left click on top of it. Note that this
will disable interaction with the 3D scene. To stop interacting with
the chart, perform another double left click. This will either enable
interaction with another chart (if clicked on top of it) or re-enable
interaction with the 3D scene.

"""

import matplotlib.pyplot as plt
import numpy as np

import pyvista as pv

###############################################################################
# Data to display
t = np.linspace(0, 5, 50)
h = np.sin(t)
v = np.cos(t)

###############################################################################
# Define a Matplotlib figure.
# Use a tight layout to keep axis labels visible on smaller figures.

f, ax = plt.subplots(tight_layout=True)
h_line = ax.plot(t[:1], h[:1])[0]
ax.set_ylim([-1, 1])
ax.set_xlabel('Time (s)')
_ = ax.set_ylabel('Height (m)')
# sphinx_gallery_defer_figures

###############################################################################
# Define plotter, add the created matplotlib figure as the first (left) chart
# to the scene, and define a second (right) chart.

p = pv.Plotter()
h_chart = pv.ChartMPL(f, size=(0.46, 0.25), loc=(0.02, 0.06))
h_chart.background_color = (1.0, 1.0, 1.0, 0.4)
p.add_chart(h_chart)
v_chart = pv.Chart2D(
    size=(0.46, 0.25), loc=(0.52, 0.06), x_label="Time (s)", y_label="Velocity (m/s)"
)
v_line = v_chart.line(t[:1], v[:1])
v_chart.y_range = (-1, 1)
v_chart.background_color = (1.0, 1.0, 1.0, 0.4)
p.add_chart(v_chart)
p.add_mesh(pv.Sphere(1), name="sphere", render=False)
p.show(auto_close=False, interactive=True, interactive_update=True)


# Method and slider to update all visuals based on the time selection
def update_time(time):
    k = np.count_nonzero(t < time)
    h_line.set_xdata(t[: k + 1])
    h_line.set_ydata(h[: k + 1])
    v_line.update(t[: k + 1], v[: k + 1])
    p.add_mesh(pv.Sphere(1, center=(0, 0, h[k])), name="sphere", render=False)
    p.update()


time_slider = p.add_slider_widget(
    update_time, [np.min(t), np.max(t)], 0, "Time", (0.25, 0.9), (0.75, 0.9), event_type='always'
)

# Start incrementing time automatically
for i in range(1, 50):
    ax.set_xlim([0, t[i]])
    time_slider.GetSliderRepresentation().SetValue(t[i])
    update_time(t[i])

p.show()  # Keep plotter open to let user play with time slider
