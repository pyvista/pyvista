"""
Chart Basics
~~~~~~~~~~~~


This example shows how different types of charts can be added to the scene. A more complex example, showing how to
combine multiple charts as overlays in the same renderer, is given in `chart_overlays`.
"""

import pyvista as pv
import numpy as np
rng = np.random.default_rng(1)

###############################################################################
# This example shows how to create a 2D scatter plot from 100 randomly sampled
# datapoints.

x = rng.standard_normal(100)
y = rng.standard_normal(100)
p = pv.Plotter()
p.background_color = (1, 1, 1)
chart = pv.Chart2D()
chart.scatter(x, y, size=10, style="+")
p.add_chart(chart)
p.show()

###############################################################################
# To connect datapoints with lines, you can create a 2D line plot as shown in
# the example below.

x = np.linspace(0, 10, 1000)
y = np.sin(x**2)
p = pv.Plotter()
p.background_color = (1, 1, 1)
chart = pv.Chart2D()
chart.line(x, y)
p.add_chart(chart)
p.show()

###############################################################################
# You can also easily combine scatter and line plots using the general
# :func:`pyvista.Chart2D.plot` function, specifying both the line and marker
# style at once.

x = np.arange(11)
y = rng.integers(-5, 6, 11)
p = pv.Plotter()
p.background_color = (1, 1, 1)
chart = pv.Chart2D()
chart.plot(x, y, 'x--b')  # Marker style 'x', striped line style '--', blue color 'b'
p.add_chart(chart)
p.show()

###############################################################################
# The following example shows how to create filled areas between two polylines.

x = np.linspace(0, 10, 1000)
y1 = np.cos(x) + np.sin(3*x)
y2 = 0.1*(x - 5)
p = pv.Plotter()
p.background_color = (1, 1, 1)
chart = pv.Chart2D()
chart.area(x, y1, y2, color=(0.1, 0.1, 0.9, 0.5))
chart.line(x, y1, color=(0.9, 0.1, 0.1), width=4, style="--")
chart.line(x, y2, color=(0.1, 0.9, 0.1), width=4, style="--")
p.add_chart(chart)
p.show()

###############################################################################
# Bar charts are also supported.

x = np.arange(1, 11)
y = rng.integers(1, 11, 10)
p = pv.Plotter()
p.background_color = (1, 1, 1)
chart = pv.Chart2D()
chart.bar(x, y)  # TODO: change chart X axis ticks/labels
p.add_chart(chart)
p.show()

# TODO: include stacked bar charts here

###############################################################################
# Beside the flexible Chart2D used in the previous examples, there are a couple
# other dedicated charts you can create. The example below shows how a pie
# chart can be created.

p = pv.Plotter()
data = np.array([8.4,6.1,2.7,2.4,0.9])
chart = pv.ChartPie(data)
chart.plot.labels = [f"slice {i}" for i in range(len(data))]
p.add_chart(chart)
p.show()

###############################################################################
# To summarize statistics of datasets, you can easily create a boxplot.

data = {f"Experiment {i}": rng.poisson(lam, 20) for i, lam in enumerate(range(2, 12, 2))}
p = pv.Plotter()
p.background_color = (1, 1, 1)
chart = pv.ChartBox(data)
p.add_chart(chart)
p.show()

###############################################################################
# If you would like to add other types of chart that are currently not
# supported by pyvista or VTK, you can resort to matplotlib to create your
# custom chart and afterwards embed it into a pyvista plotting window.
# The below example shows how you can do this.

import matplotlib.pyplot as plt

# First, create the matplotlib figure
f, ax = plt.subplots(tight_layout=True)  # Tight layout to keep axis labels visible on smaller figures
alphas = [0.5+i for i in range(5)]
betas = [*reversed(alphas)]
N = int(1e4)
data = [rng.beta(alpha, beta, N) for alpha, beta in zip(alphas, betas)]
labels = [f"$\\alpha={alpha:.1f}\\,;\\,\\beta={beta:.1f}$" for alpha, beta in zip(alphas, betas)]
ax.violinplot(data)
ax.set_xticks(np.arange(1, 1 + len(labels)))
ax.set_xticklabels(labels)
ax.set_title("$B(\\alpha, \\beta)$")

# Next, embed the figure into a pyvista plotting window
p = pv.Plotter()
chart = pv.ChartMPL(f)
p.add_chart(chart)
p.show()
