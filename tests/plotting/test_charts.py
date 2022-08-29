import numpy as np

import pyvista


def test_charts():
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x)
    chart = pyvista.Chart2D()
    _ = chart.scatter(x, y)
    _ = chart.line(x, y, 'r')
    chart.show()
