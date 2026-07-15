Charts
------
.. currentmodule:: pyvista

Charts API reference. These dedicated classes can be used to embed
charts in plotting windows.

.. data:: Chart

.. autosummary::
   :toctree: _autosummary

   Chart2D
   ChartBox
   ChartPie
   ChartMPL


To customize these charts, extra plot and utility classes are
available in the ``charts`` module. Note that users should
typically not instantiate these classes themselves, but rather
use the dedicated methods and properties from the chart
classes above.

.. currentmodule:: pyvista.plotting.charts

.. autosummary::
   :toctree: _autosummary

   Pen
   Brush
   Axis
   LinePlot2D
   ScatterPlot2D
   BarPlot
   AreaPlot
   StackPlot
   BoxPlot
   PiePlot
