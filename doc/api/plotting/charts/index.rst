Charts
------
.. currentmodule:: pyvista

Charts API reference. These dedicated classes can be used to embed
charts in plotting windows. Note that full charts functionality
requires a VTK version of at least 9.3. Most components work fine
in older versions though.

Unusable features in older VTK versions:

- Interaction with charts is bugged when multiple subplots are
  used (VTK <9.3).
- :class:`ChartBox` and :class:`ChartPie` cannot have a custom
  geometry and fill the entire scene by default (VTK <9.2).

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
