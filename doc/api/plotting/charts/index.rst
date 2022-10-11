Charts
------
Charts API reference. These dedicated classes can be used to embed
charts in plotting windows. Note that using charts requires a VTK
version of at least 9.0 and in a future release will require
``vtk>=9.2``.

.. currentmodule:: pyvista

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
