import numpy as np
from typing import Tuple

import pyvista
from pyvista import _vtk
from .tools import parse_color

try:
    import matplotlib.figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _HAS_MPL = True
except ModuleNotFoundError:
    _HAS_MPL = False


class ChartInterface(object):
    """ Pythonic interface for vtkChart instances """

    @property
    def scene(self):
        return self.GetScene()

    @property
    def renderer(self):
        return self.scene.GetRenderer()

    def render_event(self, *args, **kwargs):
        self._resize()

    def _resize(self):
        r_w, r_h = self.renderer.GetSize()
        _, _, c_w, c_h = self.GetSize()
        # Target size is calculated from specified normalized width and height and the renderer's current size
        t_w = self._size[0] * r_w
        t_h = self._size[1] * r_h
        if c_w != t_w or c_h != t_h:
            # Mismatch between current size and target size, so resize chart:
            self.SetSize(_vtk.vtkRectf(self._loc[0] * r_w, self._loc[1] * r_h, t_w, t_h))

    @property
    def size(self):
        """Chart size (in normalized coordinates)."""
        return self._size

    @size.setter
    def size(self, val):
        assert len(val) == 2 and 0 <= val[0] <= 1 and 0 <= val[1] <= 1
        self._size = val

    @property
    def loc(self):
        """Chart position w.r.t. the bottom left corner (in normalized coordinates)."""
        return self._loc

    @loc.setter
    def loc(self, val):
        assert len(val) == 2 and 0 <= val[0] <= 1 and 0 <= val[1] <= 1
        self._loc = val

    @property
    def bg_color(self):
        return self.GetBackgroundBrush().GetColor()

    @bg_color.setter
    def bg_color(self, val):
        self.GetBackgroundBrush().SetColorF(*parse_color(val))

    @property
    def visible(self):
        return self.GetVisible()

    @visible.setter
    def visible(self, val):
        self.SetVisible(val)

    def toggle(self):
        self.visible = not self.visible


class PlotInterface(object):
    """ Pythonic interface for vtkPlot instances """

    LINE_STYLES = {
        "": _vtk.vtkPen.NO_PEN,
        "-": _vtk.vtkPen.SOLID_LINE,
        "--": _vtk.vtkPen.DASH_LINE,
        ":": _vtk.vtkPen.DOT_LINE,
        "-.": _vtk.vtkPen.DASH_DOT_LINE,
        "-..": _vtk.vtkPen.DASH_DOT_DOT_LINE
    }

    MARKER_STYLES = {
        "": _vtk.vtkPlotPoints.NONE,
        "x": _vtk.vtkPlotPoints.CROSS,
        "+": _vtk.vtkPlotPoints.PLUS,
        "s": _vtk.vtkPlotPoints.SQUARE,
        "o": _vtk.vtkPlotPoints.CIRCLE,
        "d": _vtk.vtkPlotPoints.DIAMOND
    }

    @classmethod
    def parse_format(cls, fmt):
        # TODO: first draft, untested
        marker_style = ""
        line_style = ""
        color = None
        last_fmt = None
        while fmt != "" and last_fmt != fmt and (marker_style == "" or line_style == "" or color is None):
            last_fmt = fmt
            if marker_style == "":
                # Marker style is not yet set, so look for it at the start of the format string
                for style in cls.MARKER_STYLES.keys():
                    if style != "" and fmt.startswith(style):
                        # Loop over all styles such that - and -- can both be checked
                        marker_style = style
                fmt = fmt.removeprefix(marker_style)
            if line_style == "":
                # Line style is not yet set, so look for it at the start of the format string
                for style in cls.LINE_STYLES.keys():
                    if style != "" and fmt.startswith(style):
                        line_style = style
                fmt = fmt.removeprefix(line_style)
            if color is None:
                # Color is not yet set, so look for it in the remaining format string
                for i in range(len(fmt), 1, -1):
                    try:
                        parse_color(fmt[:i])
                        color = fmt[:i]
                        fmt = fmt[i:]
                        break
                    except ValueError:
                        pass
        if color is None:
            color = "b"
        return marker_style, line_style, color

    @property
    def color(self):
        return self.GetColor()

    @color.setter
    def color(self, val):
        self.SetColor(*parse_color(val))

    @property
    def width(self):
        return self.GetWidth()

    @width.setter
    def width(self, val):
        self.SetWidth(val)

    @property
    def line_style(self):
        return self._line_style

    @line_style.setter
    def line_style(self, val):
        if val in self.LINE_STYLES:
            self._line_style = val
            self.GetPen().SetLineType(self.LINE_STYLES[val])
        else:
            formatted_styles = "\", \"".join(self.LINE_STYLES.keys())
            raise ValueError(f"Invalid line style. Allowed line styles: \"{formatted_styles}\"")

    @property
    def visible(self):
        return self.GetVisible()

    @visible.setter
    def visible(self, val):
        self.SetVisible(val)

    def toggle(self):
        self.visible = not self.visible


class Chart2D(_vtk.vtkChartXY, ChartInterface):

    def __init__(self, size=(1, 1), loc=(0, 0), x_label="x", y_label="y"):
        super().__init__()
        self._lines = []
        self._scatters = []
        self.SetAutoSize(False)  # We manually set the appropriate size
        self.size = size
        self.loc = loc
        self.x_label = x_label
        self.y_label = y_label

    def render_event(self, *args, **kwargs):
        self.RecalculateBounds()
        super().render_event(*args, **kwargs)

    def plot(self, x, y, fmt):
        """ Matplotlib like plot method """
        marker_style, line_style, color = PlotInterface.parse_format(fmt)
        scatter_plot, line_plot = None, None
        if marker_style != "":
            scatter_plot = self.scatter(x, y, color, style=marker_style)
        if line_style != "":
            line_plot = self.line(x, y, color, style=line_style)
        return scatter_plot, line_plot

    def scatter(self, x, y, color="b", size=1.0, style="o"):
        plot = ScatterPlot2D(x, y, color, size, style)
        self.AddPlot(plot)
        self._scatters.append(plot)
        return plot

    def line(self, x, y, color="b", width=1.0, style="-"):
        plot = LinePlot2D(x, y, color, width, style)
        self.AddPlot(plot)
        self._lines.append(plot)
        return plot

    @property
    def x_axis(self):
        return self.GetAxis(1)

    @property
    def y_axis(self):
        return self.GetAxis(0)

    @property
    def x_label(self):
        return self.x_axis.GetTitle()

    @x_label.setter
    def x_label(self, val):
        self.x_axis.SetTitle(val)

    @property
    def y_label(self):
        return self.y_axis.GetTitle()

    @y_label.setter
    def y_label(self, val):
        self.y_axis.SetTitle(val)

    @property
    def x_range(self):
        return self.x_axis.GetRange()

    @x_range.setter
    def x_range(self, val):
        if val is None:
            self.x_axis.SetBehavior(_vtk.vtkAxis.AUTO)
        else:
            self.x_axis.SetBehavior(_vtk.vtkAxis.FIXED)
            self.x_axis.SetRange(val)

    @property
    def y_range(self):
        return self.y_axis.GetRange()

    @y_range.setter
    def y_range(self, val):
        if val is None:
            self.y_axis.SetBehavior(_vtk.vtkAxis.AUTO)
        else:
            self.y_axis.SetBehavior(_vtk.vtkAxis.FIXED)
            self.y_axis.SetRange(val)


class LinePlot2D(_vtk.vtkPlotLine, PlotInterface):

    def __init__(self, x, y, color="b", width=1.0, style="-"):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32)})
        self.SetInputData(self._table, "x", "y")
        self.update(x, y)
        self.color = color
        self.width = width
        self.line_style = style

    def update(self, x, y):
        if len(x) > 1:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False)})
            self.visible = True
        else:
            # Turn off visibility for fewer than 2 points as otherwise an error message is shown
            self.visible = False


class ScatterPlot2D(_vtk.vtkPlotPoints, PlotInterface):

    def __init__(self, x, y, color="b", size=1.0, style="o"):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32)})
        self.SetInputData(self._table, "x", "y")
        self.update(x, y)
        self.color = color
        self.marker_size = size
        self.marker_style = style

    def update(self, x, y):
        self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False)})

    @property
    def marker_size(self):
        return self.GetMarkerSize()

    @marker_size.setter
    def marker_size(self, val):
        self.SetMarkerSize(val)

    @property
    def marker_style(self):
        return self._marker_style

    @marker_style.setter
    def marker_style(self, val):
        if val in self.MARKER_STYLES:
            self._marker_style = val
            self.SetMarkerStyle(self.MARKER_STYLES[val])
        else:
            formatted_styles = "\", \"".join(self.MARKER_STYLES.keys())
            raise ValueError(f"Invalid marker style. Allowed marker styles: \"{formatted_styles}\"")


class Chart3D(_vtk.vtkChartXYZ, ChartInterface):

    def __init__(self):
        super().__init__()


class ChartMPL(_vtk.vtkImageItem, ChartInterface):

    if _HAS_MPL:
        def __init__(self, figure: matplotlib.figure.Figure, size: Tuple[int, int] = (1, 1), loc: Tuple[int, int] = (0, 0)):
            """
            Create new chart from an existing matplotlib figure.
            :param figure: The matplotlib figure to draw
            :param size: The normalized size of the chart (values between 0 and 1). None to completely fill the renderer
                         and autoresize
            :param loc: The normalized location of the chart (values between 0 and 1). None to manually set the position
                        (in pixels)
            """
            super().__init__()
            self._fig = figure
            self._canvas = FigureCanvasAgg(self._fig)  # Switch backends and store reference to figure's canvas
            self._canvas.mpl_connect('draw_event', self.redraw)  # Attach 'draw_event' callback

            self.size = size
            self.loc = loc
            self.redraw()

        def _resize(self):
            r_w, r_h = self.renderer.GetSize()
            c_w, c_h = self._canvas.get_width_height()
            # Calculate target size from specified normalized width and height and the renderer's current size
            t_w = self._size[0]*r_w
            t_h = self._size[1]*r_h
            if c_w != t_w or c_h != t_h:
                # Mismatch between canvas size and target size, so resize figure:
                f_w = t_w / self._fig.dpi
                f_h = t_h / self._fig.dpi
                self._fig.set_size_inches(f_w, f_h)
                self.position = (self._loc[0]*r_w, self._loc[1]*r_h)

    else:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChartMPL requires matplotlib")

    def redraw(self, event=None):
        if event is None:
            # Manual call, so make sure canvas is redrawn first (which will callback here with a proper event defined)
            self._canvas.draw()
        else:
            # Called from draw_event callback
            img = np.frombuffer(self._canvas.buffer_rgba(), dtype=np.uint8)  # Store figure data in numpy array
            w, h = self._canvas.get_width_height()
            img_arr = img.reshape([h, w, 4])
            img_data = pyvista.Texture(img_arr).to_image()  # Convert to vtkImageData
            self.SetImage(img_data)

    def render_event(self, *args, **kwargs):
        self._resize()  # Update figure dimensions if needed
        self.redraw()  # Redraw figure

    @property
    def bg_color(self):
        return self._bg_color

    @bg_color.setter
    def bg_color(self, val):
        color = parse_color(val) if val is not None else [1, 1, 1, 1]
        opacity = color[3] if len(color) == 4 else 1
        self._bg_color = color
        self._fig.patch.set_color(color[:3])
        self._fig.patch.set_alpha(opacity)
        for ax in self._fig.axes:
            ax.patch.set_alpha(0 if opacity < 1 else 1)  # Make axes fully transparent if opacity is lower than 1

    @property
    def position(self):
        """Chart position w.r.t the bottom left corner (in pixels)."""
        return self.GetPosition()

    @position.setter
    def position(self, val):
        assert len(val) == 2
        self.SetPosition(*val)


class Charts(object):

    def __init__(self, renderer):
        self._charts = []

        self._scene = _vtk.vtkContextScene()
        self._actor = _vtk.vtkContextActor()

        self._actor.SetScene(self._scene)
        renderer.AddActor(self._actor)
        self._scene.SetRenderer(renderer)

    def add_chart(self, chart):
        self._charts.append(chart)
        self._scene.AddItem(chart)

    def remove_chart(self, chart_or_index):
        chart = self[chart_or_index] if isinstance(chart_or_index, int) else chart_or_index
        assert chart in self._charts
        self._charts.remove(chart)
        self._scene.RemoveItem(chart)

    def __getitem__(self, index):
        """Return a chart based on an index."""
        return self._charts[index]

    def __len__(self):
        """Return number of charts."""
        return len(self._charts)

    def __iter__(self):
        """Return an iterable of charts."""
        for chart in self._charts:
            yield chart
