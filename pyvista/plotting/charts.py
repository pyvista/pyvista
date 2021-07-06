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

#region Some metaclass wrapping magic until a wrapped vtkPlotPie object can be added to a vtkChartPie
class _vtkWrapperMeta(type):

    def __call__(cls, *args, _wrap=None, **kwargs):
        # if _wrap is None:
        #     obj = cls.__new__(cls, *args, **kwargs)
        # else:
        #     obj = cls.__new__(_wrap.__class__, _wrap.__this__, *args, **kwargs)
        obj = cls.__new__(cls, *args, **kwargs)
        obj._wrapped = _wrap
        obj.__init__(*args, **kwargs)
        return obj


class _vtkWrapper(object, metaclass=_vtkWrapperMeta):

    def __getattribute__(self, item):
        unwrapped_attrs = ["_wrapped", "__class__", "__init__"]
        wrapped = super().__getattribute__("_wrapped")
        if item in unwrapped_attrs or wrapped is None:
            return super().__getattribute__(item)
        else:
            try:
                return wrapped.__getattribute__(item)
            except AttributeError:
                return super().__getattribute__(item)

    def __str__(self):
        if self._wrapped is None:
            return super().__str__()
        else:
            return "Wrapped: " + self._wrapped.__str__()
#endregion

class _ChartInterface(object):
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
        r_w, r_h = self.renderer.GetSize()  # Alternatively: self.scene.GetViewWidth(), self.scene.GetViewHeight()
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

    @property
    def title(self):
        return self.GetTitle()

    @title.setter
    def title(self, val):
        self.SetTitle(val)

    @property
    def legend(self):
        return self.GetShowLegend()

    @legend.setter
    def legend(self, val):
        self.SetShowLegend(val)


class _PlotInterface(object):
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
        # TODO: add tests for different combinations/positions of marker, line and color
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
                fmt = fmt[len(marker_style):]  # Remove marker_style from format string
            if line_style == "":
                # Line style is not yet set, so look for it at the start of the format string
                for style in cls.LINE_STYLES.keys():
                    if style != "" and fmt.startswith(style):
                        line_style = style
                fmt = fmt[len(line_style):]  # Remove line_style from format string
            if color is None:
                # Color is not yet set, so look for it in the remaining format string
                for i in range(len(fmt), 0, -1):
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
        return self._color

    @color.setter
    def color(self, val):
        self._color = parse_color(val)
        self.pen.SetColorF(*self._color)  # Deals with both color and opacity (if it is set)
        self.brush.SetColorF(*self._color)

    @property
    def width(self):
        return self.GetWidth()

    @width.setter
    def width(self, val):
        self.SetWidth(val)

    @property
    def pen(self):
        """
        :return: Retrieve vtkPen object controlling how lines in this plot are drawn.
        """
        return self.GetPen()

    @property
    def brush(self):
        """
        :return: Retrieve vtkBrush object controlling how shapes in this plot are filled.
        """
        return self.GetBrush()

    @property
    def line_style(self):
        return self._line_style

    @line_style.setter
    def line_style(self, val):
        if val in self.LINE_STYLES:
            self._line_style = val
            self.pen.SetLineType(self.LINE_STYLES[val])
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


class _MultiCompPlotInterface(object):
    """ Pythonic interface for vtkPlot instances with multiple components (e.g. BoxPlot, PiePlot, StackedBarPlot) """

    COLOR_SCHEMES = {
        "spectrum": _vtk.vtkColorSeries.SPECTRUM,
        "warm": _vtk.vtkColorSeries.WARM,
        "cool": _vtk.vtkColorSeries.COOL,
        "blues": _vtk.vtkColorSeries.BLUES,
        "wild_flower": _vtk.vtkColorSeries.WILD_FLOWER,
        "citrus": _vtk.vtkColorSeries.CITRUS,
        "div_purple_orange11": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_11,
        "div_purple_orange10": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_10,
        "div_purple_orange9": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_9,
        "div_purple_orange8": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_8,
        "div_purple_orange7": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_7,
        "div_purple_orange6": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_6,
        "div_purple_orange5": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_5,
        "div_purple_orange4": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_4,
        "div_purple_orange3": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_3,
        "div_spectral11": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_11,
        "div_spectral10": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_10,
        "div_spectral9": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_9,
        "div_spectral8": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_8,
        "div_spectral7": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_7,
        "div_spectral6": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_6,
        "div_spectral5": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_5,
        "div_spectral4": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_4,
        "div_spectral3": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_3,
        "div_brown_blue_green11": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_11,
        "div_brown_blue_green10": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_10,
        "div_brown_blue_green9": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_9,
        "div_brown_blue_green8": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_8,
        "div_brown_blue_green7": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_7,
        "div_brown_blue_green6": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_6,
        "div_brown_blue_green5": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_5,
        "div_brown_blue_green4": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_4,
        "div_brown_blue_green3": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_3,
        "seq_blue_green9": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_9,
        "seq_blue_green8": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_8,
        "seq_blue_green7": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_7,
        "seq_blue_green6": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_6,
        "seq_blue_green5": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_5,
        "seq_blue_green4": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_4,
        "seq_blue_green3": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3,
        "seq_yellow_orange_brown9": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_9,
        "seq_yellow_orange_brown8": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_8,
        "seq_yellow_orange_brown7": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_7,
        "seq_yellow_orange_brown6": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_6,
        "seq_yellow_orange_brown5": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_5,
        "seq_yellow_orange_brown4": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_4,
        "seq_yellow_orange_brown3": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_3,
        "seq_blue_purple9": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_9,
        "seq_blue_purple8": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_8,
        "seq_blue_purple7": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_7,
        "seq_blue_purple6": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_6,
        "seq_blue_purple5": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_5,
        "seq_blue_purple4": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_4,
        "seq_blue_purple3": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_3,
        "qual_accent": _vtk.vtkColorSeries.BREWER_QUALITATIVE_ACCENT,
        "qual_dark2": _vtk.vtkColorSeries.BREWER_QUALITATIVE_DARK2,
        "qual_set3": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET3,
        "qual_set2": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET2,
        "qual_set1": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET1,
        "qual_pastel2": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PASTEL2,
        "qual_pastel1": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PASTEL1,
        "qual_paired": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PAIRED,
        "custom": _vtk.vtkColorSeries.CUSTOM
    }
    _SCHEME_NAMES = {scheme_id: scheme_name for scheme_name, scheme_id in COLOR_SCHEMES.items()}
    DEFAULT_SCHEME = "qual_accent"

    def __init__(self):
        super().__init__()
        self._color_series = _vtk.vtkColorSeries()
        self._lookup_table = self._color_series.CreateLookupTable(_vtk.vtkColorSeries.CATEGORICAL)
        self._labels = _vtk.vtkStringArray()
        self.SetLabels(self._labels)
        self.scheme = self.DEFAULT_SCHEME

    @property
    def scheme(self):
        return self._SCHEME_NAMES.get(self._color_series.GetColorScheme(), "custom")

    @scheme.setter
    def scheme(self, val):
        self._color_series.SetColorScheme(self.COLOR_SCHEMES.get(val, _vtk.vtkColorSeries.CUSTOM))
        self._color_series.BuildLookupTable(self._lookup_table, _vtk.vtkColorSeries.CATEGORICAL)

    @property
    def colors(self):
        return [self._color_series.GetColor(i) for i in range(self._color_series.GetNumberOfColors())]

    @colors.setter
    def colors(self, val):
        if val is None:
            self.scheme = self.DEFAULT_SCHEME
        else:
            self._color_series.SetNumberOfColors(len(val))
            for i, color in enumerate(val):
                self._color_series.SetColor(i, parse_color(color)[:3])
            self._color_series.BuildLookupTable(self._lookup_table, _vtk.vtkColorSeries.CATEGORICAL)

    @property
    def labels(self):
        return [self._labels.GetValue(i) for i in range(self._labels.GetNumberOfValues())]

    @labels.setter
    def labels(self, val):
        self._labels.Reset()
        if val is not None:
            for label in val:
                self._labels.InsertNextValue(label)


class LinePlot2D(_vtk.vtkPlotLine, _PlotInterface):

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
            self.visible &= True
        else:
            # Turn off visibility for fewer than 2 points as otherwise an error message is shown
            self.visible = False


class ScatterPlot2D(_vtk.vtkPlotPoints, _PlotInterface):

    def __init__(self, x, y, color="b", size=10, style="o"):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32)})
        self.SetInputData(self._table, "x", "y")
        self.update(x, y)
        self.color = color
        self.marker_size = size
        self.marker_style = style

    def update(self, x, y):
        if len(x) > 0:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False)})
            self.visible &= True
        else:
            self.visible = False

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


class AreaPlot(_vtk.vtkPlotArea, _PlotInterface):

    def __init__(self, x, y1, y2, color="b"):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y1": np.empty(0, np.float32), "y2": np.empty(0, np.float32)})
        self.SetInputData(self._table)
        self.SetInputArray(0, "x")
        self.SetInputArray(1, "y1")
        self.SetInputArray(2, "y2")
        self.update(x, y1, y2)
        self.color = color

    def update(self, x, y1, y2):
        if len(x) > 0:
            self._table.update({"x": np.array(x, copy=False), "y1": np.array(y1, copy=False), "y2": np.array(y2, copy=False)})
            self.visible &= True
        else:
            self.visible = False


class BarPlot(_vtk.vtkPlotBar, _PlotInterface):

    ORIENTATIONS = {
        "H": _vtk.vtkPlotBar.HORIZONTAL,
        "V": _vtk.vtkPlotBar.VERTICAL
    }

    def __init__(self, x, y, color="b", offset=5, orientation="V"):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32)})
        self.SetInputData(self._table, "x", "y")
        self.update(x, y)
        self.color = color
        self.offset = offset
        self.orientation = orientation

    def update(self, x, y):
        if len(x) > 0:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False)})
            self.visible &= True
        else:
            self.visible = False

    @property
    def offset(self):
        return self.GetOffset()

    @offset.setter
    def offset(self, val):
        self.SetOffset(val)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, val):
        if val in self.ORIENTATIONS:
            self._orientation = val
            self.SetOrientation(self.ORIENTATIONS[val])
        else:
            formatted_orientations = "\", \"".join(self.ORIENTATIONS.keys())
            raise ValueError(f"Invalid orientation. Allowed orientations: \"{formatted_orientations}\"")


class Chart2D(_vtk.vtkChartXY, _ChartInterface):
    PLOT_TYPES = {
        "scatter": ScatterPlot2D,
        "line": LinePlot2D,
        "area": AreaPlot,
        "bar": BarPlot
    }

    def __init__(self, size=(1, 1), loc=(0, 0), x_label="x", y_label="y", grid=True):
        super().__init__()
        self._plots = {plot_type: [] for plot_type in self.PLOT_TYPES.keys()}
        self.SetAutoSize(False)  # We manually set the appropriate size
        self.size = size
        self.loc = loc
        self.x_label = x_label
        self.y_label = y_label
        self.grid = grid

    def render_event(self, *args, **kwargs):
        self.RecalculateBounds()
        super().render_event(*args, **kwargs)

    def _add_plot(self, plot_type, *args, **kwargs):
        plot = self.PLOT_TYPES[plot_type](*args, **kwargs)
        self.AddPlot(plot)
        self._plots[plot_type].append(plot)
        return plot

    def plot(self, x, y, fmt):
        """ Matplotlib like plot method """
        # TODO: make x and fmt optional, allow multiple ([x], y, [fmt]) entries
        marker_style, line_style, color = _PlotInterface.parse_format(fmt)
        scatter_plot, line_plot = None, None
        if marker_style != "":
            scatter_plot = self.scatter(x, y, color, style=marker_style)
        if line_style != "":
            line_plot = self.line(x, y, color, style=line_style)
        return scatter_plot, line_plot

    def scatter(self, x, y, color="b", size=10, style="o"):
        return self._add_plot("scatter", x, y, color=color, size=size, style=style)

    def line(self, x, y, color="b", width=1.0, style="-"):
        return self._add_plot("line", x, y, color=color, width=width, style=style)

    def area(self, x, y1, y2, color="b"):
        return self._add_plot("area", x, y1, y2, color=color)

    def bar(self, x, y, color="b", offset=5, orientation="V"):
        return self._add_plot("bar", x, y, color=color, offset=offset, orientation=orientation)

    def plots(self, plot_type=None):
        if plot_type is None:
            for plots in self._plots.values():
                for plot in plots:
                    yield plot
        else:
            for plot in self._plots[plot_type]:
                yield plot

    @property
    def x_axis(self):
        return self.GetAxis(_vtk.vtkAxis.BOTTOM)

    @property
    def y_axis(self):
        return self.GetAxis(_vtk.vtkAxis.LEFT)

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

    # TODO: ticks/labels. Maybe create vtkAxis wrapper?

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, val):
        self._grid = bool(val)
        self.x_axis.SetGridVisible(self._grid)
        self.y_axis.SetGridVisible(self._grid)


class BoxPlot(_vtk.vtkPlotBox, _PlotInterface, _MultiCompPlotInterface):

    def __init__(self, data, colors=None):
        super().__init__()
        self._table = pyvista.Table(data)
        self._quartiles = _vtk.vtkComputeQuartiles()
        self._quartiles.SetInputData(self._table)
        self.SetInputData(self._quartiles.GetOutput())
        self.update(data)
        self.SetLookupTable(self._lookup_table)
        if isinstance(colors, str):
            self.scheme = colors
        else:
            self.colors = colors

    def update(self, data):
        self._table.update(data)
        self._quartiles.Update()


class ChartBox(_vtk.vtkChartBox, _ChartInterface):

    def __init__(self, data):
        super().__init__()
        self._plot = BoxPlot(data)
        self.SetPlot(self._plot)
        self.SetColumnVisibilityAll(True)
        self.legend = True

    def render_event(self, *args, **kwargs):
        pass  # ChartBox fills entire scene by default, so no resizing is needed (nor possible at this moment)

    @property
    def plot(self):
        return self._plot

    @property
    def size(self):
        return (1, 1)

    @size.setter
    def size(self, val):
        raise ValueError("Cannot set ChartBox geometry, it fills up the entire viewport by default.")

    @property
    def loc(self):
        return (0, 0)

    @loc.setter
    def loc(self, val):
        raise ValueError("Cannot set ChartBox geometry, it fills up the entire viewport by default.")


class PiePlot(_vtkWrapper, _vtk.vtkPlotPie, _PlotInterface, _MultiCompPlotInterface):

    def __init__(self, data, labels=None, colors=None):
        super().__init__()
        self._table = pyvista.Table(data)
        self.SetInputData(self._table)
        self.SetInputArray(0, self._table.keys()[0])
        self.update(data)

        self.labels = labels

        self.SetColorSeries(self._color_series)
        if isinstance(colors, str):
            self.scheme = colors
        else:
            self.colors = colors

    def update(self, data):
        self._table.update(data)


class ChartPie(_vtk.vtkChartPie, _ChartInterface):

    def __init__(self, data, labels=None):
        super().__init__()
        self.AddPlot(0)  # We can't manually set a wrapped vtkPlotPie instance...
        self._plot = PiePlot(data, labels, _wrap=self.GetPlot(0))  # So we have to wrap the existing one
        self.legend = True

    def render_event(self, *args, **kwargs):
        pass  # ChartPie fills entire scene by default, so no resizing is needed (nor possible at this moment)

    @property
    def plot(self):
        return self._plot

    @property
    def size(self):
        return (1, 1)

    @size.setter
    def size(self, val):
        raise ValueError("Cannot set ChartPie geometry, it fills up the entire viewport by default.")

    @property
    def loc(self):
        return (0, 0)

    @loc.setter
    def loc(self, val):
        raise ValueError("Cannot set ChartPie geometry, it fills up the entire viewport by default.")


class Chart3D(_vtk.vtkChartXYZ, _ChartInterface):

    def __init__(self):
        super().__init__()


class ChartMPL(_vtk.vtkImageItem, _ChartInterface):

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
