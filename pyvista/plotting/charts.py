"""Module containing pyvista wrappers for the vtk Charts API."""

import numpy as np
from typing import Tuple, Sequence, Union, Optional

import pyvista
from pyvista import _vtk
from .tools import parse_color

try:
    import matplotlib.figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _HAS_MPL = True
except ModuleNotFoundError:
    _HAS_MPL = False

# Type definitions
Color4f = Tuple[float, float, float, float]

#region Some metaclass wrapping magic
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

class Pen(_vtkWrapper, _vtk.vtkPen):
    """Pythonic wrapper for a VTK Pen, used to draw lines."""

    LINE_STYLES = {
        "": _vtk.vtkPen.NO_PEN,
        "-": _vtk.vtkPen.SOLID_LINE,
        "--": _vtk.vtkPen.DASH_LINE,
        ":": _vtk.vtkPen.DOT_LINE,
        "-.": _vtk.vtkPen.DASH_DOT_LINE,
        "-..": _vtk.vtkPen.DASH_DOT_DOT_LINE
    }

    def __init__(self, color: Sequence = "k", width: float = 1, style: Optional[str] = "-"):
        """Initialize a new Pen instance.

        Parameters
        ----------
        color : str or sequence, optional
            Color of the lines drawn using this pen. Any color parsable by ``pyvista.parse_color`` is allowed. Defaults
            to ``"k"``.

        width : float, optional
            Width of the lines drawn using this pen. Defaults to ``1``.

        style : str, optional
            Style of the lines drawn using this pen. See ``Pen.LINE_STYLES`` for a list of allowed line styles. Defaults
            to ``"-"``.

        """
        super().__init__()
        self.color = color
        self.width = width
        self.style = style

    @property
    def color(self) -> Color4f:
        """Get the pen's current color."""
        return self._color

    @color.setter
    def color(self, val: Sequence):
        """Set the pen's color.

        Examples
        --------
        >>> import pyvista
        >>> p = pyvista.Plotter()
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.pen.color = 'r'
        >>> p.add_chart(chart)
        >>> p.show()

        """
        self._color = (0, 0, 0, 0) if val is None else parse_color(val, opacity=1)
        self.SetColorF(*self._color)

    @property
    def width(self) -> float:
        """Get the pen's current width."""
        return self.GetWidth()

    @width.setter
    def width(self, val: float):
        """Set the pen's width.

        Examples
        --------
        >>> import pyvista
        >>> p = pyvista.Plotter()
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.pen.width = 10
        >>> p.add_chart(chart)
        >>> p.show()

        """
        self.SetWidth(float(val))

    @property
    def style(self) -> Optional[str]:
        """Get the pen's current line style."""
        return self._line_style

    @style.setter
    def style(self, val: Optional[str]):
        """Set the pen's line style.

        Examples
        --------
        >>> import pyvista
        >>> p = pyvista.Plotter()
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.pen.style = '-.'
        >>> p.add_chart(chart)
        >>> p.show()

        """
        if val is None:
            val = ""
        if val in self.LINE_STYLES:
            self._line_style = val
            self.SetLineType(self.LINE_STYLES[val])
        else:
            formatted_styles = "\", \"".join(self.LINE_STYLES.keys())
            raise ValueError(f"Invalid line style. Allowed line styles: \"{formatted_styles}\"")


class Brush(_vtkWrapper, _vtk.vtkBrush):
    """Pythonic wrapper for a VTK Brush, used to fill shapes."""

    def __init__(self, color: Sequence = "k", texture=None):
        """Initialize a new Pen instance.

        Parameters
        ----------
        color : str or sequence, optional
            Fill color of the shapes drawn using this brush. Any color parsable by ``pyvista.parse_color`` is allowed.
            Defaults to ``"k"``.

        texture : ``pyvista.Texture``, optional
            Texture used to fill shapes drawn using this brush. Any object convertible to a ``pyvista.Texture`` is
            allowed. Defaults to ``None``.

        """
        super().__init__()
        self.color = color
        self.texture = texture
        self._interpolate = True  # vtkBrush textureProperties defaults to LINEAR & STRETCH
        self._repeat = False

    @property
    def color(self) -> Color4f:
        """Get the brush's current color."""
        return self._color

    @color.setter
    def color(self, val: Sequence):
        """Set the brush's color.

        Examples
        --------
        >>> import pyvista
        >>> p = pyvista.Plotter()
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.color = 'r'
        >>> p.add_chart(chart)
        >>> p.show()

        """
        self._color = (0, 0, 0, 0) if val is None else parse_color(val, opacity=1)
        self.SetColorF(*self._color)

    @property
    def texture(self) -> Optional["pyvista.Texture"]:
        """Get the brush's current texture."""
        return self._texture

    @texture.setter
    def texture(self, val):
        """Set the brush's texture.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> p = pyvista.Plotter()
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.texture = examples.download_puppy_texture()
        >>> p.add_chart(chart)
        >>> p.show()

        """
        if val is None:
            self._texture = None
            self.SetTexture(None)
        else:
            self._texture = pyvista.Texture(val)
            self.SetTexture(self._texture.to_image())

    @property
    def texture_interpolate(self) -> bool:
        """Get texture interpolation mode (NEAREST = ``False``, LINEAR = ``TRUE``)."""
        return self._interpolate

    @texture_interpolate.setter
    def texture_interpolate(self, val: bool):
        """Set texture interpolation mode (NEAREST = ``False``, LINEAR = ``TRUE``).

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> p = pyvista.Plotter()
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.texture = examples.download_puppy_texture()
        >>> plot.brush.texture_interpolate = False
        >>> p.add_chart(chart)
        >>> p.show()

        """
        self._interpolate = bool(val)
        self._update_textureprops()

    @property
    def texture_repeat(self) -> bool:
        """Get texture repeat mode (STRETCH = ``False``, REPEAT = ``TRUE``)."""
        return self._repeat

    @texture_repeat.setter
    def texture_repeat(self, val: bool):
        """Set texture repeat mode (STRETCH = ``False``, REPEAT = ``TRUE``).

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> p = pyvista.Plotter()
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.texture = examples.download_puppy_texture()
        >>> plot.brush.texture_repeat = True
        >>> p.add_chart(chart)
        >>> p.show()

        """
        self._repeat = bool(val)
        self._update_textureprops()

    def _update_textureprops(self):
        # Interpolation: NEAREST = 0x01, LINEAR = 0x02
        # Stretch/repeat: STRETCH = 0x04, REPEAT = 0x08
        self.SetTextureProperties(1+int(self._interpolate) + 4*(1+int(self._repeat)))


class Axis(_vtkWrapper, _vtk.vtkAxis):
    """Pythonic interface for a VTK Axis, used by 2D charts."""
    BEHAVIORS = {
        "auto": _vtk.vtkAxis.AUTO,
        "fixed": _vtk.vtkAxis.FIXED
    }

    def __init__(self, behavior="auto"):
        super().__init__()
        self._tick_locs = _vtk.vtkDoubleArray()
        self._tick_labels = _vtk.vtkStringArray()
        self.pen = Pen(color=(0, 0, 0), _wrap=self.GetPen())
        self.grid_pen = Pen(color=(0.95, 0.95, 0.95), _wrap=self.GetGridPen())
        self.behavior = behavior

    @property
    def label(self):
        return self.GetTitle()

    @label.setter
    def label(self, val):
        self.SetTitle(val)

    @property
    def label_visibility(self):
        return self.GetTitleVisible()

    @label_visibility.setter
    def label_visibility(self, val):
        self.SetTitleVisible(bool(val))

    @property
    def range(self):
        r = [0.0, 0.0]
        self.GetRange(r)
        return r

    @range.setter
    def range(self, val):
        if val is None:
            self.behavior = "auto"
        else:
            self.behavior = "fixed"
            self.SetRange(*val)

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, val):
        if val in self.BEHAVIORS:
            self._behavior = val
            self.SetBehavior(self.BEHAVIORS[val])
        else:
            formatted_behaviors = "\", \"".join(self.BEHAVIORS.keys())
            raise ValueError(f"Invalid behavior. Allowed behaviors: \"{formatted_behaviors}\"")

    @property
    def margins(self):
        return self.GetMargins()

    @margins.setter
    def margins(self, val):
        self.SetMargins(*val)

    @property
    def log_scale(self):
        # Returns whether a log scale is used on this axis
        return self.GetLogScaleActive()

    @log_scale.setter
    def log_scale(self, val):
        # False: log_scale will be disabled, True: axis will attempt to activate log_scale if possible
        self.SetLogScale(bool(val))

    @property
    def grid(self):
        return self.GetGridVisible()

    @grid.setter
    def grid(self, val):
        self.SetGridVisible(bool(val))

    @property
    def visible(self):
        return self.GetAxisVisible()

    @visible.setter
    def visible(self, val):
        self.SetAxisVisible(bool(val))

    def toggle(self):
        self.visible = not self.visible

    # --- Ticks ---
    @property
    def tick_count(self):
        return self.GetNumberOfTicks()

    @tick_count.setter
    def tick_count(self, val):
        self.SetNumberOfTicks(int(val))

    @property
    def tick_locations(self):
        positions = self.GetTickPositions()
        return tuple(positions.GetValue(i) for i in range(positions.GetNumberOfValues()))

    @tick_locations.setter
    def tick_locations(self, val):
        self._tick_locs.Reset()
        if val is not None:
            for loc in val:
                self._tick_locs.InsertNextValue(loc)
        self._update_ticks()

    @property
    def tick_labels(self):
        labels = self.GetTickLabels()
        return tuple(labels.GetValue(i) for i in range(labels.GetNumberOfValues()))

    @tick_labels.setter
    def tick_labels(self, val):
        # val is either None to fallback to the default labelling, a sequence to manually specify each label or a string
        # describing the label format to use for each label.
        self._tick_labels.Reset()
        self.SetNotation(_vtk.vtkAxis.STANDARD_NOTATION)
        if isinstance(val, str):
            precision = int(val[:-1])
            notation = val[-1].lower()
            if notation == "f":
                self.SetNotation(_vtk.vtkAxis.FIXED_NOTATION)
                self.SetPrecision(precision)
            elif notation == "e":
                self.SetNotation(_vtk.vtkAxis.SCIENTIFIC_NOTATION)
                self.SetPrecision(precision)
        elif isinstance(val, Sequence):
            for label in val:
                self._tick_labels.InsertNextValue(label)
        self._update_ticks()

    @property
    def tick_size(self):
        return self.GetTickLength()

    @tick_size.setter
    def tick_size(self, val):
        self.SetTickLength(val)

    @property
    def tick_labels_offset(self):
        return self.GetLabelOffset()

    @tick_labels_offset.setter
    def tick_labels_offset(self, val):
        self.SetLabelOffset(float(val))

    @property
    def tick_labels_visibility(self):
        return self.GetLabelsVisible()

    @tick_labels_visibility.setter
    def tick_labels_visibility(self, val):
        self.SetLabelsVisible(bool(val))
        self.SetRangeLabelsVisible(bool(val))

    @property
    def tick_visibility(self):
        return self.GetTicksVisible()

    @tick_visibility.setter
    def tick_visibility(self, val):
        self.SetTicksVisible(bool(val))

    def _update_ticks(self):
        locs = None if self._tick_locs.GetNumberOfValues() == 0 else self._tick_locs
        labels = None if self._tick_labels.GetNumberOfValues() == 0 else self._tick_labels
        self.SetCustomTickPositions(locs, labels)


class _CustomContextItem(_vtk.vtkPythonItem):

    class ItemWrapper(object):

        def Initialize(self, item):
            # item is the _CustomContextItem subclass instance
            return True

        def Paint(self, item, painter):
            # item is the _CustomContextItem subclass instance
            return item.paint(painter)

    def __init__(self):
        super().__init__()
        self.SetPythonObject(_CustomContextItem.ItemWrapper())  # This will also call ItemWrapper.Initialize

    def paint(self, painter: _vtk.vtkContext2D):
        return True


class _ChartBackground(_CustomContextItem):
    """ Utility class for chart backgrounds (until native VTK support is available) """

    def __init__(self, chart):
        super().__init__()
        self._chart = chart  # Chart to draw the background for
        # Default background is translucent with black border line
        self.BorderPen = Pen(color=(0, 0, 0))
        self.BackgroundBrush = Brush(color=(0, 0, 0, 0))

    def paint(self, painter: _vtk.vtkContext2D):
        if self._chart.visible:
            painter.ApplyPen(self.BorderPen)
            painter.ApplyBrush(self.BackgroundBrush)
            l, b, w, h = self._chart._geometry
            painter.DrawRect(l, b, w, h)
        return True


class _Chart(object):
    """ Common pythonic interface for vtkChart, vtkChartBox, vtkChartPie and ChartMPL instances """

    def __init__(self):
        super().__init__()
        self._background = _ChartBackground(self)
        self._x_axis = Axis()  # Not actually used for now (see note in Chart2D), but still present for the
        self._y_axis = Axis()  # Charts.toggle_interaction code
        self._z_axis = Axis()

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
        _, _, c_w, c_h = self._geometry
        # Target size is calculated from specified normalized width and height and the renderer's current size
        t_w = self._size[0] * r_w
        t_h = self._size[1] * r_h
        if c_w != t_w or c_h != t_h:
            # Mismatch between current size and target size, so resize chart:
            self._geometry = (self._loc[0] * r_w, self._loc[1] * r_h, t_w, t_h)

    @property
    def _geometry(self):
        """ Chart geometry (x and y position of bottom left corner and width and height in pixels). """
        return self.GetSize()

    @_geometry.setter
    def _geometry(self, val):
        self.SetSize(_vtk.vtkRectf(*val))

    def is_within(self, pos):
        l, b, w, h = self._geometry
        return l <= pos[0] <= l+w and b <= pos[1] <= b+h

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
    def border_color(self):
        return self._background.BorderPen.color

    @border_color.setter
    def border_color(self, val):
        self._background.BorderPen.color = val

    @property
    def border_width(self):
        return self._background.BorderPen.width

    @border_width.setter
    def border_width(self, val):
        self._background.BorderPen.width = val

    @property
    def border_style(self):
        return self._background.BorderPen.style

    @border_style.setter
    def border_style(self, val):
        self._background.BorderPen.style = val

    @property
    def background_color(self):
        # return self.GetBackgroundBrush().GetColor()
        return self._background.BackgroundBrush.color

    @background_color.setter
    def background_color(self, val):
        # self.GetBackgroundBrush().SetColorF(*parse_color(val))
        self._background.BackgroundBrush.color = val

    @property
    def background_texture(self):
        return self._background.BackgroundBrush.texture

    @background_texture.setter
    def background_texture(self, val):
        self._background.BackgroundBrush.texture = val

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


class _Plot(object):
    """ Pythonic interface for vtkPlot and vtkPlot3D instances """

    MARKER_STYLES = {
        "": _vtk.vtkPlotPoints.NONE,
        "x": _vtk.vtkPlotPoints.CROSS,
        "+": _vtk.vtkPlotPoints.PLUS,
        "s": _vtk.vtkPlotPoints.SQUARE,
        "o": _vtk.vtkPlotPoints.CIRCLE,
        "d": _vtk.vtkPlotPoints.DIAMOND
    }

    def __init__(self):
        super().__init__()
        self._pen = Pen()
        self._brush = Brush()
        if hasattr(self, "SetPen"):
            self.SetPen(self._pen)
        if hasattr(self, "SetBrush"):
            self.SetBrush(self._brush)

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
                for style in Pen.LINE_STYLES.keys():
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
        return self.pen.color

    @color.setter
    def color(self, val):
        self.pen.color = val
        self.brush.color = val

    @property
    def pen(self):
        """Pen object controlling how lines in this plot are drawn.

        Returns
        --------
        pyvista.charts.Pen
            `Pen` object controlling how lines in this plot are drawn.

        Examples
        ---------
        >>> import pyvista
        """
        return self._pen

    @property
    def brush(self):
        """
        :return: Retrieve vtkBrush object controlling how shapes in this plot are filled.
        """
        return self._brush

    @property
    def line_width(self):
        return self.pen.width

    @line_width.setter
    def line_width(self, val):
        self.pen.width = val

    @property
    def line_style(self):
        return self.pen.style

    @line_style.setter
    def line_style(self, val):
        self.pen.style = val

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, val):
        self._label = val
        self.SetLabel(self._label)

    @property
    def visible(self):
        return self.GetVisible()

    @visible.setter
    def visible(self, val):
        self.SetVisible(val)

    def toggle(self):
        self.visible = not self.visible


class _MultiCompPlot(object):
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
    DEFAULT_COLOR_SCHEME = "qual_accent"

    def __init__(self):
        super().__init__()
        self._color_series = _vtk.vtkColorSeries()
        self._lookup_table = self._color_series.CreateLookupTable(_vtk.vtkColorSeries.CATEGORICAL)
        self._labels = _vtk.vtkStringArray()
        self.SetLabels(self._labels)
        self.color_scheme = self.DEFAULT_COLOR_SCHEME

    @property
    def color_scheme(self):
        return self._SCHEME_NAMES.get(self._color_series.GetColorScheme(), "custom")

    @color_scheme.setter
    def color_scheme(self, val):
        self._color_series.SetColorScheme(self.COLOR_SCHEMES.get(val, _vtk.vtkColorSeries.CUSTOM))
        self._color_series.BuildLookupTable(self._lookup_table, _vtk.vtkColorSeries.CATEGORICAL)

    @property
    def colors(self):
        return [self._color_series.GetColor(i) for i in range(self._color_series.GetNumberOfColors())]

    @colors.setter
    def colors(self, val):
        if val is None:
            self.color_scheme = self.DEFAULT_COLOR_SCHEME
        elif isinstance(val, str):
            self.color_scheme = val
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


class LinePlot2D(_vtk.vtkPlotLine, _Plot):

    def __init__(self, x, y, color="b", width=1.0, style="-", label=""):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32)})
        self.SetInputData(self._table, "x", "y")
        self.update(x, y)
        self.color = color
        self.width = width
        self.line_style = style
        self.label = label

    def update(self, x, y):
        if len(x) > 1:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False)})
            self.visible = True
        else:
            # Turn off visibility for fewer than 2 points as otherwise an error message is shown
            self.visible = False


class ScatterPlot2D(_vtk.vtkPlotPoints, _Plot):

    def __init__(self, x, y, color="b", size=10, style="o", label=""):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32)})
        self.SetInputData(self._table, "x", "y")
        self.update(x, y)
        self.color = color
        self.marker_size = size
        self.marker_style = style
        self.label = label

    def update(self, x, y):
        if len(x) > 0:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False)})
            self.visible = True
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


class AreaPlot(_vtk.vtkPlotArea, _Plot):

    def __init__(self, x, y1, y2, color="b", label=""):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y1": np.empty(0, np.float32), "y2": np.empty(0, np.float32)})
        self.SetInputData(self._table)
        self.SetInputArray(0, "x")
        self.SetInputArray(1, "y1")
        self.SetInputArray(2, "y2")
        self.update(x, y1, y2)
        self.color = color
        self.label = label

    def update(self, x, y1, y2):
        if len(x) > 0:
            self._table.update({"x": np.array(x, copy=False), "y1": np.array(y1, copy=False), "y2": np.array(y2, copy=False)})
            self.visible = True
        else:
            self.visible = False


class BarPlot(_vtk.vtkPlotBar, _Plot, _MultiCompPlot):

    ORIENTATIONS = {
        "H": _vtk.vtkPlotBar.HORIZONTAL,
        "V": _vtk.vtkPlotBar.VERTICAL
    }

    def __init__(self, x, y, color=None, label=None, offset=5, orientation="V"):
        super().__init__()
        if not isinstance(y[0], (Sequence, np.ndarray)):
            y = (y,)
        y_data = {f"y{i}": np.empty(0, np.float32) for i in range(len(y))}
        self._table = pyvista.Table({"x": np.empty(0, np.float32), **y_data})
        self.SetInputData(self._table, "x", "y0")
        for i in range(1, len(y)):
            self.SetInputArray(i, f"y{i}")
        self.update(x, y)

        if len(y) > 1:
            self.SetColorSeries(self._color_series)
            self.colors = color  # None will use default scheme
            self.labels = label
        else:
            self.brush.color = "b" if color is None else color
            self.label = "" if label is None else label
        self.offset = offset
        self.orientation = orientation

    def update(self, x, y):
        if len(x) > 0:
            if not isinstance(y[0], (Sequence, np.ndarray)):
                y = (y,)
            y_data = {f"y{i}": np.array(y[i], copy=False) for i in range(len(y))}
            self._table.update({"x": np.array(x, copy=False), **y_data})
            self.visible = True
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


class StackPlot(_vtk.vtkPlotStacked, _Plot, _MultiCompPlot):

    def __init__(self, x, ys, colors=None, labels=None):
        super().__init__()
        if not isinstance(ys[0], (Sequence, np.ndarray)):
            ys = (ys,)
        y_data = {f"y{i}": np.empty(0, np.float32) for i in range(len(ys))}
        self._table = pyvista.Table({"x": np.empty(0, np.float32), **y_data})
        self.SetInputData(self._table, "x", "y0")
        for i in range(1, len(ys)):
            self.SetInputArray(i, f"y{i}")
        self.update(x, ys)

        self.SetColorSeries(self._color_series)
        self.colors = colors  # None will use default scheme
        self.pen.style = None  # Hide lines by default
        self.labels = labels

    def update(self, x, ys):
        if len(x) > 0:
            if not isinstance(ys[0], (Sequence, np.ndarray)):
                ys = (ys,)
            y_data = {f"y{i}": np.array(ys[i], copy=False) for i in range(len(ys))}
            self._table.update({"x": np.array(x, copy=False), **y_data})
            self.visible = True
        else:
            self.visible = False


class Chart2D(_vtk.vtkChartXY, _Chart):
    PLOT_TYPES = {
        "scatter": ScatterPlot2D,
        "line": LinePlot2D,
        "area": AreaPlot,
        "bar": BarPlot,
        "stack": StackPlot
    }

    def __init__(self, size=(1, 1), loc=(0, 0), x_label="x", y_label="y", grid=True):
        super().__init__()
        self._plots = {plot_type: [] for plot_type in self.PLOT_TYPES.keys()}
        self.SetAutoSize(False)  # We manually set the appropriate size
        self.size = size
        self.loc = loc
        # self.SetAxis(_vtk.vtkAxis.BOTTOM, self._x_axis)  # Disabled for now and replaced by a wrapper object, as for
        # self.SetAxis(_vtk.vtkAxis.LEFT, self._y_axis)  # some reason vtkChartXY.SetAxis(...) causes a crash at the end
        self._x_axis = Axis(_wrap=self.GetAxis(_vtk.vtkAxis.BOTTOM))  # of the script's execution (nonzero exit code)
        self._y_axis = Axis(_wrap=self.GetAxis(_vtk.vtkAxis.LEFT))
        self.x_label = x_label
        self.y_label = y_label
        self.grid = grid
        self.legend = True

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
        marker_style, line_style, color = _Plot.parse_format(fmt)
        scatter_plot, line_plot = None, None
        if marker_style != "":
            scatter_plot = self.scatter(x, y, color, style=marker_style)
        if line_style != "":
            line_plot = self.line(x, y, color, style=line_style)
        return scatter_plot, line_plot

    def scatter(self, x, y, color="b", size=10, style="o", label=""):
        return self._add_plot("scatter", x, y, color=color, size=size, style=style, label=label)

    def line(self, x, y, color="b", width=1.0, style="-", label=""):
        return self._add_plot("line", x, y, color=color, width=width, style=style, label=label)

    def area(self, x, y1, y2, color="b", label=""):
        return self._add_plot("area", x, y1, y2, color=color, label=label)

    def bar(self, x, y, color=None, label=None, offset=5, orientation="V"):
        return self._add_plot("bar", x, y, color=color, label=label, offset=offset, orientation=orientation)

    def stack(self, x, ys, colors=None, labels=None):
        return self._add_plot("stack", x, ys, colors=colors, labels=labels)

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
        return self._x_axis

    @property
    def y_axis(self):
        return self._y_axis

    @property
    def x_label(self):
        return self.x_axis.label

    @x_label.setter
    def x_label(self, val):
        self.x_axis.label = val

    @property
    def y_label(self):
        return self.y_axis.label

    @y_label.setter
    def y_label(self, val):
        self.y_axis.label = val

    @property
    def x_range(self):
        return self.x_axis.range

    @x_range.setter
    def x_range(self, val):
        self.x_axis.range = val

    @property
    def y_range(self):
        return self.y_axis.range

    @y_range.setter
    def y_range(self, val):
        self.y_axis.range = val

    @property
    def grid(self):
        return self.x_axis.grid and self.y_axis.grid

    @grid.setter
    def grid(self, val):
        self.x_axis.grid = val
        self.y_axis.grid = val


class BoxPlot(_vtk.vtkPlotBox, _Plot, _MultiCompPlot):

    def __init__(self, data, colors=None):
        super().__init__()
        self._table = pyvista.Table(data)
        self._quartiles = _vtk.vtkComputeQuartiles()
        self._quartiles.SetInputData(self._table)
        self.SetInputData(self._quartiles.GetOutput())
        self.update(data)
        self.SetLookupTable(self._lookup_table)
        self.colors = colors

    def update(self, data):
        self._table.update(data)
        self._quartiles.Update()


class ChartBox(_vtk.vtkChartBox, _Chart):

    def __init__(self, data):
        super().__init__()
        self._plot = BoxPlot(data)
        self.SetPlot(self._plot)
        self.SetColumnVisibilityAll(True)
        self.legend = True

    def render_event(self, *args, **kwargs):
        pass  # ChartBox fills entire scene by default, so no resizing is needed (nor possible at this moment)

    @property
    def _geometry(self):
        # Needed for background (remove once resizing is possible)
        return (0, 0, *self.renderer.GetSize())

    @_geometry.setter
    def _geometry(self, value):
        raise AttributeError(f'Cannot set the geometry of {type(self).__class__}')

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


class PiePlot(_vtkWrapper, _vtk.vtkPlotPie, _Plot, _MultiCompPlot):

    def __init__(self, data, labels=None, colors=None):
        super().__init__()
        self._table = pyvista.Table(data)
        self.SetInputData(self._table)
        self.SetInputArray(0, self._table.keys()[0])
        self.update(data)

        self.labels = labels

        self.SetColorSeries(self._color_series)
        self.colors = colors

    def update(self, data):
        self._table.update(data)


class ChartPie(_vtk.vtkChartPie, _Chart):

    def __init__(self, data, labels=None):
        super().__init__()
        self.AddPlot(0)  # We can't manually set a wrapped vtkPlotPie instance...
        self._plot = PiePlot(data, labels, _wrap=self.GetPlot(0))  # So we have to wrap the existing one
        self.legend = True

    def render_event(self, *args, **kwargs):
        pass  # ChartPie fills entire scene by default, so no resizing is needed (nor possible at this moment)

    @property
    def _geometry(self):
        # Needed for background (remove once resizing is possible)
        return (0, 0, *self.renderer.GetSize())

    @_geometry.setter
    def _geometry(self, value):
        raise AttributeError(f'Cannot set the geometry of {type(self).__class__}')

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


#region 3D charts
# Basic implementation of 3D line, scatter and volume plots, to be used in a 3D chart.
# Unfortunately, they are much less customisable than their 2D counterparts and they do not respect the enforced
# size/geometry constraints once you start interacting with them. So for now they are not further supported and not
# publicly exposed in pyvista.
class _LinePlot3D(_vtk.vtkPlotLine3D, _Plot):

    def __init__(self, x, y, z, color="b", width=1.0, style="-"):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32), "z": np.empty(0, np.float32)})
        # self.SetInputData(self._table, "x", "y", "z")
        self.update(x, y, z)
        self.color = color
        self.width = width
        self.line_style = style

    def update(self, x, y, z):
        if len(x) > 1:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False), "z": np.array(z, copy=False)})
            self.SetInputData(self._table, "x", "y", "z")  # For 3D plots a copy is made to an internal table, so we have to call this every time...
            self.visible = True
        else:
            # Turn off visibility for fewer than 2 points as otherwise an error message is shown
            self.visible = False


class _ScatterPlot3D(_vtk.vtkPlotPoints3D, _Plot):

    def __init__(self, x, y, z, color="b"):
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32), "z": np.empty(0, np.float32)})
        # self.SetInputData(self._table, "x", "y", "z")
        self.update(x, y, z)
        self.color = color

    def update(self, x, y, z):
        if len(x) > 0:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False), "z": np.array(z, copy=False)})
            self.SetInputData(self._table, "x", "y", "z")  # For 3D plots a copy is made to an internal table, so we have to call this every time...
            self.visible = True
        else:
            self.visible = False


class _SurfacePlot(_vtk.vtkPlotSurface, _Plot):

    def __init__(self, x, y, z):
        super().__init__()
        self._table = pyvista.Table(np.empty((0, 0), np.float32))
        self.update(x, y, z)

    def update(self, x, y, z):
        x = np.array(x, copy=False).reshape((-1,))
        y = np.array(y, copy=False).reshape((len(x),))
        z = np.array(z, copy=False).reshape((len(x), len(x)))
        if len(x) > 2:
            self._table = pyvista.Table(z)  # We have to create a new table, as update will not remove columns if needed (i.e. less points given)
            self.SetXRange(x[0], x[-1])
            self.SetYRange(y[0], y[-1])
            self.SetInputData(self._table)  # For 3D plots a copy is made to an internal table, so we have to call this every time...
            self.visible = True
        else:
            self.visible = False


class _Chart3D(_vtk.vtkChartXYZ, _Chart):
    PLOT_TYPES = {
        "scatter": _ScatterPlot3D,
        "line": _LinePlot3D,
        "surface": _SurfacePlot
    }

    def __init__(self, size=(0.8, 0.8), loc=(0.1, 0.1), x_label="x", y_label="y", z_label="z"):
        super().__init__()
        self._plots = {plot_type: [] for plot_type in self.PLOT_TYPES.keys()}
        self.background_color = None  # Transparent background and no border, as the 3D plots do not respect the
        self.border_style = None  # geometry bounds
        self.SetFitToScene(True)  # Some more room for axis labels
        self.size = size
        self.loc = loc
        self._geom = (0, 0, 0, 0)  # Will be properly set on first render_event
        self._x_axis = Axis(_wrap=self.GetAxis(0))
        self._y_axis = Axis(_wrap=self.GetAxis(1))
        self._z_axis = Axis(_wrap=self.GetAxis(2))
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label

    def render_event(self, *args, **kwargs):
        self.RecalculateBounds()
        super().render_event(*args, **kwargs)

    @property
    def _geometry(self):
        # TODO: replace by self.GetGeometry() once properly supported by vtk
        return self._geom

    @_geometry.setter
    def _geometry(self, val):
        self.SetGeometry(_vtk.vtkRectf(*val))
        self._geom = tuple(val)

    def _add_plot(self, plot_type, *args, **kwargs):
        plot = self.PLOT_TYPES[plot_type](*args, **kwargs)
        self.AddPlot(plot)
        self._plots[plot_type].append(plot)
        return plot

    def plot(self, x, y, z, fmt):
        """Matplotlib like plot method

        Examples
        --------
        ###############################################################################
        # Some limited functionality for 3D charts also exists. The interface is the
        # same as for the 2D counterparts. (To be added to ``chart_basics`` example
        # once supported.)

        >>> x = np.arange(11)
        >>> y = rng.integers(-5, 6, 11)
        >>> z = rng.integers(-5, 6, 11)
        >>> p = pv.Plotter()
        >>> p.background_color = (1, 1, 1)
        >>> chart = pv.Chart3D()
        >>> chart.plot(x, y, z, 'x-b')  # Show markers (marker style is ignored), solid line and blue color 'b'
        >>> p.add_chart(chart)
        >>> p.show()

        """
        # TODO: make x and fmt optional, allow multiple ([x], y, z, [fmt]) entries
        marker_style, line_style, color = _Plot.parse_format(fmt)
        scatter_plot, line_plot = None, None
        if marker_style != "":
            scatter_plot = self.scatter(x, y, z, color)
        if line_style != "":
            line_plot = self.line(x, y, z, color, style=line_style)
        return scatter_plot, line_plot

    def scatter(self, x, y, z, color="b"):
        return self._add_plot("scatter", x, y, z, color=color)

    def line(self, x, y, z, color="b", width=1.0, style="-"):
        return self._add_plot("line", x, y, z, color=color, width=width, style=style)

    def surface(self, x, y, z):
        """
        Examples
        --------
        ###############################################################################
        # 3D surfaces can be visualized on such a chart as well. (To be added to
        # ``chart_basics`` example once supported.)

        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> xx, yy = np.meshgrid(x, y)
        >>> z = np.cos(6*(xx**2+yy**2))
        >>> p = pv.Plotter()
        >>> p.background_color = (1, 1, 1)
        >>> chart = pv.Chart3D()
        >>> chart.surface(x, y, z)
        >>> p.add_chart(chart)
        >>> p.show()

        """
        return self._add_plot("surface", x, y, z)

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
        return self._x_axis

    @property
    def y_axis(self):
        return self._y_axis

    @property
    def z_axis(self):
        return self._z_axis

    @property
    def x_label(self):
        return self.x_axis.label

    @x_label.setter
    def x_label(self, val):
        self.x_axis.label = val

    @property
    def y_label(self):
        return self.y_axis.label

    @y_label.setter
    def y_label(self, val):
        self.y_axis.label = val

    @property
    def z_label(self):
        return self.z_axis.label

    @z_label.setter
    def z_label(self, val):
        self.z_axis.label = val

    @property
    def x_range(self):
        return self.x_axis.range

    @x_range.setter
    def x_range(self, val):
        self.x_axis.range = val

    @property
    def y_range(self):
        return self.y_axis.range

    @y_range.setter
    def y_range(self, val):
        self.y_axis.range = val

    @property
    def z_range(self):
        return self.z_axis.range

    @z_range.setter
    def z_range(self, val):
        self.z_axis.range = val
#endregion


class ChartMPL(_vtk.vtkImageItem, _Chart):

    def __init__(self, figure: matplotlib.figure.Figure, size: Tuple[int, int] = (1, 1), loc: Tuple[int, int] = (0, 0)):
        """
        Create new chart from an existing matplotlib figure.
        :param figure: The matplotlib figure to draw
        :param size: The normalized size of the chart (values between 0 and 1). None to completely fill the renderer
                     and autoresize
        :param loc: The normalized location of the chart (values between 0 and 1). None to manually set the position
                    (in pixels)
        """
        if not _HAS_MPL:
            raise ImportError("ChartMPL requires matplotlib")

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
    def _geometry(self):
        r_w, r_h = self.renderer.GetSize()
        t_w = self._size[0]*r_w
        t_h = self._size[1]*r_h
        return (*self.position, t_w, t_h)

    @_geometry.setter
    def _geometry(self, value):
        raise AttributeError(f'Cannot set the geometry of {type(self).__class__}')

    @property
    def background_color(self):
        return self._bg_color

    @background_color.setter
    def background_color(self, val):
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

#region Type definitions
Chart = Union[Chart2D, ChartBox, ChartPie, ChartMPL]
#endregion

class Charts:

    def __init__(self, renderer: "pyvista.Renderer"):
        self._charts = []

        self._scene = None  # Postpone creation of scene and actor objects until they are needed. Otherwise they are
        self._actor = None  # not properly cleaned up in case the Plotter's initialization fails.
        self._renderer = renderer

    def setup_scene(self):
        self._scene = _vtk.vtkContextScene()
        self._actor = _vtk.vtkContextActor()

        self._actor.SetScene(self._scene)
        self._renderer.AddActor(self._actor)
        self._scene.SetRenderer(self._renderer)

    def deep_clean(self):
        if self._scene is not None:
            for chart in self._charts:
                self.remove_chart(chart)
            self._renderer.RemoveActor(self._actor)
        self._scene = None
        self._actor = None

    def add_chart(self, chart: Chart):
        if self._scene is None:
            self.setup_scene()
        self._charts.append(chart)
        self._scene.AddItem(chart._background)
        self._scene.AddItem(chart)
        chart.SetInteractive(False)  # Charts are not interactive by default

    def remove_chart(self, chart_or_index: Union[Chart, int]):
        chart = self._charts[chart_or_index] if isinstance(chart_or_index, int) else chart_or_index
        assert chart in self._charts
        self._charts.remove(chart)
        self._scene.RemoveItem(chart)
        self._scene.RemoveItem(chart._background)

    def toggle_interaction(self, mouse_pos):
        # Mouse_pos is either False (to disable interaction with all charts) or a tuple containing the mouse position
        # (to disable interaction with all charts, except the one indicated by the mouse, if any)
        enable = False
        for chart in self._charts:
            if chart.visible and (mouse_pos is not False and chart.is_within(mouse_pos)):
                chart.SetInteractive(True)
                if chart._x_axis is not None:
                    chart._x_axis.behavior = "fixed"  # Change the chart's axis behaviour to fixed, such that the user
                if chart._y_axis is not None:
                    chart._y_axis.behavior = "fixed"  # can properly interact with the chart
                enable = True
            else:
                chart.SetInteractive(False)

        return self._scene if enable else None

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
