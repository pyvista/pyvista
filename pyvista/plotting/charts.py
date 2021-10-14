"""Module containing pyvista wrappers for the vtk Charts API."""

import numpy as np
from typing import Tuple, Sequence, Union, Optional
import inspect

import pyvista
from pyvista import _vtk
from .tools import parse_color
    

#region Some metaclass wrapping magic
class _vtkWrapperMeta(type):

    def __init__(cls, clsname, bases, attrs):
        # Restore the signature of classes inheriting from _vtkWrapper
        # Based on https://stackoverflow.com/questions/49740290/call-from-metaclass-shadows-signature-of-init
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())
        params.insert(len(params)-1 if params[-1].kind == inspect.Parameter.VAR_KEYWORD else len(params),
                      inspect.Parameter("_wrap", inspect.Parameter.KEYWORD_ONLY, default=None))
        cls.__signature__ = sig.replace(parameters=params[1:])
        super().__init__(clsname, bases, attrs)

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
    """Pythonic wrapper for a VTK Pen, used to draw lines.

    Parameters
    ----------
    color : color, optional
        Color of the lines drawn using this pen. Any color parsable by :func:`pyvista.parse_color` is allowed. Defaults
        to ``"k"``.

    width : float, optional
        Width of the lines drawn using this pen. Defaults to ``1``.

    style : str, optional
        Style of the lines drawn using this pen. See :ref:`Pen.LINE_STYLES <pen_line_styles>` for a list of allowed line styles. Defaults
        to ``"-"``.

    Other Parameters
    ----------------
    _wrap : vtk.vtkPen, optional
        Wrap an existing VTK Pen instance. Defaults to ``None`` (no wrapping).

    Notes
    -----
    .. _pen_line_styles:

    LINE_STYLES : dict
        Dictionary containing all allowed line styles as its keys.

        .. include:: ../pen_line_styles.rst

    """

    LINE_STYLES = {  # descr is used in the documentation, set to None to hide it from the docs.
        "": {"id": _vtk.vtkPen.NO_PEN, "descr": "Hidden"},
        "-": {"id": _vtk.vtkPen.SOLID_LINE, "descr": "Solid"},
        "--": {"id": _vtk.vtkPen.DASH_LINE, "descr": "Dashed"},
        ":": {"id": _vtk.vtkPen.DOT_LINE, "descr": "Dotted"},
        "-.": {"id": _vtk.vtkPen.DASH_DOT_LINE, "descr": "Dash-dot"},
        "-..": {"id": _vtk.vtkPen.DASH_DOT_DOT_LINE, "descr": "Dash-dot-dot"}
    }

    def __init__(self, color="k", width=1, style="-"):
        """Initialize a new Pen instance."""
        super().__init__()
        self.color = color
        self.width = width
        self.style = style

    @property
    def color(self):
        """Return or set the pen's color.

        Examples
        --------
        Set the pen's color to red.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.pen.color = 'r'
        >>> chart.show()

        """
        return self._color

    @color.setter
    def color(self, val):
        self._color = (0, 0, 0, 0) if val is None else parse_color(val, opacity=1)
        self.SetColorF(*self._color)

    @property
    def width(self):
        """Return or set the pen's width.

        Examples
        --------
        Set the pen's width to 10

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.pen.width = 10
        >>> chart.show()

        """
        return self.GetWidth()

    @width.setter
    def width(self, val):
        self.SetWidth(float(val))

    @property
    def style(self):
        """Return or set the pen's line style.

        See :ref:`Pen.LINE_STYLES <pen_line_styles>` for a list of allowed line styles.

        Examples
        --------
        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.pen.style = '-.'
        >>> chart.show()

        """
        return self._line_style

    @style.setter
    def style(self, val):
        if val is None:
            val = ""
        try:
            self.SetLineType(self.LINE_STYLES[val]["id"])
            self._line_style = val
        except KeyError:
            formatted_styles = "\", \"".join(self.LINE_STYLES.keys())
            raise ValueError(f"Invalid line style. Allowed line styles: \"{formatted_styles}\"")


class Brush(_vtkWrapper, _vtk.vtkBrush):
    """Pythonic wrapper for a VTK Brush, used to fill shapes.

    Parameters
    ----------
    color : color, optional
        Fill color of the shapes drawn using this brush. Any color
        parsable by :func:`pyvista.parse_color` is allowed.  Defaults
        to ``"k"``.

    texture : pyvista.Texture, optional
        Texture used to fill shapes drawn using this brush. Any
        object convertible to a :class:`pyvista.Texture` is
        allowed. Defaults to ``None``.

    Other Parameters
    ----------------
    _wrap : vtk.vtkBrush, optional
        Wrap an existing VTK Brush instance. Defaults to ``None`` (no wrapping).

    """

    def __init__(self, color="k", texture=None):
        """Initialize a new Pen instance."""
        super().__init__()
        self.color = color
        self.texture = texture
        self._interpolate = True  # vtkBrush textureProperties defaults to LINEAR & STRETCH
        self._repeat = False

    @property
    def color(self):
        """Return or set the brush's color.

        Examples
        --------
        Set the brush's color to red.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.color = 'r'
        >>> chart.show()

        """
        return self._color

    @color.setter
    def color(self, val):
        self._color = (0, 0, 0, 0) if val is None else parse_color(val, opacity=1)
        self.SetColorF(*self._color)

    @property
    def texture(self):
        """Return or set the brush's texture.

        Examples
        --------
        Set the brush's texture to the sample puppy texture.

        >>> import pyvista
        >>> from pyvista import examples
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.texture = examples.download_puppy_texture()
        >>> chart.show()

        """
        return self._texture

    @texture.setter
    def texture(self, val):
        if val is None:
            self._texture = None
            self.SetTexture(None)
        else:
            self._texture = pyvista.Texture(val)
            self.SetTexture(self._texture.to_image())

    @property
    def texture_interpolate(self):
        """Set texture interpolation mode.

        There are two modes:

        * ``False`` - NEAREST
        * ``True`` - LINEAR

        Examples
        --------
        Setup a brush with a texture.

        >>> import pyvista
        >>> from pyvista import examples
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.texture = examples.download_puppy_texture()
        >>> chart.show()

        Disable linear interpolation.

        >>> plot.brush.texture_interpolate = False
        >>> chart.show()

        """
        return self._interpolate

    @texture_interpolate.setter
    def texture_interpolate(self, val):
        self._interpolate = bool(val)
        self._update_textureprops()

    @property
    def texture_repeat(self):
        """Return or set the texture repeat mode.

        There are two modes:

        * ``False`` - STRETCH
        * ``True`` - REPEAT

        Examples
        --------
        Setup a brush with a texture.

        >>> import pyvista
        >>> from pyvista import examples
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.texture = examples.download_puppy_texture()
        >>> chart.show()

        Enable texture repeat.

        >>> plot.brush.texture_repeat = True
        >>> chart.show()

        """
        return self._repeat

    @texture_repeat.setter
    def texture_repeat(self, val):
        self._repeat = bool(val)
        self._update_textureprops()

    def _update_textureprops(self):
        # Interpolation: NEAREST = 0x01, LINEAR = 0x02
        # Stretch/repeat: STRETCH = 0x04, REPEAT = 0x08
        self.SetTextureProperties(1+int(self._interpolate) + 4*(1+int(self._repeat)))


class Axis(_vtkWrapper, _vtk.vtkAxis):
    """Pythonic interface for a VTK Axis, used by 2D charts.

    Parameters
    ----------
    label : str, optional
        Axis label. Defaults to the empty string ``""`` (no visible label).

    range : list or tuple of float, optional
        Axis range, denoting the minimum and maximum values
        displayed on this axis. Setting this to any valid value
        other than ``None`` will change this axis behavior to
        ``'fixed'``. Setting it to ``None`` will change the axis
        behavior to ``'auto'``. Defaults to ``None``
        (automatically scale axis).

    grid : bool, optional
        Flag to toggle grid lines visibility for this axis. Defaults to ``True``.

    """

    BEHAVIORS = {
        "auto": _vtk.vtkAxis.AUTO,
        "fixed": _vtk.vtkAxis.FIXED
    }

    def __init__(self, label="", range=None, grid=True):
        """Initialize a new Axis instance."""
        super().__init__()
        self._tick_locs = _vtk.vtkDoubleArray()
        self._tick_labels = _vtk.vtkStringArray()
        self.pen = Pen(color=(0, 0, 0), _wrap=self.GetPen())
        self.grid_pen = Pen(color=(0.95, 0.95, 0.95), _wrap=self.GetGridPen())
        self.label = label
        self._behavior = None  # Will be set by specifying the range below
        self.range = range
        self.grid = grid

    @property
    def label(self):
        """Return or set the axis label.

        Examples
        --------
        Set the axis label to ``"Axis Label"``.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.label = "Axis Label"
        >>> chart.show()

        """
        return self.GetTitle()

    @label.setter
    def label(self, val):
        self.SetTitle(val)

    @property
    def label_visible(self):
        """Return or set the axis label's visibility.

        Examples
        --------
        Hide the x-axis label of a 2D chart.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.label_visible = False
        >>> chart.show()

        """
        return self.GetTitleVisible()

    @label_visible.setter
    def label_visible(self, val):
        self.SetTitleVisible(bool(val))

    @property
    def range(self):
        """Return or set the axis range.

        This will automatically set the axis behavior to ``"fixed"``
        when a valid range is given. Setting the range to ``None``
        will set the axis behavior to ``"auto"``.

        Examples
        --------
        Manually specify the x-axis range of a 2D chart.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.range = [0, 5]
        >>> chart.show()

        Revert to automatic axis scaling.

        >>> chart.x_axis.range = None
        >>> chart.show()

        """
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
        """Set the axis' scaling behavior.

        Allowed behaviors are ``'auto'`` to automatically rescale the
        axis to fit all visible datapoints in the plot, or ``'fixed'``
        to use the user defined range.

        Examples
        --------
        Manually specify the x-axis range of a 2D chart.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.range = [0, 5]
        >>> chart.show()

        Revert to automatic axis scaling.

        >>> chart.x_axis.behavior = "auto"
        >>> chart.show()
        >>> print(chart.x_axis.range)
        [0.0, 2.0]

        """
        return self._behavior

    @behavior.setter
    def behavior(self, val):
        try:
            self.SetBehavior(self.BEHAVIORS[val])
            self._behavior = val
        except KeyError:
            formatted_behaviors = "\", \"".join(self.BEHAVIORS.keys())
            raise ValueError(f"Invalid behavior. Allowed behaviors: \"{formatted_behaviors}\"")

    @property
    def margin(self):
        """Return or set the axis margin.

        Examples
        --------
        Create a 2D chart.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Manually specify a larger (bottom) margin for the x-axis and a
        larger (left) margin for the y-axis.

        >>> chart.x_axis.margin = 50
        >>> chart.y_axis.margin = 50
        >>> chart.show()

        """
        return self.GetMargins()[0]

    @margin.setter
    def margin(self, val):
        # Second margin doesn't seem to have any effect? So we only expose the first entry as 'the margin'.
        m = self.GetMargins()
        self.SetMargins(val, m[1])

    @property
    def log_scale(self):
        """Flag denoting whether a log scale is used for this axis.

        Note that setting this property to ``True`` will not guarantee
        that the log scale will be enabled.  Verify whether activating
        the log scale succeeded by rereading this property.

        Examples
        --------
        Create a 2D chart.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2, 3, 4], [1e0, 1e1, 1e2, 1e3, 1e4])
        >>> chart.show()

        Try to enable the log scale on the y-axis.

        >>> chart.y_axis.log_scale = True
        >>> chart.show()
        >>> print(f"Enabling log scale {'succeeded' if chart.y_axis.log_scale else 'failed'}.")
        Enabling log scale succeeded.

        """
        return self.GetLogScaleActive()

    @log_scale.setter
    def log_scale(self, val):
        # False: log_scale will be disabled, True: axis will attempt to activate log_scale if possible
        self.SetLogScale(bool(val))

    @property
    def grid(self):
        """Return or set the axis' grid line visibility.

        Examples
        --------
        Create a 2D chart with grid lines disabled for the x-axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.grid = False
        >>> chart.show()

        """
        return self.GetGridVisible()

    @grid.setter
    def grid(self, val):
        self.SetGridVisible(bool(val))

    @property
    def visible(self):
        """Return or set the axis' visibility.

        Examples
        --------
        Create a 2D chart with no visible y-axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.y_axis.visible = False
        >>> chart.show()

        """
        return self.GetAxisVisible()

    @visible.setter
    def visible(self, val):
        self.SetAxisVisible(bool(val))

    def toggle(self):
        """Toggle the axis' visibility.

        Examples
        --------
        Create a 2D chart.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Toggle the visibility of the y-axis.

        >>> chart.y_axis.toggle()
        >>> chart.show()

        """
        self.visible = not self.visible

    # --- Ticks ---
    @property
    def tick_count(self):
        """Return or set the number of ticks drawn on this axis.

        Setting this property to a negative value or ``None`` will
        automatically determine the appropriate amount of ticks to
        draw.

        Examples
        --------
        Create a 2D chart with a reduced number of ticks on the x-axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.tick_count = 5
        >>> chart.show()

        Revert back to automatic tick behavior.

        >>> chart.x_axis.tick_count = None
        >>> chart.show()

        """
        return self.GetNumberOfTicks()

    @tick_count.setter
    def tick_count(self, val):
        if val is None or val < 0:
            val = -1
        self.SetNumberOfTicks(int(val))

    @property
    def tick_locations(self):
        """Return or set the tick locations for this axis.

        Setting this to ``None`` will revert back to the default,
        automatically determined, tick locations.

        Examples
        --------
        Create a 2D chart with custom tick locations and labels on the y-axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.y_axis.tick_locations = (0.2, 0.4, 0.6, 1, 1.5, 2, 3)
        >>> chart.y_axis.tick_labels = ["Very small", "Small", "Still small",
        ...                             "Small?", "Not large", "Large?", 
        ...                             "Very large"]
        >>> chart.show()

        Revert back to automatic tick placement.

        >>> chart.y_axis.tick_locations = None
        >>> chart.y_axis.tick_labels = None
        >>> chart.show()

        """
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
        """Return or set the tick labels for this axis.

        You can specify a sequence, to provide a unique label to every
        tick position; a string, to describe the label format to use
        for each label; or ``None``, which will revert back to the
        default tick labels.  A label format is a string consisting of
        an integer part, denoting the precision to use, and a final
        character, denoting the notation to use.

        Allowed notations:

        * ``"f"`` for fixed notation 
        * ``"e"`` for scientific notation.

        Examples
        --------
        Create a 2D chart with custom tick locations and labels on the y-axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.y_axis.tick_locations = (0.2, 0.4, 0.6, 1, 1.5, 2, 3)
        >>> chart.y_axis.tick_labels = ["Very small", "Small", "Still small",
        ...                             "Small?", "Not large", "Large?", "Very large"]
        >>> chart.show()

        Revert back to automatic tick placement.

        >>> chart.y_axis.tick_locations = None
        >>> chart.y_axis.tick_labels = None
        >>> chart.show()

        Specify a custom label format to use (fixed notation with precision 2).

        >>> chart.y_axis.tick_labels = "2f"
        >>> chart.show()

        """
        labels = self.GetTickLabels()
        return tuple(labels.GetValue(i) for i in range(labels.GetNumberOfValues()))

    @tick_labels.setter
    def tick_labels(self, val):
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
        """Return or set the size of this axis' ticks.

        Examples
        --------
        Create a 2D chart with an x-axis with an increased tick size
        and adjusted offset for the tick labels.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.tick_size += 10
        >>> chart.x_axis.tick_labels_offset += 12
        >>> chart.show()

        """
        return self.GetTickLength()

    @tick_size.setter
    def tick_size(self, val):
        self.SetTickLength(val)

    @property
    def tick_labels_offset(self):
        """Return or set the offset of the tick labels for this axis.

        Examples
        --------
        Create a 2D chart with an x-axis with an increased tick size
        and adjusted offset for the tick labels.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.tick_size += 10
        >>> chart.x_axis.tick_labels_offset += 12
        >>> chart.show()

        """
        return self.GetLabelOffset()

    @tick_labels_offset.setter
    def tick_labels_offset(self, val):
        self.SetLabelOffset(float(val))

    @property
    def tick_labels_visible(self):
        """Return or set the tick label visibility for this axis.

        Examples
        --------
        Create a 2D chart with hidden tick labels on the y-axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.y_axis.tick_labels_visible = False
        >>> chart.show()

        """
        return self.GetLabelsVisible()

    @tick_labels_visible.setter
    def tick_labels_visible(self, val):
        self.SetLabelsVisible(bool(val))
        self.SetRangeLabelsVisible(bool(val))

    @property
    def ticks_visible(self):
        """Return or set the tick visibility for this axis.

        Examples
        --------
        Create a 2D chart with hidden ticks on the y-axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.y_axis.ticks_visible = False
        >>> chart.show()

        """
        return self.GetTicksVisible()

    @ticks_visible.setter
    def ticks_visible(self, val):
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

    def paint(self, painter):
        return True


class _ChartBackground(_CustomContextItem):
    """Utility class for chart backgrounds (until native VTK support is available)."""

    def __init__(self, chart):
        super().__init__()
        self._chart = chart  # Chart to draw the background for
        # Default background is translucent with black border line
        self.BorderPen = Pen(color=(0, 0, 0))
        self.BackgroundBrush = Brush(color=(0, 0, 0, 0))

    def paint(self, painter):
        if self._chart.visible:
            painter.ApplyPen(self.BorderPen)
            painter.ApplyBrush(self.BackgroundBrush)
            l, b, w, h = self._chart._geometry
            painter.DrawRect(l, b, w, h)
            # TODO: following 'patch' is necessary until vtkPlotPie is fixed. Otherwise Pie plots will use the same
            #  opacity as the chart's background when their legend is hidden. As the default background is transparent,
            #  this will cause Pie charts to completely disappear.
            painter.GetBrush().SetOpacity(255)
        return True


class _Chart(object):
    """Common pythonic interface for vtkChart, vtkChartBox, vtkChartPie and ChartMPL instances."""

    def __init__(self, size=(1, 1), loc=(0, 0)):
        super().__init__()
        self._background = _ChartBackground(self)
        self._x_axis = Axis()  # Not actually used for now (see note in Chart2D), but still present for the
        self._y_axis = Axis()  # Charts.toggle_interaction code
        self._z_axis = Axis()
        if size is not None:
            self.size = size
        if loc is not None:
            self.loc = loc

    @property
    def _scene(self):
        """Get a reference to the vtkScene in which this chart is drawn."""
        return self.GetScene()

    @property
    def _renderer(self):
        """Get a reference to the vtkRenderer in which this chart is drawn."""
        return self._scene.GetRenderer() if self._scene is not None else None

    def _render_event(self, *args, **kwargs):
        """Update the chart right before it will be rendered."""
        self._resize()

    def _resize(self):
        """Resize this chart.

        Resize this chart such that it always occupies the specified
        geometry (matching the specified location and size).
        """
        r_w, r_h = self._renderer.GetSize()  # Alternatively: self.scene.GetViewWidth(), self.scene.GetViewHeight()
        _, _, c_w, c_h = self._geometry
        # Target size is calculated from specified normalized width and height and the renderer's current size
        t_w = self._size[0] * r_w
        t_h = self._size[1] * r_h
        if c_w != t_w or c_h != t_h:
            # Mismatch between current size and target size, so resize chart:
            self._geometry = (self._loc[0] * r_w, self._loc[1] * r_h, t_w, t_h)

    @property
    def _geometry(self):
        """Chart geometry (x and y position of bottom left corner and width and height in pixels)."""
        return tuple(self.GetSize())

    @_geometry.setter
    def _geometry(self, val):
        """Set the chart geometry."""
        self.SetSize(_vtk.vtkRectf(*val))

    def _is_within(self, pos):
        """Check whether the specified position (in pixels) lies within this chart's geometry."""
        l, b, w, h = self._geometry
        return l <= pos[0] <= l+w and b <= pos[1] <= b+h

    @property
    def size(self):
        """Return or set the chart size in normalized coordinates.

        A size of ``(1, 1)`` occupies the whole renderer.

        Examples
        --------
        Create a half-sized 2D chart centered in the middle of the
        renderer.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.size = (0.5, 0.5)
        >>> chart.loc = (0.25, 0.25)
        >>> chart.show()

        """
        return self._size

    @size.setter
    def size(self, val):
        assert len(val) == 2 and 0 <= val[0] <= 1 and 0 <= val[1] <= 1
        self._size = val

    @property
    def loc(self):
        """Return or set the chart position in normalized coordinates.

        This denotes the location of the chart's bottom left corner.

        Examples
        --------
        Create a half-sized 2D chart centered in the middle of the
        renderer.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.size = (0.5, 0.5)
        >>> chart.loc = (0.25, 0.25)
        >>> chart.show()

        """
        return self._loc

    @loc.setter
    def loc(self, val):
        assert len(val) == 2 and 0 <= val[0] <= 1 and 0 <= val[1] <= 1
        self._loc = val

    @property
    def border_color(self):
        """Return or set the chart's border color.

        Examples
        --------
        Create a half-sized 2D chart centered in the middle of the
        renderer with a thick, dashed red border .

        >>> import pyvista
        >>> chart = pyvista.Chart2D(size=(0.5, 0.5), loc=(0.25, 0.25))
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.border_color = 'r'
        >>> chart.border_width = 5
        >>> chart.border_style = '--'
        >>> chart.show()

        """
        return self._background.BorderPen.color

    @border_color.setter
    def border_color(self, val):
        self._background.BorderPen.color = val

    @property
    def border_width(self):
        """Return or set the chart's border width.

        Examples
        --------
        Create a half-sized 2D chart centered in the middle of the
        renderer with a thick, dashed red border.

        >>> import pyvista
        >>> chart = pyvista.Chart2D(size=(0.5, 0.5), loc=(0.25, 0.25))
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.border_color = 'r'
        >>> chart.border_width = 5
        >>> chart.border_style = '--'
        >>> chart.show()

        """
        return self._background.BorderPen.width

    @border_width.setter
    def border_width(self, val):
        self._background.BorderPen.width = val

    @property
    def border_style(self):
        """Return or set the chart's border style.

        Examples
        --------
        Create a half-sized 2D chart centered in the middle of the
        renderer with a thick, dashed red border.

        >>> import pyvista
        >>> chart = pyvista.Chart2D(size=(0.5, 0.5), loc=(0.25, 0.25))
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.border_color = 'r'
        >>> chart.border_width = 5
        >>> chart.border_style = '--'
        >>> chart.show()

        """
        return self._background.BorderPen.style

    @border_style.setter
    def border_style(self, val):
        self._background.BorderPen.style = val

    @property
    def background_color(self):
        """Return or set the chart's background color.

        Examples
        --------
        Create a 2D chart with a green background.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.background_color = (0.5, 0.9, 0.5)
        >>> chart.show()

        """
        # return self.GetBackgroundBrush().GetColor()
        return self._background.BackgroundBrush.color

    @background_color.setter
    def background_color(self, val):
        # self.GetBackgroundBrush().SetColorF(*parse_color(val))
        self._background.BackgroundBrush.color = val

    @property
    def background_texture(self):
        """Return or set the chart's background texture.

        Examples
        --------
        Create a 2D chart plotting an approximate satellite
        trajectory.

        >>> import pyvista
        >>> from pyvista import examples
        >>> import numpy as np
        >>> chart = pyvista.Chart2D()
        >>> x = np.linspace(0, 1, 100)
        >>> y = np.sin(6.5*x-1)
        >>> _ = chart.line(x, y, "y", 4)
        >>> chart.background_texture = examples.load_globe_texture()
        >>> chart.hide_axes()
        >>> chart.show()

        """
        return self._background.BackgroundBrush.texture

    @background_texture.setter
    def background_texture(self, val):
        self._background.BackgroundBrush.texture = val

    @property
    def visible(self):
        """Return or set the chart's visibility.

        Examples
        --------
        Create a 2D chart.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Hide it.

        >>> chart.visible = False
        >>> chart.show()

        """
        return self.GetVisible()

    @visible.setter
    def visible(self, val):
        self.SetVisible(val)

    def toggle(self):
        """Toggle the chart's visibility.

        Examples
        --------
        Create a 2D chart.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Hide it.

        >>> chart.toggle()
        >>> chart.show()

        """
        self.visible = not self.visible

    @property
    def title(self):
        """Return or set the chart's title.

        Examples
        --------
        Create a 2D chart with title 'My Chart'.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.title = 'My Chart'
        >>> chart.show()

        """
        return self.GetTitle()

    @title.setter
    def title(self, val):
        self.SetTitle(val)

    @property
    def legend_visible(self):
        """Return or set the visibility of the chart's legend.

        Examples
        --------
        Create a pie chart with custom labels.

        >>> import pyvista
        >>> chart = pyvista.ChartPie([1, 2, 3])
        >>> chart.plot.labels = ["A", "B", "C"]
        >>> chart.show()

        Hide the legend.

        >>> chart.legend_visible = False
        >>> chart.show()

        """
        return self.GetShowLegend()

    @legend_visible.setter
    def legend_visible(self, val):
        self.SetShowLegend(val)

    def show(self, off_screen=None, full_screen=None, screenshot=None,
             window_size=None, notebook=None, background='w'):
        """Show this chart in a self contained plotter.

        Parameters
        ----------
        off_screen : bool
            Plots off screen when ``True``.  Helpful for saving screenshots
            without a window popping up.  Defaults to active theme setting in
            :attr:`pyvista.global_theme.full_screen
            <pyvista.themes.DefaultTheme.full_screen`.

        full_screen : bool, optional
            Opens window in full screen.  When enabled, ignores
            ``window_size``.  Defaults to active theme setting in
            :attr:`pyvista.global_theme.full_screen
            <pyvista.themes.DefaultTheme.full_screen`.

        screenshot : str or bool, optional
            Saves screenshot to file when enabled.  See:
            :func:`Plotter.screenshot() <pyvista.Plotter.screenshot>`.
            Default ``False``.

            When ``True``, takes screenshot and returns ``numpy`` array of
            image.

        window_size : list, optional
            Window size in pixels.  Defaults to global theme
            :attr:`pyvista.global_theme.window_size
            <pyvista.themes.DefaultTheme.window_size>`.

        notebook : bool, optional
            When ``True``, the resulting plot is placed inline a
            jupyter notebook.  Assumes a jupyter console is active.

        background : str or 3 item list, optional
            Use to make the entire mesh have a single solid color.
            Either a string, RGB list, or hex color string.  For example:
            ``color='white'``, ``color='w'``, ``color=[1, 1, 1]``, or
            ``color='#FFFFFF'``.  Defaults to ``'w'``.

        Returns
        -------
        image : np.ndarray
            Numpy array of the last image when ``screenshot=True``
            is set. Optionally contains alpha values. Sized:

            * [Window height x Window width x 3] if the theme sets
              ``transparent_background=False``.
            * [Window height x Window width x 4] if the theme sets
              ``transparent_background=True``.

        Examples
        --------
        Plot a simple sine wave as a scatter and line plot.

        >>> import pyvista
        >>> import numpy as np
        >>> x = np.linspace(0, 2*np.pi, 20)
        >>> y = np.sin(x)
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.scatter(x, y)
        >>> _ = chart.line(x, y, 'r')
        >>> chart.show()

        """
        pl = pyvista.Plotter(window_size=window_size,
                             notebook=notebook,
                             off_screen=off_screen)
        pl.background_color = background
        pl.add_chart(self)
        return pl.show(screenshot=screenshot,
                       full_screen=full_screen,
        )


class _Plot(object):
    """Common pythonic interface for vtkPlot and vtkPlot3D instances."""

    def __init__(self):
        super().__init__()
        self._pen = Pen()
        self._brush = Brush()
        self._label = ""
        if hasattr(self, "SetPen"):
            self.SetPen(self._pen)
        if hasattr(self, "SetBrush"):
            self.SetBrush(self._brush)

    @property
    def color(self):
        """Return or set the plot's color.

        This is the color used by the plot's pen and brush to draw lines and shapes.

        Examples
        --------
        Set the line plot's color to red and the area plot's color
        to cyan.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> area_plot = chart.area([0, 1, 2], [2, 1, 3])
        >>> line_plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> line_plot.color = 'r'
        >>> area_plot.color = 'c'
        >>> chart.show()

        """
        return self.pen.color

    @color.setter
    def color(self, val):
        self.pen.color = val
        self.brush.color = val

    @property
    def pen(self):
        """Pen object controlling how lines in this plot are drawn.

        Returns
        -------
        pyvista.charts.Pen
            Pen object controlling how lines in this plot are drawn.

        Examples
        --------
        Increase the line width of the line plot's pen object.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.pen.width = 10
        >>> chart.show()

        """
        return self._pen

    @property
    def brush(self):
        """Brush object controlling how shapes in this plot are filled.

        Returns
        -------
        pyvista.charts.Brush
            Brush object controlling how shapes in this plot are filled.

        Examples
        --------
        Use a custom texture for the area plot's brush object.

        >>> import pyvista
        >>> from pyvista import examples
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])
        >>> plot.brush.texture = examples.download_puppy_texture()
        >>> chart.show()

        """
        return self._brush

    @property
    def line_width(self):
        """Return or set the line width of all lines drawn in this plot.

        This is equivalent to accessing/modifying the width of this plot's pen.

        Examples
        --------
        Set the line width to 10

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.line_width = 10
        >>> chart.show()

        """
        return self.pen.width

    @line_width.setter
    def line_width(self, val):
        self.pen.width = val

    @property
    def line_style(self):
        """Return or set the line style of all lines drawn in this plot.

        This is equivalent to accessing/modifying the style of this plot's pen.

        Examples
        --------
        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.line_style = '-.'
        >>> chart.show()

        """
        return self.pen.style

    @line_style.setter
    def line_style(self, val):
        self.pen.style = val

    @property
    def label(self):
        """Return or set the this plot's label, as shown in the chart's legend.

        Examples
        --------
        Create a line plot with custom label.

        >>> import pyvista
        >>> import numpy as np
        >>> x = np.linspace(0, 10, 100)
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line(x, np.sin(x), label="Sine")
        >>> chart.show()

        Modify the label.

        >>> plot.label = "y = sin(x)"
        >>> chart.show()

        """
        return self._label

    @label.setter
    def label(self, val):
        self._label = "" if val is None else val
        self.SetLabel(self._label)

    @property
    def visible(self):
        """Return or set the this plot's visibility.

        Examples
        --------
        Create a line and scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> scatter_plot, line_plot = chart.plot([0, 1, 2], [2, 1, 3], "o-")
        >>> chart.show()

        Hide the scatter plot.

        >>> scatter_plot.visible = False
        >>> chart.show()

        """
        return self.GetVisible()

    @visible.setter
    def visible(self, val):
        self.SetVisible(val)

    def toggle(self):
        """Toggle the plot's visibility.

        Examples
        --------
        Create a line and scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> scatter_plot, line_plot = chart.plot([0, 1, 2], [2, 1, 3], "o-")
        >>> chart.show()

        Hide the scatter plot.

        >>> scatter_plot.toggle()
        >>> chart.show()

        """
        self.visible = not self.visible


class _MultiCompPlot(_Plot):
    """Common pythonic interface for vtkPlot instances with multiple components.

    Example subclasses are BoxPlot, PiePlot, BarPlot and StackPlot.
    """

    COLOR_SCHEMES = {
        "spectrum": {"id": _vtk.vtkColorSeries.SPECTRUM, "descr": "black, red, blue, green, purple, orange, brown"},
        "warm": {"id": _vtk.vtkColorSeries.WARM, "descr": "dark red → yellow"},
        "cool": {"id": _vtk.vtkColorSeries.COOL, "descr": "green → blue → purple"},
        "blues": {"id": _vtk.vtkColorSeries.BLUES, "descr": "Different shades of blue"},
        "wild_flower": {"id": _vtk.vtkColorSeries.WILD_FLOWER, "descr": "blue → purple → pink"},
        "citrus": {"id": _vtk.vtkColorSeries.CITRUS, "descr": "green → yellow → orange"},
        "div_purple_orange11": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_11, "descr": "dark brown → white → dark purple"},
        "div_purple_orange10": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_10, "descr": "dark brown → white → dark purple"},
        "div_purple_orange9": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_9, "descr": "brown → white → purple"},
        "div_purple_orange8": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_8, "descr": "brown → white → purple"},
        "div_purple_orange7": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_7, "descr": "brown → white → purple"},
        "div_purple_orange6": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_6, "descr": "brown → white → purple"},
        "div_purple_orange5": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_5, "descr": "orange → white → purple"},
        "div_purple_orange4": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_4, "descr": "orange → white → purple"},
        "div_purple_orange3": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_3, "descr": "orange → white → purple"},
        "div_spectral11": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_11, "descr": "dark red → light yellow → dark blue"},
        "div_spectral10": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_10, "descr": "dark red → light yellow → dark blue"},
        "div_spectral9": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_9, "descr": "red → light yellow → blue"},
        "div_spectral8": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_8, "descr": "red → light yellow → blue"},
        "div_spectral7": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_7, "descr": "red → light yellow → blue"},
        "div_spectral6": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_6, "descr": "red → light yellow → blue"},
        "div_spectral5": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_5, "descr": "red → light yellow → blue"},
        "div_spectral4": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_4, "descr": "red → light yellow → blue"},
        "div_spectral3": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_3, "descr": "orange → light yellow → green"},
        "div_brown_blue_green11": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_11, "descr": "dark brown → white → dark blue-green"},
        "div_brown_blue_green10": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_10, "descr": "dark brown → white → dark blue-green"},
        "div_brown_blue_green9": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_9, "descr": "brown → white → blue-green"},
        "div_brown_blue_green8": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_8, "descr": "brown → white → blue-green"},
        "div_brown_blue_green7": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_7, "descr": "brown → white → blue-green"},
        "div_brown_blue_green6": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_6, "descr": "brown → white → blue-green"},
        "div_brown_blue_green5": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_5, "descr": "brown → white → blue-green"},
        "div_brown_blue_green4": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_4, "descr": "brown → white → blue-green"},
        "div_brown_blue_green3": {"id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_3, "descr": "brown → white → blue-green"},
        "seq_blue_green9": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_9, "descr": "light blue → dark green"},
        "seq_blue_green8": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_8, "descr": "light blue → dark green"},
        "seq_blue_green7": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_7, "descr": "light blue → dark green"},
        "seq_blue_green6": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_6, "descr": "light blue → green"},
        "seq_blue_green5": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_5, "descr": "light blue → green"},
        "seq_blue_green4": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_4, "descr": "light blue → green"},
        "seq_blue_green3": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3, "descr": "light blue → green"},
        "seq_yellow_orange_brown9": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_9, "descr": "light yellow → orange → dark brown"},
        "seq_yellow_orange_brown8": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_8, "descr": "light yellow → orange → brown"},
        "seq_yellow_orange_brown7": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_7, "descr": "light yellow → orange → brown"},
        "seq_yellow_orange_brown6": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_6, "descr": "light yellow → orange → brown"},
        "seq_yellow_orange_brown5": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_5, "descr": "light yellow → orange → brown"},
        "seq_yellow_orange_brown4": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_4, "descr": "light yellow → orange"},
        "seq_yellow_orange_brown3": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_3, "descr": "light yellow → orange"},
        "seq_blue_purple9": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_9, "descr": "light blue → dark purple"},
        "seq_blue_purple8": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_8, "descr": "light blue → purple"},
        "seq_blue_purple7": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_7, "descr": "light blue → purple"},
        "seq_blue_purple6": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_6, "descr": "light blue → purple"},
        "seq_blue_purple5": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_5, "descr": "light blue → purple"},
        "seq_blue_purple4": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_4, "descr": "light blue → purple"},
        "seq_blue_purple3": {"id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_3, "descr": "light blue → purple"},
        "qual_accent": {"id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_ACCENT, "descr": "pastel green, pastel purple, pastel orange, pastel yellow, blue, pink, brown, gray"},
        "qual_dark2": {"id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_DARK2, "descr": "darker shade of qual_set2"},
        "qual_set3": {"id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET3, "descr": "pastel colors: blue green, light yellow, dark purple, red, blue, orange, green, pink, gray, purple, light green, yellow"},
        "qual_set2": {"id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET2, "descr": "blue green, orange, purple, pink, green, yellow, brown, gray"},
        "qual_set1": {"id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET1, "descr": "red, blue, green, purple, orange, yellow, brown, pink, gray"},
        "qual_pastel2": {"id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PASTEL2, "descr": "pastel shade of qual_set2"},
        "qual_pastel1": {"id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PASTEL1, "descr": "pastel shade of qual_set1"},
        "qual_paired": {"id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PAIRED, "descr": "light blue, blue, light green, green, light red, red, light orange, orange, light purple, purple, light yellow"},
        "custom": {"id": _vtk.vtkColorSeries.CUSTOM, "descr": None}
    }
    _SCHEME_NAMES = {scheme_info["id"]: scheme_name for scheme_name, scheme_info in COLOR_SCHEMES.items()}
    DEFAULT_COLOR_SCHEME = "qual_accent"

    def __init__(self):
        super().__init__()
        self._color_series = _vtk.vtkColorSeries()
        self._lookup_table = self._color_series.CreateLookupTable(_vtk.vtkColorSeries.CATEGORICAL)
        self._labels = _vtk.vtkStringArray()
        self.SetLabels(self._labels)
        self.color_scheme = self.DEFAULT_COLOR_SCHEME

    @staticmethod
    def _from_c3ub(c3ub):
        """Convert vtkColor3ub to an RGB color tuple (with values in range [0;1])."""
        return tuple(float(c) / 255 for c in c3ub)

    @staticmethod
    def _to_c3ub(color):
        """Convert an RGB(A) color tuple/sequence to a vtkColor3ub object (with values in range [0;255])."""
        return _vtk.vtkColor3ub(*[int(255 * c + 0.5) for c in color[:3]])

    @property
    def color_scheme(self):
        """Return or set the plot's color scheme.

        This scheme defines the colors of the different
        components drawn by this plot.
        See the table below for the available color
        schemes.

        Notes
        -----
        .. _plot_color_schemes:

        Overview of all available color schemes.

        .. include:: ../plot_color_schemes.rst

        Examples
        --------
        Set the pie plot's color scheme to warm.

        >>> import pyvista
        >>> chart = pyvista.ChartPie([6,5,4,3,2,1])
        >>> chart.plot.color_scheme = "warm"
        >>> chart.show()

        """
        return self._SCHEME_NAMES.get(self._color_series.GetColorScheme(), "custom")

    @color_scheme.setter
    def color_scheme(self, val):
        self._color_series.SetColorScheme(self.COLOR_SCHEMES.get(val, self.COLOR_SCHEMES["custom"])["id"])
        self._color_series.BuildLookupTable(self._lookup_table, _vtk.vtkColorSeries.CATEGORICAL)
        self.brush.color = self.colors[0]

    @property
    def colors(self):
        """Return or set the plot's colors.

        These are the colors used for the different
        components drawn by this plot.

        Examples
        --------
        Set the pie plot's colors manually.

        >>> import pyvista
        >>> chart = pyvista.ChartPie([6,5,4,3,2,1])
        >>> chart.plot.colors = ["b", "g", "r", "c", "m", "y"]
        >>> chart.show()

        """
        return [self._from_c3ub(self._color_series.GetColor(i)) for i in range(self._color_series.GetNumberOfColors())]

    @colors.setter
    def colors(self, val):
        if val is None:
            self.color_scheme = self.DEFAULT_COLOR_SCHEME
        elif isinstance(val, str):
            self.color_scheme = val
        else:
            self._color_series.SetNumberOfColors(len(val))
            for i, color in enumerate(val):
                self._color_series.SetColor(i, self._to_c3ub(parse_color(color)))
            self._color_series.BuildLookupTable(self._lookup_table, _vtk.vtkColorSeries.CATEGORICAL)
        self.brush.color = self.colors[0]  # Synchronize "color" and "colors" properties

    @property
    def color(self):
        """Return or set the plot's color.

        This is the color used by the plot's brush
        to draw the different components.

        Examples
        --------
        Set the bar plot's color to red.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([0, 1, 2], [2, 1, 3])
        >>> plot.color = 'r'
        >>> chart.show()

        """
        return self.brush.color

    @color.setter
    def color(self, val):
        # Override default _Plot behaviour. This makes sure the plot's "color_scheme", "colors" and "color" properties
        # (and their internal representations through color series, lookup tables and brushes) stay synchronized.
        self.colors = [val]

    @property
    def labels(self):
        """Return or set the this plot's labels, as shown in the chart's legend.

        Examples
        --------
        Create a stacked plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.stack([0, 1, 2], [[2, 1, 3], [0, 2, 1]])
        >>> chart.show()

        Modify the labels.

        >>> plot.labels = ["A", "B"]
        >>> chart.show()

        """
        return [self._labels.GetValue(i) for i in range(self._labels.GetNumberOfValues())]

    @labels.setter
    def labels(self, val):
        self._labels.Reset()
        if val is not None:
            for label in val:
                self._labels.InsertNextValue(label)

    @property
    def label(self):
        """Return or set the this plot's label, as shown in the chart's legend.

        Examples
        --------
        Create a box plot with custom label.

        >>> import pyvista
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)  # Seeded random number generator used for data generation
        >>> normal_data = rng.normal(size=50)
        >>> chart = pyvista.ChartBox({"Normal distribution": normal_data})
        >>> chart.show()

        Modify the label.

        >>> chart.plot.label = "x ~ N(0,1)"
        >>> chart.show()

        """
        return self.labels[0] if self._labels.GetNumberOfValues() > 0 else ""

    @label.setter
    def label(self, val):
        # Override default _Plot behaviour. This makes sure the plot's "labels" and "label" properties (and their
        # internal representations) stay synchronized.
        self.labels = None if val is None else [val]


class LinePlot2D(_vtk.vtkPlotLine, _Plot):
    """Class representing a 2D line plot.

    Users should typically not directly create new plot instances, but use the dedicated 2D chart's plotting methods.

    Parameters
    ----------
    x : array_like
        X coordinates of the points through which a line should be drawn.

    y : array_like
        Y coordinates of the points through which a line should be drawn.

    color : color, optional
        Color of the line drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed. Defaults
        to ``"b"``.

    width : float, optional
        Width of the line drawn in this plot. Defaults to ``1``.

    style : str, optional
        Style of the line drawn in this plot. See :ref:`Pen.LINE_STYLES <pen_line_styles>` for a list of allowed line
        styles. Defaults to ``"-"``.

    label : str, optional
        Label of this plot, as shown in the chart's legend. Defaults to ``""``.

    """

    def __init__(self, x, y, color="b", width=1.0, style="-", label=""):
        """Initialize a new 2D line plot instance."""
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32)})
        self.SetInputData(self._table, "x", "y")
        self.update(x, y)
        self.color = color
        self.line_width = width
        self.line_style = style
        self.label = label

    @property
    def x(self):
        """Retrieve the X coordinates of the points through which a line is drawn.

        Examples
        --------
        Create a line plot and display the x coordinates.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.x
        pyvista_ndarray([0, 1, 2])
        >>> chart.show()

        """
        return self._table["x"]

    @property
    def y(self):
        """Retrieve the Y coordinates of the points through which a line is drawn.

        Examples
        --------
        Create a line plot and display the y coordinates.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> plot.y
        pyvista_ndarray([2, 1, 3])
        >>> chart.show()

        """
        return self._table["y"]

    def update(self, x, y):
        """Update this plot's points, through which a line is drawn.

        Parameters
        ----------
        x : array_like
            The new x coordinates of the points through which a line should be drawn.

        y : array_like
            The new y coordinates of the points through which a line should be drawn.

        Examples
        --------
        Create a line plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Update the line's y coordinates.

        >>> plot.update([0, 1, 2], [3, 1, 2])
        >>> chart.show()

        """
        if len(x) > 1:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False)})
            self.visible = True
        else:
            # Turn off visibility for fewer than 2 points as otherwise an error message is shown
            self.visible = False


class ScatterPlot2D(_vtk.vtkPlotPoints, _Plot):
    """Class representing a 2D scatter plot.

    Users should typically not directly create new plot instances, but use the dedicated 2D chart's plotting methods.

    Parameters
    ----------
    x : array_like
        X coordinates of the points to draw.

    y : array_like
        Y coordinates of the points to draw.

    color : color, optional
        Color of the points drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed. Defaults
        to ``"b"``.

    size : float, optional
        Size of the point markers drawn in this plot. Defaults to ``10``.

    style : str, optional
        Style of the point markers drawn in this plot. See :ref:`ScatterPlot2D.MARKER_STYLES <scatter_marker_styles>`
        for a list of allowed marker styles. Defaults to ``"o"``.

    label : str, optional
        Label of this plot, as shown in the chart's legend. Defaults to ``""``.

    Notes
    -----
    .. _scatter_marker_styles:

    MARKER_STYLES : dict
        Dictionary containing all allowed marker styles as its keys.

        .. include:: ../scatter_marker_styles.rst

    """

    MARKER_STYLES = {  # descr is used in the documentation, set to None to hide it from the docs.
        "": {"id": _vtk.vtkPlotPoints.NONE, "descr": "Hidden"},
        "x": {"id": _vtk.vtkPlotPoints.CROSS, "descr": "Cross"},
        "+": {"id": _vtk.vtkPlotPoints.PLUS, "descr": "Plus"},
        "s": {"id": _vtk.vtkPlotPoints.SQUARE, "descr": "Square"},
        "o": {"id": _vtk.vtkPlotPoints.CIRCLE, "descr": "Circle"},
        "d": {"id": _vtk.vtkPlotPoints.DIAMOND, "descr": "Diamond"}
    }

    def __init__(self, x, y, color="b", size=10, style="o", label=""):
        """Initialize a new 2D scatter plot instance."""
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y": np.empty(0, np.float32)})
        self.SetInputData(self._table, "x", "y")
        self.update(x, y)
        self.color = color
        self.marker_size = size
        self.marker_style = style
        self.label = label

    @property
    def x(self):
        """Retrieve the X coordinates of this plot's points.

        Examples
        --------
        Create a scatter plot and display the x coordinates.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.scatter([0, 1, 2], [2, 1, 3])
        >>> plot.x
        pyvista_ndarray([0, 1, 2])
        >>> chart.show()

        """
        return self._table["x"]

    @property
    def y(self):
        """Retrieve the Y coordinates of this plot's points.

        Examples
        --------
        Create a scatter plot and display the y coordinates.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.scatter([0, 1, 2], [2, 1, 3])
        >>> plot.y
        pyvista_ndarray([2, 1, 3])
        >>> chart.show()

        """
        return self._table["y"]

    def update(self, x, y):
        """Update this plot's points.

        Parameters
        ----------
        x : array_like
            The new x coordinates of the points to draw.

        y : array_like
            The new y coordinates of the points to draw.

        Examples
        --------
        Create a scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.scatter([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Update the marker locations.

        >>> plot.update([0, 1, 2], [3, 1, 2])
        >>> chart.show()

        """
        if len(x) > 0:
            self._table.update({"x": np.array(x, copy=False), "y": np.array(y, copy=False)})
            self.visible = True
        else:
            self.visible = False

    @property
    def marker_size(self):
        """Return or set the plot's marker size.

        Examples
        --------
        Create a 2D scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.scatter([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Increase the marker size.

        >>> plot.marker_size = 30
        >>> chart.show()

        """
        return self.GetMarkerSize()

    @marker_size.setter
    def marker_size(self, val):
        self.SetMarkerSize(val)

    @property
    def marker_style(self):
        """Return or set the plot's marker style.

        Examples
        --------
        Create a 2D scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.scatter([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Change the marker style.

        >>> plot.marker_style = "d"
        >>> chart.show()

        """
        return self._marker_style

    @marker_style.setter
    def marker_style(self, val):
        if val is None:
            val = ""
        try:
            self.SetMarkerStyle(self.MARKER_STYLES[val]["id"])
            self._marker_style = val
        except KeyError:
            formatted_styles = "\", \"".join(self.MARKER_STYLES.keys())
            raise ValueError(f"Invalid marker style. Allowed marker styles: \"{formatted_styles}\"")


class AreaPlot(_vtk.vtkPlotArea, _Plot):
    """Class representing a 2D area plot.

    Users should typically not directly create new plot instances, but use the dedicated 2D chart's plotting methods.

    Parameters
    ----------
    x : array_like
        X coordinates of the points outlining the area to draw.

    y1 : array_like
        Y coordinates of the points on the first outline of the area to draw.

    y2 : array_like, optional
        Y coordinates of the points on the second outline of the area to draw. Defaults to a sequence of zeros.

    color : color, optional
        Color of the area drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed. Defaults
        to ``"b"``.

    label : str, optional
        Label of this plot, as shown in the chart's legend. Defaults to ``""``.

    """

    def __init__(self, x, y1, y2=None, color="b", label=""):
        """Initialize a new 2D area plot instance."""
        super().__init__()
        self._table = pyvista.Table({"x": np.empty(0, np.float32), "y1": np.empty(0, np.float32), "y2": np.empty(0, np.float32)})
        self.SetInputData(self._table)
        self.SetInputArray(0, "x")
        self.SetInputArray(1, "y1")
        self.SetInputArray(2, "y2")
        self.update(x, y1, y2)
        self.color = color
        self.label = label

    @property
    def x(self):
        """Retrieve the X coordinates of the points outlining the drawn area.

        Examples
        --------
        Create an area plot and display the x coordinates.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [2, 1, 3], [0, 2, 0])
        >>> plot.x
        pyvista_ndarray([0, 1, 2])
        >>> chart.show()

        """
        return self._table["x"]

    @property
    def y1(self):
        """Retrieve the Y coordinates of the points on the first outline of the drawn area.

        Examples
        --------
        Create an area plot and display the y1 coordinates.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [2, 1, 3], [0, 2, 0])
        >>> plot.y1
        pyvista_ndarray([2, 1, 3])
        >>> chart.show()

        """
        return self._table["y1"]

    @property
    def y2(self):
        """Retrieve the Y coordinates of the points on the second outline of the drawn area.

        Examples
        --------
        Create an area plot and display the y2 coordinates.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [2, 1, 3], [0, 2, 0])
        >>> plot.y2
        pyvista_ndarray([0, 2, 0])
        >>> chart.show()

        """
        return self._table["y2"]

    def update(self, x, y1, y2=None):
        """Update this plot's points, outlining the area to draw.

        Parameters
        ----------
        x : array_like
            The new x coordinates of the points outlining the area.

        y1 : array_like
            The new y coordinates of the points on the first outline of the area.

        y2 : array_like, optional
            The new y coordinates of the points on the second outline of the area. Defaults to a sequence of zeros.

        Examples
        --------
        Create an area plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Update the points on the second outline of the area.

        >>> plot.update([0, 1, 2], [2, 1, 3], [1, 2, 0])
        >>> chart.show()

        """
        if len(x) > 0:
            if y2 is None:
                y2 = np.zeros_like(x)
            self._table.update({"x": np.array(x, copy=False), "y1": np.array(y1, copy=False), "y2": np.array(y2, copy=False)})
            self.visible = True
        else:
            self.visible = False


class BarPlot(_vtk.vtkPlotBar, _MultiCompPlot):
    """Class representing a 2D bar plot.

    Users should typically not directly create new plot instances, but use the dedicated 2D chart's plotting methods.

    Parameters
    ----------
    x : array_like
        Positions (along the x-axis for a vertical orientation, along the y-axis for
        a horizontal orientation) of the bars to draw.

    y : array_like
        Size of the bars to draw. Multiple bars can be stacked by passing a sequence of sequences.

    color : color, optional
        Color of the bars drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed. Defaults
        to ``"b"``.

    offset : float, optional
        Offset between the bars drawn in this plot. Defaults to ``5``.

    orientation : str, optional
        Orientation of the bars drawn in this plot. Either ``"H"`` for an horizontal orientation or ``"V"`` for a
        vertical orientation. Defaults to ``"V"``.

    label : str, optional
        Label of this plot, as shown in the chart's legend. Defaults to ``""``.

    """

    ORIENTATIONS = {
        "H": _vtk.vtkPlotBar.HORIZONTAL,
        "V": _vtk.vtkPlotBar.VERTICAL
    }

    def __init__(self, x, y, color=None, offset=5, orientation="V", label=None):
        """Initialize a new 2D bar plot instance."""
        super().__init__()
        if not isinstance(y[0], (Sequence, np.ndarray)):
            y = (y,)
        y_data = {f"y{i}": np.empty(0, np.float32) for i in range(len(y))}
        self._table = pyvista.Table({"x": np.empty(0, np.float32), **y_data})
        self.SetInputData(self._table, "x", "y0")
        for i in range(1, len(y)):
            self.SetInputArray(i+1, f"y{i}")
        self.update(x, y)

        if len(y) > 1:
            self.SetColorSeries(self._color_series)
            self.colors = color  # None will use default scheme
            self.labels = label
        else:
            self.color = "b" if color is None else color  # Use blue bars by default in single component mode
            self.label = label
        self.offset = offset
        self.orientation = orientation

    @property
    def x(self):
        """Retrieve the positions of the drawn bars.

        Examples
        --------
        Create a bar plot and display the positions.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([0, 1, 2], [[2, 1, 3], [1, 2, 0]])
        >>> plot.x
        pyvista_ndarray([0, 1, 2])
        >>> chart.show()

        """
        return self._table["x"]

    @property
    def y(self):
        """Retrieve the sizes of the drawn bars.

        Examples
        --------
        Create a bar plot and display the sizes.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([0, 1, 2], [[2, 1, 3], [1, 2, 0]])
        >>> plot.y
        (pyvista_ndarray([2, 1, 3]), pyvista_ndarray([1, 2, 0]))
        >>> chart.show()

        """
        return tuple(self._table[f"y{i}"] for i in range(self._table.n_arrays - 1))

    def update(self, x, y):
        """Update the positions and/or size of the bars in this plot.

        Parameters
        ----------
        x : array_like
            The new positions of the bars to draw.

        y : array_like
            The new sizes of the bars to draw.

        Examples
        --------
        Create a bar plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Update the bar sizes.

        >>> plot.update([0, 1, 2], [3, 1, 2])
        >>> chart.show()

        """
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
        """Return or set the offset between bars in this plot.

        Examples
        --------
        Create a bar plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Increase the offset.

        >>> plot.offset = 15
        >>> chart.show()

        """
        return self.GetOffset()

    @offset.setter
    def offset(self, val):
        self.SetOffset(val)

    @property
    def orientation(self):
        """Return or set the orientation of the bars in this plot.

        Examples
        --------
        Create a bar plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([0, 1, 2], [[2, 1, 3], [1, 3, 2]])
        >>> chart.show()

        Change the orientation to horizontal.

        >>> plot.orientation = "H"
        >>> chart.show()

        """
        return self._orientation

    @orientation.setter
    def orientation(self, val):
        if val in self.ORIENTATIONS:
            self._orientation = val
            self.SetOrientation(self.ORIENTATIONS[val])
        else:
            formatted_orientations = "\", \"".join(self.ORIENTATIONS.keys())
            raise ValueError(f"Invalid orientation. Allowed orientations: \"{formatted_orientations}\"")


class StackPlot(_vtk.vtkPlotStacked, _MultiCompPlot):
    """Class representing a 2D stack plot.

    Users should typically not directly create new plot instances, but use the dedicated 2D chart's plotting methods.

    Parameters
    ----------
    x : array_like
        X coordinates of the points outlining the stacks (areas) to draw.

    ys : list or tuple of array_like
        Size of the stacks (areas) to draw at the corresponding X coordinates. Each sequence defines the sizes of
        one stack (area), which are stacked on top of each other.

    colors : list or tuple of color, optional
        Color of the stacks (areas) drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed.
        Defaults to ``None``.

    labels : list or tuple of str, optional
        Label for each stack (area) drawn in this plot, as shown in the chart's legend. Defaults to ``[]``.

    """

    def __init__(self, x, ys, colors=None, labels=None):
        """Initialize a new 2D stack plot instance."""
        super().__init__()
        if not isinstance(ys[0], (Sequence, np.ndarray)):
            ys = (ys,)
        y_data = {f"y{i}": np.empty(0, np.float32) for i in range(len(ys))}
        self._table = pyvista.Table({"x": np.empty(0, np.float32), **y_data})
        self.SetInputData(self._table, "x", "y0")
        for i in range(1, len(ys)):
            self.SetInputArray(i+1, f"y{i}")
        self.update(x, ys)

        if len(ys) > 1:
            self.SetColorSeries(self._color_series)
            self.colors = colors  # None will use default scheme
            self.labels = labels
        else:
            self.color = "b" if colors is None else colors
            self.label = labels
        self.pen.style = None  # Hide lines by default

    @property
    def x(self):
        """Retrieve the X coordinates of the drawn stacks.

        Examples
        --------
        Create a stack plot and display the x coordinates.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.stack([0, 1, 2], [[2, 1, 3], [1, 2, 0]])
        >>> plot.x
        pyvista_ndarray([0, 1, 2])
        >>> chart.show()

        """
        return self._table["x"]

    @property
    def ys(self):
        """Retrieve the sizes of the drawn stacks.

        Examples
        --------
        Create a stack plot and display the sizes.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.stack([0, 1, 2], [[2, 1, 3], [1, 2, 0]])
        >>> plot.ys
        (pyvista_ndarray([2, 1, 3]), pyvista_ndarray([1, 2, 0]))
        >>> chart.show()

        """
        return tuple(self._table[f"y{i}"] for i in range(self._table.n_arrays - 1))

    def update(self, x, ys):
        """Update the locations and/or size of the stacks (areas) in this plot.

        Parameters
        ----------
        x : array_like
            The new x coordinates of the stacks (areas) to draw.

        ys : list or tuple of array_like
            The new sizes of the stacks (areas) to draw.

        Examples
        --------
        Create a stack plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.stack([0, 1, 2], [[2, 1, 3],[1, 2, 1]])
        >>> chart.show()

        Update the stack sizes.

        >>> plot.update([0, 1, 2], [[3, 1, 2], [0, 3, 1]])
        >>> chart.show()

        """
        if len(x) > 0:
            if not isinstance(ys[0], (Sequence, np.ndarray)):
                ys = (ys,)
            y_data = {f"y{i}": np.array(ys[i], copy=False) for i in range(len(ys))}
            self._table.update({"x": np.array(x, copy=False), **y_data})
            self.visible = True
        else:
            self.visible = False


class Chart2D(_vtk.vtkChartXY, _Chart):
    """2D chart class similar to a ``matplotlib`` figure.

    Parameters
    ----------
    size : list or tuple, optional
        Size of the chart in normalized coordinates. A size of ``(0, 0)`` is invisible, a size of ``(1, 1)`` occupies
        the whole renderer's width and height.

    loc : list or tuple, optional
        Location of the chart (its bottom left corner) in normalized coordinates. A location of ``(0, 0)`` corresponds
        to the renderer's bottom left corner, a location of ``(1, 1)`` corresponds to the renderer's top right corner.

    x_label : str, optional
        Label along the x-axis.  Defaults to ``'x'``.

    y_label : str, optional
        Label along the y-axis.  Defaults to ``'y'``.

    grid : bool, optional
        Show the background grid in the plot.  Default ``True``.

    Examples
    --------
    Plot a simple sine wave as a scatter and line plot.

    >>> import pyvista
    >>> import numpy as np
    >>> x = np.linspace(0, 2*np.pi, 20)
    >>> y = np.sin(x)
    >>> chart = pyvista.Chart2D()
    >>> _ = chart.scatter(x, y)
    >>> _ = chart.line(x, y, 'r')
    >>> chart.show()

    Create a bar plot.

    >>> rng = np.random.default_rng(1)
    >>> x = np.arange(1, 13)
    >>> y1 = rng.integers(1e2, 1e4, 12)
    >>> y2 = rng.integers(1e2, 1e4, 12)
    >>> chart = pyvista.Chart2D()
    >>> chart.background_color = 'w'
    >>> _ = chart.bar(x, y1, color="b", label="2020")
    >>> _ = chart.bar(x, y2, color="r", label="2021")
    >>> chart.x_axis.tick_locations = x
    >>> chart.x_axis.tick_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
    ...                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    >>> chart.x_label = "Month"
    >>> chart.y_axis.tick_labels = "2e"
    >>> chart.y_label = "# incidents"
    >>> chart.show()

    """

    PLOT_TYPES = {
        "scatter": ScatterPlot2D,
        "line": LinePlot2D,
        "area": AreaPlot,
        "bar": BarPlot,
        "stack": StackPlot
    }
    _PLOT_CLASSES = {plot_class: plot_type for (plot_type, plot_class) in PLOT_TYPES.items()}

    def __init__(self, size=(1, 1), loc=(0, 0), x_label="x", y_label="y", grid=True):
        """Initialize the chart."""
        super().__init__(size, loc)
        self._plots = {plot_type: [] for plot_type in self.PLOT_TYPES.keys()}
        self.SetAutoSize(False)  # We manually set the appropriate size
        # self.SetAxis(_vtk.vtkAxis.BOTTOM, self._x_axis)  # Disabled for now and replaced by a wrapper object, as for
        # self.SetAxis(_vtk.vtkAxis.LEFT, self._y_axis)  # some reason vtkChartXY.SetAxis(...) causes a crash at the end
        self._x_axis = Axis(_wrap=self.GetAxis(_vtk.vtkAxis.BOTTOM))  # of the script's execution (nonzero exit code)
        self._y_axis = Axis(_wrap=self.GetAxis(_vtk.vtkAxis.LEFT))
        self.x_label = x_label
        self.y_label = y_label
        self.grid = grid
        self.legend_visible = True

    def _render_event(self, *args, **kwargs):
        self.RecalculateBounds()
        super()._render_event(*args, **kwargs)

    def _add_plot(self, plot_type, *args, **kwargs):
        plot = self.PLOT_TYPES[plot_type](*args, **kwargs)
        self.AddPlot(plot)
        self._plots[plot_type].append(plot)
        return plot

    @classmethod
    def parse_format(cls, fmt):
        """Parse a format string and separate it into a marker style, line style and color.

        Parameters
        ----------
        fmt : str
            Format string to parse. A format string consists of any combination of a valid marker
            style, a valid line style and parsable color. The specific order does not matter. See
            :attr:`pyvista.ScatterPlot2D.MARKER_STYLES` for a list of valid marker styles,
            :attr:`pyvista.Pen.LINE_STYLES` for a list of valid line styles and
            :func:`pyvista.parse_color` for an overview of parsable colors.

        Returns
        -------
        marker_style : str
            Extracted marker style (empty string if no marker style was present in the format string).

        line_style : str
            Extracted line style (empty string if no line style was present in the format string).

        color : str
            Extracted color string (defaults to ``"b"`` if no color was present in the format string).

        Examples
        --------
        >>> import pyvista
        >>> m, l, c = pyvista.Chart2D.parse_format("x--b")

        """
        # TODO: add tests for different combinations/positions of marker, line and color
        marker_style = ""
        line_style = ""
        color = None
        last_fmt = None
        while fmt != "" and last_fmt != fmt and (marker_style == "" or line_style == "" or color is None):
            last_fmt = fmt
            if marker_style == "":
                # Marker style is not yet set, so look for it at the start of the format string
                for style in ScatterPlot2D.MARKER_STYLES.keys():
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

    def plot(self, x, y, fmt='-'):
        """Matplotlib like plot method.

        Parameters
        ----------
        x : array_like
            Values to plot on the X-axis.

        y : array_like
            Values to plot on the Y-axis.

        fmt : str, optional
            A format string, e.g. 'ro' for red circles. See the Notes
            section for a full description of the format strings.

        Returns
        -------
        scatter_plot : pyvista.ScatterPlot2D, optional
            The created scatter plot when a valid marker style
            was present in the format string, ``None`` otherwise.

        line_plot : pyvista.LinePlot2D, optional
            The created line plot when a valid line style was
            present in the format string, ``None`` otherwise.

        Notes
        -----
        This plot method shares many of the same plotting features as
        the `matplotlib.pyplot.plot
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_.
        Please reference the documentation there for a full
        description of the allowable format strings.

        Examples
        --------
        Generate a line plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _, line_plot = chart.plot(range(10), range(10))

        Generate a line and scatter plot.

        >>> scatter_plot, line_plot = chart.plot(range(10), range(10), fmt='o-')

        """
        # TODO: make x and fmt optional, allow multiple ([x], y, [fmt]) entries
        marker_style, line_style, color = self.parse_format(fmt)
        scatter_plot, line_plot = None, None
        if marker_style != "":
            scatter_plot = self.scatter(x, y, color, style=marker_style)
        if line_style != "":
            line_plot = self.line(x, y, color, style=line_style)
        return scatter_plot, line_plot

    def scatter(self, x, y, color="b", size=10, style="o", label=""):
        """Add a scatter plot to this chart.

        Parameters
        ----------
        x : array_like
            X coordinates of the points to draw.

        y : array_like
            Y coordinates of the points to draw.

        color : color, optional
            Color of the points drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed. Defaults
            to ``"b"``.

        size : float, optional
            Size of the point markers drawn in this plot. Defaults to ``10``.

        style : str, optional
            Style of the point markers drawn in this plot. See ``pyvista.ScatterPlot2D.MARKER_STYLES`` for a list of
            allowed marker styles. Defaults to ``"o"``.

        label : str, optional
            Label of this plot, as shown in the chart's legend. Defaults to ``""``.

        Returns
        -------
        scatter_plot : pyvista.ScatterPlot2D
            The created scatter plot.

        Examples
        --------
        Generate a scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.scatter([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        """
        return self._add_plot("scatter", x, y, color=color, size=size, style=style, label=label)

    def line(self, x, y, color="b", width=1.0, style="-", label=""):
        """Add a line plot to this chart.

        Parameters
        ----------
        x : array_like
            X coordinates of the points through which a line should be drawn.

        y : array_like
            Y coordinates of the points through which a line should be drawn.

        color : color, optional
            Color of the line drawn in this plot. Any color parsable by :func:`pyvista.parse_color` is allowed. Defaults
            to ``"b"``.

        width : float, optional
            Width of the line drawn in this plot. Defaults to ``1``.

        style : str, optional
            Style of the line drawn in this plot. See :attr:`Pen.LINE_STYLES` for a list of allowed line styles.
            Defaults to ``"-"``.

        label : str, optional
            Label of this plot, as shown in the chart's legend. Defaults to ``""``.

        Returns
        -------
        line_plot : :class:`LinePlot2D <pyvista.plotting.charts.LinePlot2D>`
            The created line plot.

        Examples
        --------
        Generate a line plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        """
        return self._add_plot("line", x, y, color=color, width=width, style=style, label=label)

    def area(self, x, y1, y2=None, color="b", label=""):
        """Add an area plot to this chart.

        Parameters
        ----------
        x : array_like
            X coordinates of the points outlining the area to draw.

        y1 : array_like
            Y coordinates of the points on the first outline of the area to draw.

        y2 : array_like, optional
            Y coordinates of the points on the second outline of the area to draw. Defaults to a sequence of zeros.

        color : color, optional
            Color of the area drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed. Defaults
            to ``"b"``.

        label : str, optional
            Label of this plot, as shown in the chart's legend. Defaults to ``""``.

        Returns
        -------
        area_plot : pyvista.AreaPlot
            The created area plot.

        Examples
        --------
        Generate an area plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        """
        return self._add_plot("area", x, y1, y2, color=color, label=label)

    def bar(self, x, y, color=None, offset=5, orientation="V", label=None):
        """Add a bar plot to this chart.

        Parameters
        ----------
        x : array_like
            Positions (along the x-axis for a vertical orientation, along the y-axis for
            a horizontal orientation) of the bars to draw.

        y : array_like
            Size of the bars to draw. Multiple bars can be stacked by passing a sequence of sequences.

        color : color, optional
            Color of the bars drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed. Defaults
            to ``"b"``.

        offset : float, optional
            Offset between the bars drawn in this plot. Defaults to ``5``.

        orientation : str, optional
            Orientation of the bars drawn in this plot. Either ``"H"`` for an horizontal orientation or ``"V"`` for a
            vertical orientation. Defaults to ``"V"``.

        label : str, optional
            Label of this plot, as shown in the chart's legend. Defaults to ``""``.

        Returns
        -------
        bar_plot : pyvista.BarPlot
            The created bar plot.

        Examples
        --------
        Generate a bar plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        """
        return self._add_plot("bar", x, y, color=color, offset=offset, orientation=orientation, label=label)

    def stack(self, x, ys, colors=None, labels=None):
        """Add a stack plot to this chart.

        Parameters
        ----------
        x : array_like
            X coordinates of the points outlining the stacks (areas) to draw.

        ys : list or tuple of array_like
            Size of the stacks (areas) to draw at the corresponding X coordinates. Each sequence defines the sizes of
            one stack (area), which are stacked on top of each other.

        colors : list or tuple of color, optional
            Color of the stacks (areas) drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed.
            Defaults to ``None``.

        labels : list or tuple of str, optional
            Label for each stack (area) drawn in this plot, as shown in the chart's legend. Defaults to ``[]``.

        Returns
        -------
        stack_plot : pyvista.StackPlot
            The created stack plot.

        Examples
        --------
        Generate a stack plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.stack([0, 1, 2], [[2, 1, 3],[1, 2, 1]])
        >>> chart.show()

        """
        return self._add_plot("stack", x, ys, colors=colors, labels=labels)

    def plots(self, plot_type=None):
        """Return all plots of the specified type in this chart.

        Parameters
        ----------
        plot_type : str, optional
            The type of plots to return. Allowed types are
            ``"scatter"``, ``"line"``, ``"area"``, ``"bar"``
            and ``"stack"``.
            Defaults to ``None``, which will return all plots,
            regardless of their type.

        Yields
        ------
        plot : ScatterPlot2D, LinePlot2D, AreaPlot, BarPlot or StackPlot
            One of the plots (of the specified type) in this chart.

        Examples
        --------
        Create a 2D chart with a line and scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> scatter_plot, line_plot = chart.plot([0, 1, 2], [2, 1, 3], "o-")

        Retrieve all plots in the chart.

        >>> plots = [*chart.plots()]
        >>> scatter_plot in plots and line_plot in plots
        True

        Retrieve all line plots in the chart.

        >>> line_plots = [*chart.plots("line")]
        >>> line_plot == line_plots[0]
        True

        """
        if plot_type is None:
            for plots in self._plots.values():
                for plot in plots:
                    yield plot
        else:
            for plot in self._plots[plot_type]:
                yield plot

    def remove_plot(self, plot):
        """Remove the given plot from this chart.

        Parameters
        ----------
        plot : ScatterPlot2D, LinePlot2D, AreaPlot, BarPlot or StackPlot
            The plot to remove.

        Examples
        --------
        Create a 2D chart with a line and scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> scatter_plot, line_plot = chart.plot([0, 1, 2], [2, 1, 3], "o-")
        >>> chart.show()

        Remove the scatter plot from the chart.

        >>> chart.remove_plot(scatter_plot)
        >>> chart.show()

        """
        try:
            plot_type = self._PLOT_CLASSES[type(plot)]
            self._plots[plot_type].remove(plot)
            self.RemovePlotInstance(plot)
        except (KeyError, ValueError):
            raise ValueError("The given plot is not part of this chart.")

    def clear(self, plot_type=None):
        """Remove all plots of the specified type from this chart.

        Parameters
        ----------
        plot_type : str, optional
            The type of the plots to remove. Allowed types are
            ``"scatter"``, ``"line"``, ``"area"``, ``"bar"``
            and ``"stack"``.
            Defaults to ``None``, which will remove all plots,
            regardless of their type.

        Examples
        --------
        Create a 2D chart with multiple line and scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.plot([0, 1, 2], [2, 1, 3], "o-b")
        >>> _ = chart.plot([-2, -1, 0], [3, 1, 2], "d-r")
        >>> chart.show()

        Remove all scatter plots from the chart.

        >>> chart.clear("scatter")
        >>> chart.show()

        """
        if plot_type is None:
            for plots in self._plots.values():
                for plot in plots:
                    self.remove_plot(plot)
        else:
            for plot in self._plots[plot_type]:
                self.remove_plot(plot)

    @property
    def x_axis(self):
        """Return this chart's horizontal (x) axis.

        Examples
        --------
        Create a 2D plot and hide the x axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.toggle()
        >>> chart.show()

        """
        return self._x_axis

    @property
    def y_axis(self):
        """Return this chart's vertical (y) axis.

        Examples
        --------
        Create a 2D plot and hide the y axis.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.y_axis.toggle()
        >>> chart.show()

        """
        return self._y_axis

    @property
    def x_label(self):
        """Return or set the label of this chart's x axis.

        Examples
        --------
        Create a 2D plot and set custom axis labels.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_label = "Horizontal axis"
        >>> chart.y_label = "Vertical axis"
        >>> chart.show()

        """
        return self.x_axis.label

    @x_label.setter
    def x_label(self, val):
        self.x_axis.label = val

    @property
    def y_label(self):
        """Return or set the label of this chart's y axis.

        Examples
        --------
        Create a 2D plot and set custom axis labels.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_label = "Horizontal axis"
        >>> chart.y_label = "Vertical axis"
        >>> chart.show()

        """
        return self.y_axis.label

    @y_label.setter
    def y_label(self, val):
        self.y_axis.label = val

    @property
    def x_range(self):
        """Return or set the range of this chart's x axis.

        Examples
        --------
        Create a 2D plot and set custom axis ranges.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_range = [-2, 2]
        >>> chart.y_range = [0, 5]
        >>> chart.show()

        """
        return self.x_axis.range

    @x_range.setter
    def x_range(self, val):
        self.x_axis.range = val

    @property
    def y_range(self):
        """Return or set the range of this chart's y axis.

        Examples
        --------
        Create a 2D plot and set custom axis ranges.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_range = [-2, 2]
        >>> chart.y_range = [0, 5]
        >>> chart.show()

        """
        return self.y_axis.range

    @y_range.setter
    def y_range(self, val):
        self.y_axis.range = val

    @property
    def grid(self):
        """Enable or disable the chart grid.

        Examples
        --------
        Disable the grid.

        >>> import pyvista
        >>> import numpy as np
        >>> x = np.linspace(0, 2*np.pi, 20)
        >>> y = np.sin(x)
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line(x, y, 'r')
        >>> chart.grid = False
        >>> chart.show()

        Enable the grid

        >>> import pyvista
        >>> import numpy as np
        >>> x = np.linspace(0, 2*np.pi, 20)
        >>> y = np.sin(x)
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line(x, y, 'r')
        >>> chart.grid = True
        >>> chart.show()

        """
        return self.x_axis.grid and self.y_axis.grid

    @grid.setter
    def grid(self, val):
        self.x_axis.grid = val
        self.y_axis.grid = val

    def hide_axes(self):
        """Hide the x- and y-axis of this chart.

        This includes all labels, ticks and the grid.

        Examples
        --------
        Create a 2D plot and hide the axes.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.hide_axes()
        >>> chart.show()

        """
        for axis in (self.x_axis, self.y_axis):
            axis.visible = False
            axis.label_visible = False
            axis.ticks_visible = False
            axis.tick_labels_visible = False
            axis.grid = False


class BoxPlot(_vtk.vtkPlotBox, _MultiCompPlot):
    """Class representing a box plot.

    Users should typically not directly create new plot instances, but use the dedicated ``ChartBox`` class.

    Parameters
    ----------
    data : list or tuple of array_like
        Dataset(s) from which the relevant statistics will be calculated used to draw the box plot.

    colors : list or tuple of color, optional
        Color of the boxes drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed.
        Defaults to ``None``.

    """

    def __init__(self, data, colors=None):
        """Initialize a new box plot instance."""
        super().__init__()
        self._table = pyvista.Table(data)
        self._quartiles = _vtk.vtkComputeQuartiles()
        self._quartiles.SetInputData(self._table)
        self.SetInputData(self._quartiles.GetOutput())
        self.update(data)
        self.SetLookupTable(self._lookup_table)
        self.colors = colors

    # TODO: property to retrieve original datasets? And a property to retrieve calculated quartile statistics?

    def update(self, data):
        """Update the plot's underlying dataset(s).

        Parameters
        ----------
        data : list or tuple of array_like
            The new dataset(s) used in this box plot.

        Examples
        --------
        Create a box plot from a standard Gaussian dataset.

        >>> import pyvista
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)  # Seeded random number generator for data generation
        >>> chart = pyvista.ChartBox([rng.normal(size=100)])
        >>> chart.show()

        Update the box plot (shift the standard Gaussian distribution).

        >>> chart.plot.update([rng.normal(loc=2, size=100)])
        >>> chart.show()

        """
        self._table.update(data)
        self._quartiles.Update()


class ChartBox(_vtk.vtkChartBox, _Chart):
    """Dedicated chart for drawing box plots.

    Parameters
    ----------
    data : list or tuple of array_like
        Dataset(s) from which the relevant statistics will be calculated used to draw the box plot.

    """

    def __init__(self, data):
        """Initialize a new chart containing box plots."""
        super().__init__(None, None)
        self._plot = BoxPlot(data)
        self.SetPlot(self._plot)
        self.SetColumnVisibilityAll(True)
        self.legend_visible = True

    def _render_event(self, *args, **kwargs):
        pass  # ChartBox fills entire scene by default, so no resizing is needed (nor possible at this moment)

    @property
    def _geometry(self):
        # Needed for background (remove once resizing is possible)
        return (0, 0, *self._renderer.GetSize())

    @_geometry.setter
    def _geometry(self, value):
        raise AttributeError(f'Cannot set the geometry of {type(self).__class__}')

    @property
    def plot(self):
        """Return the BoxPlot instance associated with this chart.

        Examples
        --------
        Create a box plot from a standard Gaussian dataset.

        >>> import pyvista
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)  # Seeded random number generator for data generation
        >>> chart = pyvista.ChartBox([rng.normal(size=100)])
        >>> chart.show()

        Update the box plot (shift the standard Gaussian distribution).

        >>> chart.plot.update([rng.normal(loc=2, size=100)])
        >>> chart.show()

        """
        return self._plot

    @property
    def size(self):
        """Return or set the chart size in normalized coordinates.

        A size of ``(1, 1)`` occupies the whole renderer.
        Note: The size of a ChartBox instance cannot be modified, it fills up the entire viewport by default.
        """
        return (1, 1)

    @size.setter
    def size(self, val):
        raise ValueError("Cannot set ChartBox geometry, it fills up the entire viewport by default.")

    @property
    def loc(self):
        """Return or set the chart position in normalized coordinates.

        This denotes the location of the chart's bottom left corner.
        Note: The location of a ChartBox instance cannot be modified, it fills up the entire viewport by default.
        """
        return (0, 0)

    @loc.setter
    def loc(self, val):
        raise ValueError("Cannot set ChartBox geometry, it fills up the entire viewport by default.")


class PiePlot(_vtkWrapper, _vtk.vtkPlotPie, _MultiCompPlot):
    """Class representing a pie plot.

    Users should typically not directly create new plot instances, but use the dedicated ``ChartPie`` class.

    Parameters
    ----------
    data : array_like
        Relative size of each pie segment.

    labels : list or tuple of str, optional
        Label for each pie segment drawn in this plot, as shown in the chart's legend. Defaults to ``[]``.

    colors : list or tuple of color, optional
        Color of the segments drawn in this plot. Any color parsable by ``pyvista.parse_color`` is allowed.
        Defaults to ``None``.

    """

    def __init__(self, data, labels=None, colors=None):
        """Initialize a new pie plot instance."""
        super().__init__()
        self._table = pyvista.Table(data)
        self.SetInputData(self._table)
        self.SetInputArray(0, self._table.keys()[0])
        self.update(data)

        self.labels = labels

        self.SetColorSeries(self._color_series)
        self.colors = colors

    @property
    def data(self):
        """Retrieve the sizes of the drawn segments.

        Examples
        --------
        Create a pie plot and display the sizes.

        >>> import pyvista
        >>> chart = pyvista.ChartPie([1, 2, 3])
        >>> chart.plot.data
        pyvista_ndarray([1, 2, 3])
        >>> chart.show()

        """
        return self._table[0]

    def update(self, data):
        """Update the size of the pie segments.

        Parameters
        ----------
        data : array_like
            The new relative size of each pie segment.

        Examples
        --------
        Create a pie plot with segments of increasing size.

        >>> import pyvista
        >>> chart = pyvista.ChartPie([1, 2, 3, 4, 5])
        >>> chart.show()

        Update the pie plot (segments of equal size).

        >>> chart.plot.update([1, 1, 1, 1, 1])
        >>> chart.show()

        """
        self._table.update(data)


class ChartPie(_vtk.vtkChartPie, _Chart):
    """Dedicated chart for drawing pie plots.

    Parameters
    ----------
    data : array_like
        Relative size of each pie segment.

    labels : list or tuple of str, optional
        Label for each pie segment drawn in this plot, as shown in the chart's legend. Defaults to ``[]``.

    """

    def __init__(self, data, labels=None):
        """Initialize a new chart containing a pie plot."""
        super().__init__(None, None)
        self.AddPlot(0)  # We can't manually set a wrapped vtkPlotPie instance...
        self._plot = PiePlot(data, labels, _wrap=self.GetPlot(0))  # So we have to wrap the existing one
        self.legend_visible = True

    def _render_event(self, *args, **kwargs):
        pass  # ChartPie fills entire scene by default, so no resizing is needed (nor possible at this moment)

    @property
    def _geometry(self):
        # Needed for background (remove once resizing is possible)
        return (0, 0, *self._renderer.GetSize())

    @_geometry.setter
    def _geometry(self, value):
        raise AttributeError(f'Cannot set the geometry of {type(self).__class__}')

    @property
    def plot(self):
        """Return the BoxPlot instance associated with this chart.

        Examples
        --------
        Create a pie plot with segments of increasing size.

        >>> import pyvista
        >>> chart = pyvista.ChartPie([1, 2, 3, 4, 5])
        >>> chart.show()

        Update the pie plot (segments of equal size).

        >>> chart.plot.update([1, 1, 1, 1, 1])
        >>> chart.show()

        """
        return self._plot

    @property
    def size(self):
        """Return or set the chart size in normalized coordinates.

        A size of ``(1, 1)`` occupies the whole renderer.
        Note: The size of a ChartPie instance cannot be modified, it fills up the entire viewport by default.
        """
        return (1, 1)

    @size.setter
    def size(self, val):
        raise ValueError("Cannot set ChartPie geometry, it fills up the entire viewport by default.")

    @property
    def loc(self):
        """Return or set the chart position in normalized coordinates.

        This denotes the location of the chart's bottom left corner.
        Note: The location of a ChartPie instance cannot be modified, it fills up the entire viewport by default.
        """
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
        self.line_width = width
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
        super().__init__(size, loc)
        self._plots = {plot_type: [] for plot_type in self.PLOT_TYPES.keys()}
        self.background_color = None  # Transparent background and no border, as the 3D plots do not respect the
        self.border_style = None  # geometry bounds
        self.SetFitToScene(True)  # Some more room for axis labels
        self._geom = (0, 0, 0, 0)  # Will be properly set on first render_event
        self._x_axis = Axis(_wrap=self.GetAxis(0))
        self._y_axis = Axis(_wrap=self.GetAxis(1))
        self._z_axis = Axis(_wrap=self.GetAxis(2))
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label

    def _render_event(self, *args, **kwargs):
        self.RecalculateBounds()
        super()._render_event(*args, **kwargs)

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
        """Matplotlib like plot method.

        Examples
        --------
        ###############################################################################
        # Some limited functionality for 3D charts also exists. The interface is the
        # same as for the 2D counterparts. (To be added to ``chart_basics`` example
        # once supported.)

        # x = np.arange(11)
        # y = rng.integers(-5, 6, 11)
        # z = rng.integers(-5, 6, 11)
        # p = pv.Plotter()
        # p.background_color = (1, 1, 1)
        # chart = pv.Chart3D()
        # chart.plot(x, y, z, 'x-b')  # Show markers (marker style is ignored), solid line and blue color 'b'
        # p.add_chart(chart)
        # p.show()

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
        """See example below.

        Examples
        --------
        ###############################################################################
        # 3D surfaces can be visualized on such a chart as well. (To be added to
        # ``chart_basics`` example once supported.)

        # x = np.linspace(-1, 1, 100)
        # y = np.linspace(-1, 1, 100)
        # xx, yy = np.meshgrid(x, y)
        # z = np.cos(6*(xx**2+yy**2))
        # p = pv.Plotter()
        # p.background_color = (1, 1, 1)
        # chart = pv.Chart3D()
        # chart.surface(x, y, z)
        # p.add_chart(chart)
        # p.show()

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
    """Create new chart from an existing matplotlib figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The matplotlib figure to draw.

    size : list or tuple, optional
        Size of the chart in normalized coordinates. A size of ``(0, 0)`` is invisible, a size of ``(1, 1)`` occupies
        the whole renderer's width and height.

    loc : list or tuple, optional
        Location of the chart (its bottom left corner) in normalized coordinates. A location of ``(0, 0)`` corresponds
        to the renderer's bottom left corner, a location of ``(1, 1)`` corresponds to the renderer's top right corner.

    """

    def __init__(self, figure, size=(1, 1), loc=(0, 0)):
        """Initialize chart."""
        try:
            import matplotlib.figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ModuleNotFoundError:
            raise ImportError("ChartMPL requires matplotlib")

        super().__init__(size, loc)
        self._fig = figure
        self._canvas = FigureCanvasAgg(self._fig)  # Switch backends and store reference to figure's canvas
        # Make figure and axes fully transparent, as the background is already dealt with by self._background.
        self._fig.patch.set_alpha(0)
        for ax in self._fig.axes:
            ax.patch.set_alpha(0)
        self._canvas.mpl_connect('draw_event', self._redraw)  # Attach 'draw_event' callback

        self.redraw()

    def _resize(self):
        r_w, r_h = self._renderer.GetSize()
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

    def redraw(self):
        """Redraw the chart after manually updating the matplotlib figure."""
        # TODO: not sure whether this is still needed, as calling plotter.update() seems to do the trick already.
        # Manual call, so make sure canvas is redrawn first (which will callback to _redraw with a proper event defined)
        self._canvas.draw()

    def _redraw(self, event):
        # Called from draw_event callback
        img = np.frombuffer(self._canvas.buffer_rgba(), dtype=np.uint8)  # Store figure data in numpy array
        w, h = self._canvas.get_width_height()
        img_arr = img.reshape([h, w, 4])
        img_data = pyvista.Texture(img_arr).to_image()  # Convert to vtkImageData
        self.SetImage(img_data)

    def _render_event(self, *args, **kwargs):
        self._resize()  # Update figure dimensions if needed
        self.redraw()  # Redraw figure

    @property
    def _geometry(self):
        r_w, r_h = self._renderer.GetSize()
        t_w = self._size[0]*r_w
        t_h = self._size[1]*r_h
        return (*self.position, t_w, t_h)

    @_geometry.setter
    def _geometry(self, value):
        raise AttributeError(f'Cannot set the geometry of {type(self).__class__}')

    # Below code can be used to customize the chart's background without a _ChartBackground instance
    # @property
    # def background_color(self):
    #     return self._bg_color
    #
    # @background_color.setter
    # def background_color(self, val):
    #     color = parse_color(val) if val is not None else [1, 1, 1, 1]
    #     opacity = color[3] if len(color) == 4 else 1
    #     self._bg_color = color
    #     self._fig.patch.set_color(color[:3])
    #     self._fig.patch.set_alpha(opacity)
    #     for ax in self._fig.axes:
    #         ax.patch.set_alpha(0 if opacity < 1 else 1)  # Make axes fully transparent if opacity is lower than 1

    @property
    def position(self):
        """Chart position w.r.t the bottom left corner (in pixels)."""
        return self.GetPosition()

    @position.setter
    def position(self, val):
        assert len(val) == 2
        self.SetPosition(*val)

    @property
    def title(self):
        """Return or set the chart's title.

        Examples
        --------
        Create a matplotlib chart with title 'My Chart'.

        .. pyvista-plot::

        >>> import pyvista
        >>> import matplotlib.pyplot as plt
        >>> f, ax = plt.subplots()
        >>> _ = ax.plot([0, 1, 2], [2, 1, 3])
        >>> chart = pyvista.ChartMPL(f)
        >>> chart.title = 'My Chart'
        >>> chart.show()

        """
        return self._fig._suptitle.get_text()

    @title.setter
    def title(self, val):
        self._fig.suptitle(val)

    @property
    def legend_visible(self):
        """Return or set the visibility of the chart's legend.

        Examples
        --------
        Create a matplotlib chart with custom labels and show the legend.

        .. pyvista-plot::

        >>> import pyvista
        >>> import matplotlib.pyplot as plt
        >>> f, ax = plt.subplots()
        >>> _ = ax.plot([0, 1, 2], [2, 1, 3], label="Line")
        >>> _ = ax.scatter([0, 1, 2], [3, 2, 1], label="Points")
        >>> chart = pyvista.ChartMPL(f)
        >>> chart.legend_visible = True
        >>> chart.show()

        Hide the legend.

        >>> chart.legend_visible = False
        >>> chart.show()

        """
        legend = self._fig.axes[0].get_legend()
        return False if legend is None else legend.get_visible()

    @legend_visible.setter
    def legend_visible(self, val):
        legend = self._fig.axes[0].get_legend()
        if legend is None:
            legend = self._fig.axes[0].legend()
        legend.set_visible(val)


class Charts:
    """Collection of charts for a renderer.

    Users should typically not directly create new instances of this class, but use the dedicated
    ``Plotter.add_chart`` method.

    """

    def __init__(self, renderer):
        """Create a new collection of charts for the given renderer."""
        self._charts = []

        # Postpone creation of scene and actor objects until they are
        # needed.
        self._scene = None
        self._actor = None
        self._renderer = renderer

    def _setup_scene(self):
        """Set up a new context scene and actor for these charts."""
        self._scene = _vtk.vtkContextScene()
        self._actor = _vtk.vtkContextActor()

        self._actor.SetScene(self._scene)
        self._renderer.AddActor(self._actor)
        self._scene.SetRenderer(self._renderer)

    def deep_clean(self):
        """Remove all references to the chart objects and internal objects."""
        if self._scene is not None:
            for chart in self._charts:
                chart._x_axis = None
                chart._y_axis = None
                chart._z_axis = None
                self.remove_chart(chart)
            self._renderer.RemoveActor(self._actor)
        self._scene = None
        self._actor = None

    def add_chart(self, chart):
        """Add a chart to the collection."""
        if self._scene is None:
            self._setup_scene()
        self._charts.append(chart)
        self._scene.AddItem(chart._background)
        self._scene.AddItem(chart)
        chart.SetInteractive(False)  # Charts are not interactive by default

    def remove_chart(self, chart_or_index):
        """Remove a chart from the collection."""
        chart = self._charts[chart_or_index] if isinstance(chart_or_index, int) else chart_or_index
        assert chart in self._charts
        self._charts.remove(chart)
        self._scene.RemoveItem(chart)
        self._scene.RemoveItem(chart._background)

    def toggle_interaction(self, mouse_pos):
        """Disable interaction with all charts, except the one indicated by the mouse position.

        Returns the scene if one of the charts got activated, None otherwise.
        """
        # Mouse_pos is either False (to disable interaction with all charts) or a tuple containing the mouse position
        # (to disable interaction with all charts, except the one indicated by the mouse, if any)
        enable = False
        for chart in self._charts:
            if chart.visible and (mouse_pos is not False and chart._is_within(mouse_pos)):
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
