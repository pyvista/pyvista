"""Module containing pyvista wrappers for the vtk Charts API."""

from functools import wraps
import inspect
import itertools
import re
from typing import Dict, Optional, Sequence
import weakref

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np

import pyvista
from pyvista.report import vtk_version_info

from . import _vtk
from ._typing import Chart
from .colors import COLOR_SCHEMES, SCHEME_NAMES, Color, color_synonyms, hexcolors


# region Some metaclass wrapping magic
class _vtkWrapperMeta(type):
    def __init__(cls, clsname, bases, attrs):
        # Restore the signature of classes inheriting from _vtkWrapper
        # Based on https://stackoverflow.com/questions/49740290/call-from-metaclass-shadows-signature-of-init
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())
        params.insert(
            len(params) - 1 if params[-1].kind == inspect.Parameter.VAR_KEYWORD else len(params),
            inspect.Parameter("_wrap", inspect.Parameter.KEYWORD_ONLY, default=None),
        )
        cls.__signature__ = sig.replace(parameters=params[1:])
        super().__init__(clsname, bases, attrs)

    def __call__(cls, *args, _wrap=None, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj._wrapped = _wrap
        obj.__init__(*args, **kwargs)
        return obj


class _vtkWrapper(metaclass=_vtkWrapperMeta):
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


# endregion


# region Documentation substitution
class DocSubs:
    """Helper class to easily substitute the docstrings of the listed member functions or properties."""

    # The substitutions to use for this (sub)class
    _DOC_SUBS: Optional[Dict[str, str]] = None
    # Internal dictionary to store registered member functions/properties and their (to be substituted) docs.
    _DOC_STORE = {}  # type: ignore
    # Tag used to mark members that require docstring substitutions.
    _DOC_TAG = ":DOC_SUBS:"

    def __init_subclass__(cls, **kwargs):
        """Initialize subclasses."""
        # First substitute all members for this class (marked in a super class)
        if cls._DOC_SUBS is not None:
            subs = {**cls._DOC_SUBS}
            if "cls" not in subs:
                subs["cls"] = cls.__name__
            for member_name, (m, d) in cls._DOC_STORE.items():
                if member_name not in cls.__dict__:
                    # If the member is not part of the subclass' __dict__, we have to generate a wrapping
                    # function or property and add it to the subclass' __dict__. Otherwise, the docstring
                    # of the superclass would be used for the substitutions.
                    mem_sub = cls._wrap_member(m)
                    mem_sub.__doc__ = d
                    setattr(cls, member_name, mem_sub)
                # Get the member function/property and substitute its docstring.
                member = getattr(cls, member_name)
                member.__doc__ = member.__doc__.format(**subs)

        # Secondly, register all members of this class that require substitutions in subclasses
        # Create copy of registered members so far
        # TODO: B010
        setattr(cls, "_DOC_STORE", {**cls._DOC_STORE})  # noqa: B010
        for member_name, member in cls.__dict__.items():
            if member.__doc__ and member.__doc__.startswith(cls._DOC_TAG):
                # New method/property to register in this class (denoting their docstring should be
                # substituted in subsequent child classes).
                cls._DOC_STORE[member_name] = (member, member.__doc__[len(cls._DOC_TAG) :])
                # Overwrite original docstring to prevent doctest issues
                member.__doc__ = """Docstring to be specialized in subclasses."""

    @staticmethod
    def _wrap_member(member):
        if callable(member):

            @wraps(member)
            def mem_sub(*args, **kwargs):
                return member(*args, **kwargs)

        elif isinstance(member, property):
            mem_sub = property(member.fget, member.fset, member.fdel)
        else:
            raise NotImplementedError(
                "Members other than methods and properties are currently not supported."
            )
        return mem_sub


def doc_subs(member):
    """Doc subs wrapper.

    Only common attribute between methods and properties that we can
    modify is __doc__, so use that to mark members that need doc
    substitutions.
    Still, only methods can be marked for doc substitution (as for
    properties the docstring seems to be overwritten when specifying
    setters or deleters), hence this decorator should be applied
    before the property decorator. And 'type: ignore' comments are
    necessary because mypy cannot handle decorated properties (see
    https://github.com/python/mypy/issues/1362)
    """
    # Ensure we are operating on a method
    if not callable(member):  # pragma: no cover
        raise ValueError('`member` must be a callable.')
    member.__doc__ = DocSubs._DOC_TAG + member.__doc__
    return member


# endregion


class Pen(_vtkWrapper, _vtk.vtkPen):
    """Pythonic wrapper for a VTK Pen, used to draw lines.

    Parameters
    ----------
    color : ColorLike, default: "k"
        Color of the lines drawn using this pen. Any color parsable by
        :class:`pyvista.Color` is allowed.

    width : float, default: 1
        Width of the lines drawn using this pen.

    style : str, default: "-"
        Style of the lines drawn using this pen. See
        :ref:`Pen.LINE_STYLES <pen_line_styles>` for a list of allowed
        line styles.

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
        "-..": {"id": _vtk.vtkPen.DASH_DOT_DOT_LINE, "descr": "Dash-dot-dot"},
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
        self._color = Color(val, default_color="black")
        self.SetColor(*self._color.int_rgba)

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
    color : ColorLike, default: "k"
        Fill color of the shapes drawn using this brush. Any color
        parsable by :class:`pyvista.Color` is allowed.

    texture : pyvista.Texture, optional
        Texture used to fill shapes drawn using this brush. Any object
        convertible to a :class:`pyvista.Texture` is allowed. Defaults to
        ``None``.

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
        self._color = Color(val, default_color="black")
        self.SetColor(*self._color.int_rgba)

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
        Set up a brush with a texture.

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
        Set up a brush with a texture.

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
        self.SetTextureProperties(1 + int(self._interpolate) + 4 * (1 + int(self._repeat)))


class Axis(_vtkWrapper, _vtk.vtkAxis):
    """Pythonic interface for a VTK Axis, used by 2D charts.

    Parameters
    ----------
    label : str, default: ""
        Axis label.

    range : sequence[float], optional
        Axis range, denoting the minimum and maximum values
        displayed on this axis. Setting this to any valid value
        other than ``None`` will change this axis behavior to
        ``'fixed'``. Setting it to ``None`` will change the axis
        behavior to ``'auto'``.

    grid : bool, default: True
        Flag to toggle grid lines visibility for this axis.

    Attributes
    ----------
    pen : Pen
        Pen used to draw the axis.

    grid_pen : Pen
        Pen used to draw the grid lines.

    Other Parameters
    ----------------
    _wrap : vtk.vtkAxis, optional
        Wrap an existing VTK Axis instance. Defaults to ``None`` (no wrapping).

    """

    BEHAVIORS = {"auto": _vtk.vtkAxis.AUTO, "fixed": _vtk.vtkAxis.FIXED}

    def __init__(self, label="", range=None, grid=True):
        """Initialize a new Axis instance."""
        super().__init__()
        self._tick_locs = _vtk.vtkDoubleArray()
        self._tick_labels = _vtk.vtkStringArray()
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            # SetPen and SetGridPen methods are not available for older VTK versions,
            # so fallback to using wrapper objects.
            self.pen = Pen(color=(0, 0, 0), _wrap=self.GetPen())
            self.grid_pen = Pen(color=(0.95, 0.95, 0.95), _wrap=self.GetGridPen())
        else:
            self.pen = Pen(color=(0, 0, 0))
            self.grid_pen = Pen(color=(0.95, 0.95, 0.95))
            self.SetPen(self.pen)
            self.SetGridPen(self.grid_pen)
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
    def label_size(self):
        """Return or set the size of the axis label font.

        Examples
        --------
        Set the x-axis label font size of a 2D chart to 20.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.label_size = 20
        >>> chart.x_axis.label_size
        20
        >>> chart.show()

        """
        return self.GetTitleProperties().GetFontSize()

    @label_size.setter
    def label_size(self, size):
        self.GetTitleProperties().SetFontSize(size)

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
        >>> chart.x_axis.range
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
        >>> chart.background_color = 'c'
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
        >>> chart.y_axis.log_scale
        True

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
        ...                             "Small?", "Not large", "Large?",
        ...                             "Very large"]
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
    def tick_label_size(self):
        """Return or set the size of the axis tick label font.

        Examples
        --------
        Set the x-axis tick label font size of a 2D chart to 20.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line([0, 1, 2], [2, 1, 3])
        >>> chart.x_axis.tick_label_size = 20
        >>> chart.x_axis.tick_label_size
        20
        >>> chart.show()

        """
        return self.GetLabelProperties().GetFontSize()

    @tick_label_size.setter
    def tick_label_size(self, size):
        self.GetLabelProperties().SetFontSize(size)

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
    class ItemWrapper:
        def Initialize(self, item):
            # item is the _CustomContextItem subclass instance
            return True

        def Paint(self, item, painter):
            # item is the _CustomContextItem subclass instance
            return item.paint(painter)

    def __init__(self):
        super().__init__()
        # This will also call ItemWrapper.Initialize
        self.SetPythonObject(_CustomContextItem.ItemWrapper())

    def paint(self, painter):
        return True


class _ChartBackground(_CustomContextItem):
    """Utility class for chart backgrounds."""

    def __init__(self, chart):
        super().__init__()
        # Note: This SHOULD be a weakref proxy, as otherwise the garbage collector will not clean up unused charts
        # (because of the cyclic references between charts and their background).
        self._chart = weakref.proxy(chart)  # Weakref proxy to the chart to draw the background for
        # Default background is translucent with black border line
        self.BorderPen = Pen(color=(0, 0, 0))
        self.BackgroundBrush = Brush(color=(0, 0, 0, 0))
        # Default active background is slightly more opaque with yellow border line
        self.ActiveBorderPen = Pen(color=(0.8, 0.8, 0.2))
        self.ActiveBackgroundBrush = Brush(color=(1.0, 1.0, 1.0, 0.4))

    def paint(self, painter):
        if self._chart.visible:
            painter.ApplyPen(self.ActiveBorderPen if self._chart._interactive else self.BorderPen)
            painter.ApplyBrush(
                self.ActiveBackgroundBrush if self._chart._interactive else self.BackgroundBrush
            )
            l, b, w, h = self._chart._geometry
            painter.DrawRect(l, b, w, h)
            if vtk_version_info < (9, 2, 0):  # pragma: no cover
                # Following 'patch' is necessary for earlier VTK versions. Otherwise Pie plots will use the same opacity
                # as the chart's background when their legend is hidden. As the default background is transparent,
                # this will cause Pie charts to completely disappear.
                painter.GetBrush().SetOpacity(255)
                painter.GetBrush().SetTexture(None)
        return True


class _Chart(DocSubs):
    """Common pythonic interface for vtkChart, vtkChartBox, vtkChartPie and ChartMPL instances."""

    # Subclasses should specify following substitutions: 'chart_name', 'chart_args', 'chart_init' and 'chart_set_labels'.
    _DOC_SUBS: Optional[Dict[str, str]] = None

    def __init__(self, size=(1, 1), loc=(0, 0)):
        super().__init__()
        self._background = _ChartBackground(self)
        self._x_axis = Axis()
        self._y_axis = Axis()
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

    def _render_event(self, *args, plotter_render=False, **kwargs):
        """Update the chart right before it will be rendered."""
        # Only resize on real VTK render events (plotter.render calls will afterwards invoke a proper render event)
        if not plotter_render:
            self._resize()

    def _resize(self):
        """Resize this chart.

        Resize this chart such that it always occupies the specified
        geometry (matching the specified location and size).

        Returns
        -------
        bool
            ``True`` if the chart was resized, ``False`` otherwise.

        """
        # edge race case
        if self._renderer is None:  # pragma: no cover
            return

        r_w, r_h = self._renderer.GetSize()
        # Alternatively: self.scene.GetViewWidth(), self.scene.GetViewHeight()
        _, _, c_w, c_h = (int(g) for g in self._geometry)
        # Target size is calculated from specified normalized width and height and the renderer's current size
        t_w = int(self._size[0] * r_w)
        t_h = int(self._size[1] * r_h)
        resize = c_w != t_w or c_h != t_h
        if resize:
            # Mismatch between current size and target size, so resize chart:
            self._geometry = (int(self._loc[0] * r_w), int(self._loc[1] * r_h), t_w, t_h)
        return resize

    @property
    def _geometry(self):
        """Chart geometry (x and y position of bottom left corner and width and height in pixels)."""
        return tuple(self.GetSize())

    @_geometry.setter
    def _geometry(self, val):
        """Set the chart geometry."""
        self.SetSize(_vtk.vtkRectf(*val))

    @property
    def _interactive(self):
        """Return or set the chart's interactivity.

        Notes
        -----
        Users should not set this property directly, but use the
        :func:`Renderer.set_chart_interaction` method instead.

        """
        return self.GetInteractive()

    @_interactive.setter
    def _interactive(self, val):
        self.SetInteractive(val)

    def _is_within(self, pos):
        """Check whether the specified position (in pixels) lies within this chart's geometry."""
        l, b, w, h = self._geometry
        return l <= pos[0] <= l + w and b <= pos[1] <= b + h

    @property  # type: ignore
    @doc_subs
    def size(self):
        """Return or set the chart size in normalized coordinates.

        A size of ``(1, 1)`` occupies the whole renderer.

        Examples
        --------
        Create a half-sized {chart_name} centered in the middle of the
        renderer.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.size = (0.5, 0.5)
        >>> chart.loc = (0.25, 0.25)
        >>> chart.show()

        """
        return self._size

    @size.setter
    def size(self, val):
        if not (len(val) == 2 and 0 <= val[0] <= 1 and 0 <= val[1] <= 1):
            raise ValueError(f'Invalid size {val}.')
        self._size = val

    @property  # type: ignore
    @doc_subs
    def loc(self):
        """Return or set the chart position in normalized coordinates.

        This denotes the location of the chart's bottom left corner.

        Examples
        --------
        Create a half-sized {chart_name} centered in the middle of the
        renderer.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.size = (0.5, 0.5)
        >>> chart.loc = (0.25, 0.25)
        >>> chart.show()

        """
        return self._loc

    @loc.setter
    def loc(self, val):
        if not (len(val) == 2 and 0 <= val[0] <= 1 and 0 <= val[1] <= 1):
            raise ValueError(f'Invalid loc {val}.')
        self._loc = val

    @property  # type: ignore
    @doc_subs
    def border_color(self):
        """Return or set the chart's border color.

        Examples
        --------
        Create a {chart_name} with a thick, dashed red border.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.border_color = 'r'
        >>> chart.border_width = 5
        >>> chart.border_style = '--'
        >>> chart.show(interactive=False)

        """
        return self._background.BorderPen.color

    @border_color.setter
    def border_color(self, val):
        self._background.BorderPen.color = val

    @property  # type: ignore
    @doc_subs
    def border_width(self):
        """Return or set the chart's border width.

        Examples
        --------
        Create a {chart_name} with a thick, dashed red border.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.border_color = 'r'
        >>> chart.border_width = 5
        >>> chart.border_style = '--'
        >>> chart.show(interactive=False)

        """
        return self._background.BorderPen.width

    @border_width.setter
    def border_width(self, val):
        self._background.BorderPen.width = val
        self._background.ActiveBorderPen.width = val

    @property  # type: ignore
    @doc_subs
    def border_style(self):
        """Return or set the chart's border style.

        Examples
        --------
        Create a {chart_name} with a thick, dashed red border.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.border_color = 'r'
        >>> chart.border_width = 5
        >>> chart.border_style = '--'
        >>> chart.show(interactive=False)

        """
        return self._background.BorderPen.style

    @border_style.setter
    def border_style(self, val):
        self._background.BorderPen.style = val
        self._background.ActiveBorderPen.style = val

    @property  # type: ignore
    @doc_subs
    def active_border_color(self):
        """Return or set the chart's border color in interactive mode.

        Examples
        --------
        Create a {chart_name} with a thick, dashed red border.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.border_color = 'r'
        >>> chart.border_width = 5
        >>> chart.border_style = '--'
        >>> chart.show(interactive=False)

        Set the active border color to yellow and activate the chart.

        >>> chart.active_border_color = 'y'
        >>> chart.show(interactive=True)

        """
        return self._background.ActiveBorderPen.color

    @active_border_color.setter
    def active_border_color(self, val):
        self._background.ActiveBorderPen.color = val

    @property  # type: ignore
    @doc_subs
    def background_color(self):
        """Return or set the chart's background color.

        Examples
        --------
        Create a {chart_name} with a green background.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.background_color = (0.5, 0.9, 0.5)
        >>> chart.show(interactive=False)

        """
        return self._background.BackgroundBrush.color

    @background_color.setter
    def background_color(self, val):
        self._background.BackgroundBrush.color = val

    @property  # type: ignore
    @doc_subs
    def background_texture(self):
        """Return or set the chart's background texture.

        Examples
        --------
        Create a {chart_name} with an emoji as its background.

        >>> import pyvista
        >>> from pyvista import examples
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.background_texture = examples.download_emoji_texture()
        >>> chart.show(interactive=False)

        """
        return self._background.BackgroundBrush.texture

    @background_texture.setter
    def background_texture(self, val):
        self._background.BackgroundBrush.texture = val
        self._background.ActiveBackgroundBrush.texture = val

    @property  # type: ignore
    @doc_subs
    def active_background_color(self):
        """Return or set the chart's background color in interactive mode.

        Examples
        --------
        Create a {chart_name} with a green background.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.background_color = (0.5, 0.9, 0.5)
        >>> chart.show(interactive=False)

        Set the active background color to blue and activate the chart.

        >>> chart.active_background_color = 'b'
        >>> chart.show(interactive=True)

        """
        return self._background.ActiveBackgroundBrush.color

    @active_background_color.setter
    def active_background_color(self, val):
        self._background.ActiveBackgroundBrush.color = val

    @property  # type: ignore
    @doc_subs
    def visible(self):
        """Return or set the chart's visibility.

        Examples
        --------
        Create a {chart_name}.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.show()

        Hide it.

        >>> chart.visible = False
        >>> chart.show()

        """
        return self.GetVisible()

    @visible.setter
    def visible(self, val):
        self.SetVisible(val)

    @doc_subs
    def toggle(self):
        """Toggle the chart's visibility.

        Examples
        --------
        Create a {chart_name}.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.show()

        Hide it.

        >>> chart.toggle()
        >>> chart.show()

        """
        self.visible = not self.visible

    @property  # type: ignore
    @doc_subs
    def title(self):
        """Return or set the chart's title.

        Examples
        --------
        Create a {chart_name} with title 'My Chart'.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.title = 'My Chart'
        >>> chart.show()

        """
        return self.GetTitle()

    @title.setter
    def title(self, val):
        self.SetTitle(val)

    @property  # type: ignore
    @doc_subs
    def legend_visible(self):
        """Return or set the visibility of the chart's legend.

        Examples
        --------
        Create a {chart_name} with custom labels.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> {chart_set_labels}
        >>> chart.show()

        Hide the legend.

        >>> chart.legend_visible = False
        >>> chart.show()

        """
        return self.GetShowLegend()

    @legend_visible.setter
    def legend_visible(self, val):
        self.SetShowLegend(val)

    @doc_subs
    def show(
        self,
        interactive=True,
        off_screen=None,
        full_screen=None,
        screenshot=None,
        window_size=None,
        notebook=None,
        background='w',
        dev_kwargs={},
    ):
        """Show this chart in a self contained plotter.

        Parameters
        ----------
        interactive : bool, default: True
            Enable interaction with the chart. Interaction is not enabled
            when plotting off screen.

        off_screen : bool, optional
            Plots off screen when ``True``.  Helpful for saving screenshots
            without a window popping up. Defaults to active theme setting.

        full_screen : bool, optional
            Opens window in full screen.  When enabled, ignores
            ``window_size``. Defaults to active theme setting.

        screenshot : str | bool, default: False
            Saves screenshot to file when enabled.  See:
            :func:`Plotter.screenshot() <pyvista.Plotter.screenshot>`.

            When ``True``, takes screenshot and returns ``numpy`` array of
            image.

        window_size : list, optional
            Window size in pixels. Defaults to active theme setting.

        notebook : bool, optional
            When ``True``, the resulting plot is placed inline a
            jupyter notebook.  Assumes a jupyter console is active.

        background : ColorLike, default: "w"
            Use to make the entire mesh have a single solid color.
            Either a string, RGB list, or hex color string.  For example:
            ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
            ``color='#FFFFFF'``.

        dev_kwargs : dict, optional
            Optional developer keyword arguments.

        Returns
        -------
        np.ndarray
            Numpy array of the last image when ``screenshot=True``
            is set. Optionally contains alpha values. Sized:

            * [Window height x Window width x 3] if the theme sets
              ``transparent_background=False``.
            * [Window height x Window width x 4] if the theme sets
              ``transparent_background=True``.

        Examples
        --------
        Create a simple {chart_name} and show it.

        >>> import pyvista
        >>> chart = pyvista.{cls}({chart_args}){chart_init}
        >>> chart.show()

        """
        if off_screen is None:
            off_screen = pyvista.OFF_SCREEN
        pl = pyvista.Plotter(window_size=window_size, notebook=notebook, off_screen=off_screen)
        pl.background_color = background
        pl.add_chart(self)
        if interactive and (not off_screen or pyvista.BUILDING_GALLERY):  # pragma: no cover
            pl.set_chart_interaction(self)
        return pl.show(
            screenshot=screenshot,
            full_screen=full_screen,
            **dev_kwargs,
        )


class _Plot(DocSubs):
    """Common pythonic interface for vtkPlot and vtkPlot3D instances."""

    # Subclasses should specify following substitutions: 'plot_name', 'chart_init' and 'plot_init'.
    _DOC_SUBS: Optional[Dict[str, str]] = None

    def __init__(self, chart):
        super().__init__()
        self._chart = weakref.proxy(chart)
        self._pen = Pen()
        self._brush = Brush()
        self._label = ""
        if hasattr(self, "SetPen"):
            self.SetPen(self._pen)
        if hasattr(self, "SetBrush"):
            self.SetBrush(self._brush)

    @property  # type: ignore
    @doc_subs
    def color(self):
        """Return or set the plot's color.

        This is the color used by the plot's pen and brush to draw lines and shapes.

        Examples
        --------
        Set the {plot_name}'s color to red.

        >>> import pyvista
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> plot.color = 'r'
        >>> chart.show()

        """
        return self.pen.color

    @color.setter
    def color(self, val):
        self.pen.color = val
        self.brush.color = val

    @property  # type: ignore
    @doc_subs
    def pen(self):
        """Pen object controlling how lines in this plot are drawn.

        Returns
        -------
        Pen
            Pen object controlling how lines in this plot are drawn.

        Examples
        --------
        Increase the line width of the {plot_name}'s pen object.

        >>> import pyvista
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> plot.line_style = '-'  # Make sure all lines are visible
        >>> plot.pen.width = 10
        >>> chart.show()

        """
        return self._pen

    @property  # type: ignore
    @doc_subs
    def brush(self):
        """Brush object controlling how shapes in this plot are filled.

        Returns
        -------
        Brush
            Brush object controlling how shapes in this plot are filled.

        Examples
        --------
        Use a custom texture for the {plot_name}'s brush object.

        >>> import pyvista
        >>> from pyvista import examples
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> plot.brush.texture = examples.download_puppy_texture()
        >>> chart.show()

        """
        return self._brush

    @property  # type: ignore
    @doc_subs
    def line_width(self):
        """Return or set the line width of all lines drawn in this plot.

        This is equivalent to accessing/modifying the width of this plot's pen.

        Examples
        --------
        Set the line width to 10

        >>> import pyvista
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> plot.line_style = '-'  # Make sure all lines are visible
        >>> plot.line_width = 10
        >>> chart.show()

        """
        return self.pen.width

    @line_width.setter
    def line_width(self, val):
        self.pen.width = val

    @property  # type: ignore
    @doc_subs
    def line_style(self):
        """Return or set the line style of all lines drawn in this plot.

        This is equivalent to accessing/modifying the style of this plot's pen.

        Examples
        --------
        Set a custom line style.

        >>> import pyvista
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> plot.line_style = '-.'
        >>> chart.show()

        """
        return self.pen.style

    @line_style.setter
    def line_style(self, val):
        self.pen.style = val

    @property  # type: ignore
    @doc_subs
    def label(self):
        """Return or set the this plot's label, as shown in the chart's legend.

        Examples
        --------
        Create a {plot_name} with custom label.

        >>> import pyvista
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> plot.label = "My awesome plot"
        >>> chart.show()

        """
        return self._label

    @label.setter
    def label(self, val):
        self._label = "" if val is None else val
        self.SetLabel(self._label)

    @property  # type: ignore
    @doc_subs
    def visible(self):
        """Return or set the this plot's visibility.

        Examples
        --------
        Create a {plot_name}.

        >>> import pyvista
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> chart.show()

        Hide it.

        >>> plot.visible = False
        >>> chart.show()

        """
        return self.GetVisible()

    @visible.setter
    def visible(self, val):
        self.SetVisible(val)

    @doc_subs
    def toggle(self):
        """Toggle the plot's visibility.

        Examples
        --------
        Create a {plot_name}.

        >>> import pyvista
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> chart.show()

        Hide it.

        >>> plot.toggle()
        >>> chart.show()

        """
        self.visible = not self.visible


class _MultiCompPlot(_Plot):
    """Common pythonic interface for vtkPlot instances with multiple components.

    Example subclasses are BoxPlot, PiePlot, BarPlot and StackPlot.
    """

    DEFAULT_COLOR_SCHEME = "qual_accent"

    # Subclasses should specify following substitutions: 'plot_name', 'chart_init', 'plot_init', 'multichart_init' and 'multiplot_init'.
    _DOC_SUBS: Optional[Dict[str, str]] = None

    def __init__(self, chart):
        super().__init__(chart)
        self._color_series = _vtk.vtkColorSeries()
        self._lookup_table = self._color_series.CreateLookupTable(_vtk.vtkColorSeries.CATEGORICAL)
        self._labels = _vtk.vtkStringArray()
        self.SetLabels(self._labels)
        self.color_scheme = self.DEFAULT_COLOR_SCHEME

    @property  # type: ignore
    @doc_subs
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
        Set the {plot_name}'s color scheme to warm.

        >>> import pyvista
        >>> chart = {multichart_init}
        >>> plot = {multiplot_init}
        >>> plot.color_scheme = "warm"
        >>> chart.show()

        """
        return SCHEME_NAMES.get(self._color_series.GetColorScheme(), "custom")

    @color_scheme.setter
    def color_scheme(self, val):
        self._color_series.SetColorScheme(COLOR_SCHEMES.get(val, COLOR_SCHEMES["custom"])["id"])
        self._color_series.BuildLookupTable(self._lookup_table, _vtk.vtkColorSeries.CATEGORICAL)
        self.brush.color = self.colors[0]

    @property  # type: ignore
    @doc_subs
    def colors(self):
        """Return or set the plot's colors.

        These are the colors used for the different
        components drawn by this plot.

        Examples
        --------
        Set the {plot_name}'s colors manually.

        >>> import pyvista
        >>> chart = {multichart_init}
        >>> plot = {multiplot_init}
        >>> plot.colors = ["b", "g", "r", "c"]
        >>> chart.show()

        """
        return [
            Color(self._color_series.GetColor(i))
            for i in range(self._color_series.GetNumberOfColors())
        ]

    @colors.setter
    def colors(self, val):
        if val is None:
            self.color_scheme = self.DEFAULT_COLOR_SCHEME
            # Setting color_scheme already sets brush.color
        elif isinstance(val, str):
            self.color_scheme = val
            # Setting color_scheme already sets brush.color
        else:
            try:
                self._color_series.SetNumberOfColors(len(val))
                for i, color in enumerate(val):
                    self._color_series.SetColor(i, Color(color).vtk_c3ub)
                self._color_series.BuildLookupTable(
                    self._lookup_table, _vtk.vtkColorSeries.CATEGORICAL
                )
                self.brush.color = self.colors[0]  # Synchronize "color" and "colors" properties
            except ValueError as e:
                self.color_scheme = self.DEFAULT_COLOR_SCHEME
                raise ValueError(
                    "Invalid colors specified, falling back to default color scheme."
                ) from e

    @property  # type: ignore
    @doc_subs
    def color(self):
        """Return or set the plot's color.

        This is the color used by the plot's brush
        to draw the different components.

        Examples
        --------
        Set the {plot_name}'s color to red.

        >>> import pyvista
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> plot.color = 'r'
        >>> chart.show()

        """
        return self.brush.color

    @color.setter
    def color(self, val):
        # Override default _Plot behaviour. This makes sure the plot's "color_scheme", "colors" and "color" properties
        # (and their internal representations through color series, lookup tables and brushes) stay synchronized.
        self.colors = [val]

    @property  # type: ignore
    @doc_subs
    def labels(self):
        """Return or set the this plot's labels, as shown in the chart's legend.

        Examples
        --------
        Create a {plot_name}.

        >>> import pyvista
        >>> chart = {multichart_init}
        >>> plot = {multiplot_init}
        >>> chart.show()

        Modify the labels.

        >>> plot.labels = ["A", "B", "C", "D"]
        >>> chart.show()

        """
        return [self._labels.GetValue(i) for i in range(self._labels.GetNumberOfValues())]

    @labels.setter
    def labels(self, val):
        self._labels.Reset()
        if isinstance(val, str):
            val = [val]
        try:
            if val is not None:
                for label in val:
                    self._labels.InsertNextValue(label)
        except TypeError:
            raise ValueError("Invalid labels specified.")

    @property  # type: ignore
    @doc_subs
    def label(self):
        """Return or set the this plot's label, as shown in the chart's legend.

        Examples
        --------
        Create a {plot_name} with custom label.

        >>> import pyvista
        >>> import numpy as np
        >>> chart = {chart_init}
        >>> plot = {plot_init}
        >>> chart.show()

        Modify the label.

        >>> plot.label = "My awesome plot"
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
    chart : Chart2D
        The chart containing this plot.

    x : array_like
        X coordinates of the points through which a line should be drawn.

    y : array_like
        Y coordinates of the points through which a line should be drawn.

    color : ColorLike, default: "b"
        Color of the line drawn in this plot. Any color parsable by :class:`pyvista.Color` is allowed.

    width : float, default: 1
        Width of the line drawn in this plot.

    style : str, default: "-"
        Style of the line drawn in this plot. See :ref:`Pen.LINE_STYLES <pen_line_styles>` for a list of allowed line
        styles.

    label : str, default: ""
        Label of this plot, as shown in the chart's legend.

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

    _DOC_SUBS = {
        "plot_name": "2D line plot",
        "chart_init": "pyvista.Chart2D()",
        "plot_init": "chart.line([0, 1, 2], [2, 1, 3])",
    }

    def __init__(self, chart, x, y, color="b", width=1.0, style="-", label=""):
        """Initialize a new 2D line plot instance."""
        super().__init__(chart)
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
    chart : Chart2D
        The chart containing this plot.

    x : array_like
        X coordinates of the points to draw.

    y : array_like
        Y coordinates of the points to draw.

    color : ColorLike, default: "b"
        Color of the points drawn in this plot. Any color parsable by :class:`pyvista.Color` is allowed.

    size : float, default: 10
        Size of the point markers drawn in this plot.

    style : str, default: "o"
        Style of the point markers drawn in this plot. See :ref:`ScatterPlot2D.MARKER_STYLES <scatter_marker_styles>`
        for a list of allowed marker styles.

    label : str, default: ""
        Label of this plot, as shown in the chart's legend.

    Notes
    -----
    .. _scatter_marker_styles:

    MARKER_STYLES : dict
        Dictionary containing all allowed marker styles as its keys.

        .. include:: ../scatter_marker_styles.rst

    Examples
    --------
    Plot a simple sine wave as a scatter plot.

    >>> import pyvista
    >>> import numpy as np
    >>> x = np.linspace(0, 2*np.pi, 20)
    >>> y = np.sin(x)
    >>> chart = pyvista.Chart2D()
    >>> _ = chart.scatter(x, y)
    >>> chart.show()

    """

    MARKER_STYLES = {  # descr is used in the documentation, set to None to hide it from the docs.
        "": {"id": _vtk.vtkPlotPoints.NONE, "descr": "Hidden"},
        "x": {"id": _vtk.vtkPlotPoints.CROSS, "descr": "Cross"},
        "+": {"id": _vtk.vtkPlotPoints.PLUS, "descr": "Plus"},
        "s": {"id": _vtk.vtkPlotPoints.SQUARE, "descr": "Square"},
        "o": {"id": _vtk.vtkPlotPoints.CIRCLE, "descr": "Circle"},
        "d": {"id": _vtk.vtkPlotPoints.DIAMOND, "descr": "Diamond"},
    }
    _DOC_SUBS = {
        "plot_name": "2D scatter plot",
        "chart_init": "pyvista.Chart2D()",
        "plot_init": "chart.scatter([0, 1, 2, 3, 4], [2, 1, 3, 4, 2])",
    }

    def __init__(self, chart, x, y, color="b", size=10, style="o", label=""):
        """Initialize a new 2D scatter plot instance."""
        super().__init__(chart)
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
        >>> plot = chart.scatter([0, 1, 2, 3, 4], [2, 1, 3, 4, 2])
        >>> plot.x
        pyvista_ndarray([0, 1, 2, 3, 4])
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
        >>> plot = chart.scatter([0, 1, 2, 3, 4], [2, 1, 3, 4, 2])
        >>> plot.y
        pyvista_ndarray([2, 1, 3, 4, 2])
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
        >>> plot = chart.scatter([0, 1, 2, 3, 4], [2, 1, 3, 4, 2])
        >>> chart.show()

        Update the marker locations.

        >>> plot.update([0, 1, 2, 3, 4], [3, 2, 4, 2, 1])
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
        >>> plot = chart.scatter([0, 1, 2, 3, 4], [2, 1, 3, 4, 2])
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
        >>> plot = chart.scatter([0, 1, 2, 3, 4], [2, 1, 3, 4, 2])
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
    chart : Chart2D
        The chart containing this plot.

    x : array_like
        X coordinates of the points outlining the area to draw.

    y1 : array_like
        Y coordinates of the points on the first outline of the area to draw.

    y2 : array_like, optional
        Y coordinates of the points on the second outline of the area to
        draw. Defaults to ``numpy.zeros_like(x)``.

    color : ColorLike, default: "b"
        Color of the area drawn in this plot. Any color parsable by :class:`pyvista.Color` is allowed.

    label : str, default: ""
        Label of this plot, as shown in the chart's legend.

    Examples
    --------
    Create an area plot showing the minimum and maximum precipitation observed in each month.

    >>> import pyvista
    >>> import numpy as np
    >>> x = np.arange(12)
    >>> p_min = [11, 0, 16, 2, 23, 18, 25, 17, 9, 12, 14, 21]
    >>> p_max = [87, 64, 92, 73, 91, 94, 107, 101, 84, 88, 95, 103]
    >>> chart = pyvista.Chart2D()
    >>> _ = chart.area(x, p_min, p_max)
    >>> chart.x_axis.tick_locations = x
    >>> chart.x_axis.tick_labels = ["Jan", "Feb", "Mar", "Apr", "May",
    ...                             "Jun", "Jul", "Aug", "Sep", "Oct",
    ...                             "Nov", "Dec"]
    >>> chart.x_axis.label = "Month"
    >>> chart.y_axis.label = "Precipitation [mm]"
    >>> chart.show()

    """

    _DOC_SUBS = {
        "plot_name": "area plot",
        "chart_init": "pyvista.Chart2D()",
        "plot_init": "chart.area([0, 1, 2], [0, 0, 1], [1, 3, 2])",
    }

    def __init__(self, chart, x, y1, y2=None, color="b", label=""):
        """Initialize a new 2D area plot instance."""
        super().__init__(chart)
        self._table = pyvista.Table(
            {
                "x": np.empty(0, np.float32),
                "y1": np.empty(0, np.float32),
                "y2": np.empty(0, np.float32),
            }
        )
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
        >>> plot = chart.area([0, 1, 2], [2, 1, 3], [1, 0, 1])
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
        >>> plot = chart.area([0, 1, 2], [2, 1, 3], [1, 0, 1])
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
        >>> plot = chart.area([0, 1, 2], [2, 1, 3], [1, 0, 1])
        >>> plot.y2
        pyvista_ndarray([1, 0, 1])
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
            The new y coordinates of the points on the second outline of the
            area. Default ``numpy.zeros_like(x)``.

        Examples
        --------
        Create an area plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.area([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        Update the points on the second outline of the area.

        >>> plot.update([0, 1, 2], [2, 1, 3], [1, 0, 1])
        >>> chart.show()

        """
        if len(x) > 0:
            if y2 is None:
                y2 = np.zeros_like(x)
            self._table.update(
                {
                    "x": np.array(x, copy=False),
                    "y1": np.array(y1, copy=False),
                    "y2": np.array(y2, copy=False),
                }
            )
            self.visible = True
        else:
            self.visible = False


class BarPlot(_vtk.vtkPlotBar, _MultiCompPlot):
    """Class representing a 2D bar plot.

    Users should typically not directly create new plot instances, but use the dedicated 2D chart's plotting methods.

    Parameters
    ----------
    chart : Chart2D
        The chart containing this plot.

    x : array_like
        Positions (along the x-axis for a vertical orientation, along the y-axis for
        a horizontal orientation) of the bars to draw.

    y : array_like
        Size of the bars to draw. Multiple bars can be stacked by passing a sequence of sequences.

    color : ColorLike, default: "b"
        Color of the bars drawn in this plot. Any color parsable by :class:`pyvista.Color` is allowed.

    orientation : str, default: "V"
        Orientation of the bars drawn in this plot. Either ``"H"`` for an horizontal orientation or ``"V"`` for a
        vertical orientation.

    label : str, default: ""
        Label of this plot, as shown in the chart's legend.

    Examples
    --------
    Create a stacked bar chart showing the average time spent on activities
    throughout the week.

    >>> import pyvista
    >>> import numpy as np
    >>> x = np.arange(1, 8)
    >>> y_s = [7, 8, 7.5, 8, 7.5, 9, 10]
    >>> y_h = [2, 3, 2, 2.5, 1.5, 4, 6.5]
    >>> y_w = [8, 8, 7, 8, 7, 0, 0]
    >>> y_r = [5, 2.5, 4.5, 3.5, 6, 9, 6.5]
    >>> y_t = [2, 2.5, 3, 2, 2, 2, 1]
    >>> labels = ["Sleep", "Household", "Work", "Relax", "Transport"]
    >>> chart = pyvista.Chart2D()
    >>> _ = chart.bar(x, [y_s, y_h, y_w, y_r, y_t], label=labels)
    >>> chart.x_axis.tick_locations = x
    >>> chart.x_axis.tick_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    >>> chart.x_label = "Day of week"
    >>> chart.y_label = "Average time spent"
    >>> chart.grid = False  # Disable the grid lines
    >>> chart.show()

    """

    ORIENTATIONS = {"H": _vtk.vtkPlotBar.HORIZONTAL, "V": _vtk.vtkPlotBar.VERTICAL}
    _DOC_SUBS = {
        "plot_name": "bar plot",
        "chart_init": "pyvista.Chart2D()",
        "plot_init": "chart.bar([1, 2, 3], [2, 1, 3])",
        "multichart_init": "pyvista.Chart2D()",
        "multiplot_init": "chart.bar([1, 2, 3], [[2, 1, 3], [1, 0, 2], [0, 3, 1], [3, 2, 0]])",
    }

    def __init__(self, chart, x, y, color=None, orientation="V", label=None):
        """Initialize a new 2D bar plot instance."""
        super().__init__(chart)
        if not isinstance(y[0], (Sequence, np.ndarray)):
            y = (y,)
        y_data = {f"y{i}": np.empty(0, np.float32) for i in range(len(y))}
        self._table = pyvista.Table({"x": np.empty(0, np.float32), **y_data})
        self.SetInputData(self._table, "x", "y0")
        for i in range(1, len(y)):
            self.SetInputArray(i + 1, f"y{i}")
        self.update(x, y)

        if len(y) > 1:
            self.SetColorSeries(self._color_series)
            self.colors = color  # None will use default scheme
            self.labels = label
        else:
            # Use blue bars by default in single component mode
            self.color = "b" if color is None else color
            self.label = label
        self.orientation = orientation

    @property
    def x(self):
        """Retrieve the positions of the drawn bars.

        Examples
        --------
        Create a bar plot and display the positions.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([1, 2, 3], [[2, 1, 3], [1, 2, 0]])
        >>> plot.x
        pyvista_ndarray([1, 2, 3])
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
        >>> plot = chart.bar([1, 2, 3], [[2, 1, 3], [1, 2, 0]])
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
        >>> plot = chart.bar([1, 2, 3], [2, 1, 3])
        >>> chart.show()

        Update the bar sizes.

        >>> plot.update([1, 2, 3], [3, 1, 2])
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
    def orientation(self):
        """Return or set the orientation of the bars in this plot.

        Examples
        --------
        Create a bar plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([1, 2, 3], [[2, 1, 3], [1, 3, 2]])
        >>> chart.show()

        Change the orientation to horizontal.

        >>> plot.orientation = "H"
        >>> chart.show()

        """
        return self._orientation

    @orientation.setter
    def orientation(self, val):
        try:
            self.SetOrientation(self.ORIENTATIONS[val])
            self._orientation = val
        except KeyError:
            formatted_orientations = "\", \"".join(self.ORIENTATIONS.keys())
            raise ValueError(
                f"Invalid orientation. Allowed orientations: \"{formatted_orientations}\""
            )


class StackPlot(_vtk.vtkPlotStacked, _MultiCompPlot):
    """Class representing a 2D stack plot.

    Users should typically not directly create new plot instances, but use the dedicated 2D chart's plotting methods.

    Parameters
    ----------
    chart : Chart2D
        The chart containing this plot.

    x : array_like
        X coordinates of the points outlining the stacks (areas) to draw.

    ys : sequence[array_like]
        Size of the stacks (areas) to draw at the corresponding X
        coordinates. Each sequence defines the sizes of one stack
        (area), which are stacked on top of each other.

    colors : sequence[ColorLike], optional
        Color of the stacks (areas) drawn in this plot. Any color
        parsable by :class:`pyvista.Color` is allowed.

    labels : sequence[str], default: []
        Label for each stack (area) drawn in this plot, as shown in
        the chart's legend.

    Examples
    --------
    Create a stack plot showing the amount of vehicles sold per type.

    >>> import pyvista
    >>> import numpy as np
    >>> year = [f"{y}" for y in np.arange(2011, 2021)]
    >>> x = np.arange(len(year))
    >>> n_e = [1739, 4925, 9515, 21727, 31452, 29926, 40648,
    ...        57761, 76370, 93702]
    >>> n_h = [5563, 7642, 11937, 13905, 22807, 46700, 60875,
    ...        53689, 46650, 50321]
    >>> n_f = [166556, 157249, 151552, 138183, 129669,
    ...        113985, 92965, 73683, 57097, 29499]
    >>> chart = pyvista.Chart2D()
    >>> plot = chart.stack(x, [n_e, n_h, n_f])
    >>> plot.labels = ["Electric", "Hybrid", "Fossil"]
    >>> chart.x_axis.label = "Year"
    >>> chart.x_axis.tick_locations = x
    >>> chart.x_axis.tick_labels = year
    >>> chart.y_axis.label = "New car sales"
    >>> chart.show()

    """

    _DOC_SUBS = {
        "plot_name": "stack plot",
        "chart_init": "pyvista.Chart2D()",
        "plot_init": "chart.stack([0, 1, 2], [2, 1, 3])",
        "multichart_init": "pyvista.Chart2D()",
        "multiplot_init": "chart.stack([0, 1, 2], [[2, 1, 3], [1, 0, 2], [0, 3, 1], [3, 2, 0]])",
    }

    def __init__(self, chart, x, ys, colors=None, labels=None):
        """Initialize a new 2D stack plot instance."""
        super().__init__(chart)
        if not isinstance(ys[0], (Sequence, np.ndarray)):
            ys = (ys,)
        y_data = {f"y{i}": np.empty(0, np.float32) for i in range(len(ys))}
        self._table = pyvista.Table({"x": np.empty(0, np.float32), **y_data})
        self.SetInputData(self._table, "x", "y0")
        for i in range(1, len(ys)):
            self.SetInputArray(i + 1, f"y{i}")
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

        ys : sequence[array_like]
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
    size : sequence[float], default: (1, 1)
        Size of the chart in normalized coordinates. A size of ``(0,
        0)`` is invisible, a size of ``(1, 1)`` occupies the whole
        renderer's width and height.

    loc : sequence[float], default: (0, 0)
        Location of the chart (its bottom left corner) in normalized
        coordinates. A location of ``(0, 0)`` corresponds to the
        renderer's bottom left corner, a location of ``(1, 1)``
        corresponds to the renderer's top right corner.

    x_label : str, default: "x"
        Label along the x-axis.

    y_label : str, default: "y"
        Label along the y-axis.

    grid : bool, default: True
        Show the background grid in the plot.

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

    Combine multiple types of plots in the same chart.

    >>> rng = np.random.default_rng(1)
    >>> x = np.arange(1, 8)
    >>> y = rng.integers(5, 15, 7)
    >>> e = np.abs(rng.normal(scale=2, size=7))
    >>> z = rng.integers(0, 5, 7)
    >>> chart = pyvista.Chart2D()
    >>> _ = chart.area(x, y-e, y+e, color=(0.12, 0.46, 0.71, 0.2))
    >>> _ = chart.line(x, y, color="tab:blue", style="--", label="Scores")
    >>> _ = chart.scatter(x, y, color="tab:blue", style="d")
    >>> _ = chart.bar(x, z, color="tab:orange", label="Violations")
    >>> chart.x_axis.tick_locations = x
    >>> chart.x_axis.tick_labels = ["Mon", "Tue", "Wed", "Thu", "Fri",
    ...                             "Sat", "Sun"]
    >>> chart.x_label = "Day of week"
    >>> chart.show()

    """

    PLOT_TYPES = {
        "scatter": ScatterPlot2D,
        "line": LinePlot2D,
        "area": AreaPlot,
        "bar": BarPlot,
        "stack": StackPlot,
    }
    _PLOT_CLASSES = {plot_class: plot_type for (plot_type, plot_class) in PLOT_TYPES.items()}
    _DOC_SUBS = {
        "chart_name": "2D chart",
        "chart_args": "",
        "chart_init": """
        >>> plot = chart.line([0, 1, 2], [2, 1, 3])""",
        "chart_set_labels": 'plot.label = "My awesome plot"',
    }

    def __init__(self, size=(1, 1), loc=(0, 0), x_label="x", y_label="y", grid=True):
        """Initialize the chart."""
        super().__init__(size, loc)
        self._plots = {plot_type: [] for plot_type in self.PLOT_TYPES.keys()}
        self.SetAutoSize(False)  # We manually set the appropriate size
        # Overwrite custom X and Y axis using a wrapper object, as using the
        # SetAxis method causes a crash at the end of the script's execution (nonzero exit code).
        self._x_axis = Axis(_wrap=self.GetAxis(_vtk.vtkAxis.BOTTOM))
        self._y_axis = Axis(_wrap=self.GetAxis(_vtk.vtkAxis.LEFT))
        # Note: registering the axis prevents the nonzero exit code at the end, however
        # this results in memory leaks in the plotting tests.
        # self.SetAxis(_vtk.vtkAxis.BOTTOM, self._x_axis)
        # self.SetAxis(_vtk.vtkAxis.LEFT, self._y_axis)
        # self.Register(self._x_axis)
        # self.Register(self._y_axis)
        self.x_label = x_label
        self.y_label = y_label
        self.grid = grid
        self.legend_visible = True

    def _render_event(self, *args, plotter_render=False, **kwargs):
        if plotter_render:
            # TODO: should probably be called internally by VTK when plot data or axis behavior/logscale is changed?
            self.RecalculateBounds()
        super()._render_event(*args, plotter_render=plotter_render, **kwargs)

    def _add_plot(self, plot_type, *args, **kwargs):
        """Add a plot of the given type to this chart."""
        plot = self.PLOT_TYPES[plot_type](self, *args, **kwargs)
        self.AddPlot(plot)
        self._plots[plot_type].append(plot)
        return plot

    @classmethod
    def _parse_format(cls, fmt):
        """Parse a format string and separate it into a marker style, line style and color.

        Parameters
        ----------
        fmt : str
            Format string to parse. A format string consists of any
            combination of a valid marker style, a valid line style
            and parsable color. The specific order does not
            matter. See :attr:`pyvista.ScatterPlot2D.MARKER_STYLES`
            for a list of valid marker styles,
            :attr:`pyvista.Pen.LINE_STYLES` for a list of valid line
            styles and :class:`pyvista.Color` for an overview of
            parsable colors.

        Returns
        -------
        marker_style : str
            Extracted marker style (empty string if no marker style
            was present in the format string).

        line_style : str
            Extracted line style (empty string if no line style was
            present in the format string).

        color : str
            Extracted color string (defaults to ``"b"`` if no color
            was present in the format string).

        Examples
        --------
        >>> import pyvista
        >>> m, l, c = pyvista.Chart2D._parse_format("x--b")

        """
        marker_style = ""
        line_style = ""
        color = None
        # Note: All colors, marker styles and line styles are sorted in decreasing order of length to be able to find
        # the largest match first (e.g. find 'darkred' and '--' first instead of 'red' and '-')
        colors = sorted(
            itertools.chain(hexcolors.keys(), color_synonyms.keys()),
            key=len,
            reverse=True,
        )
        marker_styles = sorted(ScatterPlot2D.MARKER_STYLES.keys(), key=len, reverse=True)
        line_styles = sorted(Pen.LINE_STYLES.keys(), key=len, reverse=True)
        hex_pattern = "(#|0x)[A-Fa-f0-9]{6}([A-Fa-f0-9]{2})?"  # Match RGB(A) hex string
        # Extract color from format string
        match = re.search(hex_pattern, fmt)  # Start with matching hex strings
        if match is not None:
            color = match.group()
        else:  # Proceed with matching color strings
            for c in colors:
                if c in fmt:
                    color = c
                    break
        if color is not None:
            fmt = fmt.replace(color, "", 1)  # Remove found color from format string
        else:
            color = "b"
        # Extract marker style from format string
        for style in marker_styles[:-1]:  # Last style is empty string
            if style in fmt:
                marker_style = style
                fmt = fmt.replace(
                    marker_style, "", 1
                )  # Remove found marker_style from format string
                break
        # Extract line style from format string
        for style in line_styles[:-1]:  # Last style is empty string
            if style in fmt:
                line_style = style
                fmt = fmt.replace(line_style, "", 1)  # Remove found line_style from format string
                break
        return marker_style, line_style, color

    def plot(self, x, y=None, fmt='-'):
        """Matplotlib like plot method.

        Parameters
        ----------
        x : array_like
            Values to plot on the X-axis. In case ``y`` is ``None``,
            these are the values to plot on the Y-axis instead.

        y : array_like, optional
            Values to plot on the Y-axis.

        fmt : str, default: "-"
            A format string, e.g. ``'ro'`` for red circles. See the Notes
            section for a full description of the format strings.

        Returns
        -------
        scatter_plot : plotting.charts.ScatterPlot2D, optional
            The created scatter plot when a valid marker style
            was present in the format string, ``None`` otherwise.

        line_plot : plotting.charts.LinePlot2D, optional
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
        >>> chart.show()

        Generate a line and scatter plot.

        >>> chart = pyvista.Chart2D()
        >>> scatter_plot, line_plot = chart.plot(range(10), fmt='o-')
        >>> chart.show()

        """
        if y is None:
            y = x
            x = np.arange(len(y))
        elif isinstance(y, str):
            fmt = y
            y = x
            x = np.arange(len(y))
        marker_style, line_style, color = self._parse_format(fmt)
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

        color : ColorLike, default: "b"
            Color of the points drawn in this plot. Any color parsable
            by :class:`pyvista.Color` is allowed.

        size : float, default: 10
            Size of the point markers drawn in this plot.

        style : str, default: "o"
            Style of the point markers drawn in this plot. See
            :ref:`ScatterPlot2D.MARKER_STYLES <scatter_marker_styles>`
            for a list of allowed marker styles.

        label : str, default: ""
            Label of this plot, as shown in the chart's legend.

        Returns
        -------
        plotting.charts.ScatterPlot2D
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

        color : ColorLike, default: "b"
            Color of the line drawn in this plot. Any color parsable
            by :class:`pyvista.Color` is allowed.

        width : float, default: 1
            Width of the line drawn in this plot.

        style : str, default: "-"
            Style of the line drawn in this plot. See
            :ref:`Pen.LINE_STYLES <pen_line_styles>` for a list of
            allowed line styles.

        label : str, default: ""
            Label of this plot, as shown in the chart's legend.

        Returns
        -------
        plotting.charts.LinePlot2D
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
            Y coordinates of the points on the second outline of the
            area to draw. Defaults to ``np.zeros_like(x)``.

        color : ColorLike, default: "b"
            Color of the area drawn in this plot. Any color parsable
            by :class:`pyvista.Color` is allowed.

        label : str, default: ""
            Label of this plot, as shown in the chart's legend.

        Returns
        -------
        plotting.charts.AreaPlot
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

    def bar(self, x, y, color=None, orientation="V", label=None):
        """Add a bar plot to this chart.

        Parameters
        ----------
        x : array_like
            Positions (along the x-axis for a vertical orientation,
            along the y-axis for a horizontal orientation) of the bars
            to draw.

        y : array_like
            Size of the bars to draw. Multiple bars can be stacked by
            passing a sequence of sequences.

        color : ColorLike, default: "b"
            Color of the bars drawn in this plot. Any color parsable
            by :class:`pyvista.Color` is allowed.

        orientation : str, default: "V"
            Orientation of the bars drawn in this plot. Either ``"H"``
            for an horizontal orientation or ``"V"`` for a vertical
            orientation.

        label : str, default: ""
            Label of this plot, as shown in the chart's legend.

        Returns
        -------
        plotting.charts.BarPlot
            The created bar plot.

        Examples
        --------
        Generate a bar plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> plot = chart.bar([0, 1, 2], [2, 1, 3])
        >>> chart.show()

        """
        return self._add_plot("bar", x, y, color=color, orientation=orientation, label=label)

    def stack(self, x, ys, colors=None, labels=None):
        """Add a stack plot to this chart.

        Parameters
        ----------
        x : array_like
            X coordinates of the points outlining the stacks (areas) to draw.

        ys : sequence[array_like]
            Size of the stacks (areas) to draw at the corresponding X
            coordinates. Each sequence defines the sizes of one stack
            (area), which are stacked on top of each other.

        colors : sequence[ColorLike], optional
            Color of the stacks (areas) drawn in this plot. Any color
            parsable by :class:`pyvista.Color` is allowed.

        labels : sequence[str], default: []
            Label for each stack (area) drawn in this plot, as shown
            in the chart's legend.

        Returns
        -------
        plotting.charts.StackPlot
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
            If no type is provided (``None``), all plots are returned,
            regardless of their type.

        Yields
        ------
        plot : plotting.charts.ScatterPlot2D | plotting.charts.LinePlot2D | plotting.charts.AreaPlot | plotting.charts.BarPlot | plotting.charts.StackPlot
            One of the plots (of the specified type) in this chart.

        Examples
        --------
        Create a 2D chart with a line and scatter plot.

        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> scatter_plot, line_plot = chart.plot([0, 1, 2], [2, 1, 3], "o-")
        >>> chart.show()

        Retrieve all plots in the chart.

        >>> plots = [*chart.plots()]
        >>> scatter_plot in plots and line_plot in plots
        True

        Retrieve all line plots in the chart.

        >>> line_plots = [*chart.plots("line")]
        >>> line_plot == line_plots[0]
        True

        """
        plot_types = self.PLOT_TYPES.keys() if plot_type is None else [plot_type]
        for plot_type in plot_types:
            yield from self._plots[plot_type]

    def remove_plot(self, plot):
        """Remove the given plot from this chart.

        Parameters
        ----------
        plot : plotting.charts.ScatterPlot2D | plotting.charts.LinePlot2D | plotting.charts.AreaPlot | plotting.charts.BarPlot | plotting.charts.StackPlot
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

            If no type is provided (``None``), all plots are removed,
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
        plot_types = self.PLOT_TYPES.keys() if plot_type is None else [plot_type]
        for plot_type in plot_types:
            # Make a copy, as this list will be modified by remove_plot
            plots = [*self._plots[plot_type]]
            for plot in plots:
                self.remove_plot(plot)

    @property
    def x_axis(self):
        """Return this chart's horizontal (x) :class:`Axis <plotting.charts.Axis>`.

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
        """Return this chart's vertical (y) :class:`Axis <plotting.charts.Axis>`.

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
        Create a 2D chart with the grid disabled.

        >>> import pyvista
        >>> import numpy as np
        >>> x = np.linspace(0, 2*np.pi, 20)
        >>> y = np.sin(x)
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.line(x, y, 'r')
        >>> chart.grid = False
        >>> chart.show()

        Enable the grid

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

    Users should typically not directly create new plot instances, but
    use the dedicated ``ChartBox`` class.

    Parameters
    ----------
    chart : ChartBox
        The chart containing this plot.

    data : sequence[array_like]
        Dataset(s) from which the relevant statistics will be
        calculated used to draw the box plot.

    colors : sequence[ColorLike], optional
        Color of the boxes drawn in this plot. Any color parsable by
        :class:`pyvista.Color` is allowed. If omitted (``None``), the
        default color scheme is used.

    labels : sequence[str], default: []
        Label for each box drawn in this plot, as shown in the chart's
        legend.

    Examples
    --------
    Create boxplots for datasets sampled from shifted normal distributions.

    >>> import pyvista
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)  # Seeded random number generator used for data generation
    >>> normal_data = [rng.normal(i, size=50) for i in range(5)]
    >>> chart = pyvista.ChartBox(normal_data, labels=[f"x ~ N({i},1)" for i in range(5)])
    >>> chart.show()

    """

    _DOC_SUBS = {
        "plot_name": "box plot",
        "chart_init": "pyvista.ChartBox([[0, 1, 1, 2, 3, 3, 4]])",
        "plot_init": "chart.plot",
        "multichart_init": "pyvista.ChartBox([[0, 1, 1, 2, 3, 4, 5], [0, 1, 2, 2, 3, 4, 5], [0, 1, 2, 3, 3, 4, 5], [0, 1, 2, 3, 4, 4, 5]])",
        "multiplot_init": "chart.plot",
    }

    def __init__(self, chart, data, colors=None, labels=None):
        """Initialize a new box plot instance."""
        super().__init__(chart)
        self._table = pyvista.Table(
            {f"data_{i}": np.array(d, copy=False) for i, d in enumerate(data)}
        )
        self._quartiles = _vtk.vtkComputeQuartiles()
        self._quartiles.SetInputData(self._table)
        self.SetInputData(self._quartiles.GetOutput())
        self.update(data)
        self.SetLookupTable(self._lookup_table)
        self.colors = colors
        self.labels = labels

    @property
    def data(self):
        """Retrieve the datasets of which the boxplots are drawn.

        Examples
        --------
        Create a box plot and display the datasets.

        >>> import pyvista
        >>> chart = pyvista.ChartBox([[0, 1, 1, 2, 3, 3, 4]])
        >>> chart.plot.data
        (pyvista_ndarray([0, 1, 1, 2, 3, 3, 4]),)
        >>> chart.show()

        """
        return tuple(self._table[f"data_{i}"] for i in range(self._table.n_arrays))

    @property
    def stats(self):
        """Retrieve the statistics (quartiles and extremum values) of the datasets of which the boxplots are drawn.

        Examples
        --------
        Create a box plot and display the statistics.

        >>> import pyvista
        >>> chart = pyvista.ChartBox([[0, 1, 1, 2, 3, 3, 4]])
        >>> chart.plot.stats
        (pyvista_ndarray([0., 1., 2., 3., 4.]),)
        >>> chart.show()

        """
        stats_table = pyvista.Table(self._quartiles.GetOutput())
        return tuple(stats_table[f"data_{i}"] for i in range(stats_table.n_arrays))

    def update(self, data):
        """Update the plot's underlying dataset(s).

        Parameters
        ----------
        data : sequence[array_like]
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
        self._table.update({f"data_{i}": np.array(d, copy=False) for i, d in enumerate(data)})
        self._quartiles.Update()


class ChartBox(_vtk.vtkChartBox, _Chart):
    """Dedicated chart for drawing box plots.

    Parameters
    ----------
    data : sequence[array_like]
        Dataset(s) from which the relevant statistics will be
        calculated used to draw the box plot.

    colors : sequence[ColorLike], optional
        Color used for each drawn boxplot. If omitted (``None``), the
        default color scheme is used.

    labels : sequence[str], default: []
        Label for each drawn boxplot, as shown in the chart's
        legend.

    size : sequence[float], optional
        Size of the chart in normalized coordinates. A size of ``(0,
        0)`` is invisible, a size of ``(1, 1)`` occupies the whole
        renderer's width and height.

    loc : sequence[float], optional
        Location of the chart (its bottom left corner) in normalized
        coordinates. A location of ``(0, 0)`` corresponds to the
        renderer's bottom left corner, a location of ``(1, 1)``
        corresponds to the renderer's top right corner.

    Examples
    --------
    Create boxplots for datasets sampled from shifted normal distributions.

    >>> import pyvista
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)  # Seeded random number generator used for data generation
    >>> normal_data = [rng.normal(i, size=50) for i in range(5)]
    >>> chart = pyvista.ChartBox(normal_data, labels=[f"x ~ N({i},1)" for i in range(5)])
    >>> chart.show()

    """

    _DOC_SUBS = {
        "chart_name": "boxplot chart",
        "chart_args": "[[0, 1, 1, 2, 3, 3, 4]]",
        "chart_init": "",
        "chart_set_labels": 'chart.plot.label = "Data label"',
    }

    def __init__(self, data, colors=None, labels=None, size=None, loc=None):
        """Initialize a new chart containing box plots."""
        if vtk_version_info >= (9, 2, 0):
            self.SetAutoSize(False)  # We manually set the appropriate size
            if size is None:
                size = (1, 1)
            if loc is None:
                loc = (0, 0)
        super().__init__(size, loc)
        self._plot = BoxPlot(self, data, colors, labels)
        self.SetPlot(self._plot)
        self.SetColumnVisibilityAll(True)
        self.legend_visible = True

    def _render_event(self, *args, **kwargs):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            # In older VTK versions, ChartBox fills the entire scene, so
            # no resizing is needed (nor possible).
            pass
        else:
            super()._render_event(*args, **kwargs)

    @property
    def _geometry(self):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            return (0, 0, *self._renderer.GetSize())
        else:
            return _Chart._geometry.fget(self)

    @_geometry.setter
    def _geometry(self, value):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            raise AttributeError(f'Cannot set the geometry of {type(self).__class__}')
        else:
            _Chart._geometry.fset(self, value)

    @property
    def plot(self):
        """Return the :class:`BoxPlot <plotting.charts.BoxPlot>` instance associated with this chart.

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

        Notes
        -----
        Customisable ChartBox geometry is only supported in VTK v9.2
        or newer. For older VTK versions, the size cannot be modified,
        filling up the entire viewport by default.

        Examples
        --------
        Create a half-sized boxplot chart centered in the middle of the
        renderer.

        >>> import pyvista
        >>> chart = pyvista.ChartBox([[0, 1, 1, 2, 3, 3, 4]])
        >>> chart.size = (0.5, 0.5)
        >>> chart.loc = (0.25, 0.25)
        >>> chart.show()

        """
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            return (1, 1)
        else:
            return _Chart.size.fget(self)

    @size.setter
    def size(self, val):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            raise ValueError(
                "Cannot set ChartBox geometry, it fills up the entire viewport by default. "
                "Upgrade to VTK v9.2 or newer."
            )
        else:
            _Chart.size.fset(self, val)

    @property
    def loc(self):
        """Return or set the chart position in normalized coordinates.

        This denotes the location of the chart's bottom left corner.

        Notes
        -----
        Customisable ChartBox geometry is only supported in VTK v9.2
        or newer. For older VTK versions, the location cannot be modified,
        filling up the entire viewport by default.

        Examples
        --------
        Create a half-sized boxplot chart centered in the middle of the
        renderer.

        >>> import pyvista
        >>> chart = pyvista.ChartBox([[0, 1, 1, 2, 3, 3, 4]])
        >>> chart.size = (0.5, 0.5)
        >>> chart.loc = (0.25, 0.25)
        >>> chart.show()

        """
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            return (0, 0)
        else:
            return _Chart.loc.fget(self)

    @loc.setter
    def loc(self, val):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            raise ValueError(
                "Cannot set ChartBox geometry, it fills up the entire viewport by default. "
                "Upgrade to VTK v9.2 or newer."
            )
        else:
            _Chart.loc.fset(self, val)


class PiePlot(_vtkWrapper, _vtk.vtkPlotPie, _MultiCompPlot):
    """Class representing a pie plot.

    Users should typically not directly create new plot instances, but
    use the dedicated :class:`ChartPie` class.

    Parameters
    ----------
    chart : ChartPie
        The chart containing this plot.

    data : array_like
        Relative size of each pie segment.

    colors : sequence[ColorLike], optional
        Color of the segments drawn in this plot. Any color parsable
        by :class:`pyvista.Color` is allowed. If omitted (``None``),
        the default color scheme is used.

    labels : sequence[str], default: []
        Label for each pie segment drawn in this plot, as shown in the
        chart's legend.

    Other Parameters
    ----------------
    _wrap : vtk.vtkPlotPie, optional
        Wrap an existing VTK PlotPie instance (no wrapping when ``None``).

    Examples
    --------
    Create a pie plot showing the usage of tax money.

    >>> import pyvista
    >>> x = [128.3, 32.9, 31.8, 29.3, 21.2]
    >>> l = ["Social benefits", "Governance", "Economic policy", "Education", "Other"]
    >>> chart = pyvista.ChartPie(x, labels=l)
    >>> chart.show()

    """

    _DOC_SUBS = {
        "plot_name": "pie plot",
        "chart_init": "pyvista.ChartPie([4, 3, 2, 1])",
        "plot_init": "chart.plot",
        "multichart_init": "pyvista.ChartPie([4, 3, 2, 1])",
        "multiplot_init": "chart.plot",
    }

    def __init__(self, chart, data, colors=None, labels=None):
        """Initialize a new pie plot instance."""
        super().__init__(chart)
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

    colors : sequence[ColorLike], optional
        Color used for each pie segment drawn in this plot. If
        omitted (``None``), the default color scheme is used.

    labels : sequence[str], default: []
        Label for each pie segment drawn in this plot, as shown in the
        chart's legend.

    size : sequence[float], optional
        Size of the chart in normalized coordinates. A size of ``(0,
        0)`` is invisible, a size of ``(1, 1)`` occupies the whole
        renderer's width and height.

    loc : sequence[float], optional
        Location of the chart (its bottom left corner) in normalized
        coordinates. A location of ``(0, 0)`` corresponds to the
        renderer's bottom left corner, a location of ``(1, 1)``
        corresponds to the renderer's top right corner.

    Examples
    --------
    Create a pie plot showing the usage of tax money.

    >>> import pyvista
    >>> x = [128.3, 32.9, 31.8, 29.3, 21.2]
    >>> l = ["Social benefits", "Governance", "Economic policy", "Education", "Other"]
    >>> chart = pyvista.ChartPie(x, labels=l)
    >>> chart.show()

    """

    _DOC_SUBS = {
        "chart_name": "pie chart",
        "chart_args": "[5, 4, 3, 2, 1]",
        "chart_init": "",
        "chart_set_labels": 'chart.plot.labels = ["A", "B", "C", "D", "E"]',
    }

    def __init__(self, data, colors=None, labels=None, size=None, loc=None):
        """Initialize a new chart containing a pie plot."""
        if vtk_version_info >= (9, 2, 0):
            self.SetAutoSize(False)  # We manually set the appropriate size
            if size is None:
                size = (1, 1)
            if loc is None:
                loc = (0, 0)
        super().__init__(size, loc)
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            # SetPlot method is not available for older VTK versions,
            # so fallback to using a wrapper object.
            self.AddPlot(0)
            self._plot = PiePlot(self, data, colors, labels, _wrap=self.GetPlot(0))
        else:
            self._plot = PiePlot(self, data, colors, labels)
            self.SetPlot(self._plot)
        self.legend_visible = True

    def _render_event(self, *args, **kwargs):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            # In older VTK versions, ChartPie fills the entire scene, so
            # no resizing is needed (nor possible).
            pass
        else:
            super()._render_event(*args, **kwargs)

    @property
    def _geometry(self):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            return (0, 0, *self._renderer.GetSize())
        else:
            return _Chart._geometry.fget(self)

    @_geometry.setter
    def _geometry(self, value):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            raise AttributeError(f'Cannot set the geometry of {type(self).__class__}')
        else:
            _Chart._geometry.fset(self, value)

    @property
    def plot(self):
        """Return the :class:`PiePlot <plotting.charts.PiePlot>` instance associated with this chart.

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

        Notes
        -----
        Customisable ChartPie geometry is only supported in VTK v9.2
        or newer. For older VTK versions, the size cannot be modified,
        filling up the entire viewport by default.

        Examples
        --------
        Create a half-sized pie chart centered in the middle of the
        renderer.

        >>> import pyvista
        >>> chart = pyvista.ChartPie([5, 4, 3, 2, 1])
        >>> chart.size = (0.5, 0.5)
        >>> chart.loc = (0.25, 0.25)
        >>> chart.show()

        """
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            return (1, 1)
        else:
            return _Chart.size.fget(self)

    @size.setter
    def size(self, val):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            raise ValueError(
                "Cannot set ChartPie geometry, it fills up the entire viewport by default. "
                "Upgrade to VTK v9.2 or newer."
            )
        else:
            _Chart.size.fset(self, val)

    @property
    def loc(self):
        """Return or set the chart position in normalized coordinates.

        This denotes the location of the chart's bottom left corner.

        Notes
        -----
        Customisable ChartPie geometry is only supported in VTK v9.2
        or newer. For older VTK versions, the location cannot be modified,
        filling up the entire viewport by default.

        Examples
        --------
        Create a half-sized pie chart centered in the middle of the
        renderer.

        >>> import pyvista
        >>> chart = pyvista.ChartPie([5, 4, 3, 2, 1])
        >>> chart.size = (0.5, 0.5)
        >>> chart.loc = (0.25, 0.25)
        >>> chart.show()

        """
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            return (0, 0)
        else:
            return _Chart.loc.fget(self)

    @loc.setter
    def loc(self, val):
        if vtk_version_info < (9, 2, 0):  # pragma: no cover
            raise ValueError(
                "Cannot set ChartPie geometry, it fills up the entire viewport by default. "
                "Upgrade to VTK v9.2 or newer."
            )
        else:
            _Chart.loc.fset(self, val)


# region 3D charts
# A basic implementation of 3D line, scatter and volume plots, to be used in a 3D chart was provided in this section
# but removed in commit 8ef8daea5d105e85f256d4e9af584aeea3c85040 of PR #1432. Unfortunately, these charts are much less
# customisable than their 2D counterparts and they do not respect the enforced size/geometry constraints once you start
# interacting with them.
# endregion


class ChartMPL(_vtk.vtkImageItem, _Chart):
    """Create new chart from an existing matplotlib figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure, optional
        The matplotlib figure to draw. If no figure is
        provided ( ``None`` ), a new figure is created.

    size : sequence[float], default: (1, 1)
        Size of the chart in normalized coordinates. A size of ``(0,
        0)`` is invisible, a size of ``(1, 1)`` occupies the whole
        renderer's width and height.

    loc : sequence[float], default: (0, 0)
        Location of the chart (its bottom left corner) in normalized
        coordinates. A location of ``(0, 0)`` corresponds to the
        renderer's bottom left corner, a location of ``(1, 1)``
        corresponds to the renderer's top right corner.

    redraw_on_render : bool, default: True
        Flag indicating whether the chart should be redrawn when
        the plotter is rendered. For static charts, setting this
        to ``False`` can improve performance.

    Examples
    --------
    Plot streamlines of a vector field with varying colors (based on `this example <https://matplotlib.org/stable/gallery/images_contours_and_fields/plot_streamplot.html>`_).

    .. pyvista-plot::

    >>> import pyvista
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> w = 3
    >>> Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    >>> U = -1 - X**2 + Y
    >>> V = 1 + X - Y**2
    >>> speed = np.sqrt(U**2 + V**2)

    >>> f, ax = plt.subplots()
    >>> strm = ax.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')
    >>> _ = f.colorbar(strm.lines)
    >>> _ = ax.set_title('Streamplot with varying Color')
    >>> plt.tight_layout()

    >>> chart = pyvista.ChartMPL(f)
    >>> chart.show()

    """

    _DOC_SUBS = {
        "chart_name": "matplotlib chart",
        "chart_args": "",
        "chart_init": """
        >>> plots = chart.figure.axes[0].plot([0, 1, 2], [2, 1, 3])""",
        "chart_set_labels": 'plots[0].label = "My awesome plot"',
    }

    def __init__(self, figure=None, size=(1, 1), loc=(0, 0), redraw_on_render=True):
        """Initialize chart."""
        super().__init__(size, loc)
        if figure is None:
            figure, _ = plt.subplots()
        self._fig = figure
        self._canvas = FigureCanvasAgg(
            self._fig
        )  # Switch backends and store reference to figure's canvas
        # Make figure and axes fully transparent, as the background is already dealt with by self._background.
        self._fig.patch.set_alpha(0)
        for ax in self._fig.axes:
            ax.patch.set_alpha(0)
        self._canvas.mpl_connect('draw_event', self._redraw)  # Attach 'draw_event' callback
        self._redraw_on_render = redraw_on_render

        self._redraw()

        # Close the underlying matplotlib figure when creating the sphinx gallery.
        # This prevents the charts from being drawn twice in example scripts:
        # once as a pyvista plot (fetched by the 'pyvista' scraper) and once as a
        # matplotlib figure (fetched by the 'matplotlib' scraper).
        # See #1999 and #2031.
        if pyvista.BUILDING_GALLERY:  # pragma: no cover
            plt.close(self._fig)

    @property
    def figure(self):
        """Retrieve the matplotlib figure associated with this chart.

        Examples
        --------
        Create a matplotlib chart from an existing figure.

        .. pyvista-plot::

        >>> import pyvista
        >>> import matplotlib.pyplot as plt
        >>> f, ax = plt.subplots()
        >>> _ = ax.plot([0, 1, 2], [2, 1, 3])
        >>> chart = pyvista.ChartMPL(f)
        >>> chart.figure is f
        True
        >>> chart.show()
        """
        return self._fig

    @property
    def redraw_on_render(self):
        """Return or set the chart's redraw-on-render behavior.

        Notes
        -----
        When disabled, the chart will only be redrawn when the
        Plotter window is resized or the matplotlib figure is
        manually redrawn using ``fig.canvas.draw()``.
        When enabled, the chart will also be automatically
        redrawn whenever the Plotter is rendered using
        ``plotter.render()``.

        """
        return self._redraw_on_render

    @redraw_on_render.setter
    def redraw_on_render(self, val):
        self._redraw_on_render = bool(val)

    def _resize(self):
        r_w, r_h = self._renderer.GetSize()
        c_w, c_h = (int(s) for s in self._canvas.get_width_height())
        # Calculate target size from specified normalized width and height and the renderer's current size
        t_w = int(self._size[0] * r_w)
        t_h = int(self._size[1] * r_h)
        resize = c_w != t_w or c_h != t_h
        if resize:
            # Mismatch between canvas size and target size, so resize figure:
            f_w = t_w / self._fig.dpi
            f_h = t_h / self._fig.dpi
            self._fig.set_size_inches(f_w, f_h)
            self.position = (int(self._loc[0] * r_w), int(self._loc[1] * r_h))
        return resize

    def _redraw(self, event=None):
        """Redraw the chart."""
        if event is None:
            # Manual call, so make sure canvas is redrawn first (which will callback to _redraw with a proper event defined)
            self._canvas.draw()
        else:
            # Called from draw_event callback
            img = np.frombuffer(
                self._canvas.buffer_rgba(), dtype=np.uint8
            )  # Store figure data in numpy array
            w, h = self._canvas.get_width_height()
            img_arr = img.reshape([h, w, 4])
            img_data = pyvista.Texture(img_arr).to_image()  # Convert to vtkImageData
            self.SetImage(img_data)

    def _render_event(self, *args, plotter_render=False, **kwargs):
        # Redraw figure when geometry has changed (self._resize call
        # already updated figure dimensions in that case) OR the
        # plotter's render method was called and redraw_on_render is
        # enabled.
        if (plotter_render and self.redraw_on_render) or (not plotter_render and self._resize()):
            self._redraw()

    @property
    def _geometry(self):
        r_w, r_h = self._renderer.GetSize()
        t_w = self._size[0] * r_w
        t_h = self._size[1] * r_h
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
    #     color = Color(val).int_rgba if val is not None else [1.0, 1.0, 1.0, 1.0]
    #     opacity = color[3]
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
        if len(val) != 2:
            raise ValueError(f'Invalid position {val}, must be length 2.')
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

    Users should typically not directly create new instances of this
    class, but use the dedicated ``Plotter.add_chart`` method.

    """

    def __init__(self, renderer):
        """Create a new collection of charts for the given renderer."""
        self._charts = []

        # Postpone creation of scene and actor objects until they are
        # needed.
        self._scene = None
        self._actor = None

        # a weakref.proxy would be nice here, but that doesn't play
        # nicely with SetRenderer, so instead we'll use a weak reference
        # plus a property to call it
        self.__renderer = weakref.ref(renderer)

    @property
    def _renderer(self):
        """Return the weakly dereferenced renderer, maybe None."""
        return self.__renderer()

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
            charts = [*self._charts]  # Make a copy, as this list will be modified by remove_chart
            for chart in charts:
                self.remove_chart(chart)
            if self._renderer is not None:
                self._renderer.RemoveActor(self._actor)
        self._scene = None
        self._actor = None

    def add_chart(self, *charts):
        """Add charts to the collection."""
        if self._scene is None:
            self._setup_scene()
        for chart in charts:
            self._charts.append(chart)
            if chart._background is not None:
                self._scene.AddItem(chart._background)
            self._scene.AddItem(chart)
            chart._interactive = False  # Charts are not interactive by default

    def set_interaction(self, interactive, toggle=False):
        """
        Set or toggle interaction with charts for this renderer.

        Interaction with other charts in this renderer is disabled when ``toggle``
        is ``False``.

        Parameters
        ----------
        interactive : bool | Chart | int | list[Chart] | list[int]
            Following parameter values are accepted:

            * A boolean to enable (``True``) or disable (``False``) interaction
              with all charts.
            * The chart or its index to enable interaction with. Interaction
              with multiple charts can be enabled by passing a list of charts
              or indices.

        toggle : bool, default: False
            Instead of enabling interaction with the provided chart(s), interaction
            with the provided chart(s) is toggled. Only applicable when ``interactive``
            is not a boolean.

        Returns
        -------
        list of Chart
            The list of all interactive charts for this renderer.

        """
        if isinstance(interactive, bool):
            # Disable toggle and convert to list of charts
            toggle = False
            interactive = self._charts if interactive else []
        if not isinstance(interactive, list):
            # Convert single chart parameter to list
            interactive = [interactive]
        # Convert to list of Charts
        charts = [
            self._charts[coi] if isinstance(coi, int) and 0 <= coi < len(self) else coi
            for coi in interactive
        ]
        interactive_charts = []

        for chart in self._charts:
            # Determine whether to enable interaction with the current chart.
            if toggle:
                enable = not chart._interactive if chart in charts else chart._interactive
            else:
                enable = chart in charts

            chart._interactive = enable
            if enable:
                interactive_charts.append(chart)

        return interactive_charts

    def remove_chart(self, chart_or_index):
        """Remove a chart from the collection."""
        chart = self._charts[chart_or_index] if isinstance(chart_or_index, int) else chart_or_index
        if chart not in self._charts:  # pragma: no cover
            raise ValueError('chart_index not present in charts collection.')
        self._charts.remove(chart)
        self._scene.RemoveItem(chart)
        if chart._background is not None:
            self._scene.RemoveItem(chart._background)

    def get_charts_by_pos(self, pos):
        """Retrieve visible charts indicated by the given mouse position.

        Parameters
        ----------
        pos : sequence[float]
            Tuple containing the mouse position.

        Returns
        -------
        list of Chart
            Visible charts indicated by the given mouse position.

        """
        return [chart for chart in self._charts if chart.visible and chart._is_within(pos)]

    def __getitem__(self, index) -> Chart:
        """Return a chart based on an index."""
        return self._charts[index]

    def __len__(self):
        """Return number of charts."""
        return len(self._charts)

    def __iter__(self):
        """Return an iterable of charts."""
        yield from self._charts

    def __del__(self):
        """Clean up before being destroyed."""
        self.deep_clean()
