"""Contains the pyvista.Text class."""
from __future__ import annotations

import pyvista as pv
from pyvista.core.utilities.misc import _check_range, no_new_attr

from . import _vtk
from .colors import Color
from .themes import Theme
from .tools import FONTS


@no_new_attr
class CornerAnnotation(_vtk.vtkCornerAnnotation):
    """text annotation in four corners.

    This is an annotation object that manages four text actors / mappers to provide annotation in the four corners of a viewport.

    Examples
    --------
    Create text annotation in four corners.

    >>> from pyvista import CornerAnnotation
    >>> text = CornerAnnotation()
    >>> prop = text.prop
    """

    def __init__(self, prop=None):
        """Initialize a new text annotation descriptor."""
        super().__init__()
        if prop is None:
            self.prop = TextProperty()

    def get_text(self, position):
        """Get the text to be displayed for each corner."""
        return self.GetText(position)

    def set_text(self, position, obj):
        """Set the text to be displayed for each corner."""
        corner_mappings = {
            'lower_left': self.LowerLeft,
            'lower_right': self.LowerRight,
            'upper_left': self.UpperLeft,
            'upper_right': self.UpperRight,
            'lower_edge': self.LowerEdge,
            'upper_edge': self.UpperEdge,
            'left_edge': self.LeftEdge,
            'right_edge': self.RightEdge,
        }
        corner_mappings['ll'] = corner_mappings['lower_left']
        corner_mappings['lr'] = corner_mappings['lower_right']
        corner_mappings['ul'] = corner_mappings['upper_left']
        corner_mappings['ur'] = corner_mappings['upper_right']
        corner_mappings['top'] = corner_mappings['upper_edge']
        corner_mappings['bottom'] = corner_mappings['lower_edge']
        corner_mappings['right'] = corner_mappings['right_edge']
        corner_mappings['r'] = corner_mappings['right_edge']
        corner_mappings['left'] = corner_mappings['left_edge']
        corner_mappings['l'] = corner_mappings['left_edge']
        if isinstance(position, str):
            position = corner_mappings[position]
        elif position is True:
            position = corner_mappings['upper_left']
        self.SetText(position, obj)

    @property
    def prop(self):
        """Return or set the property of this actor."""
        return self.GetTextProperty()

    @prop.setter
    def prop(self, obj: TextProperty):
        self.SetTextProperty(obj)


@no_new_attr
class Text(_vtk.vtkTextActor):
    """Define text by default theme.

    Examples
    --------
    Create a text with text's property.

    >>> from pyvista import Text
    >>> text = Text()
    >>> prop = text.prop
    """

    def __init__(self, prop=None):
        """Initialize a new text descriptor."""
        super().__init__()
        if prop is None:
            self.prop = TextProperty()

    @property
    def input(self):
        r"""Set the text string to be displayed.

        "\n" is recognized as a carriage return/linefeed (line separator).
        The characters must be in the UTF-8 encoding.
        """
        return self.GetInput()

    @input.setter
    def input(self, obj: str):
        self.SetInput(obj)

    @property
    def prop(self):
        """Return or set the property of this actor."""
        return self.GetTextProperty()

    @prop.setter
    def prop(self, obj: TextProperty):
        self.SetTextProperty(obj)


@no_new_attr
class TextProperty(_vtk.vtkTextProperty):
    """Define text's property.

    Examples
    --------
    Create a text's property.

    >>> from pyvista import TextProperty
    >>> prop = TextProperty()
    >>> prop.opacity = 0.5
    >>> prop.background_color = "b"
    >>> prop.background_opacity = 0.5
    >>> prop.show_frame = True
    >>> prop.frame_color = "b"
    >>> prop.frame_width = 10
    >>> prop.frame_color
    Color(name='blue', hex='#0000ffff', opacity=255)

    """

    _theme = Theme()
    _color_set = None
    _background_color_set = None
    _font_family = None

    def __init__(self, theme=None, color=None):
        """Initialize text's property."""
        super().__init__()
        if theme is None:
            # copy global theme to ensure local property theme is fixed
            # after creation.
            self._theme.load_theme(pv.global_theme)
        else:
            self._theme.load_theme(theme)
        self.color = color

    @property
    def color(self) -> Color:
        """Return or set the color of text's property.

        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

        Examples
        --------
        Set the color to blue.

        >>> import pyvista as pv
        >>> prop = pv.TextProperty()
        >>> prop.color = 'b'
        >>> prop.color
        Color(name='blue', hex='#0000ffff', opacity=255)

        """
        return Color(self.GetColor())

    @color.setter
    def color(self, value):
        self._color_set = value is not None
        rgb_color = Color(value, default_color=self._theme.font.color)
        self.SetColor(rgb_color.float_rgb)

    @property
    def opacity(self) -> float:
        """Return or set the opacity of text's property.

        Opacity of the text. A single float value that will be applied globally
        opacity of the text and uniformly applied everywhere. Between 0 and 1.

        Examples
        --------
        Set opacity to ``0.5``.

        >>> import pyvista as pv
        >>> prop = pv.TextProperty()
        >>> prop.opacity = 0.5
        >>> prop.opacity
        0.5

        """
        return self.GetOpacity()

    @opacity.setter
    def opacity(self, value: float):
        _check_range(value, (0, 1), 'opacity')
        self.SetOpacity(value)

    @property
    def background_color(self):
        """Return or set the background color of text's property.

        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

        Examples
        --------
        Set the background color to blue.

        >>> import pyvista as pv
        >>> prop = pv.TextProperty()
        >>> prop.background_color = 'b'
        >>> prop.background_color
        Color(name='blue', hex='#0000ffff', opacity=255)

        """
        return Color(self.GetBackgroundColor())

    @background_color.setter
    def background_color(self, value):
        self._background_color_set = value is not None
        rgb_color = Color(value)
        self.SetBackgroundColor(rgb_color.float_rgb)

    @property
    def background_opacity(self):
        """Return or set the background opacity of text's property.

        Background opacity of the text. A single float value that will be applied globally
        background opacity of the text and uniformly applied everywhere. Between 0 and 1.

        Examples
        --------
        Set background opacity to ``0.5``.

        >>> import pyvista as pv
        >>> prop = pv.TextProperty()
        >>> prop.background_opacity = 0.5
        >>> prop.background_opacity
        0.5

        """
        return self.GetBackgroundOpacity()

    @background_opacity.setter
    def background_opacity(self, value: float):
        _check_range(value, (0, 1), 'background_opacity')
        self.SetBackgroundOpacity(value)

    @property
    def show_frame(self) -> bool:
        """Return or set the visibility of frame.

        Shows or hides the frame.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.TextProperty()
        >>> prop.show_frame = True
        >>> prop.show_frame
        True

        """
        return bool(self.GetFrame())

    @show_frame.setter
    def show_frame(self, value: bool):
        self.SetFrame(value)

    @property
    def frame_color(self) -> Color:
        """Return or set the frame color of this property.

        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

        Examples
        --------
        Set the frame color to blue.

        >>> import pyvista as pv
        >>> prop = pv.TextProperty()
        >>> prop.frame_color = 'b'
        >>> prop.frame_color
        Color(name='blue', hex='#0000ffff', opacity=255)

        """
        return Color(self.GetFrameColor())

    @frame_color.setter
    def frame_color(self, value):
        self.SetFrameColor(Color(value).float_rgb)

    @property
    def frame_width(self) -> int:
        """Set/Get the width of the frame.

        The width is expressed in pixels. The default is 1 pixel.

        Examples
        --------
        Change the frame width to ``10``.

        >>> import pyvista as pv
        >>> prop = pv.TextProperty()
        >>> prop.frame_width = 10
        >>> prop.frame_width
        10

        """
        return self.GetFrameWidth()

    @frame_width.setter
    def frame_width(self, value: int):
        self.SetFrameWidth(value)

    @property
    def font_family(self) -> str | None:
        """Set/Get the font family."""
        return self._font_family

    @font_family.setter
    def font_family(self, font: str | None):
        if font is None:
            font = self._theme.font.family
        self._font_family = font
        self.SetFontFamily(FONTS[self._font_family].value)
