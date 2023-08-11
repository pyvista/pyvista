"""Contains the pyvista.Text class."""
from __future__ import annotations

import pyvista as pv
from pyvista.core.utilities.misc import no_new_attr

from . import _vtk
from .colors import Color


class Text(_vtk.vtkTextActor):
    """Define text by default theme.

    Examples
    --------
    Create a text with text property.

    >>> from pyvista import Text
    >>> text = Text("text")
    >>> prop = text.prop
    """

    def __init__(self, text, prop=None):
        """Initialize a new text descriptor."""
        self._input = text
        if prop is None:
            self.prop = TextProperty()

    @property
    def prop(self):
        """Return or set the property of this actor."""
        return self.GetTextProperty()

    @prop.setter
    def prop(self, obj: TextProperty):
        self.SetTextProperty(obj)


@no_new_attr
class TextProperty(_vtk.vtkTextProperty):
    """Define text property.

    Examples
    --------
    Create a text property.

    >>> from pyvista import TextProperty
    >>> prop = TextProperty()
    >>> prop.opacity = 0.5
    >>> prop.background_color = "b"
    >>> prop.background_opacity = 0.5
    >>> prop.frame = True
    >>> prop.frame_color = "b"
    >>> prop.frame_width = 10.0
    >>> assert prop.color == "b"
    """

    _theme = None
    _color_set = None

    def __init__(self, color=None):
        """Initialize text property."""
        self._theme = pv.themes.Theme()
        self.color = color

    @property
    def color(self) -> Color:
        """Return or set the color of this property.

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

        Visualize setting the property to blue.

        >>> prop.color = 'b'
        >>> prop.plot()

        Visualize setting the color using an RGB value.

        >>> prop.color = (0.5, 0.5, 0.1)
        >>> prop.plot()

        """
        return Color(self.GetColor())

    @color.setter
    def color(self, value):
        self._color_set = value is not None
        rgb_color = Color(value, default_color=self._theme.color)
        self.SetColor(rgb_color.float_rgb)
