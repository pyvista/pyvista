"""Contains the pyvista.Text class."""
from __future__ import annotations

from . import _vtk


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
