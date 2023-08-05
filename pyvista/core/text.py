"""Contains the pyvista.Text class."""


class Text(TextActor):
    """Define text by default theme.

    Examples
    --------
    Create a text with text property.

    >>> from pyvista import Text
    >>> text = Text("text")
    >>> prop = text.prop
    """

class Text(TextProperty):
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
