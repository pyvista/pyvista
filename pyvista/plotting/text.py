"""Contains the pyvista.Text class."""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core._typing_core import BoundsTuple
from pyvista.core.utilities.misc import _check_range
from pyvista.core.utilities.misc import _NameMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin

from . import _vtk
from .colors import Color
from .prop3d import _Prop3DMixin
from .themes import Theme
from .tools import FONTS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista import Property
    from pyvista.core._typing_core import VectorLike

    from ._typing import ColorLike

HorizontalOptions = Literal['left', 'center', 'right']
VerticalOptions = Literal['bottom', 'center', 'top']


class CornerAnnotation(
    _NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _NameMixin, _vtk.vtkCornerAnnotation
):
    """Text annotation in four corners.

    This is an annotation object that manages four text actors / mappers to provide
    annotation in the four corners of a viewport.

    Parameters
    ----------
    position : str | bool
        Position of the text.

    text : str
        Text input.

    prop : pyvista.TextProperty, optional
        Text property.

    linear_font_scale_factor : float, optional
        Linear font scale factor.

    name : str, optional
        The name of this actor used when tracking on a plotter.

        .. versionadded:: 0.45

    Examples
    --------
    Create text annotation in four corners.

    >>> from pyvista import CornerAnnotation
    >>> text = CornerAnnotation(0, 'text')
    >>> prop = text.prop

    """

    @_deprecate_positional_args(allowed=['position', 'text'])
    def __init__(  # noqa: PLR0917
        self, position, text, prop=None, linear_font_scale_factor=None, name=None
    ):
        """Initialize a new text annotation descriptor."""
        super().__init__()
        self.set_text(position, text)
        if prop is None:
            self.prop = TextProperty()
        if linear_font_scale_factor is not None:
            self.linear_font_scale_factor = linear_font_scale_factor
        self._name = name

    def get_text(self, position):
        """Get the text to be displayed for each corner.

        Parameters
        ----------
        position : str | bool
            Position of the text.

        Returns
        -------
        str
            Text to be displayed for each corner.

        """
        return self.GetText(position)

    def set_text(self, position, text):
        """Set the text to be displayed for each corner.

        Parameters
        ----------
        position : str | bool
            Position of the text.

        text : str
            Text to be displayed for each corner.

        """
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
        self.SetText(position, text)

    @property
    def prop(self) -> TextProperty:
        """Property of this actor.

        Returns
        -------
        pyvista.TextProperty
            Property of this actor.

        """
        return self.GetTextProperty()

    @prop.setter
    def prop(self, prop: TextProperty):
        self.SetTextProperty(prop)

    @property
    def linear_font_scale_factor(self) -> float:
        """Font scaling factors.

        Returns
        -------
        float
            Font scaling factors.

        """
        return self.GetLinearFontScaleFactor()

    @linear_font_scale_factor.setter
    def linear_font_scale_factor(self, factor: float):
        self.SetLinearFontScaleFactor(factor)


class Text(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _NameMixin, _vtk.vtkTextActor):
    r"""Define text by default theme.

    Parameters
    ----------
    text : str, optional
        Text string to be displayed.
        "\n" is recognized as a carriage return/linefeed (line separator).
        The characters must be in the UTF-8 encoding.

    position : Sequence[float], optional
        The position coordinate.

    prop : pyvista.TextProperty, optional
        The property of this actor.

    name : str, optional
        The name of this actor used when tracking on a plotter.

        .. versionadded:: 0.45

    Examples
    --------
    Create a text with text's property.

    >>> from pyvista import Text
    >>> text = Text()
    >>> prop = text.prop

    """

    @_deprecate_positional_args(allowed=['text'])
    def __init__(  # noqa: PLR0917
        self, text=None, position=None, prop=None, name=None
    ):
        """Initialize a new text descriptor."""
        super().__init__()
        if text is not None:
            self.input = text
        if position is not None:
            self.position = position
        if prop is None:
            self.prop = TextProperty()
        self._name = name

    @property
    def input(self):
        r"""Text string to be displayed.

        Returns
        -------
        str
            Text string to be displayed.
            "\n" is recognized as a carriage return/linefeed (line separator).
            The characters must be in the UTF-8 encoding.

        """
        return self.GetInput()

    @input.setter  # noqa: A003
    def input(self, text: str):
        self.SetInput(text)

    @property
    def prop(self):
        """Property of this actor.

        Returns
        -------
        pyvista.TextProperty
            Property of this actor.

        """
        return self.GetTextProperty()

    @prop.setter
    def prop(self, prop: TextProperty):
        self.SetTextProperty(prop)

    @property
    def position(self):
        """Position coordinate.

        Returns
        -------
        Sequence[float]
            Position coordinate.

        """
        return self.GetPosition()

    @position.setter
    def position(self, position: Sequence[float]):
        self.SetPosition(position[0], position[1])


class Label(_Prop3DMixin, Text):
    """2D label actor with a 3D position coordinate.

    Unlike :class:`~pyvista.Text`, which uses 2D viewport coordinates to position text
    in a plot, this class instead uses a 3D position coordinate. This class may be
    positioned, oriented, and transformed in a manner similar to a 3D
    :class:`~pyvista.Actor`.

    In addition, this class supports an additional :attr:`relative_position` attribute.
    In general, it is recommended to simply use :attr:`~pyvista.Prop3D.position` when positioning a
    :class:`Label` by itself. However, if the position of the label depends on the
    positioning of another actor, both :attr:`~pyvista.Prop3D.position` and
    :attr:`relative_position` may be used together.
    In these cases, the :attr:`~pyvista.Prop3D.position` of the label and actor
    should be kept in-sync. See the examples below.

    Parameters
    ----------
    text : str, optional
        Text string to be displayed.

    position : VectorLike[float]
        Position of the text in XYZ coordinates.

    relative_position : VectorLike[float]
        Position of the text in XYZ coordinates relative to its :attr:`~pyvista.Prop3D.position`.

    size : int
        Size of the text label.

    prop : pyvista.TextProperty, optional
        The property of this actor.

    name : str, optional
        The name of this actor used when tracking on a plotter.

        .. versionadded:: 0.45

    See Also
    --------
    pyvista.Plotter.add_point_labels

    Examples
    --------
    Create a label for a point of interest. Here we add a label to the tip of a cone.

    >>> import pyvista as pv
    >>> cone_dataset = pv.Cone()
    >>> tip = (0.5, 0, 0)
    >>> label = pv.Label('tip', position=tip)

    Plot the mesh and label.

    >>> pl = pv.Plotter()
    >>> cone_actor = pl.add_mesh(cone_dataset)
    >>> _ = pl.add_actor(label)
    >>> pl.show()

    The previous example set the label's position as the cone's tip explicitly.
    However, this means that the two actors now have different positions.

    >>> cone_actor.position
    (0.0, 0.0, 0.0)
    >>> label.position
    (0.5, 0.0, 0.0)

    And if we change the 3D orientation of the cone and label, the label is no longer
    positioned at the tip.

    >>> cone_actor.orientation = 0, 0, 90
    >>> label.orientation = 0, 0, 90
    >>>
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(cone_actor)
    >>> _ = pl.add_actor(label)
    >>> pl.show()

    This is because rotations by :class:`pyvista.Prop3D` are applied **before** the
    actor is moved to its final position, and therefore the label's position is not
    considered in the rotation. Hence, the final position of the label remains at
    ``(0.5, 0.0, 0.0)`` as it did earlier, despite changing its orientation.

    If we want the position of the label to have the same positioning *relative* to the
    cone, we can instead set its :attr:`relative_position`.

    First, reset the label's position to match the cone's position.

    >>> label.position = cone_actor.position
    >>> label.position
    (0.0, 0.0, 0.0)

    Now set its :attr:`relative_position` to the tip of the cone.

    >>> label.relative_position = tip
    >>> label.relative_position
    (0.5, 0.0, 0.0)

    Plot the results. The label is now correctly positioned at the tip of the cone.
    This is because the :attr:`relative_position` is considered as part of the
    rotation.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(cone_actor)
    >>> _ = pl.add_actor(label)
    >>> pl.show()

    As long as the label and cone's :class:`pyvista.Prop3D` attributes are modified
    together and synchronized, the label will remain at the tip of the cone.

    Modify the position of the label and tip.

    >>> cone_actor.position = (1.0, 2.0, 3.0)
    >>> label.position = (1.0, 2.0, 3.0)
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(cone_actor)
    >>> _ = pl.add_actor(label)
    >>> _ = pl.add_axes_at_origin()
    >>> pl.show()

    """

    def __init__(
        self,
        text: str | None = None,
        position: VectorLike[float] = (0.0, 0.0, 0.0),
        relative_position: VectorLike[float] = (0.0, 0.0, 0.0),
        *,
        size: int = 50,
        prop: Property | None = None,
        name: str = 'Label',
    ):
        Text.__init__(self, text=text, prop=prop)
        self.GetPositionCoordinate().SetCoordinateSystemToWorld()
        self.SetTextScaleModeToNone()  # Use font size to control size of text
        self._name = name

        _Prop3DMixin.__init__(self)
        self.relative_position = relative_position
        self.position = position
        self.size = size

    @property
    def _label_position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Position of the label in xyz space.

        This is the "true" position of the label. Internally this is loosely
        equal to :attr:`~pyvista.Prop3D.position` + :attr:`relative_position`.
        """
        return self.GetPositionCoordinate().GetValue()

    @_label_position.setter
    def _label_position(self, position: VectorLike[float]):
        valid_position = _validation.validate_array3(position)
        self.GetPositionCoordinate().SetValue(valid_position)

    @property
    def size(self) -> int:  # numpydoc ignore=RT01
        """Size of the text label.

        Notes
        -----
        The text property's font size used to control the size of the label.

        """
        return self.prop.font_size

    @size.setter
    def size(self, size: int):
        self.prop.font_size = size

    @property
    def relative_position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Position of the label relative to its :attr:`~pyvista.Prop3D.position`."""
        return tuple(self._relative_position.tolist())

    @relative_position.setter
    def relative_position(self, position: VectorLike[float]):
        self._relative_position = _validation.validate_array3(position, dtype_out=float)
        self._post_set_update()

    def _post_set_update(self):
        # Update the label's underlying text position
        matrix4x4 = self._transformation_matrix
        vector4 = (*self.relative_position, 1)
        new_position = (matrix4x4 @ vector4)[:3]
        self._label_position = new_position

    def _get_bounds(self) -> BoundsTuple:
        # Define its 3D position as its bounds
        x, y, z = self._label_position
        return BoundsTuple(x, x, y, y, z, z)


class TextProperty(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkTextProperty):
    """Define text's property.

    Parameters
    ----------
    theme : pyvista.plotting.themes.Theme, optional
        Plot-specific theme.

    color : pyvista.ColorLike, optional
        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

    font_family : str | None, optional
        Font family or None.

    orientation : float, optional
        Text's orientation (in degrees).

    font_size : int, optional
        Font size.

    font_file : str, optional
        Font file path.

    shadow : bool, optional
        If enable the shadow.

    justification_horizontal : str, optional
        Text's horizontal justification.
        Should be either "left", "center" or "right".

    justification_vertical : str, optional
        Text's vertical justification.
        Should be either "bottom", "center" or "top".

    italic : bool, default: False
        Italicises title and bar labels.

    bold : bool, default: True
        Bolds title and bar labels.

    background_color : pyvista.Color, optional
        Background color of text.

    background_opacity : pyvista.Color, optional
        Background opacity of text.

    Examples
    --------
    Create a text's property.

    >>> from pyvista import TextProperty
    >>> prop = TextProperty()
    >>> prop.opacity = 0.5
    >>> prop.background_color = 'b'
    >>> prop.background_opacity = 0.5
    >>> prop.show_frame = True
    >>> prop.frame_color = 'b'
    >>> prop.frame_width = 10
    >>> prop.frame_color
    Color(name='blue', hex='#0000ffff', opacity=255)

    """

    _theme = Theme()
    _color_set = None
    _background_color_set = None
    _font_family = None

    @_deprecate_positional_args(allowed=['theme'])
    def __init__(  # noqa: PLR0917
        self,
        theme=None,
        color=None,
        font_family=None,
        orientation=None,
        font_size=None,
        font_file=None,
        shadow: bool = False,  # noqa: FBT001, FBT002
        justification_horizontal=None,
        justification_vertical=None,
        italic: bool = False,  # noqa: FBT001, FBT002
        bold: bool = False,  # noqa: FBT001, FBT002
        background_color=None,
        background_opacity=None,
    ):
        """Initialize text's property."""
        super().__init__()
        if theme is None:
            # copy global theme to ensure local property theme is fixed
            # after creation.
            self._theme.load_theme(pv.global_theme)
        else:
            self._theme.load_theme(theme)
        self.color = color
        self.font_family = font_family
        if orientation is not None:
            self.orientation = orientation
        if font_size is not None:
            self.font_size = font_size
        if font_file is not None:
            self.set_font_file(font_file)
        if shadow:
            self.enable_shadow()
        if justification_horizontal is not None:
            self.justification_horizontal = justification_horizontal
        if justification_vertical is not None:
            self.justification_vertical = justification_vertical
        self.italic = italic
        self.bold = bold
        if background_color is not None:
            self.background_color = background_color
        if background_opacity is not None:
            self.background_opacity = background_opacity

    @property
    def color(self) -> Color:
        """Color of text's property.

        Returns
        -------
        pyvista.Color
            Color of text's property.

        """
        return Color(self.GetColor())

    @color.setter
    def color(self, color: ColorLike):
        self._color_set = color is not None
        rgb_color = Color(color, default_color=self._theme.font.color)
        self.SetColor(rgb_color.float_rgb)

    @property
    def opacity(self) -> float:
        """Opacity of text's property.

        Returns
        -------
        float
            Opacity of the text. A single float value that will be applied globally
            opacity of the text and uniformly applied everywhere. Between 0 and 1.

        """
        return self.GetOpacity()

    @opacity.setter
    def opacity(self, opacity: float):
        _check_range(opacity, (0, 1), 'opacity')
        self.SetOpacity(opacity)

    @property
    def background_color(self) -> Color:
        """Background color of text's property.

        Returns
        -------
        pyvista.Color
            Background color of text's property.

        """
        return Color(self.GetBackgroundColor())

    @background_color.setter
    def background_color(self, color: ColorLike):
        self._background_color_set = color is not None
        rgb_color = Color(color)
        self.SetBackgroundColor(rgb_color.float_rgb)

    @property
    def background_opacity(self) -> float:
        """Background opacity of text's property.

        Returns
        -------
        float
            Background opacity of the text. A single float value that will be applied globally.
            Background opacity of the text and uniformly applied everywhere. Between 0 and 1.

        """
        return self.GetBackgroundOpacity()

    @background_opacity.setter
    def background_opacity(self, opacity: float):
        _check_range(opacity, (0, 1), 'background_opacity')
        self.SetBackgroundOpacity(opacity)

    @property
    def show_frame(self) -> bool:
        """Visibility of frame.

        Returns
        -------
        bool:
            If shows the frame.

        """
        return bool(self.GetFrame())

    @show_frame.setter
    def show_frame(self, frame: bool):
        self.SetFrame(frame)

    @property
    def frame_color(self) -> Color:
        """Frame color of text property.

        Returns
        -------
        pyvista.Color
            Frame color of text property.

        """
        return Color(self.GetFrameColor())

    @frame_color.setter
    def frame_color(self, color):
        self.SetFrameColor(Color(color).float_rgb)

    @property
    def frame_width(self) -> int:
        """Width of the frame.

        Returns
        -------
        int
            Width of the frame. The width is expressed in pixels.
            The default is 1 pixel.

        """
        return self.GetFrameWidth()

    @frame_width.setter
    def frame_width(self, width: int):
        self.SetFrameWidth(width)

    @property
    def font_family(self) -> str | None:
        """Font family.

        Returns
        -------
        output : str | None
            Font family or None.

        """
        return self._font_family

    @font_family.setter
    def font_family(self, font_family: str | None):
        if font_family is None:
            font_family = self._theme.font.family
        self._font_family = font_family
        self.SetFontFamily(FONTS[self._font_family].value)

    @property
    def font_size(self) -> int:
        """Font size.

        Returns
        -------
        int
            Font size.

        """
        return self.GetFontSize()

    @font_size.setter
    def font_size(self, font_size: int):
        self.SetFontSize(font_size)

    def enable_shadow(self) -> None:
        """Enable the shadow."""
        self.SetShadow(True)

    @property
    def orientation(self) -> float:
        """Text's orientation (in degrees).

        Returns
        -------
        float
            Text's orientation (in degrees).

        """
        return self.GetOrientation()

    @orientation.setter
    def orientation(self, orientation: float):
        self.SetOrientation(orientation)

    def set_font_file(self, font_file: str):
        """Set the font file.

        Parameters
        ----------
        font_file : str
            Font file path.

        """
        path = pathlib.Path(font_file)
        path = path.resolve()
        if not Path(path).is_file():
            msg = f'Unable to locate {path}'
            raise FileNotFoundError(msg)
        self.SetFontFamily(_vtk.VTK_FONT_FILE)
        self.SetFontFile(str(path))

    @property
    def justification_horizontal(self) -> str:
        """Text's justification horizontal.

        Returns
        -------
        str
            Text's horizontal justification.
            Should be either "left", "center" or "right".

        """
        justification = self.GetJustificationAsString().lower()
        if justification == 'centered':
            justification = 'center'
        return justification

    @justification_horizontal.setter
    def justification_horizontal(self, justification: str):
        if justification.lower() == 'left':
            self.SetJustificationToLeft()
        elif justification.lower() == 'center':
            self.SetJustificationToCentered()
        elif justification.lower() == 'right':
            self.SetJustificationToRight()
        else:
            msg = (
                f'Invalid {justification} for justification_horizontal. '
                'Should be either "left", "center" or "right".'
            )
            raise ValueError(msg)

    @property
    def justification_vertical(self) -> str:
        """Text's vertical justification.

        Returns
        -------
        str
            Text's vertical justification.
            Should be either "bottom", "center" or "top".

        """
        justification = self.GetVerticalJustificationAsString().lower()
        if justification == 'centered':
            justification = 'center'
        return justification

    @justification_vertical.setter
    def justification_vertical(self, justification: str):
        if justification.lower() == 'bottom':
            self.SetVerticalJustificationToBottom()
        elif justification.lower() == 'center':
            self.SetVerticalJustificationToCentered()
        elif justification.lower() == 'top':
            self.SetVerticalJustificationToTop()
        else:
            msg = (
                f'Invalid {justification} for justification_vertical. '
                'Should be either "bottom", "center" or "top".'
            )
            raise ValueError(msg)

    @property
    def italic(self) -> bool:
        """Italic of text's property.

        Returns
        -------
        bool
            If text is italic.

        """
        return bool(self.GetItalic())

    @italic.setter
    def italic(self, italic: bool):
        self.SetItalic(italic)

    @property
    def bold(self) -> bool:
        """Bold of text's property.

        Returns
        -------
        bool
            If text is bold.

        """
        return bool(self.GetBold())

    @bold.setter
    def bold(self, bold: bool):
        self.SetBold(bold)

    def shallow_copy(self, to_copy: TextProperty) -> None:
        """Create a shallow copy of the text's property.

        Parameters
        ----------
        to_copy : pyvista.TextProperty
            Text's property to copy from.

        """
        self.ShallowCopy(to_copy)
