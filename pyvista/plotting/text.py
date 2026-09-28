"""Contains the pyvista.Text class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast
from typing import get_args

import pyvista_validation as _validation

import pyvista as pv
from pyvista import _vtk
from pyvista.core._typing_core import BoundsTuple
from pyvista.core._vtk_utilities import DisableVtkSnakeCase
from pyvista.core.utilities.misc import _NameMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin

from .colors import Color
from .prop3d import _Prop3DMixin
from .themes import Theme
from .tools import FONTS

if TYPE_CHECKING:
    from pyvista.core._typing_core import VectorLike

    from ._typing import ColorLike

HorizontalOptions = Literal['left', 'center', 'right']
VerticalOptions = Literal['bottom', 'center', 'top']

# The places `Plotter.add_text` names to draw text in, as opposed to the coordinate it
# also accepts
TextPositionOptions = Literal[
    'lower_left',
    'lower_right',
    'upper_left',
    'upper_right',
    'lower_edge',
    'upper_edge',
    'left_edge',
    'right_edge',
]

CornerOptions = Literal[
    TextPositionOptions,
    'll',
    'lr',
    'ul',
    'ur',
    'top',
    'bottom',
    'right',
    'r',
    'left',
    'l',
]

_CORNERS: dict[CornerOptions, int] = {
    'lower_left': _vtk.vtkCornerAnnotation.LowerLeft,
    'lower_right': _vtk.vtkCornerAnnotation.LowerRight,
    'upper_left': _vtk.vtkCornerAnnotation.UpperLeft,
    'upper_right': _vtk.vtkCornerAnnotation.UpperRight,
    'lower_edge': _vtk.vtkCornerAnnotation.LowerEdge,
    'upper_edge': _vtk.vtkCornerAnnotation.UpperEdge,
    'left_edge': _vtk.vtkCornerAnnotation.LeftEdge,
    'right_edge': _vtk.vtkCornerAnnotation.RightEdge,
    'll': _vtk.vtkCornerAnnotation.LowerLeft,
    'lr': _vtk.vtkCornerAnnotation.LowerRight,
    'ul': _vtk.vtkCornerAnnotation.UpperLeft,
    'ur': _vtk.vtkCornerAnnotation.UpperRight,
    'top': _vtk.vtkCornerAnnotation.UpperEdge,
    'bottom': _vtk.vtkCornerAnnotation.LowerEdge,
    'right': _vtk.vtkCornerAnnotation.RightEdge,
    'r': _vtk.vtkCornerAnnotation.RightEdge,
    'left': _vtk.vtkCornerAnnotation.LeftEdge,
    'l': _vtk.vtkCornerAnnotation.LeftEdge,
}


def _resolve_corner(position: CornerOptions | int) -> int:
    """Return the corner a position names, or the corner index itself."""
    if isinstance(position, str):
        _validation.check_contains(list(_CORNERS), must_contain=position, name='Position')
        return _CORNERS[position]
    if position is True:
        return _CORNERS['upper_left']
    return int(position)


class CornerAnnotation(_NoNewAttrMixin, DisableVtkSnakeCase, _NameMixin, _vtk.vtkCornerAnnotation):
    """Text annotation in four corners.

    This is an annotation object that manages four text actors / mappers to provide
    annotation in the four corners of a viewport.

    Parameters
    ----------
    position : str | bool | int
        Position of the text. Either the name of a corner, ``True`` for the upper
        left one, or the index of a corner.

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

    >>> import pyvista as pv
    >>> text = pv.CornerAnnotation(0, 'text')
    >>> prop = text.prop

    """

    def __init__(
        self,
        position: CornerOptions | int,
        text: str,
        *,
        prop: TextProperty | None = None,
        linear_font_scale_factor: float | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a new text annotation descriptor."""
        super().__init__()
        self.set_text(position, text)
        self.prop = TextProperty() if prop is None else prop
        if linear_font_scale_factor is not None:
            self.linear_font_scale_factor = linear_font_scale_factor
        self._name = name

    def get_text(self, position: CornerOptions | int) -> str:
        """Get the text to be displayed for each corner.

        Parameters
        ----------
        position : str | bool | int
            Position of the text. Either the name of a corner, ``True`` for the
            upper left one, or the index of a corner.

        Returns
        -------
        str
            Text to be displayed for each corner.

        """
        return self.GetText(_resolve_corner(position))

    def set_text(self, position: CornerOptions | int, text: str) -> None:
        """Set the text to be displayed for each corner.

        Parameters
        ----------
        position : str | bool | int
            Position of the text. Either the name of a corner, ``True`` for the
            upper left one, or the index of a corner.

        text : str
            Text to be displayed for each corner.

        """
        self.SetText(_resolve_corner(position), text)

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
    def prop(self, prop: TextProperty) -> None:
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
    def linear_font_scale_factor(self, factor: float) -> None:
        self.SetLinearFontScaleFactor(factor)


class Text(_NoNewAttrMixin, DisableVtkSnakeCase, _NameMixin, _vtk.vtkTextActor):
    r"""Define text by default theme.

    Parameters
    ----------
    text : str, optional
        Text string to be displayed.
        ``\n`` is recognized as a carriage return/linefeed (line separator).
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

    >>> import pyvista as pv
    >>> text = pv.Text()
    >>> prop = text.prop

    """

    def __init__(
        self,
        text: str | None = None,
        *,
        position: VectorLike[float] | None = None,
        prop: TextProperty | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a new text descriptor."""
        super().__init__()
        if text is not None:
            self.input = text
        if position is not None:
            self.position = position
        self.prop = TextProperty() if prop is None else prop
        self._name = name

    @property
    def input(self) -> str:
        r"""Text string to be displayed.

        Returns
        -------
        str
            Text string to be displayed.
            ``\n`` is recognized as a carriage return/linefeed (line separator).
            The characters must be in the UTF-8 encoding.

        """
        return self.GetInput()

    @input.setter  # noqa: A003
    def input(self, text: str) -> None:
        self.SetInput(text)

    @property
    def prop(self) -> TextProperty:
        """Property of this actor.

        Returns
        -------
        pyvista.TextProperty
            Property of this actor.

        """
        return cast('TextProperty', self.GetTextProperty())

    @prop.setter
    def prop(self, prop: TextProperty) -> None:
        self.SetTextProperty(prop)

    @property
    def position(self) -> tuple[float, float]:
        """Position coordinate.

        Returns
        -------
        tuple[float, float]
            Position coordinate.

        """
        return self.GetPosition()

    @position.setter
    def position(self, position: VectorLike[float]) -> None:
        self.SetPosition(float(position[0]), float(position[1]))


class Label(_Prop3DMixin, Text):  # type: ignore[misc]
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
        prop: TextProperty | None = None,
        name: str = 'Label',
    ) -> None:
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
        """Position of the label in ``xyz`` space.

        This is the "true" position of the label. Internally this is loosely
        equal to :attr:`~pyvista.Prop3D.position` + :attr:`relative_position`.
        """
        return self.GetPositionCoordinate().GetValue()

    @_label_position.setter
    def _label_position(self, position: VectorLike[float]) -> None:
        valid_position = _validation.validate_array3(position, dtype_out=float, to_tuple=True)
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
    def size(self, size: int) -> None:
        self.prop.font_size = size

    @property
    def relative_position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Position of the label relative to its :attr:`~pyvista.Prop3D.position`."""
        return tuple(self._relative_position.tolist())

    @relative_position.setter
    def relative_position(self, position: VectorLike[float]) -> None:
        self._relative_position = _validation.validate_array3(position, dtype_out=float)
        self._post_set_update()

    def _post_set_update(self) -> None:
        """Move the underlying text to the transformed relative position."""
        matrix4x4 = self._transformation_matrix
        vector4 = (*self.relative_position, 1)
        new_position = (matrix4x4 @ vector4)[:3]
        self._label_position = new_position

    def _get_bounds(self) -> BoundsTuple:
        # Define its 3D position as its bounds
        x, y, z = self._label_position
        return BoundsTuple(x, x, y, y, z, z)


class TextProperty(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkTextProperty):
    """Define text's property.

    Parameters
    ----------
    theme : pyvista.plotting.themes.Theme, optional
        Plot-specific theme.

    color : ColorLike, optional
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

    bold : bool, default: False
        Bolds title and bar labels.

    background_color : ColorLike, optional
        Background color of text.

    background_opacity : float, optional
        Background opacity of text.

    Examples
    --------
    Create a text's property.

    >>> import pyvista as pv
    >>> prop = pv.TextProperty()
    >>> prop.opacity = 0.5
    >>> prop.background_color = 'b'
    >>> prop.background_opacity = 0.5
    >>> prop.show_frame = True
    >>> prop.frame_color = 'b'
    >>> prop.frame_width = 10
    >>> prop.frame_color
    Color(name='blue', hex='#0000ffff', opacity=255)

    """

    _color_set: bool | None = None
    _background_color_set: bool | None = None
    _font_family: str | None = None

    def __init__(
        self,
        theme: Theme | None = None,
        *,
        color: ColorLike | None = None,
        font_family: str | None = None,
        orientation: float | None = None,
        font_size: int | None = None,
        font_file: str | Path | None = None,
        shadow: bool = False,
        justification_horizontal: HorizontalOptions | None = None,
        justification_vertical: VerticalOptions | None = None,
        italic: bool = False,
        bold: bool = False,
        background_color: ColorLike | None = None,
        background_opacity: float | None = None,
    ) -> None:
        """Initialize text's property."""
        super().__init__()
        self._theme = Theme._from_theme(pv.global_theme if theme is None else theme)
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
    def color(self, color: ColorLike | None) -> None:
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
    def opacity(self, opacity: float) -> None:
        _validation.check_range(opacity, [0.0, 1.0], name='opacity')
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
    def background_color(self, color: ColorLike | None) -> None:
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
    def background_opacity(self, opacity: float) -> None:
        _validation.check_range(opacity, [0.0, 1.0], name='background_opacity')
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
    def show_frame(self, frame: bool) -> None:
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
    def frame_color(self, color: ColorLike) -> None:
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
    def frame_width(self, width: int) -> None:
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
    def font_family(self, font_family: str | None) -> None:
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
    def font_size(self, font_size: int) -> None:
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
    def orientation(self, orientation: float) -> None:
        self.SetOrientation(orientation)

    def set_font_file(self, font_file: str | Path) -> None:
        """Set the font file.

        Parameters
        ----------
        font_file : str | Path
            Font file path.

        """
        path = Path(font_file)
        path = path.resolve()
        if not Path(path).is_file():
            msg = f'Unable to locate {path}'
            raise FileNotFoundError(msg)
        self.SetFontFamily(_vtk.VTK_FONT_FILE)
        self.SetFontFile(str(path))

    @property
    def justification_horizontal(self) -> HorizontalOptions:
        """Text's justification horizontal.

        Returns
        -------
        str
            Text's horizontal justification.
            Should be either "left", "center" or "right".

        """
        justifications: dict[str, HorizontalOptions] = {
            'left': 'left',
            'centered': 'center',
            'right': 'right',
        }
        return justifications[self.GetJustificationAsString().lower()]

    @justification_horizontal.setter
    def justification_horizontal(self, justification: HorizontalOptions) -> None:
        value = justification.lower()
        _validation.check_contains(
            list(get_args(HorizontalOptions)),
            must_contain=value,
            name='justification_horizontal',
        )
        if value == 'left':
            self.SetJustificationToLeft()
        elif value == 'center':
            self.SetJustificationToCentered()
        else:
            self.SetJustificationToRight()

    @property
    def justification_vertical(self) -> VerticalOptions:
        """Text's vertical justification.

        Returns
        -------
        str
            Text's vertical justification.
            Should be either "bottom", "center" or "top".

        """
        justifications: dict[str, VerticalOptions] = {
            'bottom': 'bottom',
            'centered': 'center',
            'top': 'top',
        }
        return justifications[self.GetVerticalJustificationAsString().lower()]

    @justification_vertical.setter
    def justification_vertical(self, justification: VerticalOptions) -> None:
        value = justification.lower()
        _validation.check_contains(
            list(get_args(VerticalOptions)),
            must_contain=value,
            name='justification_vertical',
        )
        if value == 'bottom':
            self.SetVerticalJustificationToBottom()
        elif value == 'center':
            self.SetVerticalJustificationToCentered()
        else:
            self.SetVerticalJustificationToTop()

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
    def italic(self, italic: bool) -> None:
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
    def bold(self, bold: bool) -> None:
        self.SetBold(bold)

    def shallow_copy(self, to_copy: TextProperty) -> None:
        """Create a shallow copy of the text's property.

        Parameters
        ----------
        to_copy : pyvista.TextProperty
            Text's property to copy from.

        """
        self.ShallowCopy(to_copy)


# Where each of the positions `Plotter.add_text` accepts sits in a viewport, and how
# text is anchored to it, as a fraction of the size of the viewport
_TEXT_MARGIN = 0.02

_TextPlacement = tuple[float, float, HorizontalOptions, VerticalOptions]

_TEXT_POSITIONS: dict[TextPositionOptions, _TextPlacement] = {
    'lower_left': (_TEXT_MARGIN, _TEXT_MARGIN, 'left', 'bottom'),
    'lower_right': (1 - _TEXT_MARGIN, _TEXT_MARGIN, 'right', 'bottom'),
    'upper_left': (_TEXT_MARGIN, 1 - _TEXT_MARGIN, 'left', 'top'),
    'upper_right': (1 - _TEXT_MARGIN, 1 - _TEXT_MARGIN, 'right', 'top'),
    'lower_edge': (0.5, _TEXT_MARGIN, 'center', 'bottom'),
    'upper_edge': (0.5, 1 - _TEXT_MARGIN, 'center', 'top'),
    'left_edge': (_TEXT_MARGIN, 0.5, 'left', 'center'),
    'right_edge': (1 - _TEXT_MARGIN, 0.5, 'right', 'center'),
}
