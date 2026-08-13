"""Module with enum options classes for plotting."""

from __future__ import annotations

from enum import Enum

from pyvista.core.utilities.misc import AnnotatedIntEnum


class ShaderType(str, Enum):
    """Shader types for GLSL shader replacements.

    .. versionadded:: 0.48
    """

    def __new__(cls, value, doc):
        """Override method to include member documentation."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.__doc__ = doc
        return obj

    VERTEX = 'vertex', 'Vertex shader.'
    FRAGMENT = 'fragment', 'Fragment shader.'
    GEOMETRY = 'geometry', 'Geometry shader.'


class PointSpriteShape(str, Enum):
    """Point sprite shape options for fragment shader rendering.

    .. versionadded:: 0.48
    """

    def __new__(cls, value, doc):
        """Override method to include member documentation."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.__doc__ = doc
        return obj

    CIRCLE = 'circle', 'Circular disc.'
    TRIANGLE = 'triangle', 'Upward-pointing triangle.'
    HEXAGON = 'hexagon', 'Regular hexagon.'
    DIAMOND = 'diamond', 'Diamond (rotated square).'
    ASTERISK = 'asterisk', 'Five-pointed asterisk.'
    STAR = 'star', 'Five-pointed star.'


class InterpolationType(AnnotatedIntEnum):
    """Lighting interpolation types.

    Members accept either their ``int`` value or ``str`` annotation, e.g.
    ``InterpolationType.from_any('Flat')``.
    """

    FLAT = (0, 'Flat', 'Flat interpolation type.')
    GOURAUD = (1, 'Gouraud', 'Gouraud interpolation type.')
    PHONG = (2, 'PHONG', 'Phong interpolation type.')
    PBR = (3, 'Physically based rendering', 'Physically based rendering interpolation type.')

    @classmethod
    def from_str(cls, input_str):
        """Create from string.

        Create an instance of InterpolationType from a string.

        Parameters
        ----------
        input_str : str
            The string representation of the interpolation type.  Accepts
            aliases such as ``'pbr'`` for ``'Physically based rendering'``.

        Returns
        -------
        InterpolationType
            Interpolation type as defined by the input string.

        """
        aliases = {
            'pbr': 'Physically based rendering',
        }
        if input_str in aliases:
            input_str = aliases[input_str]
        return super().from_str(input_str)


class RepresentationType(AnnotatedIntEnum):
    """Types of representations the models can have.

    Members accept either their ``int`` value or ``str`` annotation, e.g.
    ``RepresentationType.from_any('Points')``.
    """

    POINTS = (0, 'Points')
    WIREFRAME = (1, 'Wireframe')
    SURFACE = (2, 'Surface')


class ElementType(AnnotatedIntEnum):
    """Types of elemental geometries.

    Members accept either their ``int`` value or ``str`` annotation, e.g.
    ``ElementType.from_any('Cell')``.
    """

    MESH = (0, 'Mesh')
    CELL = (1, 'Cell')
    FACE = (2, 'Face')
    EDGE = (3, 'Edge')
    POINT = (4, 'Point')


class PickerType(AnnotatedIntEnum):
    """Types of pickers.

    Members accept either their ``int`` value or ``str`` annotation, e.g.
    ``PickerType.from_any('Volume')``.
    """

    AREA = (0, 'Area')
    CELL = (1, 'Cell')
    HARDWARE = (2, 'Hardware')
    POINT = (3, 'Point')
    PROP = (4, 'Prop')
    RENDERED = (5, 'Rendered')
    RESLICE = (6, 'Reslice')
    SCENE = (7, 'Scene')
    VOLUME = (8, 'Volume')
    WORLD = (9, 'World')


class StereoType(AnnotatedIntEnum):
    """Types of stereo rendering.

    Members accept either their ``int`` value or ``str`` annotation, e.g.
    ``StereoType.from_any('Anaglyph')``.
    """

    CRYSTAL_EYES = (1, 'Crystal Eyes')
    RED_BLUE = (2, 'Red Blue')
    INTERLACED = (3, 'Interlaced')
    LEFT = (4, 'Left')
    RIGHT = (5, 'Right')
    DRESDEN = (6, 'Dresden')
    ANAGLYPH = (7, 'Anaglyph')
    CHECKERBOARD = (8, 'Checkerboard')
    SPLITVIEWPORT_HORIZONTAL = (9, 'Split Viewport Horizontal')
    FAKE = (10, 'Fake')
    EMULATE = (11, 'Emulate')
    ZSPACE_INSPIRE = (12, 'ZSpace Inspire')
