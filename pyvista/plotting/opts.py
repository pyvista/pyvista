"""Module with enum options classes for plotting."""

from __future__ import annotations

from enum import Enum

from pyvista.core.utilities.misc import AnnotatedIntEnum


class ShaderType(str, Enum):
    """Shader types for GLSL shader replacements.

    .. versionadded:: 0.48

    Attributes
    ----------
    VERTEX : str
        Vertex shader.
    FRAGMENT : str
        Fragment shader.
    GEOMETRY : str
        Geometry shader.

    """

    VERTEX = 'vertex'
    FRAGMENT = 'fragment'
    GEOMETRY = 'geometry'


class PointSpriteShape(str, Enum):
    """Point sprite shape options for fragment shader rendering.

    .. versionadded:: 0.48

    Attributes
    ----------
    CIRCLE : str
        Circular disc.
    TRIANGLE : str
        Upward-pointing triangle.
    HEXAGON : str
        Regular hexagon.
    DIAMOND : str
        Diamond (rotated square).
    ASTERISK : str
        Five-pointed asterisk.
    STAR : str
        Five-pointed star.

    """

    CIRCLE = 'circle'
    TRIANGLE = 'triangle'
    HEXAGON = 'hexagon'
    DIAMOND = 'diamond'
    ASTERISK = 'asterisk'
    STAR = 'star'


class InterpolationType(AnnotatedIntEnum):
    """Lighting interpolation types.

    Attributes
    ----------
    FLAT : (int, str)
        Flat interpolation type.
    GOURAUD : (int, str)
        Gouraud interpolation type.
    PHONG : (int, str)
        Phong interpolation type.
    PBR : (int, str)
        Physically based rendering interpolation type.

    """

    FLAT = (0, 'Flat')
    GOURAUD = (1, 'Gouraud')
    PHONG = (2, 'PHONG')
    PBR = (3, 'Physically based rendering')

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
    """Types of representations the models can have."""

    POINTS = (0, 'Points')
    WIREFRAME = (1, 'Wireframe')
    SURFACE = (2, 'Surface')


class ElementType(AnnotatedIntEnum):
    """Types of elemental geometries."""

    MESH = (0, 'Mesh')
    CELL = (1, 'Cell')
    FACE = (2, 'Face')
    EDGE = (3, 'Edge')
    POINT = (4, 'Point')


class PickerType(AnnotatedIntEnum):
    """Types of pickers."""

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
