"""Module with enum options classes for plotting."""
from pyvista.core.utilities.misc import AnnotatedIntEnum


class InterpolationType(AnnotatedIntEnum):
    """Lighting interpolation types."""

    FLAT = (0, 'Flat')
    GOURAUD = (1, 'Gouraud')
    PHONG = (2, 'PHONG')
    PBR = (3, 'Physically based rendering')

    @classmethod
    def from_str(cls, input_str):
        """Create from string."""
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
