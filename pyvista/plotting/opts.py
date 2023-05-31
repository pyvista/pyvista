"""Module with enum options classes for plotting."""
from pyvista.utilities.helpers import AnnotatedIntEnum


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
