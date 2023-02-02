"""Module with enum options classes for plotting."""
from enum import Enum


class AnnotatedIntEnum(int, Enum):
    """Annotated enum type."""

    def __new__(cls, value, annotation):
        """Initialize."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.annotation = annotation
        return obj

    @classmethod
    def from_str(cls, input_str):
        """Create from string."""
        for value in cls:
            if value.annotation.lower() == input_str.lower():
                return value
        raise ValueError(f"{cls.__name__} has no value matching {input_str}")

    @classmethod
    def from_any(cls, value):
        """Create from string, int, etc."""
        if isinstance(value, cls):
            return value
        elif isinstance(value, int):
            return cls(value)
        elif isinstance(value, str):
            return cls.from_str(value)
        else:
            raise ValueError(f"{cls.__name__} has no value matching {value}")


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
