"""Module with enum options classes for plotting."""
from enum import Enum


class ShaderOpts(Enum):
    """Types of shader methods available."""

    FLAT = 0
    GOURAUD = 1
    PHONG = 2
    PBR = 3


class RepresentationOpts(Enum):
    """Types of representations the models can have."""

    POINTS = 0
    WIREFRAME = 1
    SURFACE = 2
