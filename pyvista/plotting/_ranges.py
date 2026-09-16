"""Range checks for the plotting values which restrict their range."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import pyvista_validation as _validation

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista.core._typing_core import VectorLike


def _checker(name: str, rng: VectorLike[float]) -> Callable[[float], None]:
    """Return a range check for one named value."""
    return functools.partial(_validation.check_range, rng=rng, name=name)


check_ambient = _checker('ambient', [0.0, 1.0])
check_anisotropy = _checker('anisotropy', [0.0, 1.0])
check_anisotropy_rotation = _checker('anisotropy_rotation', [0.0, 1.0])
check_cap_opacity = _checker('cap_opacity', [0.0, 1.0])
check_decimate = _checker('decimate', [0.0, 1.0])
check_diffuse = _checker('diffuse', [0.0, 1.0])
check_edge_opacity = _checker('edge_opacity', [0.0, 1.0])
check_index_of_refraction = _checker('index_of_refraction', [1.0, np.inf])
check_line_width = _checker('line_width', [0.0, np.inf])
check_metallic = _checker('metallic', [0.0, 1.0])
check_opacity = _checker('opacity', [0.0, 1.0])
check_point_size = _checker('point_size', [0.0, np.inf])
check_roughness = _checker('roughness', [0.0, 1.0])
check_specular = _checker('specular', [0.0, 1.0])
check_specular_power = _checker('specular_power', [0.0, 128.0])
