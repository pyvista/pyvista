"""Information about cell quality measures."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal
from typing import NoReturn
from typing import get_args

import numpy as np

from pyvista.core import _vtk_core as _vtk
from pyvista.core.celltype import CellType

_CellQualityLiteral = Literal[
    'area',
    'aspect_frobenius',
    'aspect_gamma',
    'aspect_ratio',
    'collapse_ratio',
    'condition',
    'diagonal',
    'dimension',
    'distortion',
    'jacobian',
    'max_angle',
    'max_aspect_frobenius',
    'max_edge_ratio',
    'med_aspect_frobenius',
    'min_angle',
    'oddy',
    'radius_ratio',
    'relative_size_squared',
    'scaled_jacobian',
    'shape',
    'shape_and_size',
    'shear',
    'shear_and_size',
    'skew',
    'stretch',
    'taper',
    'volume',
    'warpage',
]
_CellQualityOptions = get_args(_CellQualityLiteral)


@dataclass
class CellQualityInfo:
    """Information about a cell's quality measure."""

    cell_type: CellType
    quality_measure: _CellQualityLiteral
    acceptable_range: tuple[float, float] | None
    normal_range: tuple[float, float]
    full_range: tuple[float, float]
    unit_cell_value: float | None


def sqrt(num: float) -> float:  # noqa: D103
    return num**0.5


INF = float('inf')
ANGLE = (180 / np.pi) * np.arccos(1 / 3)
R22 = sqrt(2) / 2
R33 = sqrt(3) / 3

_CELL_QUALITY_INFO = [
    CellQualityInfo(CellType.TRIANGLE, 'area', (0, INF), (0, INF), (0, INF), sqrt(3) / 4),
    CellQualityInfo(CellType.TRIANGLE, 'aspect_ratio', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TRIANGLE, 'aspect_frobenius', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TRIANGLE, 'condition', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TRIANGLE, 'distortion', (0.5, 1), (0, 1), (-INF, INF), 1),
    # CellQualityInfo(CellType.TRIANGLE, 'edge_ratio', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TRIANGLE, 'max_angle', (60, 90), (60, 180), (0, 180), 60),
    CellQualityInfo(CellType.TRIANGLE, 'min_angle', (30, 60), (0, 60), (0, 360), 60),
    CellQualityInfo(
        CellType.TRIANGLE, 'scaled_jacobian', (0.5, 2 * R33), (-2 * R33, 2 * R33), (-INF, INF), 1
    ),
    CellQualityInfo(CellType.TRIANGLE, 'radius_ratio', (1, 3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TRIANGLE, 'shape', (0.25, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.TRIANGLE, 'shape_and_size', (0.25, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.QUAD, 'area', (0, INF), (0, INF), (-INF, INF), 1),
    CellQualityInfo(CellType.QUAD, 'aspect_ratio', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.QUAD, 'condition', (1, 4), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.QUAD, 'distortion', (0.5, 1), (0, 1), (-INF, INF), 1),
    # CellQualityInfo(CellType.QUAD, 'edge_ratio', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.QUAD, 'jacobian', (0, INF), (0, INF), (-INF, INF), 1),
    CellQualityInfo(CellType.QUAD, 'max_aspect_frobenius', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.QUAD, 'max_angle', (90, 135), (90, 360), (0, 360), 90),
    CellQualityInfo(CellType.QUAD, 'max_edge_ratio', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.QUAD, 'med_aspect_frobenius', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.QUAD, 'min_angle', (45, 90), (0, 90), (0, 360), 90),
    CellQualityInfo(CellType.QUAD, 'oddy', (0, 0.5), (0, INF), (0, INF), 0),
    CellQualityInfo(CellType.QUAD, 'radius_ratio', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.QUAD, 'relative_size_squared', (0.3, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.QUAD, 'scaled_jacobian', (0.3, 1), (-1, 1), (-1, 1), 1),
    CellQualityInfo(CellType.QUAD, 'shape', (0.3, 1), (0, 1), (0, 1), 1),
    CellQualityInfo(CellType.QUAD, 'shape_and_size', (0.2, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.QUAD, 'shear', (0.3, 1), (0, 1), (0, 1), 1),
    CellQualityInfo(CellType.QUAD, 'shear_and_size', (0.2, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.QUAD, 'skew', (0.5, 1), (0, 1), (0, 1), 1),
    CellQualityInfo(CellType.QUAD, 'stretch', (0.25, 1), (0, 1), (0, INF), 1),
    CellQualityInfo(CellType.QUAD, 'taper', (0, 0.7), (0, INF), (0, INF), 0),
    CellQualityInfo(CellType.QUAD, 'warpage', (0, 0.7), (0, 2), (0, INF), 0),
    # CellQualityInfo(CellType.TETRA, 'edge_ratio', (1, 3), (1, INF), (1, INF), 1),
    # CellQualityInfo(CellType.TETRA, 'aspect_beta', (1, 3), (1, INF), (1, INF), 1),
    # CellQualityInfo(CellType.TETRA, 'aspect_delta', (0.1, INF), (0, INF), (0, INF), 1),
    CellQualityInfo(CellType.TETRA, 'aspect_frobenius', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TETRA, 'aspect_gamma', (1, 3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TETRA, 'aspect_ratio', (1, 3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TETRA, 'collapse_ratio', (0.1, INF), (0, INF), (0, INF), sqrt(6) / 3),
    # CellQualityInfo(CellType.TETRA, 'condition', (1, 3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TETRA, 'distortion', (0.5, 1), (0, 1), (-INF, INF), 0),
    CellQualityInfo(CellType.TETRA, 'jacobian', (0, INF), (0, INF), (-INF, INF), sqrt(2) / 2),
    CellQualityInfo(CellType.TETRA, 'min_angle', (40, ANGLE), (0, ANGLE), (0, 360), ANGLE),
    CellQualityInfo(CellType.TETRA, 'radius_ratio', (1, 3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.TETRA, 'relative_size_squared', (0.3, 1), (0, 1), (0, 1), None),
    # CellQualityInfo(CellType.TETRA, 'scaled_jacobian', (0.5, R22), (-R22, R22), (-INF, INF), 1),
    CellQualityInfo(CellType.TETRA, 'shape', (0.3, 1), (0, 1), (0, 1), 1),
    CellQualityInfo(CellType.TETRA, 'shape_and_size', (0.2, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.TETRA, 'volume', (0, INF), (-INF, INF), (-INF, INF), sqrt(2) / 12),
    CellQualityInfo(CellType.HEXAHEDRON, 'diagonal', (0.65, 1), (0, 1), (1, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'dimension', None, (0, INF), (0, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'distortion', (0.5, 1), (0, 1), (-INF, INF), 1),
    # CellQualityInfo(CellType.HEXAHEDRON, 'edge_ratio', None, (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'jacobian', (0, INF), (0, INF), (-INF, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'max_edge_ratio', (1, 1.3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'max_aspect_frobenius', (1, 3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'med_aspect_frobenius', (1, 3), (1, INF), (1, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'oddy', (0, 0.5), (0, INF), (0, INF), 0),
    CellQualityInfo(CellType.HEXAHEDRON, 'relative_size_squared', (0.5, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.HEXAHEDRON, 'scaled_jacobian', (0.5, 1), (-1, 1), (-1, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'shape', (0.3, 1), (0, 1), (0, 1), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'shape_and_size', (0.2, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.HEXAHEDRON, 'shear', (0.3, 1), (0, 1), (0, 1), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'shear_and_size', (0.2, 1), (0, 1), (0, 1), None),
    CellQualityInfo(CellType.HEXAHEDRON, 'skew', (0, 0.5), (0, 1), (0, INF), 0),
    CellQualityInfo(CellType.HEXAHEDRON, 'stretch', (0.25, 1), (0, 1), (0, INF), 1),
    CellQualityInfo(CellType.HEXAHEDRON, 'taper', (0, 0.5), (0, INF), (0, INF), 0),
    CellQualityInfo(CellType.HEXAHEDRON, 'volume', (0, INF), (0, INF), (-INF, INF), 2),
    CellQualityInfo(CellType.PYRAMID, 'volume', (0, INF), (-INF, INF), (-INF, INF), sqrt(2) / 6),
    CellQualityInfo(CellType.WEDGE, 'volume', (0, INF), (-INF, INF), (-INF, INF), sqrt(3) / 4),
]

# Create lookup dict
_CELL_QUALITY_LOOKUP: dict[CellType, dict[_CellQualityLiteral, CellQualityInfo]] = {}
for info in _CELL_QUALITY_INFO:
    _CELL_QUALITY_LOOKUP.setdefault(info.cell_type, {})
    _CELL_QUALITY_LOOKUP[info.cell_type][info.quality_measure] = info


def cell_quality_info(cell_type: CellType, measure: _CellQualityLiteral) -> CellQualityInfo:
    """Return information about a cell's quality measure.

    This function returns information about a quality measure for a specified
    :class:`~pyvista.CellType`. The following is provided for each measure:

    - ``acceptable_range``: Well-behaved cells have values in this range.
    - ``normal_range``: All cells except those with degeneracies have values in this range.
    - ``full_range``: All cells including degenerate ones have values in this range.
    - ``unit_cell_value``: The quality measure value for a reference unit cell (e.g.
      equilateral triangle for triangles).

    This information is extracted from the `Verdict Library Reference Manual <https://public.kitware.com/Wiki/images/6/6b/VerdictManual-revA.pdf>`_.
    The info can help inform if a particular cell is of high or low quality.

    See the tables below for a summary of all cell quality info available from this
    function.

    .. dropdown:: Cell Quality Info

        .. include:: api/core/cell_quality/cell_quality_info_table_TRIANGLE.rst

        .. include:: api/core/cell_quality/cell_quality_info_table_QUAD.rst

        .. include:: api/core/cell_quality/cell_quality_info_table_HEXAHEDRON.rst

        .. include:: api/core/cell_quality/cell_quality_info_table_TETRA.rst

        .. include:: api/core/cell_quality/cell_quality_info_table_WEDGE.rst

        .. include:: api/core/cell_quality/cell_quality_info_table_PYRAMID.rst


    Parameters
    ----------
    cell_type : CellType
        Cell type to get information about.

    measure : str
        Quality measure to get information about.

    Returns
    -------
    CellQualityInfo
        Dataclass with information about the quality measure for a specific cell type.

    Raises
    ------
    ValueError
        If info is not available for the specified cell type or measure.

    """

    def raise_error(item_: str, valid_options_: list[str]) -> NoReturn:
        msg = (
            f'Cell quality info is not available for {item_}. Valid options are:\n{valid_options_}'
        )
        raise ValueError(msg)

    # Lookup measures available for the cell type
    try:
        measures = _CELL_QUALITY_LOOKUP[cell_type]
    except KeyError:
        item = f'cell type {cell_type.name!r}'
        valid_options = [typ.name for typ in _CELL_QUALITY_LOOKUP.keys()]
        raise_error(item, valid_options)

    # Lookup the measure info
    try:
        return measures[measure]
    except KeyError:
        item = f'{cell_type.name!r} measure {measure!r}'
        valid_options = list(measures.keys())
        raise_error(item, valid_options)


def _get_cell_quality_measures() -> dict[str, str]:
    """Return a dict with snake case quality measure keys and vtkCellQuality attribute setter names."""
    # Get possible quality measures dynamically
    str_start = 'SetQualityMeasureTo'
    measures = {}
    for attr in dir(_vtk.vtkCellQuality):
        if attr.startswith(str_start):
            # Get the part after 'SetQualityMeasureTo'
            measure_name = attr[len(str_start) :]
            # Convert to snake case
            # Add underscore before uppercase letters, except the first one
            measure_name = re.sub(r'([a-z])([A-Z])', r'\1_\2', measure_name).lower()
            measures[measure_name] = attr
    return measures
