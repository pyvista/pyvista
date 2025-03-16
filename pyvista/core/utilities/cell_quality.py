"""Information about cell quality measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import NoReturn
from typing import get_args

import numpy as np

from pyvista.core import _vtk_core as _vtk
from pyvista.core.celltype import _CELL_TYPE_INFO
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
    measure: str
    acceptable_range: tuple[float, float] | None
    normal_range: tuple[float, float]
    full_range: tuple[float, float]
    unit_cell_value: float | None


def sqrt(num: float) -> float:  # noqa: D103
    return num**0.5


INF = float('inf')
TET_ANGLE = (180 / np.pi) * np.arccos(1 / 3)
R22 = sqrt(2) / 2
R33 = sqrt(3) / 3

_INFO = [
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
    CellQualityInfo(
        CellType.TETRA,
        'min_angle',
        (40, TET_ANGLE),
        (0, TET_ANGLE),
        (0, 360),
        TET_ANGLE,
    ),
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

_INFO_LOOKUP = {}


def _init_lookup(lookup: dict | None) -> None:
    """Populate info lookup dict."""
    from pyvista import examples  # Avoid circular import

    for info in _INFO:
        # Validate info by loading the cell as a mesh and computing its cell quality
        example_name = _CELL_TYPE_INFO[info.cell_type.name].example
        cell_mesh = getattr(examples.cells, example_name)()
        null_value = -1

        # Suppress errors for invalid metrics
        verbosity = _vtk.vtkLogger.GetCurrentVerbosityCutoff()
        _vtk.vtkLogger.SetStderrVerbosity(_vtk.vtkLogger.VERBOSITY_OFF)

        qual = cell_mesh.compute_cell_quality(info.measure, null_value=null_value)

        # Restore the original vtkLogger verbosity level
        _vtk.vtkLogger.SetStderrVerbosity(verbosity)

        # Ensure the measure is valid for this cell type
        assert qual.active_scalars[0] != null_value, (
            f'Measure {info.measure!r} is not valid for cell type {info.cell_type.name!r}'
        )

        # Measure is valid, populate dict
        lookup.setdefault(info.cell_type, {})
        lookup[info.cell_type][info.measure] = info


def cell_quality_info(cell_type: CellType, measure: str) -> CellQualityInfo:
    """Return information about a cell's quality measure.

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

    """

    def raise_error(item_: str, valid_options_: list[str]) -> NoReturn:
        raise ValueError(
            f'Cell quality info is not available for {item_}. Valid options are:\n{valid_options_}'
        )

    if _INFO_LOOKUP == {}:
        _init_lookup(_INFO_LOOKUP)

    # Lookup measures available for the cell type
    try:
        measures = _INFO_LOOKUP[cell_type]
    except KeyError:
        item = f'cell type {cell_type.name!r}'
        valid_options = [typ.name for typ in _INFO_LOOKUP.keys()]
        raise_error(item, valid_options)

    # Lookup the measure info
    try:
        return measures[measure]
    except KeyError:
        item = f'{cell_type.name!r} measure {measure!r}'
        valid_options = list(measures.keys())
        raise_error(item, valid_options)
