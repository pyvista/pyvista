"""Information about cell quality measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
from typing import NoReturn
from typing import get_args

import numpy as np

from pyvista.core.celltype import _CELL_TYPE_INFO
from pyvista.core.celltype import CellType

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    CellQualityInfo(CellType.TRIANGLE, 'area', (0.0, INF), (0.0, INF), (0.0, INF), sqrt(3.0) / 4.0),
    CellQualityInfo(CellType.TRIANGLE, 'aspect_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.TRIANGLE, 'aspect_frobenius', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.TRIANGLE, 'condition', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.TRIANGLE, 'distortion', (0.5, 1.0), (0.0, 1.0), (-INF, INF), 1.0),
    # CellQualityInfo(CellType.TRIANGLE, 'edge_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(
        CellType.TRIANGLE, 'max_angle', (60.0, 90.0), (60.0, 180.0), (0.0, 180.0), 60.0
    ),
    CellQualityInfo(CellType.TRIANGLE, 'min_angle', (30.0, 60.0), (0.0, 60.0), (0.0, 360.0), 60.0),
    CellQualityInfo(
        CellType.TRIANGLE, 'scaled_jacobian', (0.5, 2 * R33), (-2 * R33, 2 * R33), (-INF, INF), 1.0
    ),
    CellQualityInfo(CellType.TRIANGLE, 'radius_ratio', (1.0, 3.0), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.TRIANGLE, 'shape', (0.25, 1.0), (0.0, 1.0), (0.0, 1.0), None),
    CellQualityInfo(CellType.TRIANGLE, 'shape_and_size', (0.25, 1.0), (0.0, 1.0), (0.0, 1.0), None),
    CellQualityInfo(CellType.QUAD, 'area', (0.0, INF), (0.0, INF), (-INF, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'aspect_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'condition', (1.0, 4), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'distortion', (0.5, 1.0), (0.0, 1.0), (-INF, INF), 1.0),
    # CellQualityInfo(CellType.QUAD, 'edge_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'jacobian', (0.0, INF), (0.0, INF), (-INF, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'max_aspect_frobenius', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'max_angle', (90.0, 135.0), (90.0, 360.0), (0.0, 360.0), 90.0),
    CellQualityInfo(CellType.QUAD, 'max_edge_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'med_aspect_frobenius', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'min_angle', (45.0, 90.0), (0.0, 90.0), (0.0, 360.0), 90.0),
    CellQualityInfo(CellType.QUAD, 'oddy', (0.0, 0.5), (0.0, INF), (0.0, INF), 0.0),
    CellQualityInfo(CellType.QUAD, 'radius_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(
        CellType.QUAD, 'relative_size_squared', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), None
    ),
    CellQualityInfo(CellType.QUAD, 'scaled_jacobian', (0.3, 1.0), (-1, 1.0), (-1, 1.0), 1.0),
    CellQualityInfo(CellType.QUAD, 'shape', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    CellQualityInfo(CellType.QUAD, 'shape_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), None),
    CellQualityInfo(CellType.QUAD, 'shear', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    CellQualityInfo(CellType.QUAD, 'shear_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), None),
    # CellQualityInfo(CellType.QUAD, 'skew', (0.5, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    CellQualityInfo(CellType.QUAD, 'stretch', (0.25, 1.0), (0.0, 1.0), (0.0, INF), 1.0),
    CellQualityInfo(CellType.QUAD, 'taper', (0.0, 0.7), (0.0, INF), (0.0, INF), 0.0),
    # CellQualityInfo(CellType.QUAD, 'warpage', (0.0, 0.7), (0.0, 2.0), (0.0, INF), 0.0),
    # CellQualityInfo(CellType.TETRA, 'aspect_beta', (1.0, 3), (1.0, INF), (1.0, INF), 1.0),
    # CellQualityInfo(CellType.TETRA, 'aspect_delta', (0.1, INF), (0.0, INF), (0.0, INF), 1.0),
    CellQualityInfo(CellType.TETRA, 'aspect_frobenius', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.TETRA, 'aspect_gamma', (1.0, 3.0), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.TETRA, 'aspect_ratio', (1.0, 3.0), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(
        CellType.TETRA, 'collapse_ratio', (0.1, INF), (0.0, INF), (0.0, INF), sqrt(6.0) / 3.0
    ),
    CellQualityInfo(CellType.TETRA, 'condition', (1.0, 3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.TETRA, 'distortion', (0.5, 1.0), (0.0, 1.0), (-INF, INF), 1.0),
    # CellQualityInfo(CellType.TETRA, 'edge_ratio', (1.0, 3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.TETRA, 'jacobian', (0.0, INF), (0.0, INF), (-INF, INF), R22),
    CellQualityInfo(CellType.TETRA, 'min_angle', (40, ANGLE), (0.0, ANGLE), (0.0, 360), ANGLE),
    CellQualityInfo(CellType.TETRA, 'radius_ratio', (1.0, 3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(
        CellType.TETRA, 'relative_size_squared', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), None
    ),
    CellQualityInfo(CellType.TETRA, 'scaled_jacobian', (0.5, R22), (-R22, R22), (-INF, INF), 1.0),
    CellQualityInfo(CellType.TETRA, 'shape', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    CellQualityInfo(CellType.TETRA, 'shape_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), None),
    CellQualityInfo(
        CellType.TETRA, 'volume', (0.0, INF), (-INF, INF), (-INF, INF), sqrt(2.0) / 12.0
    ),
    CellQualityInfo(CellType.HEXAHEDRON, 'diagonal', (0.65, 1.0), (0.0, 1.0), (1.0, INF), 1.0),
    CellQualityInfo(CellType.HEXAHEDRON, 'dimension', None, (0.0, INF), (0.0, INF), R33),
    CellQualityInfo(CellType.HEXAHEDRON, 'distortion', (0.5, 1.0), (0.0, 1.0), (-INF, INF), 1.0),
    # CellQualityInfo(CellType.HEXAHEDRON, 'edge_ratio', None, (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(CellType.HEXAHEDRON, 'jacobian', (0.0, INF), (0.0, INF), (-INF, INF), 1.0),
    CellQualityInfo(CellType.HEXAHEDRON, 'max_edge_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    CellQualityInfo(
        CellType.HEXAHEDRON, 'max_aspect_frobenius', (1.0, 3), (1.0, INF), (1.0, INF), 1.0
    ),
    CellQualityInfo(
        CellType.HEXAHEDRON, 'med_aspect_frobenius', (1.0, 3), (1.0, INF), (1.0, INF), 1.0
    ),
    CellQualityInfo(CellType.HEXAHEDRON, 'oddy', (0.0, 0.5), (0.0, INF), (0.0, INF), 0.0),
    CellQualityInfo(
        CellType.HEXAHEDRON, 'relative_size_squared', (0.5, 1.0), (0.0, 1.0), (0.0, 1.0), None
    ),
    CellQualityInfo(CellType.HEXAHEDRON, 'scaled_jacobian', (0.5, 1.0), (-1, 1.0), (-1, INF), 1.0),
    CellQualityInfo(CellType.HEXAHEDRON, 'shape', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    CellQualityInfo(
        CellType.HEXAHEDRON, 'shape_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), None
    ),
    CellQualityInfo(CellType.HEXAHEDRON, 'shear', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    CellQualityInfo(
        CellType.HEXAHEDRON, 'shear_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), None
    ),
    CellQualityInfo(CellType.HEXAHEDRON, 'skew', (0.0, 0.5), (0.0, 1.0), (0.0, INF), 0.0),
    CellQualityInfo(CellType.HEXAHEDRON, 'stretch', (0.25, 1.0), (0.0, 1.0), (0.0, INF), 1.0),
    CellQualityInfo(CellType.HEXAHEDRON, 'taper', (0.0, 0.5), (0.0, INF), (0.0, INF), 0.0),
    CellQualityInfo(CellType.HEXAHEDRON, 'volume', (0.0, INF), (0.0, INF), (-INF, INF), 1.0),
    CellQualityInfo(
        CellType.PYRAMID, 'volume', (0.0, INF), (-INF, INF), (-INF, INF), sqrt(2.0) / 6.0
    ),
    CellQualityInfo(
        CellType.WEDGE, 'volume', (0.0, INF), (-INF, INF), (-INF, INF), sqrt(3.0) / 4.0
    ),
]

# Create lookup dict
_CELL_QUALITY_LOOKUP: dict[CellType, dict[_CellQualityLiteral, CellQualityInfo]] = {}
for info in _CELL_QUALITY_INFO:
    _CELL_QUALITY_LOOKUP.setdefault(info.cell_type, {})
    _CELL_QUALITY_LOOKUP[info.cell_type][info.quality_measure] = info


_CELL_TYPE_NAMES = [typ.name for typ in _CELL_QUALITY_LOOKUP.keys()]


def cell_quality_info(cell_type: CellType | str, measure: _CellQualityLiteral) -> CellQualityInfo:
    """Return information about a cell's quality measure.

    This function returns information about a quality measure for a specified
    :class:`~pyvista.CellType`. The following is provided for each measure:

    - ``acceptable_range``: Well-behaved cells have values in this range.
    - ``normal_range``: All cells except those with degeneracies have values in this range.
    - ``full_range``: All cells including degenerate ones have values in this range.
    - ``unit_cell_value``: The quality measure value for a reference unit cell (e.g.
      equilateral triangle with edge length of one for triangles).

    This information is extracted from the `Verdict Library Reference Manual <https://public.kitware.com/Wiki/images/6/6b/VerdictManual-revA.pdf>`_.
    The info can help inform if a particular cell is of high or low quality.

    See the tables below for a summary of all cell quality info available from this
    function.

    .. include:: /api/core/cell_quality/cell_quality_info_table_TRIANGLE.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_QUAD.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_HEXAHEDRON.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_TETRA.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_WEDGE.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_PYRAMID.rst


    .. note::

        Some fields for some measures have a value of ``None``. This is done in cases
        where an acceptable range may be application-specific, or where the unit cell
        value is dependent on the data (e.g. any size-dependent measures).

    .. note::

        Information is not available for all valid quality measures. Only a subset
        is provided here. If information about a measure is missing and you have
        knowledge about its acceptable range, normal range, etc., please consider
        submitting a pull request on GitHub at https://github.com/pyvista/pyvista.

    Parameters
    ----------
    cell_type : CellType | str
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

    Examples
    --------
    Get cell quality info for :attr:`~pyvista.CellType.TRIANGLE` cells and the
    ``'scaled_jacobian'`` quality measure.

    >>> import pyvista as pv
    >>> info_tri = pv.cell_quality_info(pv.CellType.TRIANGLE, 'scaled_jacobian')
    >>> info_tri
    CellQualityInfo(cell_type=<CellType.TRIANGLE: 5>, quality_measure='scaled_jacobian', acceptable_range=(0.5, 1.1547005383792515), normal_range=(-1.1547005383792515, 1.1547005383792515), full_range=(-inf, inf), unit_cell_value=1.0)

    Show the acceptable range for this measure.

    >>> info_tri.acceptable_range
    (0.5, 1.1547005383792515)

    Show the value of this measure for equilateral triangles with edge length of one.

    >>> info_tri.unit_cell_value
    1.0

    Get info for the same measure but for :attr:`~pyvista.CellType.QUAD` cells.

    >>> info_quad = pv.cell_quality_info(pv.CellType.QUAD, 'scaled_jacobian')
    >>> info_quad
    CellQualityInfo(cell_type=<CellType.QUAD: 9>, quality_measure='scaled_jacobian', acceptable_range=(0.3, 1.0), normal_range=(-1, 1.0), full_range=(-1, 1.0), unit_cell_value=1.0)

    Show the acceptable range. Note that it differs for quads compared to triangles.

    >>> info_quad.acceptable_range
    (0.3, 1.0)

    Show the value of this measure for a square cell with edge length of one.

    >>> info_quad.unit_cell_value
    1.0

    """

    def raise_error(item_: str, valid_options_: Sequence[str]) -> NoReturn:
        msg = (
            f'Cell quality info is not available for {item_}. Valid options are:\n{valid_options_}'
        )
        raise ValueError(msg)

    if isinstance(cell_type, str):
        upper = cell_type.upper()
        if upper not in _CELL_TYPE_NAMES:
            item = f'cell type {upper!r}'
            raise_error(item, _CELL_TYPE_NAMES)
        value = CellType(_CELL_TYPE_INFO[upper].value)
    else:
        value = CellType(cell_type)

    # Lookup measures available for the cell type
    try:
        measures = _CELL_QUALITY_LOOKUP[value]
    except KeyError:
        item = f'cell type {value.name!r}'
        raise_error(item, _CELL_TYPE_NAMES)

    # Lookup the measure info
    try:
        return measures[measure]
    except KeyError:
        item = f'{value.name!r} measure {measure!r}'
        valid_options = list(measures.keys())
        raise_error(item, valid_options)
