"""Information about cell quality measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
from typing import NoReturn

import numpy as np

from pyvista.core.celltype import _CELL_TYPE_INFO
from pyvista.core.celltype import CellType
from pyvista.core.utilities.misc import _NoNewAttrMixin

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

_CellTypesLiteral = Literal[
    CellType.TRIANGLE,
    CellType.QUAD,
    CellType.TETRA,
    CellType.HEXAHEDRON,
    CellType.PYRAMID,
    CellType.WEDGE,
]
_CellTypeNamesLiteral = Literal[
    'TRIANGLE',
    'triangle',
    'QUAD',
    'quad',
    'TETRA',
    'tetra',
    'HEXAHEDRON',
    'hexahedron',
    'PYRAMID',
    'pyramid',
    'WEDGE',
    'wedge',
]


@dataclass
class CellQualityInfo(_NoNewAttrMixin):
    """Information about a cell's quality measure."""

    cell_type: _CellTypesLiteral
    quality_measure: _CellQualityLiteral
    acceptable_range: tuple[float, float]
    normal_range: tuple[float, float]
    full_range: tuple[float, float]
    unit_cell_value: float


def sqrt(num: float) -> float:  # noqa: D103
    return num**0.5


# Define aliases to help definitions fit on one line
INF = float('inf')
ANGLE = float((180 / np.pi) * np.arccos(1 / 3))
R22 = sqrt(2) / 2
R33 = sqrt(3) / 3

TRIANGLE: Literal[CellType.TRIANGLE] = CellType.TRIANGLE
QUAD: Literal[CellType.QUAD] = CellType.QUAD
TETRA: Literal[CellType.TETRA] = CellType.TETRA
HEXAHEDRON: Literal[CellType.HEXAHEDRON] = CellType.HEXAHEDRON
PYRAMID: Literal[CellType.PYRAMID] = CellType.PYRAMID
WEDGE: Literal[CellType.WEDGE] = CellType.WEDGE

Info = CellQualityInfo

_CELL_QUALITY_INFO = [
    Info(TRIANGLE, 'area', (0.0, INF), (0.0, INF), (0.0, INF), sqrt(3.0) / 4.0),
    Info(TRIANGLE, 'aspect_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(TRIANGLE, 'aspect_frobenius', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(TRIANGLE, 'condition', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(TRIANGLE, 'distortion', (0.5, 1.0), (0.0, 1.0), (-INF, INF), 1.0),
    Info(TRIANGLE, 'max_angle', (60.0, 90.0), (60.0, 180.0), (0.0, 180.0), 60.0),
    Info(TRIANGLE, 'min_angle', (30.0, 60.0), (0.0, 60.0), (0.0, 360.0), 60.0),
    Info(TRIANGLE, 'scaled_jacobian', (0.5, 2 * R33), (-2 * R33, 2 * R33), (-INF, INF), 1.0),
    Info(TRIANGLE, 'radius_ratio', (1.0, 3.0), (1.0, INF), (1.0, INF), 1.0),
    Info(TRIANGLE, 'shape', (0.25, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(TRIANGLE, 'shape_and_size', (0.25, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(QUAD, 'area', (0.0, INF), (0.0, INF), (-INF, INF), 1.0),
    Info(QUAD, 'aspect_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(QUAD, 'condition', (1.0, 4), (1.0, INF), (1.0, INF), 1.0),
    Info(QUAD, 'distortion', (0.5, 1.0), (0.0, 1.0), (-INF, INF), 1.0),
    Info(QUAD, 'jacobian', (0.0, INF), (0.0, INF), (-INF, INF), 1.0),
    Info(QUAD, 'max_aspect_frobenius', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(QUAD, 'max_angle', (90.0, 135.0), (90.0, 360.0), (0.0, 360.0), 90.0),
    Info(QUAD, 'max_edge_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(QUAD, 'med_aspect_frobenius', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(QUAD, 'min_angle', (45.0, 90.0), (0.0, 90.0), (0.0, 360.0), 90.0),
    Info(QUAD, 'oddy', (0.0, 0.5), (0.0, INF), (0.0, INF), 0.0),
    Info(QUAD, 'radius_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(QUAD, 'relative_size_squared', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(QUAD, 'scaled_jacobian', (0.3, 1.0), (-1.0, 1.0), (-1.0, 1.0), 1.0),
    Info(QUAD, 'shape', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(QUAD, 'shape_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(QUAD, 'shear', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(QUAD, 'shear_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(QUAD, 'skew', (0.0, 0.7), (0.0, 1.0), (0.0, 1.0), 0.0),
    Info(QUAD, 'stretch', (0.25, 1.0), (0.0, 1.0), (0.0, INF), 1.0),
    Info(QUAD, 'taper', (0.0, 0.7), (0.0, INF), (0.0, INF), 0.0),
    Info(QUAD, 'warpage', (0.5, 1.0), (0.0, 2.0), (0.0, INF), 1.0),
    Info(TETRA, 'aspect_frobenius', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(TETRA, 'aspect_gamma', (1.0, 3.0), (1.0, INF), (1.0, INF), 1.0),
    Info(TETRA, 'aspect_ratio', (1.0, 3.0), (1.0, INF), (1.0, INF), 1.0),
    Info(TETRA, 'collapse_ratio', (0.1, INF), (0.0, INF), (0.0, INF), sqrt(6.0) / 3.0),
    Info(TETRA, 'condition', (1.0, 3), (1.0, INF), (1.0, INF), 1.0),
    Info(TETRA, 'distortion', (0.5, 1.0), (0.0, 1.0), (-INF, INF), 1.0),
    Info(TETRA, 'jacobian', (0.0, INF), (0.0, INF), (-INF, INF), R22),
    Info(TETRA, 'min_angle', (40, ANGLE), (0.0, ANGLE), (0.0, 360), ANGLE),
    Info(TETRA, 'radius_ratio', (1.0, 3), (1.0, INF), (1.0, INF), 1.0),
    Info(TETRA, 'relative_size_squared', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(TETRA, 'scaled_jacobian', (0.5, 1.0), (-1.0, 1.0), (-INF, INF), 1.0),
    Info(TETRA, 'shape', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(TETRA, 'shape_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(TETRA, 'volume', (0.0, INF), (-INF, INF), (-INF, INF), sqrt(2.0) / 12.0),
    Info(HEXAHEDRON, 'diagonal', (0.65, 1.0), (0.0, 1.0), (0.0, INF), 1.0),
    Info(HEXAHEDRON, 'dimension', (0.0, INF), (0.0, INF), (0.0, INF), R33),
    Info(HEXAHEDRON, 'distortion', (0.5, 1.0), (0.0, 1.0), (-INF, INF), 1.0),
    Info(HEXAHEDRON, 'jacobian', (0.0, INF), (0.0, INF), (-INF, INF), 1.0),
    Info(HEXAHEDRON, 'max_edge_ratio', (1.0, 1.3), (1.0, INF), (1.0, INF), 1.0),
    Info(HEXAHEDRON, 'max_aspect_frobenius', (1.0, 3), (1.0, INF), (1.0, INF), 1.0),
    Info(HEXAHEDRON, 'med_aspect_frobenius', (1.0, 3), (1.0, INF), (1.0, INF), 1.0),
    Info(HEXAHEDRON, 'oddy', (0.0, 0.5), (0.0, INF), (0.0, INF), 0.0),
    Info(HEXAHEDRON, 'relative_size_squared', (0.5, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(HEXAHEDRON, 'scaled_jacobian', (0.5, 1.0), (-1.0, 1.0), (-1.0, INF), 1.0),
    Info(HEXAHEDRON, 'shape', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(HEXAHEDRON, 'shape_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(HEXAHEDRON, 'shear', (0.3, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(HEXAHEDRON, 'shear_and_size', (0.2, 1.0), (0.0, 1.0), (0.0, 1.0), 1.0),
    Info(HEXAHEDRON, 'skew', (0.0, 0.5), (0.0, 1.0), (0.0, INF), 0.0),
    Info(HEXAHEDRON, 'stretch', (0.25, 1.0), (0.0, 1.0), (0.0, INF), 1.0),
    Info(HEXAHEDRON, 'taper', (0.0, 0.5), (0.0, INF), (0.0, INF), 0.0),
    Info(HEXAHEDRON, 'volume', (0.0, INF), (0.0, INF), (-INF, INF), 1.0),
    Info(PYRAMID, 'volume', (0.0, INF), (-INF, INF), (-INF, INF), sqrt(2.0) / 6.0),
    Info(WEDGE, 'volume', (0.0, INF), (-INF, INF), (-INF, INF), sqrt(3.0) / 4.0),
]

# Create lookup dict
_CELL_QUALITY_LOOKUP: dict[CellType, dict[_CellQualityLiteral, CellQualityInfo]] = {}
for info in _CELL_QUALITY_INFO:
    _CELL_QUALITY_LOOKUP.setdefault(info.cell_type, {})
    _CELL_QUALITY_LOOKUP[info.cell_type][info.quality_measure] = info


_CELL_TYPE_NAMES = [typ.name for typ in _CELL_QUALITY_LOOKUP.keys()]


def cell_quality_info(
    cell_type: _CellTypesLiteral | _CellTypeNamesLiteral,
    quality_measure: _CellQualityLiteral,
) -> CellQualityInfo:
    """Return information about a cell's quality measure.

    This function returns information about a quality measure computed by
    :meth:`~pyvista.DataObjectFilters.cell_quality` for a specified
    :class:`~pyvista.CellType`. The following is provided for each measure:

    - ``acceptable_range``: Well-behaved cells have values in this range.
    - ``normal_range``: All cells except those with degeneracies have values in this range.
    - ``full_range``: All cells including degenerate ones have values in this range.
    - ``unit_cell_value``: The quality measure value for a reference unit cell (e.g.
      equilateral triangle with edge length of one for triangles).

    This information can help inform if a particular cell is of high or low quality.

    See the tables below for a summary of all cell quality info available from this
    function.

    .. include:: /api/core/cell_quality/cell_quality_info_table_TRIANGLE.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_QUAD.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_HEXAHEDRON.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_TETRA.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_WEDGE.rst

    .. include:: /api/core/cell_quality/cell_quality_info_table_PYRAMID.rst


    .. note::

        The information returned by this function is based on the
        `Verdict Library Reference Manual <https://public.kitware.com/Wiki/images/6/6b/VerdictManual-revA.pdf>`_.
        Since this reference has `known errors <https://gitlab.kitware.com/vtk/vtk/-/issues/19644>`_,
        some values have been adjusted so that the returned values are correct.

    .. note::

        Information is not available for all valid quality measures computed by
        :meth:`~pyvista.DataObjectFilters.cell_quality`. Only a subset
        is provided here. If information about a measure is missing and you have
        knowledge about its acceptable range, normal range, etc., please consider
        submitting a pull request on GitHub at https://github.com/pyvista/pyvista.

    Parameters
    ----------
    cell_type : CellType | str
        Cell type to get information about. May be a :class:`~pyvista.CellType` or the
        name of a cell type as a string.

    quality_measure : str
        Quality measure to get information about. May be any quality measure from
        :ref:`cell_quality_measures_table`.

    Returns
    -------
    CellQualityInfo
        Dataclass with information about the quality measure for a specific cell type.

    Raises
    ------
    ValueError
        If info is not available for the specified cell type or measure.

    See Also
    --------
    :meth:`~pyvista.DataObjectFilters.cell_quality`

    Examples
    --------
    Get cell quality info for :attr:`~pyvista.CellType.TRIANGLE` cells and the
    ``'scaled_jacobian'`` quality measure.

    >>> import pyvista as pv
    >>> info_tri = pv.cell_quality_info(pv.CellType.TRIANGLE, 'scaled_jacobian')
    >>> info_tri  # doctest: +NORMALIZE_WHITESPACE
    CellQualityInfo(cell_type=<CellType.TRIANGLE: 5>,
                    quality_measure='scaled_jacobian',
                    acceptable_range=(0.5, 1.1547005383792515),
                    normal_range=(-1.1547005383792515, 1.1547005383792515),
                    full_range=(-inf, inf),
                    unit_cell_value=1.0)

    Show the acceptable range for this measure.

    >>> info_tri.acceptable_range
    (0.5, 1.1547005383792515)

    Show the value of this measure for equilateral triangles with edge length of one.

    >>> info_tri.unit_cell_value
    1.0

    Get info for the same measure but for :attr:`~pyvista.CellType.QUAD` cells.

    >>> info_quad = pv.cell_quality_info(pv.CellType.QUAD, 'scaled_jacobian')
    >>> info_quad  # doctest: +NORMALIZE_WHITESPACE
    CellQualityInfo(cell_type=<CellType.QUAD: 9>,
                    quality_measure='scaled_jacobian',
                    acceptable_range=(0.3, 1.0),
                    normal_range=(-1.0, 1.0),
                    full_range=(-1.0, 1.0),
                    unit_cell_value=1.0)

    Show the acceptable range. Note that it differs for quads compared to triangles.

    >>> info_quad.acceptable_range
    (0.3, 1.0)

    Show the value of this measure for a square cell with edge length of one.

    >>> info_quad.unit_cell_value
    1.0

    See :ref:`mesh_quality_example` for more examples using this function.

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
        return measures[quality_measure]
    except KeyError:
        item = f'{value.name!r} measure {quality_measure!r}'
        valid_options = list(measures.keys())
        raise_error(item, valid_options)
