"""`pyvista report` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Annotated
from typing import get_args

from cyclopts import Parameter

import pyvista as pv
from pyvista.core.celltype import CellType
from pyvista.core.filters.data_object import _LiteralMeshValidationFields
from pyvista.core.filters.data_object import _MeshValidator
from pyvista.core.utilities.reader import _mesh_types

from .app import app
from .utils import HELP_FORMATTER
from .utils import _console_error
from .utils import _converter_files

if TYPE_CHECKING:
    from collections.abc import Iterable

fields_help = """
Field(s) to validate. Specify individual field(s) or group(s) of fields:

**Data fields**

- ``cell_data_wrong_length``: Ensure the length of each cell data array matches _n_cells_.
- ``point_data_wrong_length``: Ensure the length of each point data array matches _n_points_.

**Point fields**

- ``non_finite_points``: Ensure all points have real values (i.e. no _NaN_ or _Inf_).
- ``unused_points``: Ensure all points are referenced by at least one cell.

**Cell fields**

- ``coincident_points``: Ensure there are no duplicate coordinates or repeated use of the
  same connectivity entry.
- ``degenerate_faces``: Ensure faces do not collapse to a line or a point through repeated
  collocated vertices.
- ``intersecting_edges``: Ensure two edges of a 2D cell do not intersect.
- ``intersecting_faces``: Ensure two faces of a 3D cell do not intersect.
- ``invalid_point_references``: Ensure all points referenced by cells are valid point ids
  that can be indexed by a dataset's _points_.
- ``inverted_faces``: Ensure the faces of a cell point in the direction required by its
  cell type.
- ``negative_size``: Ensure 1D, 2D, and 3D cells have positive length, area, and volume,
  respectively.
- ``non_contiguous_edges``: Ensure edges around the perimeter of a 2D cell are contiguous.
- ``non_convex``: Ensure all 2D and 3D cells are convex.
- ``non_planar_faces``: Ensure vertices for a face all lie in the same plane.
- ``wrong_number_of_points``: Ensure each cell has the minimum number of points needed to
  describe it.
- ``zero_size``: Ensure 1D, 2D, and 3D cells have non-zero length, area, and volume,
  respectively.

**Field groups**

- ``data``: Validate all data fields.
- ``points``: Validate all point fields.
- ``cells``: Validate all cell fields.
- ``memory_safe``: Validate all fields that, if invalid, may cause a segmentation
  fault and crash Python. Includes ``cell_data_wrong_length``,
  ``point_data_wrong_length``, and ``invalid_point_references``.
"""
exclude_help = """
Field(s) to exclude from the validation. This is similar to using FIELDS, but is subtractive
instead of additive.
"""


def _format_color(text: str, names: Iterable[str], style: str) -> str:
    """Highlight all occurrences of `names` in `text` using Rich style markup."""
    for name in names:
        text = text.replace(name, f'[{style}]{name}[/{style}]')
    return text


@app.command(
    usage=f'Usage: [bold]{pv.__name__} validate MESH-PATH [FIELDS...] [--exclude FIELDS...]',
    help_formatter=HELP_FORMATTER,
    help="Validate a mesh's array data, points, and cells.",
)
def _validate(
    mesh_path: Annotated[
        str,
        Parameter(
            help='Mesh to validate. Must be readable with ``pyvista.read``.',
            converter=_converter_files,
        ),
    ],
    fields: Annotated[
        list[_LiteralMeshValidationFields] | None,
        Parameter(
            name='fields',
            consume_multiple=True,
            negative=[],
            help=fields_help,
        ),
    ] = None,
    *,
    exclude: Annotated[
        list[_LiteralMeshValidationFields] | None,
        Parameter(
            name=('exclude', '-e'),
            consume_multiple=True,
            negative=[],
            help=exclude_help,
        ),
    ] = None,
) -> None:
    mesh = mesh_path[0].mesh  # type: ignore[attr-defined]
    path = mesh_path[0].path  # type: ignore[attr-defined]
    try:
        report = pv.DataObjectFilters.validate_mesh(
            mesh, validation_fields=fields, exclude_fields=exclude
        )
    except Exception as e:  # noqa: BLE001
        msg = (
            f'Failed to validate {mesh.__class__.__name__} mesh read from path {str(path)!r}\n{e}'
        )
        _console_error(app=app, message=msg)
    else:
        report_string = str(report)

        # Highlight cell types in yellow
        cell_names = {celltype.name for celltype in CellType}
        report_string = _format_color(report_string, cell_names, 'yellow')

        # Highlight mesh types in purple
        mesh_names = {*get_args(_mesh_types), 'ExplicitStructuredGrid'}
        report_string = _format_color(report_string, mesh_names, 'purple')

        # Make section headings bold
        section_headings = _MeshValidator._SECTION_HEADINGS
        report_string = _format_color(report_string, section_headings, 'bold')

        # Highlight invalid fields in message as red
        # Reverse sort to ensure we replace things like 'unused_points' before 'unused_point'
        norm_field_names = sorted(_MeshValidator._NORMALIZED_FIELD_NAMES)[::-1]
        report_string = _format_color(report_string, norm_field_names, 'red')

        app.console.print(report_string, markup=True)
