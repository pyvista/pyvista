"""`pyvista report` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Annotated
from typing import cast
from typing import get_args

from cyclopts import Parameter

import pyvista as pv
from pyvista.core.filters.data_object import _LiteralMeshValidationFields
from pyvista.core.filters.data_object import _ReportBodyOptions

from .app import app
from .utils import HELP_FORMATTER
from .utils import _console_error
from .utils import _converter_files

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cyclopts import Token

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
report_help = """
Show report. Control the body of the report with:
- ``fields`` to show all validation fields.
- ``message`` to show the error message (if any).

``message`` is used by default if no tokens are passed.
"""


def _converter_report(
    type_: type,  # noqa: ARG001
    tokens: Sequence[Token],
) -> list[_ReportBodyOptions]:
    values: list[str] = [t.value for t in tokens]
    n_values = len(values)
    if n_values > 1:
        msg = f'Invalid value for {tokens[0].keyword}: accepts 0 or 1 arguments. Got {n_values}.'
        raise ValueError(msg)
    value = values[0]
    allowed = get_args(_ReportBodyOptions)
    if value in allowed:
        return cast('list[_ReportBodyOptions]', [value])
    msg = f'expected one of {str(allowed)[1:-1]} or no value. Got {value!r}.'
    raise ValueError(msg)


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
    report: Annotated[
        list[_ReportBodyOptions] | None,
        Parameter(
            name=('report', '-r'),
            consume_multiple=True,
            converter=_converter_report,
            show_default=False,
            negative=[],
            help=report_help,
        ),
    ] = None,
) -> None:
    mesh = mesh_path[0].mesh  # type: ignore[attr-defined]
    path = mesh_path[0].path  # type: ignore[attr-defined]
    report_body = report[0] if report else 'message'
    class_name = mesh.__class__.__name__
    try:
        out = pv.DataObjectFilters.validate_mesh(
            mesh, validation_fields=fields, exclude_fields=exclude, report_body=report_body
        )
    except Exception as e:  # noqa: BLE001
        msg = f'Failed to validate {class_name} mesh read from path {str(path)!r}\n{e}'
        _console_error(app=app, message=msg)
    else:
        if report is not None:
            # Print the full report
            app.console.print(str(out))
        elif (message := out.message) is not None:
            # Only print the error message
            message = message.replace('mesh is not valid', f'mesh {path.name!r} is not valid')
            app.console.print(message)
        else:
            # Mesh is valid
            app.console.print(f'[green]{class_name} mesh {path.name!r} is valid![/green]')
