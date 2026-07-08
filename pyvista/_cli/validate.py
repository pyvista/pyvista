"""`pyvista report` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Annotated
from typing import cast
from typing import get_args

from cyclopts import Parameter
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

import pyvista as pv
from pyvista.core.filters.data_object import _LiteralMeshValidationFields
from pyvista.core.filters.data_object import _MeshValidator
from pyvista.core.filters.data_object import _ReportBodyOptions

from .app import CLI_APP
from .utils import HELP_FORMATTER
from .utils import print_error_and_exit
from .utils import read_mesh
from .utils import skip_unreadable
from .utils import validate_paths

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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
tolerance_help = """
Value used for most floating point equality checks throughout the cell checking process, e.g. for
checking coincident points or intersecting edges.
"""
planarity_tolerance_help = """
Allowed relative distance a planar polyhedral cell face may protrude out of its plane compared to
the largest distance between a face center and any of its corner points.
"""
size_tolerance_help = """
Value used for evaluating the size of a cell. Cells with an absolute size less than or equal to
this value are flagged as having zero size, and cells with a size less than this value are flagged
as having negative size.
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


@CLI_APP.command(
    usage=f'Usage: [bold]{pv.__name__} validate PATH... [--fields FIELD...] [--exclude FIELD...]',
    help_formatter=HELP_FORMATTER,
    help='Validate data, points, and cells for one or more mesh files.',
)
def _validate(
    paths: Annotated[
        list[str],
        Parameter(
            consume_multiple=True,
            help=(
                'Mesh(es) to validate. Must be readable with ``pyvista.read``. '
                'Glob patterns (``*``, ``?``, ``[...]``) are expanded; '
                'every match is validated in turn.'
            ),
        ),
    ],
    /,
    *,
    fields: Annotated[
        list[_LiteralMeshValidationFields] | None,
        Parameter(
            name=('fields', '-f'),
            consume_multiple=True,
            negative=[],
            help=fields_help,
        ),
    ] = None,
    exclude: Annotated[
        list[_LiteralMeshValidationFields] | None,
        Parameter(
            name=('exclude', '-e'),
            consume_multiple=True,
            negative=[],
            help=exclude_help,
        ),
    ] = None,
    tolerance: Annotated[
        float | None,
        Parameter(
            name='tolerance',
            show_default=False,
            help=tolerance_help,
        ),
    ] = None,
    planarity_tolerance: Annotated[
        float | None,
        Parameter(
            name='planarity-tolerance',
            show_default=False,
            help=planarity_tolerance_help,
        ),
    ] = None,
    size_tolerance: Annotated[
        float | None,
        Parameter(
            name='size-tolerance',
            show_default=False,
            help=size_tolerance_help,
        ),
    ] = None,
    report: Annotated[
        list[_ReportBodyOptions] | None,
        Parameter(
            name='report',
            consume_multiple=True,
            converter=_converter_report,
            show_default=False,
            negative=[],
            help=report_help,
        ),
    ] = None,
    skip_unreadable: skip_unreadable = False,
) -> None:
    valid_paths = validate_paths(paths)
    report_body = report[0] if report else 'message'
    n_paths = len(valid_paths)

    if n_paths == 1:
        path = valid_paths[0]
        mesh = read_mesh(
            path,
            on_error='suppress' if skip_unreadable else 'exit',
        )
        if mesh is not None:
            _validate_one(
                mesh,
                path,
                announce=True,
                fields=fields,
                exclude=exclude,
                tolerance=tolerance,
                planarity_tolerance=planarity_tolerance,
                size_tolerance=size_tolerance,
                report=report,
                report_body=report_body,
            )
    else:
        _validate_many(
            valid_paths,
            skip_unreadable=skip_unreadable,
            fields=fields,
            exclude=exclude,
            tolerance=tolerance,
            planarity_tolerance=planarity_tolerance,
            size_tolerance=size_tolerance,
            report=report,
            report_body=report_body,
        )


def _check_mesh_type(mesh: object, path: Path) -> None:
    """Exit with a console error if ``mesh`` is not a supported type."""
    if not isinstance(mesh, (pv.DataSet, pv.MultiBlock)):
        msg = (
            f'Cannot validate {type(mesh).__name__} read from path {str(path)!r}: '
            f'only DataSet and MultiBlock meshes are supported.'
        )
        print_error_and_exit(message=msg)


def _validate_one(
    mesh: pv.DataObject,
    path: Path,
    *,
    fields: list[_LiteralMeshValidationFields] | None,
    exclude: list[_LiteralMeshValidationFields] | None,
    tolerance: float | None,
    planarity_tolerance: float | None,
    size_tolerance: float | None,
    report: list[_ReportBodyOptions] | None,
    report_body: _ReportBodyOptions,
    announce: bool,
) -> str | None:
    """Validate a single mesh and optionally print its result to the console.

    Returns None if the mesh is valid, and the report as a string otherwise .
    """
    class_name = mesh.__class__.__name__
    _check_mesh_type(mesh, path)
    try:
        out = pv.DataObjectFilters.validate_mesh(  # type: ignore[type-var]
            mesh,
            name=path.name,
            validation_fields=fields,
            exclude_fields=exclude,
            tolerance=tolerance,
            planarity_tolerance=planarity_tolerance,
            size_tolerance=size_tolerance,
            report_body=report_body,
        )
    except Exception as e:  # noqa: BLE001
        msg = f'Failed to validate {class_name} mesh read from path {str(path)!r}\n{e}'
        print_error_and_exit(message=msg)

    if report is not None:
        report_string = str(out)
        invalid_fields = out.invalid_fields if report_body == 'fields' else None
        output = _MeshValidator._colorize_output(report_string, invalid_fields)
        announcement = output
        console = CLI_APP.console  # Actual program output for stdout
    elif (message := out.message) is not None:
        output = _MeshValidator._colorize_output(message)
        announcement = output
        console = CLI_APP.console  # Actual program output for stdout
    else:
        # Mesh is valid,
        announcement = _mesh_is_valid_message(class_name, path)
        console = CLI_APP.error_console  # print to stderr for user info only
        output = None
    if announce:
        console.print(announcement)
    return output


def _mesh_is_valid_message(class_name: str, path: Path) -> str:
    return f'[green]{class_name} mesh {path.name!r} is valid![/green]'


def _validate_many(
    paths: list[Path],
    *,
    skip_unreadable: bool,
    fields: list[_LiteralMeshValidationFields] | None,
    exclude: list[_LiteralMeshValidationFields] | None,
    tolerance: float | None,
    planarity_tolerance: float | None,
    size_tolerance: float | None,
    report: list[_ReportBodyOptions] | None,
    report_body: _ReportBodyOptions,
) -> None:
    """Validate each mesh under a progress bar and report a summary."""
    columns = (
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('<'),
        TimeRemainingColumn(),
    )

    n_valid = 0
    n_invalid = 0
    skipped: list[Path] = []
    invalid_output: list[str] = []

    with Progress(*columns, console=CLI_APP.error_console, transient=False) as progress:
        task = progress.add_task('Validating', total=len(paths))
        for path in paths:
            progress.update(task, description=f'Validating [cyan]{path.name}[/cyan]')
            mesh = read_mesh(
                path,
                on_error='suppress' if skip_unreadable else 'exit+hint',
            )
            if mesh is None:
                skipped.append(path)
            else:
                output = _validate_one(
                    mesh,
                    path,
                    fields=fields,
                    exclude=exclude,
                    tolerance=tolerance,
                    planarity_tolerance=planarity_tolerance,
                    size_tolerance=size_tolerance,
                    report=report,
                    report_body=report_body,
                    announce=False,
                )

                if output:
                    n_invalid += 1
                    invalid_output.append(output)
                else:
                    n_valid += 1
            progress.update(task, advance=1)

    n_total = n_valid + n_invalid
    if n_invalid:
        msg = f'[red]{n_invalid} invalid[/red] meshes out of {n_total} meshes validated.'
        CLI_APP.error_console.print(msg)
        CLI_APP.error_console.print('\n'.join(invalid_output))
    elif n_total:
        msg = (
            '[green]1 mesh is valid.[/green]'
            if n_total == 1
            else f'[green]All {n_total} meshes are valid.[/green]'
        )
        CLI_APP.error_console.print(msg)

    if n_skipped := len(skipped):
        s = 's' if n_skipped > 1 else ''
        msg = f'\n[yellow]{n_skipped} file{s} skipped (unreadable):[/yellow]'
        CLI_APP.error_console.print(msg)
        for path in skipped:
            CLI_APP.error_console.print(f'  {path}')
