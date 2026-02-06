"""A set of common filters that can be applied to any DataSet or MultiBlock."""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Sized
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import fields
import functools
import itertools
import re
import reprlib
from typing import TYPE_CHECKING
from typing import Generic
from typing import Literal
from typing import TypeVar
from typing import cast
from typing import get_args
import warnings

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._version import version_info
from pyvista._warn_external import warn_external
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core import _DataSetOrMultiBlockType
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output
from pyvista.core.utilities.helpers import _NormalsLiteral
from pyvista.core.utilities.helpers import _validate_plane_origin_and_normal
from pyvista.core.utilities.helpers import generate_plane
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import _reciprocal
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.state_manager import _update_alg
from pyvista.core.utilities.transform import Transform

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import ClassVar

    from pyvista import DataSet
    from pyvista import DataSetAttributes
    from pyvista import MultiBlock
    from pyvista import PolyData
    from pyvista import RotationLike
    from pyvista import TransformLike
    from pyvista import VectorLike
    from pyvista import pyvista_ndarray
    from pyvista.core._typing_core import _DataSetType
    from pyvista.core._typing_core import _MultiBlockType
    from pyvista.core.utilities.cell_quality import _CellQualityLiteral

    _MeshType_co = TypeVar('_MeshType_co', DataSet, MultiBlock, covariant=True)


# Matches https://github.com/Kitware/VTK/blob/ac6cb2b3550b7de9c9cfcd731098d453e9fab1b7/Common/DataModel/vtkCellStatus.h#L16-L28
_CELL_VALIDATOR_BIT_FIELD = dict(
    wrong_number_of_points=0x01,
    intersecting_edges=0x02,
    intersecting_faces=0x04,
    non_contiguous_edges=0x08,
    non_convex=0x10,
    inverted_faces=0x20,
    non_planar_faces=0x40,
    degenerate_faces=0x80,
    coincident_points=0x100,
)


class _SENTINEL: ...


_ExtractSurfaceOptions = Literal['geometry', 'dataset_surface', None]  # noqa: PYI061


class _MeshValidator(Generic[_DataSetOrMultiBlockType]):
    _ActionOptions = Literal['warn', 'error']
    _DataFields = Literal[
        'point_data_wrong_length',
        'cell_data_wrong_length',
    ]
    _CellFields = Literal[
        'wrong_number_of_points',
        'intersecting_edges',
        'intersecting_faces',
        'non_contiguous_edges',
        'non_convex',
        'inverted_faces',
        'non_planar_faces',
        'degenerate_faces',
        'coincident_points',
        'invalid_point_references',
    ]
    _PointFields = Literal['unused_points', 'non_finite_points']
    _MemorySafeFields = Literal[
        'invalid_point_references',
        'point_data_wrong_length',
        'cell_data_wrong_length',
    ]
    _DefaultFieldGroups = Literal['data', 'cells', 'points']
    _OtherFieldGroups = Literal['memory_safe']
    _AllValidationOptions = (
        _DataFields | _CellFields | _PointFields | _DefaultFieldGroups | _OtherFieldGroups
    )
    _DEFAULT_MESH_VALIDATION_ARGS = get_args(_DefaultFieldGroups)

    _allowed_data_fields = get_args(_DataFields)
    _allowed_cell_fields = get_args(_CellFields)
    _allowed_point_fields = get_args(_PointFields)
    _allowed_field_groups = (*get_args(_DefaultFieldGroups), *get_args(_OtherFieldGroups))

    @dataclass
    class _FieldSummary:
        name: str
        message: str
        values: Sequence[str | int] | None

    def __init__(
        self,
        mesh: _DataSetOrMultiBlockType,
        validation_fields: _AllValidationOptions
        | Sequence[_AllValidationOptions] = _DEFAULT_MESH_VALIDATION_ARGS,
    ) -> None:
        data_fields, cell_fields, point_fields = _MeshValidator._validate_fields(validation_fields)
        self._validation_report = _MeshValidator._generate_report(
            mesh, data_fields=data_fields, cell_fields=cell_fields, point_fields=point_fields
        )

    @staticmethod
    def _validate_fields(
        validation_fields,
    ) -> tuple[tuple[_DataFields, ...], tuple[_CellFields, ...], tuple[_PointFields, ...]]:
        # Validate inputs
        allowed_data_fields = _MeshValidator._allowed_data_fields
        allowed_cell_fields = _MeshValidator._allowed_cell_fields
        allowed_point_fields = _MeshValidator._allowed_point_fields
        data_fields_to_validate: list[_MeshValidator._DataFields] = []
        cell_fields_to_validate: list[_MeshValidator._CellFields] = []
        point_fields_to_validate: list[_MeshValidator._PointFields] = []

        if validation_fields == _MeshValidator._DEFAULT_MESH_VALIDATION_ARGS:
            # Default values, no need to validate
            data_fields_to_validate.extend(allowed_data_fields)
            cell_fields_to_validate.extend(allowed_cell_fields)
            point_fields_to_validate.extend(allowed_point_fields)
        else:
            allowed_fields_or_groups = (
                *allowed_data_fields,
                *allowed_cell_fields,
                *allowed_point_fields,
                *_MeshValidator._allowed_field_groups,
            )
            if isinstance(validation_fields, str):
                validation_fields = (validation_fields,)
            for field_or_group in validation_fields:
                _validation.check_contains(
                    allowed_fields_or_groups, must_contain=field_or_group, name='validation_fields'
                )

            # Inputs are valid, but we need to categorize them
            input_fields = list(validation_fields)
            if 'memory_safe' in validation_fields:
                # Replace memory_safe group with the actual field names used
                input_fields.remove('memory_safe')
                memory_safe_fields = (
                    field
                    for field in get_args(_MeshValidator._MemorySafeFields)
                    if field not in validation_fields
                )
                input_fields.extend(memory_safe_fields)

            for field_or_group in input_fields:
                if field_or_group == 'data':
                    data_fields_to_validate.extend(allowed_data_fields)
                elif field_or_group == 'cells':
                    cell_fields_to_validate.extend(allowed_cell_fields)
                elif field_or_group == 'points':
                    point_fields_to_validate.extend(allowed_point_fields)
                elif field_or_group in allowed_data_fields:
                    data_fields_to_validate.append(field_or_group)
                elif field_or_group in allowed_cell_fields:
                    cell_fields_to_validate.append(field_or_group)
                elif field_or_group in allowed_point_fields:
                    point_fields_to_validate.append(field_or_group)
                else:  # pragma: no cover
                    msg = (
                        f'Something went wrong! Invalid field or group {field_or_group}. '
                        'This code should not be reachable.'
                    )
                    raise RuntimeError(msg)

        return (
            tuple(data_fields_to_validate),
            tuple(cell_fields_to_validate),
            tuple(point_fields_to_validate),
        )

    @staticmethod
    def _generate_report(
        mesh: _DataSetOrMultiBlockType,
        *,
        data_fields: tuple[_DataFields, ...],
        cell_fields: tuple[_CellFields, ...],
        point_fields: tuple[_PointFields, ...],
    ) -> _MeshValidationReport[_DataSetOrMultiBlockType]:
        with warnings.catch_warnings():
            # Ignore any warnings caused by wrapping alg outputs
            warnings.filterwarnings(
                'ignore',
                category=pv.InvalidMeshWarning,
            )
            if isinstance(mesh, pv.DataSet):
                return _MeshValidator._validate_dataset(  # type: ignore[return-value]
                    mesh,
                    data_fields=data_fields,
                    cell_fields=cell_fields,
                    point_fields=point_fields,
                )
            else:
                return _MeshValidator._validate_multiblock(  # type: ignore[return-value]
                    mesh,
                    data_fields=data_fields,
                    cell_fields=cell_fields,
                    point_fields=point_fields,
                )

    @staticmethod
    def _validate_dataset(
        mesh: _DataSetType,
        *,
        data_fields: tuple[_DataFields, ...],
        cell_fields: tuple[_CellFields, ...],
        point_fields: tuple[_PointFields, ...],
    ) -> _MeshValidationReport[_DataSetType]:
        validated_mesh = mesh.copy(deep=False)
        field_summaries: dict[str, _MeshValidator._FieldSummary] = {}
        # Validate data arrays
        if data_fields:
            for summary in _MeshValidator._validate_data(mesh, data_fields):
                field_summaries[summary.name] = summary

        # Validate cells
        if cell_fields:
            summaries, validated = _MeshValidator._validate_cells(mesh, cell_fields)
            if validated:
                validated_mesh = validated  # Store the output from cell_validator
            for summary in summaries:
                field_summaries[summary.name] = summary

        # Validate points
        if point_fields:
            for summary in _MeshValidator._validate_points(mesh, point_fields):
                field_summaries[summary.name] = summary

        message_body: list[str] = [
            summary.message for summary in field_summaries.values() if summary.values
        ]
        message = _MeshValidator._create_message(validated_mesh, message_body)
        dataclass_fields = {issue.name: issue.values for issue in field_summaries.values()}
        return _MeshValidationReport(
            _mesh=validated_mesh,
            _message=message,
            _subreports=None,
            **dataclass_fields,  # type: ignore[arg-type]
        )

    @staticmethod
    def _validate_multiblock(
        mesh: _MultiBlockType,
        *,
        data_fields: tuple[_DataFields, ...],
        cell_fields: tuple[_CellFields, ...],
        point_fields: tuple[_PointFields, ...],
    ) -> _MeshValidationReport[_MultiBlockType]:
        validated_mesh = mesh.copy(deep=False)

        # Generate reports and error messages for each block
        reports: list[_MeshValidationReport[DataSet] | None] = []
        message_body: list[str] = []
        bullet = _MeshValidator._message_bullet
        for i, block in enumerate(mesh):
            if block is None:
                reports.append(None)
            else:
                report = _MeshValidator._generate_report(
                    block,
                    data_fields=data_fields,
                    cell_fields=cell_fields,
                    point_fields=point_fields,
                )
                reports.append(report)
                validated_mesh.replace(i, report.mesh)

                if (msg := report.message) is not None:
                    prefix = f'Block id {i} {validated_mesh.get_block_name(i)!r}'
                    indented = msg.replace(bullet, '  ' + bullet)
                    message_body.append(f'{prefix} {indented}')

        # Iterate over fields in order and identify blocks with invalid fields
        dataclass_fields: dict[str, Sequence[int | str]] = {}
        for field in [
            *data_fields,
            *cell_fields,
            *point_fields,
        ]:
            invalid_block_ids: list[int] = []
            for i, report in enumerate(reports):  # type: ignore[assignment]
                if report is not None and field in report.invalid_fields:  # type: ignore[redundant-expr]
                    invalid_block_ids.append(i)
            dataclass_fields[field] = invalid_block_ids

        message = _MeshValidator._create_message(validated_mesh, message_body)
        return _MeshValidationReport(
            _mesh=validated_mesh,
            _message=message,
            _subreports=tuple(reports),
            **dataclass_fields,  # type: ignore[arg-type]
        )

    @staticmethod
    def _normalize_field_name(name: str) -> str:
        return name.replace('_', ' ').replace('non ', 'non-')

    @staticmethod
    def _validate_cells(
        mesh: _DataSetType, validation_fields: tuple[_CellFields, ...]
    ) -> tuple[list[_MeshValidator._FieldSummary], _DataSetType | None]:
        """Validate cells and only return summary objects for the requested fields."""
        summaries: list[_MeshValidator._FieldSummary] = []
        validated_mesh = None
        mutable_validation_fields = list(validation_fields)
        if 'invalid_point_references' in mutable_validation_fields:
            mutable_validation_fields.remove('invalid_point_references')
            summary = _MeshValidator._validate_invalid_point_references(mesh)
            summaries.append(summary)
        if mutable_validation_fields:
            validated_mesh = mesh.cell_validator()
            for name in mutable_validation_fields:
                array = validated_mesh.field_data[name].tolist()
                msg = _MeshValidator._invalid_cell_msg(name, array)
                summary = _MeshValidator._FieldSummary(name=name, message=msg, values=array)
                summaries.append(summary)
        return summaries, validated_mesh

    @staticmethod
    def _validate_invalid_point_references(
        mesh: DataSet,
    ) -> _MeshValidator._FieldSummary:
        def _find_cells_with_invalid_point_refs() -> list[int]:
            """Return cell IDs that reference points that do not exist."""
            if hasattr(mesh, 'dimensions'):
                return []  # Cells are implicitly defined and cannot be invalid
            grid = (
                mesh if isinstance(mesh, pv.UnstructuredGrid) else mesh.cast_to_unstructured_grid()
            )

            # Find indices in the connectivity array that are invalid
            conn = grid.cell_connectivity
            invalid_indices = np.where((conn < 0) | (conn >= grid.n_points))[0]
            if len(invalid_indices) == 0:
                return []

            # Map invalid connectivity indices back to cell IDs using offsets
            # Each invalid index belongs to the cell whose start offset <= index < next offset
            cell_ids = np.searchsorted(grid.offset, invalid_indices, side='right') - 1
            return np.unique(cell_ids).tolist()

        name = 'invalid_point_references'
        array = _find_cells_with_invalid_point_refs()
        msg = _MeshValidator._invalid_cell_msg(name, array)
        return _MeshValidator._FieldSummary(name=name, message=msg, values=array)

    @staticmethod
    def _invalid_cell_msg(name: str, array: list[int]) -> str:
        if not array:
            return ''
        name_norm = _MeshValidator._normalize_field_name(name)
        # Need to write name either before of after the word "cell"
        if name == 'non_convex':
            before = f' {name_norm} '
            after = ''
        else:
            before = ' '
            after = f' with {name_norm}'
        s = 's' if len(array) > 1 else ''
        return (
            f'Mesh has {len(array)}{before}cell{s}{after}. '
            f'Invalid cell id{s}: {reprlib.repr(array)}'
        )

    @staticmethod
    def _validate_data(
        mesh: DataSet, validation_fields: tuple[_DataFields, ...]
    ) -> list[_MeshValidator._FieldSummary]:
        """Validate data arrays and only return summary objects for the requested fields."""

        def join_limited(items, max_items=4):
            if len(items) <= max_items:
                return ', '.join(items)
            return ', '.join(items[:max_items]) + ', ...'

        def _invalid_array_length_msg(
            invalid_arrays: dict[str, int], kind: str, expected: int
        ) -> str:
            if not invalid_arrays:
                return ''
            n_arrays = len(invalid_arrays)
            s = 's' if n_arrays > 1 else ''
            msg_template = (
                'Mesh has {n_arrays} {kind} array{s} with incorrect length '
                '(length must be {expected}). Invalid array{s}: {details}'
            )
            details = join_limited(
                [f'{name!r} ({length})' for name, length in invalid_arrays.items()]
            )
            return msg_template.format(
                n_arrays=n_arrays,
                kind=kind,
                kind_lower=kind.lower() + 's',
                expected=expected,
                details=details,
                s=s,
            )

        def _validate_array_lengths(arrays: DataSetAttributes, expected: int) -> dict[str, int]:
            return {name: len(arrays[name]) for name in arrays if len(arrays[name]) != expected}

        summaries: list[_MeshValidator._FieldSummary] = []
        for (
            name,
            kind,
            data,
            expected_n,
        ) in [
            ('point_data_wrong_length', 'point', mesh.point_data, mesh.n_points),
            ('cell_data_wrong_length', 'cell', mesh.cell_data, mesh.n_cells),
        ]:
            if name in validation_fields:
                invalid_arrays: dict[str, int] = _validate_array_lengths(data, expected_n)
                message = _invalid_array_length_msg(
                    invalid_arrays=invalid_arrays, kind=kind, expected=expected_n
                )
                issue = _MeshValidator._FieldSummary(
                    name=name, message=message, values=list(invalid_arrays.keys())
                )
                summaries.append(issue)
        return summaries

    @staticmethod
    def _validate_points(
        mesh: DataSet, validation_fields: tuple[_PointFields, ...]
    ) -> list[_MeshValidator._FieldSummary]:
        """Validate points and only return summary objects for the requested fields."""

        def get_unused_point_ids() -> list[int]:
            if hasattr(mesh, 'dimensions'):
                return []  # Cells are implicitly defined and cannot have unused points
            grid = (
                mesh if isinstance(mesh, pv.UnstructuredGrid) else mesh.cast_to_unstructured_grid()
            )
            all_points = np.arange(grid.n_points)
            # Note: This may not include points used by Polyhedron cells
            used_points = np.unique(grid.cell_connectivity)
            return np.setdiff1d(all_points, used_points, assume_unique=True).tolist()

        def get_non_finite_point_ids() -> list[int]:
            if isinstance(mesh, pv.Grid):
                return []  # Points are implicitly defined and cannot be non-finite
            mask = ~np.isfinite(mesh.points).all(axis=1)
            return np.where(mask)[0].tolist()

        def invalid_points_msg(name_: str, array: list[int], info_: str) -> str:
            if not array:
                return ''
            name_norm = _MeshValidator._normalize_field_name(name_)
            name_norm = name_norm.removesuffix('s')
            n_ids = len(array)
            s = 's' if n_ids > 1 else ''
            return (
                f'Mesh has {n_ids} {name_norm}{s}{info_}. Invalid point id{s}: '
                f'{reprlib.repr(array)}'
            )

        summaries: list[_MeshValidator._FieldSummary] = []
        for name, func, info in [
            ('unused_points', get_unused_point_ids, ' not referenced by any cell(s)'),
            ('non_finite_points', get_non_finite_point_ids, ''),
        ]:
            if name in validation_fields:
                point_ids = func()
                msg = invalid_points_msg(name, point_ids, info)
                issue = _MeshValidator._FieldSummary(name=name, message=msg, values=point_ids)
                summaries.append(issue)
        return summaries

    _message_bullet = ' - '

    @staticmethod
    def _create_message(obj: object, message_body: list[str]) -> str:
        bullet = _MeshValidator._message_bullet
        body = bullet + f'\n{bullet}'.join(message_body)
        header = _MeshValidator._create_message_header(obj)
        return f'{header}\n{body}'

    @staticmethod
    def _create_message_header(obj: object) -> str:
        return f'{obj.__class__.__name__} mesh is not valid due to the following problems:'

    @property
    def validation_report(self) -> _MeshValidationReport[_DataSetOrMultiBlockType]:
        return self._validation_report


# Create alias for reuse/export to other modules
_MeshValidationOptions = (
    _MeshValidator._AllValidationOptions | Sequence[_MeshValidator._AllValidationOptions]
)


@dataclass(frozen=True)
class _MeshValidationReport(_NoNewAttrMixin, Generic[_DataSetOrMultiBlockType]):
    """Dataclass to report mesh validation results."""

    # Non-fields
    _mesh: InitVar[_DataSetOrMultiBlockType]
    _message: InitVar[str | None]
    _subreports: InitVar[tuple[_MeshValidationReport[DataSet] | None, ...] | None]

    # Data fields
    point_data_wrong_length: list[str] | None = None
    cell_data_wrong_length: list[str] | None = None

    # Cell fields
    wrong_number_of_points: list[int] | None = None
    intersecting_edges: list[int] | None = None
    intersecting_faces: list[int] | None = None
    non_contiguous_edges: list[int] | None = None
    non_convex: list[int] | None = None
    inverted_faces: list[int] | None = None
    non_planar_faces: list[int] | None = None
    degenerate_faces: list[int] | None = None
    coincident_points: list[int] | None = None
    invalid_point_references: list[int] | None = None

    # Point fields
    unused_points: list[int] | None = None
    non_finite_points: list[int] | None = None

    def __post_init__(
        self,
        _mesh: _DataSetOrMultiBlockType,
        _message: str | None,
        _subreports: tuple[_MeshValidationReport[DataSet] | None, ...] | None,
    ) -> None:
        object.__setattr__(self, '_mesh', _mesh)
        object.__setattr__(self, '_message', _message)
        object.__setattr__(self, '_subreports', _subreports)

    @property
    def mesh(self) -> _DataSetOrMultiBlockType:
        return self._mesh  # type: ignore[attr-defined]

    @property
    def message(self) -> str | None:
        return None if self.is_valid else self._message  # type: ignore[attr-defined]

    @property
    def is_valid(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if the mesh is valid."""
        return not self.invalid_fields

    @property
    def invalid_fields(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """Return any field names which have values."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name))

    def __getitem__(self, index: int) -> _MeshValidationReport[_DataSetType] | None:
        subreports: tuple[_MeshValidationReport[_DataSetType] | None, ...] | None = (
            self._subreports  # type: ignore[attr-defined]
        )
        if subreports is None:
            msg = 'Indexing mesh validation reports is only supported for composite meshes.'
            raise TypeError(msg)
        return subreports[index]

    def __len__(self) -> int:
        subreports: tuple[_MeshValidationReport[DataSet] | None, ...] | None = self._subreports  # type: ignore[attr-defined]
        if subreports is None:
            msg = 'Length of mesh validation report is only defined for composite meshes.'
            raise TypeError(msg)
        return len(subreports)

    def __str__(self) -> str:
        """Include all validation results in a printable string."""
        summary_fields = ['is_valid', 'invalid_fields']
        dataset_fields = [f.name for f in fields(self)]

        def compute_label_width() -> int:
            max_width = 0
            for name in [*summary_fields, *dataset_fields]:
                width = len(name)
                if (value := getattr(self, name)) and isinstance(value, Sized):
                    num_digits = len(str(len(value)))
                    len_space_plus_brackets = 3  # Value will be printed inside two brackets
                    width += num_digits + len_space_plus_brackets
                max_width = max(max_width, width)

            return max_width

        indent = ' ' * 4
        label_width = compute_label_width()
        lines: list[str] = []

        title = 'Mesh Validation Report'
        lines.append(title)
        lines.append('-' * len(title))

        def emit_group(name: str, field_names: Sequence[str]) -> None:
            if all(getattr(self, field) is None for field in field_names):
                return
            lines.append(f'{name}:')
            for field in field_names:
                value = getattr(self, field)
                if value is not None or field in summary_fields:
                    label = _MeshValidator._normalize_field_name(field).capitalize()
                    n_values = ''
                    try:
                        length = len(value)
                        if length:
                            n_values = f' ({length!s})'
                    except TypeError:
                        pass

                    lines.append(
                        f'{indent}{label + n_values:<{label_width}} : {reprlib.repr(value)}'
                    )

        def emit_mesh_info() -> None:
            lines.append('Mesh:')
            for key, value in mesh_items.items():
                lines.append(f'{indent}{key:<{label_width}} : {value}')

        mesh = self.mesh
        mesh_items: dict[str, str | int] = {'Type': mesh.__class__.__name__}
        # Set report content based on mesh type
        if isinstance(mesh, pv.DataSet):
            mesh_items['N Points'] = mesh.n_points
            mesh_items['N Cells'] = mesh.n_cells
            data_text = 'Invalid data arrays'
            cell_text = 'Invalid cell ids'
            point_text = 'Invalid point ids'
        else:
            mesh_items['N Blocks'] = mesh.n_blocks
            data_text = 'Blocks with invalid data arrays'
            cell_text = 'Blocks with invalid cells'
            point_text = 'Blocks with invalid points'

        emit_mesh_info()
        emit_group('Report summary', summary_fields)
        emit_group(data_text, _MeshValidator._allowed_data_fields)
        emit_group(cell_text, _MeshValidator._allowed_cell_fields)
        emit_group(point_text, _MeshValidator._allowed_point_fields)

        return '\n'.join(lines)


@abstract_class
class DataObjectFilters:
    """A set of common filters that can be applied to any DataSet or MultiBlock."""

    points: pyvista_ndarray

    def validate_mesh(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        validation_fields: _MeshValidationOptions | None = None,
        action: _MeshValidator._ActionOptions | None = None,
    ) -> _MeshValidationReport[_DataSetOrMultiBlockType]:
        """Validate this mesh's array data, cells, and points.

        This method returns a ``MeshValidationReport`` dataclass with information about the
        validity of a mesh. The dataclass contains validation fields which are specific to the
        mesh's data, cells, and points. By default, all validation fields below are checked and
        included in the report. Optionally, only a subset of fields may be requested, and a
        warning or error may be raised if the mesh is not valid.

        **Data validation fields**

        Fields specific to :attr:`~pyvista.DataSet.point_data` and
        :attr:`~pyvista.DataSet.cell_data` arrays.

        - ``point_data_wrong_length``: Ensure the length of each point data array matches
          :attr:`~pyvista.DataSet.n_points`.
        - ``cell_data_wrong_length``: Ensure the length of each cell data array matches
          :attr:`~pyvista.DataSet.n_cells`.

        .. note::
            When setting new arrays using PyVista's API, similar array validation checks are
            `already` implicitly performed. As such, these checks may be redundant in many cases.
            They are most useful for validating `newly` loaded or :func:`wrapped <pyvista.wrap>`
            meshes.

        **Cell validation fields**

        - ``wrong_number_of_points``: Ensure each cell has the minimum number of points needed to
          describe it.
        - ``intersecting_edges``: Ensure two edges of a 2D cell do not intersect.
        - ``intersecting_faces``: Ensure two faces of a 3D cell do not intersect.
        - ``non_contiguous_edges``: Ensure edges around the perimeter of a 2D cell are contiguous.
        - ``non_convex``: Ensure all 2D and 3D cells are convex.
        - ``inverted_faces``: Ensure the faces of a cell point in the direction required by its
          :class:`~pyvista.CellType`.
        - ``non_planar_faces``: Ensure vertices for a face all lie in the same plane.
        - ``degenerate_faces``: Ensure faces do not collapse to a line or a point through repeated
          collocated vertices.
        - ``coincident_points``: Ensure there are no duplicate coordinates or repeated use of the
          same connectivity entry.
        - ``invalid_point_references``: Ensure all points referenced by cells are valid point ids
          that can be indexed.

        .. note::
          Other than ``invalid_point_references``, all cell fields are computed using
          :meth:`~pyvista.DataObjectFilters.cell_validator`.

        **Point validation fields**

        - ``unused_points``: Ensure all points are referenced by at least one cell.
        - ``non_finite_points``: Ensure all points have real values (i.e. no ``NaN`` or ``Inf``).

        For each field, its value is:

        - ``None`` if the field is omitted from the report,
        - an empty list ``[]`` if the field is included but there is no issue to report for it, or
        - a list of invalid items (e.g. invalid array names or cell/point ids).

        In addition to the validation fields above, the report includes properties for
        convenience:

        - ``is_valid``: evaluates to ``True`` when all fields are ``None`` or empty.
        - ``invalid_fields``: tuple of validation field names where problems were detected.
        - ``mesh``: a shallow copy of the validated mesh. If any cell fields are included which
          are computed by :meth:`~pyvista.DataObjectFilters.cell_validator`, this mesh includes the
          output from that filter.
        - ``message``: message string generated by the report. The message contains a compact
          summary of any problems detected, and is formatted for printing to console. This is the
          message used when the ``action`` keyword is set for emitting warnings or raising errors.
          This value is ``None`` if the mesh is valid.

        Validating composite :class:`~pyvista.MultiBlock` is also supported. In this case, all
        mesh blocks are validated separately and the results are aggregated and reported per-block.

        .. versionadded:: 0.47

        Parameters
        ----------
        validation_fields : str | sequence[str], default: ('data', 'cells', 'points')
            Select which field(s) to include in the validation report. All data, cell, and point
            fields are included by default. Specify individual fields by name, or use group name(s)
            to include multiple related validation fields:

            - ``'data'`` to include all data fields
            - ``'cells'`` to include all cell fields
            - ``'points'`` to include all point fields
            - ``'memory_safe'`` to include all fields that, if invalid, may cause a segmentation
              fault and crash Python. This option includes ``point_data_wrong_length``,
              ``cell_data_wrong_length``, and ``invalid_point_references``.

            Fields that are excluded from the report will have a value of ``None``.

        action : 'warn' | 'error', optional
            Issue a warning or raise an error if the mesh is not valid for the specified fields.
            By default, no action is taken.

        Returns
        -------
        MeshValidationReport
            Report dataclass with information about mesh validity.

        See Also
        --------
        :meth:`~pyvista.DataObjectFilters.cell_validator`
        :meth:`~pyvista.DataObjectFilters.cell_quality`
        :meth:`~pyvista.UnstructuredGridFilters.remove_unused_points`
        :ref:`mesh_validation_example`

        Examples
        --------
        Create a :func:`~pyvista.Sphere` and check if it's a valid mesh.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = pv.Sphere()
        >>> report = mesh.validate_mesh()
        >>> report.is_valid
        True

        Print the full report.

        >>> print(report)
        Mesh Validation Report
        ----------------------
        Mesh:
            Type                     : PolyData
            N Points                 : 842
            N Cells                  : 1680
        Report summary:
            Is valid                 : True
            Invalid fields           : ()
        Invalid data arrays:
            Point data wrong length  : []
            Cell data wrong length   : []
        Invalid cell ids:
            Wrong number of points   : []
            Intersecting edges       : []
            Intersecting faces       : []
            Non-contiguous edges     : []
            Non-convex               : []
            Inverted faces           : []
            Non-planar faces         : []
            Degenerate faces         : []
            Coincident points        : []
            Invalid point references : []
        Invalid point ids:
            Unused points            : []
            Non-finite points        : []

        Load a mesh with invalid cells, e.g. :func:`~pyvista.examples.downloads.download_cow`
        and validate it. Use ``'cells'`` to only validate the cells specifically.

        >>> mesh = examples.download_cow()
        >>> report = mesh.validate_mesh('cells')

        Show the report. Note that only cell validation fields are reported (array and point
        fields are omitted).

        >>> print(report)
        Mesh Validation Report
        ----------------------
        Mesh:
            Type                     : PolyData
            N Points                 : 2903
            N Cells                  : 3263
        Report summary:
            Is valid                 : False
            Invalid fields (1)       : ('non_convex',)
        Invalid cell ids:
            Wrong number of points   : []
            Intersecting edges       : []
            Intersecting faces       : []
            Non-contiguous edges     : []
            Non-convex (3)           : [1013, 1532, 3250]
            Inverted faces           : []
            Non-planar faces         : []
            Degenerate faces         : []
            Coincident points        : []
            Invalid point references : []

        >>> report.is_valid
        False

        Show what the issue(s) are.

        >>> report.invalid_fields
        ('non_convex',)

        Show the cell ids of the non-convex cells.

        >>> report.non_convex
        [1013, 1532, 3250]

        Access the same underlying mesh array of non-convex cell ids that was internally computed
        by :meth:`~pyvista.DataObjectFilters.cell_validator`.

        >>> report.mesh.field_data['non_convex']
        pyvista_ndarray([1013, 1532, 3250])

        Print the message generated by the report. This is the message used when the
        ``action`` keyword is set for emitting warnings or raising errors.

        >>> print(report.message)
        PolyData mesh is not valid due to the following problems:
         - Mesh has 3 non-convex cells. Invalid cell ids: [1013, 1532, 3250]

        Show a validation report for cells with intersecting edges and unused points only.

        >>> report = mesh.validate_mesh(['intersecting_edges', 'unused_points'])
        >>> print(report)
        Mesh Validation Report
        ----------------------
        Mesh:
            Type                     : PolyData
            N Points                 : 2903
            N Cells                  : 3263
        Report summary:
            Is valid                 : True
            Invalid fields           : ()
        Invalid cell ids:
            Intersecting edges       : []
        Invalid point ids:
            Unused points            : []

        Even though other fields are invalid (i.e. ``non_convex``), for `these` specific
        validation fields the mesh is considered valid.

        >>> report.is_valid
        True

        Do minimal validation to ensure the mesh properties are "memory_safe". This helps to avoid
        a segmentation fault which may be caused by invalid memory accesses by VTK. In this case,
        we use ``action`` to raise an error if the mesh is not valid.

        >>> _ = mesh.validate_mesh('memory_safe', action='error')

        Validate the mesh as a :class:`~pyvista.MultiBlock` instead.

        >>> multi = mesh.cast_to_multiblock()
        >>> report = multi.validate_mesh()

        Instead of reporting problems with specific arrays, point ids, or cell ids, the errors
        are reported by block id. Here, block id ``0`` is reported as having non-convex cells.

        >>> print(report)
        Mesh Validation Report
        ----------------------
        Mesh:
            Type                     : MultiBlock
            N Blocks                 : 1
        Report summary:
            Is valid                 : False
            Invalid fields (1)       : ('non_convex',)
        Blocks with invalid data arrays:
            Point data wrong length  : []
            Cell data wrong length   : []
        Blocks with invalid cells:
            Wrong number of points   : []
            Intersecting edges       : []
            Intersecting faces       : []
            Non-contiguous edges     : []
            Non-convex (1)           : [0]
            Inverted faces           : []
            Non-planar faces         : []
            Degenerate faces         : []
            Coincident points        : []
            Invalid point references : []
        Blocks with invalid points:
            Unused points            : []
            Non-finite points        : []

        The report message still contains specifics about the invalid cell ids though.

        >>> print(report.message)
        MultiBlock mesh is not valid due to the following problems:
         - Block id 0 'Block-00' PolyData mesh is not valid due to the following problems:
           - Mesh has 3 non-convex cells. Invalid cell ids: [1013, 1532, 3250]

        And subreports for each block can be accessed with indexing.

        >>> len(report)
        1
        >>> subreport = report[0]
        >>> subreport.non_convex
        [1013, 1532, 3250]

        """
        input_fields = (
            _MeshValidator._DEFAULT_MESH_VALIDATION_ARGS
            if validation_fields is None
            else validation_fields
        )
        if action is not None:
            allowed = get_args(_MeshValidator._ActionOptions)
            _validation.check_contains(allowed, must_contain=action, name='action')
        report = _MeshValidator(self, input_fields).validation_report
        if action is not None and (message := report.message) is not None:
            if action == 'warn':
                warn_external(message, pv.InvalidMeshWarning)
            else:  # action == 'error':
                raise pv.InvalidMeshError(message)
        return report

    def _validate_mesh(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        validate: Literal[True] | _MeshValidationOptions,
    ):
        """Validate mesh using a bool or named fields and raise error."""
        validation_fields = None if validate is True else validate
        self.validate_mesh(validation_fields, action='error')

    def cell_validator(self: _DataSetOrMultiBlockType):  # type:ignore[misc]
        """Check the validity of each cell in this dataset.

        Use :vtk:`vtkCellValidator` to determine the status of each cell. The status is encoded
        as a bit field cell data array ``'validity_state'``. The cell states are:

        - ``valid`` (``0x00``): Cell is valid and has no issues.
        - ``wrong_number_of_points`` (``0x01``): Cell does not have the minimum number of points
          needed to describe it.
        - ``intersecting_edges`` (``0x02``): 2D cell has two edges that intersect.
        - ``intersecting_faces`` (``0x04``): 3D cell has two faces that intersect.
        - ``non_contiguous_edges`` (``0x08``): 2D cell's perimeter edges are not contiguous.
        - ``non_convex`` (``0x10``): 2D or 3D cell is not convex.
        - ``inverted_faces`` (``0x20``): Cell face(s) do not point in the direction required by
          its :class:`~pyvista.CellType`.
        - ``non_planar_faces`` (``0x40``): Vertices for a face do not all lie in the same plane.
        - ``degenerate_faces`` (``0x80``): Face(s) collapse to a line or a point through repeated
          collocated vertices.
        - ``coincident_points`` (``0x100``): Cell has duplicate coordinates or repeated use of
          the same connectivity entry.

        For convenience, a field data array for each state is also appended. The array names match
        the state names above, except for the ``'valid'`` state; instead, an array with
        ``'invalid'`` cells is stored. Each field data array contains the indices of cells with
        the specified state.

        Refer to :vtk:`vtkCellValidator` for more details about each state.

        .. versionadded:: 0.47

        Returns
        -------
        DataSet
            Dataset with field data of cell validity.

        See Also
        --------
        :meth:`~pyvista.DataObjectFilters.validate_mesh`
        :meth:`~pyvista.DataObjectFilters.cell_quality`
        :ref:`mesh_validation_example`

        Examples
        --------
        Load a mesh with invalid cells.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_cow()

        Validate the cells and show the included arrays.

        >>> validated = mesh.cell_validator()
        >>> validated.array_names  # doctest: +NORMALIZE_WHITESPACE
        ['validity_state',
         'invalid',
         'wrong_number_of_points',
         'intersecting_edges',
         'intersecting_faces',
         'non_contiguous_edges',
         'non_convex',
         'inverted_faces',
         'non_planar_faces',
         'degenerate_faces',
         'coincident_points']

        Show unique scalar values.

        >>> np.unique(validated.cell_data['validity_state'])
        pyvista_ndarray([ 0, 16], dtype=int16)

        The ``0`` cells are valid, and the cells with value ``16`` (i.e. hex ``0x10``) have a
        nonconvex state. We confirm this by printing the ``'non_convex'`` array, which shows there
        are three invalid cells.

        >>> validated.field_data['non_convex']
        pyvista_ndarray([1013, 1532, 3250])

        We can also show all invalid cells. This matches the nonconvex ids, which confirms
        these are the only invalid cells.

        >>> validated.field_data['invalid']
        pyvista_ndarray([1013, 1532, 3250])

        Plot the cell states using :meth:`~pyvista.DataSetFilters.color_labels`. Orient the
        camera to show the underside of the cow where two of the invalid cells are located.

        >>> colored, color_map = validated.color_labels(
        ...     scalars='validity_state', return_dict=True
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(colored)
        >>> _ = pl.add_legend(color_map)
        >>> pl.view_xz()
        >>> pl.camera.zoom(2.5)
        >>> pl.show()

        Extract the invalid cells and plot them along with the original mesh as wireframe for
        context. Orient the camera to focus on the cow's left eye where the third invalid cell is
        located.

        >>> invalid_cells = mesh.extract_cells(validated['invalid'])

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, style='wireframe', color='light gray')
        >>> _ = pl.add_mesh(invalid_cells, color='lime')
        >>> pl.camera_position = pv.CameraPosition(
        ...     position=(5.1, 1.8, -4.9),
        ...     focal_point=(4.7, 1.8, 0.38),
        ...     viewup=(0.0, 1.0, 0.0),
        ... )
        >>> pl.show()

        """
        cell_validator = _vtk.vtkCellValidator()
        cell_validator.SetInputData(self)
        cell_validator.Update()
        output = _get_output(cell_validator)

        def _process_output_arrays(mesh: DataSet):
            # Rename output scalars and make them active
            validity_state = mesh.cell_data['ValidityState']
            mesh.cell_data['validity_state'] = validity_state
            del mesh.cell_data['ValidityState']
            mesh.set_active_scalars('validity_state', preference='cell')

            # Extract indices of invalid cells and store as field data
            mesh.field_data['invalid'] = np.where(validity_state != 0)[0]
            for name, value in _CELL_VALIDATOR_BIT_FIELD.items():
                mesh.field_data[name] = np.where(validity_state & value)[0]

        if isinstance(output, pv.DataSet):
            _process_output_arrays(output)
        else:
            output.generic_filter(_process_output_arrays)
        return output

    @_deprecate_positional_args(allowed=['trans'])
    def transform(  # noqa: PLR0917
        self: _MeshType_co,
        trans: TransformLike,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool | None = None,  # noqa: FBT001
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Transform this mesh with a 4x4 transform.

        .. warning::
            When using ``transform_all_input_vectors=True``, there is
            no distinction in VTK between vectors and arrays with
            three components.  This may be an issue if you have scalar
            data with three components (e.g. RGB data).  This will be
            improperly transformed as if it was vector data rather
            than scalar data.  One possible (albeit ugly) workaround
            is to store the three components as separate scalar
            arrays.

        .. warning::
            In general, transformations give non-integer results. This
            method converts integer-typed vector data to float before
            performing the transformation. This applies to the points
            array, as well as any vector-valued data that is affected
            by the transformation. To prevent subtle bugs arising from
            in-place transformations truncating the result to integers,
            this conversion always applies to the input mesh.

        .. warning::
            Shear transformations are not supported for :class:`~pyvista.ImageData` or
            :class:`~pyvista.RectilinearGrid`, and rotations are not supported for
            :class:`~pyvista.RectilinearGrid`. If present, a ``ValueError`` is raised.
            To fully support these transformations, the input should be cast to
            :class:`~pyvista.StructuredGrid` `before` applying this filter.

        .. note::
            Transforming :class:`~pyvista.ImageData` modifies its
            :class:`~pyvista.ImageData.origin`,
            :class:`~pyvista.ImageData.spacing`, and
            :class:`~pyvista.ImageData.direction_matrix` properties.

        .. deprecated:: 0.45.0
            `inplace` was previously defaulted to `True`. In the future this will change
            to `False`.

        .. versionchanged:: 0.45.0
            Transforming :class:`~pyvista.ImageData` now returns ``ImageData``.
            Previously, :class:`~pyvista.StructuredGrid` was returned.

        .. versionchanged:: 0.46.0
            Transforming :class:`~pyvista.RectilinearGrid` now returns ``RectilinearGrid``.
            Previously, :class:`~pyvista.StructuredGrid` was returned.

        .. versionchanged:: 0.47.0
            An error is now raised instead of a warning if a transformation cannot be
            applied.

        Parameters
        ----------
        trans : TransformLike
            Accepts a vtk transformation object or a 4x4
            transformation matrix.

        transform_all_input_vectors : bool, default: False
            When ``True``, all arrays with three components are
            transformed. Otherwise, only the normals and vectors are
            transformed.  See the warning for more details.

        inplace : bool, default: True
            When ``True``, modifies the dataset inplace.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        output : DataSet | MultiBlock
            Transformed dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform
            Describe linear transformations via a 4x4 matrix.
        pyvista.Prop3D.transform
            Transform an actor.

        Examples
        --------
        Translate a mesh by ``(50, 100, 200)``.

        >>> import numpy as np
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()

        Here a 4x4 :class:`numpy.ndarray` is used, but any :class:`~pyvista.TransformLike`
        is accepted.

        >>> transform_matrix = np.array(
        ...     [
        ...         [1, 0, 0, 50],
        ...         [0, 1, 0, 100],
        ...         [0, 0, 1, 200],
        ...         [0, 0, 0, 1],
        ...     ]
        ... )
        >>> transformed = mesh.transform(transform_matrix, inplace=False)
        >>> transformed.plot(show_edges=True)

        """
        # Deprecated v0.45, convert to error in v0.48, remove v0.51
        if inplace is None:
            # if inplace is None user has not explicitly opted into inplace behavior
            if version_info >= (0, 48):  # pragma: no cover
                msg = (
                    'Convert this deprecation warning into an error '
                    'and update the docstring default value/type for inplace.'
                )
                raise RuntimeError(msg)
            if version_info >= (0, 51):  # pragma: no cover
                msg = 'Remove this deprecation and update the docstring value/type for inplace.'
                raise RuntimeError(msg)

            msg = (
                f'The default value of `inplace` for the filter '
                f'`{self.__class__.__name__}.transform` will change in the future. '
                'Previously it defaulted to `True`, but will change to `False`. '
                'Explicitly set `inplace` to `True` or `False` to silence this warning.'
            )
            warn_external(msg, PyVistaDeprecationWarning)
            inplace = True  # The old default behavior

        if isinstance(self, pv.MultiBlock):
            return self.generic_filter(
                'transform',
                trans=trans,
                transform_all_input_vectors=transform_all_input_vectors,
                inplace=inplace,
                progress_bar=progress_bar,
            )

        t = trans if isinstance(trans, Transform) else Transform(trans)

        if t.matrix[3, 3] == 0:
            msg = 'Transform element (3,3), the inverse scale term, is zero'
            raise ValueError(msg)

        # vtkTransformFilter truncates the result if the input is an integer type
        # so convert input points and relevant vectors to float
        # (creating a new copy would be harmful much more often)
        converted_ints = False
        if not np.issubdtype(self.points.dtype, np.floating):
            self.points = self.points.astype(np.float32)
            converted_ints = True
        if transform_all_input_vectors:
            # all vector-shaped data will be transformed
            point_vectors: list[str | None] = [
                name for name, data in self.point_data.items() if data.shape == (self.n_points, 3)
            ]
            cell_vectors: list[str | None] = [
                name for name, data in self.cell_data.items() if data.shape == (self.n_cells, 3)
            ]
        else:
            # we'll only transform active vectors and normals
            point_vectors = [
                self.point_data.active_vectors_name,
                self.point_data.active_normals_name,
            ]
            cell_vectors = [
                self.cell_data.active_vectors_name,
                self.cell_data.active_normals_name,
            ]
        # dynamically convert each self.point_data[name] etc. to float32
        all_vectors = [point_vectors, cell_vectors]
        all_dataset_attrs = [self.point_data, self.cell_data]
        for vector_names, dataset_attrs in zip(all_vectors, all_dataset_attrs, strict=True):
            for vector_name in vector_names:
                if vector_name is None:
                    continue
                vector_arr = dataset_attrs[vector_name]
                if not np.issubdtype(vector_arr.dtype, np.floating):
                    dataset_attrs[vector_name] = vector_arr.astype(np.float32)
                    converted_ints = True
        if converted_ints:
            warn_external(
                'Integer points, vector and normal data (if any) of the input mesh '
                'have been converted to ``np.float32``. This is necessary in order '
                'to transform properly.',
            )

        # vtkTransformFilter doesn't respect active scalars.  We need to track this
        active_point_scalars_name: str | None = self.point_data.active_scalars_name
        active_cell_scalars_name: str | None = self.cell_data.active_scalars_name

        # vtkTransformFilter sometimes doesn't transform all vector arrays
        # when there are active point/cell scalars. Use this workaround
        self.active_scalars_name = None

        f = _vtk.vtkTransformFilter()
        f.SetInputDataObject(self)
        f.SetTransform(t)
        f.SetTransformAllInputVectors(transform_all_input_vectors)

        _update_alg(f, progress_bar=progress_bar, message='Transforming')
        vtk_filter_output = pv.core.filters._get_output(f)

        output = self if inplace else self.__class__()

        if isinstance(output, pv.ImageData):
            # vtkTransformFilter returns a StructuredGrid for legacy code (before VTK 9)
            # but VTK 9+ supports oriented images.
            # To keep an ImageData -> ImageData mapping, we copy the transformed data
            # from the filter output but manually transform the structure
            output.copy_structure(self)  # type: ignore[arg-type]
            current_matrix = output.index_to_physical_matrix
            new_matrix = pv.Transform(current_matrix).compose(t).matrix
            output.index_to_physical_matrix = new_matrix

            output.point_data.update(vtk_filter_output.point_data, copy=not inplace)
            output.cell_data.update(vtk_filter_output.cell_data, copy=not inplace)
            output.field_data.update(vtk_filter_output.field_data, copy=not inplace)

        elif isinstance(output, pv.RectilinearGrid):
            # vtkTransformFilter returns a StructuredGrid, but we can return
            # RectilinearGrid if we ignore shear and rotations
            # Follow similar decomposition performed by ImageData.index_to_physical_matrix
            T, R, N, S, K = t.decompose()

            if not np.allclose(K, np.eye(3)):
                msg = (
                    'The transformation has a shear component which is not supported by '
                    'RectilinearGrid.\nCast to StructuredGrid first to support shear '
                    'transformations, or use `Transform.decompose()`\nto remove this component.'
                )
                raise ValueError(msg)

            # Lump scale and reflection together
            scale = S * N
            if not np.allclose(np.abs(R), np.eye(3)):
                msg = (
                    'The transformation has a non-diagonal rotation component which is not '
                    'supported by\nRectilinearGrid. Cast to StructuredGrid first to fully '
                    'support rotations, or use\n`Transform.decompose()` to remove this component.'
                )
                raise ValueError(msg)
            else:
                # Lump any reflections from the rotation into the scale
                scale *= np.diagonal(R)

            # Apply transformation to structure
            tx, ty, tz = T
            sx, sy, sz = scale
            output.x = self.x * sx + tx
            output.y = self.y * sy + ty
            output.z = self.z * sz + tz

            # Copy data arrays from the vtkTransformFilter's output
            output.point_data.update(vtk_filter_output.point_data, copy=not inplace)
            output.cell_data.update(vtk_filter_output.cell_data, copy=not inplace)
            output.field_data.update(vtk_filter_output.field_data, copy=not inplace)

        elif inplace:
            output.copy_from(vtk_filter_output, deep=False)
        else:
            # The output from the transform filter contains a shallow copy
            # of the original dataset except for the point arrays.  Here
            # we perform a copy so the two are completely unlinked.
            output.copy_from(vtk_filter_output, deep=True)

        # Make the previously active scalars active again
        self.point_data.active_scalars_name = active_point_scalars_name
        if output is not self:
            output.point_data.active_scalars_name = active_point_scalars_name
        self.cell_data.active_scalars_name = active_cell_scalars_name
        if output is not self:
            output.cell_data.active_scalars_name = active_cell_scalars_name

        return output

    @_deprecate_positional_args(allowed=['normal'])
    def reflect(  # noqa: PLR0917
        self: _MeshType_co,
        normal: VectorLike[float],
        point: VectorLike[float] | None = None,
        inplace: bool = False,  # noqa: FBT001, FBT002
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Reflect a dataset across a plane.

        Parameters
        ----------
        normal : array_like[float]
            Normal direction for reflection.

        point : array_like[float]
            Point which, along with ``normal``, defines the reflection
            plane. If not specified, this is the origin.

        inplace : bool, default: False
            When ``True``, modifies the dataset inplace.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise,
            only the points, normals and active vectors are transformed.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        output : DataSet | MultiBlock
            Reflected dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.reflect
            Concatenate a reflection matrix with a transformation.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh = mesh.reflect((0, 0, 1), point=(0, 0, -100))
        >>> mesh.plot(show_edges=True)

        See the :ref:`reflect_example` for more examples using this filter.

        """
        t = Transform().reflect(normal, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
            progress_bar=progress_bar,
        )

    @_deprecate_positional_args(allowed=['angle'])
    def rotate_x(  # noqa: PLR0917
        self: _MeshType_co,
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the x-axis.

        point : VectorLike[float], optional
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate_x
            Concatenate a rotation about the x-axis with a transformation.

        Examples
        --------
        Rotate a mesh 30 degrees about the x-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_x(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate_x(angle, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['angle'])
    def rotate_y(  # noqa: PLR0917
        self: _MeshType_co,
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the y-axis.

        point : VectorLike[float], optional
            Point to rotate about.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise, only
            the points, normals and active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate_y
            Concatenate a rotation about the y-axis with a transformation.

        Examples
        --------
        Rotate a cube 30 degrees about the y-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_y(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate_y(angle, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['angle'])
    def rotate_z(  # noqa: PLR0917
        self: _MeshType_co,
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the z-axis.

        point : VectorLike[float], optional
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate_z
            Concatenate a rotation about the z-axis with a transformation.

        Examples
        --------
        Rotate a mesh 30 degrees about the z-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_z(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate_z(angle, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['vector', 'angle'])
    def rotate_vector(  # noqa: PLR0917
        self: _MeshType_co,
        vector: VectorLike[float],
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about a vector.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        vector : VectorLike[float]
            Vector to rotate about.

        angle : float
            Angle to rotate.

        point : VectorLike[float], optional
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate_vector
            Concatenate a rotation about a vector with a transformation.

        Examples
        --------
        Rotate a mesh 30 degrees about the ``(1, 1, 1)`` axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_vector((1, 1, 1), 30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate_vector(vector, angle, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['rotation'])
    def rotate(  # noqa: PLR0917
        self: _MeshType_co,
        rotation: RotationLike,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about a point with a rotation matrix or ``Rotation`` object.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        rotation : RotationLike
            3x3 rotation matrix or a SciPy ``Rotation`` object.

        point : VectorLike[float], optional
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate
            Concatenate a rotation matrix with a transformation.

        Examples
        --------
        Define a rotation. Here, a 3x3 matrix is used which rotates about the z-axis by
        60 degrees.

        >>> import pyvista as pv
        >>> rotation = [
        ...     [0.5, -0.8660254, 0.0],
        ...     [0.8660254, 0.5, 0.0],
        ...     [0.0, 0.0, 1.0],
        ... ]

        Use the rotation to rotate a cone about its tip.

        >>> mesh = pv.Cone()
        >>> tip = (0.5, 0.0, 0.0)
        >>> rot = mesh.rotate(rotation, point=tip)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate(rotation, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def translate(  # type: ignore[misc]
        self: _MeshType_co,
        xyz: VectorLike[float],
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Translate the mesh.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        xyz : VectorLike[float]
            A vector of three floats.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Translated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.translate
            Concatenate a translation matrix with a transformation.

        Examples
        --------
        Create a sphere and translate it by ``(2, 1, 2)``.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.center
        (0.0, 0.0, 0.0)
        >>> trans = mesh.translate((2, 1, 2), inplace=False)
        >>> trans.center
        (2.0, 1.0, 2.0)

        """
        transform = Transform().translate(xyz)
        return self.transform(
            transform,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['xyz'])
    def scale(  # noqa: PLR0917
        self: _MeshType_co,
        xyz: float | VectorLike[float],
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
        point: VectorLike[float] | None = None,
    ) -> _MeshType_co:
        """Scale the mesh.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        xyz : float | VectorLike[float]
            A vector sequence defining the scale factors along x, y, and z. If
            a scalar, the same uniform scale is used along all three axes.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise, only
            the points, normals and active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        point : VectorLike[float], optional
            Point to scale from. Defaults to origin.

        Returns
        -------
        output : DataSet | MultiBlock
            Scaled dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.scale
            Concatenate a scale matrix with a transformation.

        pyvista.DataObjectFilters.resize
            Resize a mesh.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh1 = examples.download_teapot()
        >>> mesh2 = mesh1.scale([10.0, 10.0, 10.0], inplace=False)

        Plot meshes side-by-side

        >>> pl = pv.Plotter(shape=(1, 2))
        >>> # Create plot with unscaled mesh
        >>> pl.subplot(0, 0)
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> # Create plot with scaled mesh
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> pl.show(cpos='xy')

        """
        transform = Transform().scale(xyz, point=point)
        return self.transform(
            transform,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def resize(  # type: ignore[misc]
        self: _MeshType_co,
        *,
        bounds: VectorLike[float] | None = None,
        bounds_size: float | VectorLike[float] | None = None,
        length: float | None = None,
        center: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ) -> _MeshType_co:
        """Resize the dataset's bounds.

        This filter rescales and translates the mesh to fit specified bounds. This is useful for
        normalizing datasets, changing units, or fitting datasets into specific coordinate ranges.

        It has three independent use cases:

        #. Use ``bounds`` to set the mesh's :attr:`~pyvista.DataSet.bounds` directly.
        #. Use ``bounds_size`` to set the mesh's ``bounds_size`` directly.
        #. Use ``length`` to set the mesh's diagonal ``length`` directly.

        By default, the ``bounds_size`` and ``length`` options resize the mesh so that its
        ``center`` is unchanged. Optionally, ``center`` may be set explicitly for these cases.

        .. versionadded:: 0.46

        See Also
        --------
        :meth:`scale`, :meth:`translate`
            Scale and/or translate a mesh. Used internally by :meth:`resize`.

        Parameters
        ----------
        bounds : VectorLike[float], optional
            Target :attr:`~pyvista.DataSet.bounds` for the resized dataset in the format
            ``[xmin, xmax, ymin, ymax, zmin, zmax]``. If provided, the dataset is scaled and
            translated to fit exactly within these bounds. Cannot be used together with
            ``bounds_size``, ``length``, or ``center``.

        bounds_size : float | VectorLike[float], optional
            Target size of the :attr:`~pyvista.DataSet.bounds` for the resized dataset. Use a
            single float to specify the size of all three axes, or a 3-element vector to set the
            size of each axis independently. Cannot be used together with ``bounds`` or ``length``.

        length : float, optional
            Target length of the :attr:`~pyvista.DataSet.bounds` for the resized dataset.
            Cannot be used together with ``bounds`` or ``bounds_size``.

            .. versionadded:: 0.47

        center : VectorLike[float], optional
            Center of the resized dataset in ``[x, y, z]``. By default, the mesh's
            :attr:`~pyvista.DataSet.center` is used. Only used when ``bounds_size`` or ``length``
            is specified.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed as part of the resize. Otherwise, only
            the points, normals and active vectors are transformed.

        inplace : bool, default: False
            If True, the dataset is modified in place. If False, a new dataset is returned.

        Returns
        -------
        output : DataSet | MultiBlock
            Resized dataset. Return type matches input.

        Examples
        --------
        Load a mesh with asymmetric bounds and show them.

        >>> import pyvista as pv
        >>> mesh = pv.Cube(
        ...     x_length=1.0, y_length=2.0, z_length=3.0, center=(1.0, 2.0, 3.0)
        ... )
        >>> mesh.bounds
        BoundsTuple(x_min = 0.5,
                    x_max = 1.5,
                    y_min = 1.0,
                    y_max = 3.0,
                    z_min = 1.5,
                    z_max = 4.5)

        Resize it to fit specific bounds.

        >>> resized = mesh.resize(bounds=[-1, 2, -3, 4, -5, 6])
        >>> resized.bounds
        BoundsTuple(x_min = -1.0,
                    x_max =  2.0,
                    y_min = -3.0,
                    y_max =  4.0,
                    z_min = -5.0,
                    z_max =  6.0)

        Resize the mesh so its diagonal length is ``4.0``. The mesh's center is unchanged.

        >>> resized = mesh.resize(length=4.0)
        >>> resized.length
        4.0
        >>> resized.center
        (1.0, 2.0, 3.0)

        Resize the mesh so its size is ``4.0``. The mesh's center is again unchanged.

        >>> resized = mesh.resize(bounds_size=4.0)
        >>> resized.bounds_size
        (4.0, 4.0, 4.0)
        >>> resized.center
        (1.0, 2.0, 3.0)
        >>> resized.bounds
        BoundsTuple(x_min = -1.0,
                    x_max =  3.0,
                    y_min =  0.0,
                    y_max =  4.0,
                    z_min =  1.0,
                    z_max =  5.0)

        Specify a different size for each axis and set the desired center.

        >>> resized = mesh.resize(bounds_size=(2.0, 1.0, 0.5), center=(1.0, 0.5, 0.25))
        >>> resized.bounds_size
        (2.0, 1.0, 0.5)
        >>> resized.center
        (1.0, 0.5, 0.25)

        Center the mesh at the origin and normalize its bounds to ``1.0``.

        >>> resized = mesh.resize(bounds_size=1.0, center=(0.0, 0.0, 0.0))
        >>> resized.bounds
        BoundsTuple(x_min = -0.5,
                    x_max =  0.5,
                    y_min = -0.5,
                    y_max =  0.5,
                    z_min = -0.5,
                    z_max =  0.5)

        """
        if self.is_empty:
            return self.copy()
        bounds_set = bounds is not None
        length_set = length is not None
        bounds_size_set = bounds_size is not None
        n_set = bounds_set + length_set + bounds_size_set
        if n_set == 0:
            msg = (
                '`bounds`, `bounds_size`, and `length` cannot all be None. '
                'Choose one resizing method.'
            )
            raise ValueError(msg)
        elif n_set > 1:
            msg = (
                'Cannot specify more than one resizing method. Choose either `bounds`, '
                '`bounds_size`, or `length` independently.'
            )
            raise ValueError(msg)

        if bounds is not None:
            if center is not None:
                msg = '`center` can only be used with the `bounds_size` and `length` parameters.'
                raise ValueError(msg)

            target_bounds3x2 = _validation.validate_array(
                bounds, must_have_shape=6, reshape_to=(3, 2), name='bounds'
            )
            target_size = np.diff(target_bounds3x2.T, axis=0)[0]
            current_center = np.array(self.center)
            target_center = np.mean(target_bounds3x2, axis=1)

        else:
            ensure_positive = dict(must_be_in_range=[0, np.inf], strict_lower_bound=True)
            if bounds_size is not None:
                target_size = _validation.validate_array3(
                    bounds_size,
                    broadcast=True,
                    name='bounds_size',
                    **ensure_positive,  # type: ignore[arg-type]
                )
            else:
                valid_length = _validation.validate_number(
                    cast('float', length),
                    name='length',
                    **ensure_positive,  # type: ignore[arg-type]
                )
                target_size = np.array(self.bounds_size) * valid_length / self.length
            current_center = np.array(self.center)
            target_center = (
                current_center
                if center is None
                else _validation.validate_array3(center, name='center')
            )

        current_size = self.bounds_size
        scale_factors = target_size * _reciprocal(current_size, value_if_division_by_zero=1.0)

        # Apply transformation
        transform = pv.Transform()
        transform.translate(-current_center)
        transform.scale(scale_factors)
        transform.translate(target_center)
        return self.transform(
            transform, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    @_deprecate_positional_args
    def flip_x(
        self: _MeshType_co,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Flip mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : sequence[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`~pyvista.DataSet.center`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Flipped dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.flip_x
            Concatenate a reflection about the x-axis with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_x(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xy')

        """
        if point is None:
            point = self.center
        t = Transform().reflect((1, 0, 0), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args
    def flip_y(
        self: _MeshType_co,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Flip mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`~pyvista.DataSet.center`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Flipped dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.flip_y
            Concatenate a reflection about the y-axis with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_y(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xy')

        """
        if point is None:
            point = self.center
        t = Transform().reflect((0, 1, 0), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args
    def flip_z(
        self: _MeshType_co,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Flip mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`~pyvista.DataSet.center`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Flipped dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.flip_z
            Concatenate a reflection about the z-axis with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot().rotate_x(90, inplace=False)
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_z(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xz')

        """
        if point is None:
            point = self.center
        t = Transform().reflect((0, 0, 1), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['normal'])
    def flip_normal(  # noqa: PLR0917
        self: _MeshType_co,
        normal: VectorLike[float],
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Flip mesh about the normal.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        normal : VectorLike[float]
           Normal vector to flip about.

        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`~pyvista.DataSet.center`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset flipped about its normal. Return type matches input.

        See Also
        --------
        pyvista.Transform.reflect
            Concatenate a reflection matrix with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_normal([1.0, 1.0, 1.0], inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xy')

        """
        if point is None:
            point = self.center
        t = Transform().reflect(normal, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def _clip_with_function(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        function: _vtk.vtkImplicitFunction,
        *,
        invert: bool = True,
        value: float = 0.0,
        return_clipped: bool = False,
        progress_bar: bool = False,
        crinkle: bool = False,
    ):
        """Clip using an implicit function (internal helper)."""
        if crinkle:
            active_scalars_info = _Crinkler.add_cell_ids(self)

        # Need to cast PointSet to PolyData since vtkTableBasedClipDataSet is broken
        # with vtk 9.4.X, see https://gitlab.kitware.com/vtk/vtk/-/issues/19649
        apply_vtk_94x_patch = (
            isinstance(self, pv.PointSet)
            and pv.vtk_version_info >= (9, 4)
            and pv.vtk_version_info < (9, 5)
        )
        mesh_in = self.cast_to_poly_points() if apply_vtk_94x_patch else self

        if isinstance(mesh_in, pv.PolyData):
            alg: _vtk.vtkClipPolyData | _vtk.vtkTableBasedClipDataSet = _vtk.vtkClipPolyData()
        # elif isinstance(self, vtk.vtkImageData):
        #     alg = vtk.vtkClipVolume()
        #     alg.SetMixed3DCellGeneration(True)
        else:
            alg = _vtk.vtkTableBasedClipDataSet()
        alg.SetInputDataObject(mesh_in)  # Use the grid as the data we desire to cut
        alg.SetValue(value)
        alg.SetClipFunction(function)  # the implicit function
        alg.SetInsideOut(invert)  # invert the clip if needed
        alg.SetGenerateClippedOutput(return_clipped)
        _update_alg(alg, progress_bar=progress_bar, message='Clipping with Function')

        def _maybe_cast_to_point_set(in_):
            return in_.cast_to_pointset() if apply_vtk_94x_patch else in_

        if return_clipped:
            a = _get_output(alg, oport=0)
            b = _get_output(alg, oport=1)
            if crinkle:
                a, b = _Crinkler.extract_crinkle_cells(self, a, b, active_scalars_info)
            return _maybe_cast_to_point_set(a), _maybe_cast_to_point_set(b)
        clipped = _get_output(alg)
        if crinkle:
            clipped = _Crinkler.extract_crinkle_cells(self, clipped, None, active_scalars_info)
        return _maybe_cast_to_point_set(clipped)

    @_deprecate_positional_args(allowed=['normal'])
    def clip(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        normal: VectorLike[float] | _NormalsLiteral | None = None,
        origin: VectorLike[float] | None = None,
        invert: bool = True,  # noqa: FBT001, FBT002
        value: float = 0.0,
        inplace: bool = False,  # noqa: FBT001, FBT002
        return_clipped: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        crinkle: bool = False,  # noqa: FBT001, FBT002
        plane: PolyData | None = None,
    ):
        """Clip a dataset by a plane by specifying the origin and normal.

        The origin and normal may be set explicitly or implicitly using a
        :func:`~pyvista.Plane`.

        If no parameters are given, the clip will occur in the center
        of the dataset along the x-axis.

        Parameters
        ----------
        normal : VectorLike[float] | str, optional
            Length-3 vector for the normal vector direction. Can also
            be specified as a string conventional direction such as
            ``'x'`` for ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.
            The ``'x'`` direction is used by default.

        origin : VectorLike[float], optional
            The center ``(x, y, z)`` coordinate of the plane on which the clip
            occurs. The default is the center of the dataset.

        invert : bool, default: True
            If ``True``, remove mesh parts in the ``normal`` direction from ``origin``.
            If ``False``, remove parts in the opposite direction.

        value : float, default: 0.0
            Set the clipping value along the normal direction.

        inplace : bool, default: False
            Updates mesh in-place.

        return_clipped : bool, default: False
            Return both unclipped and clipped parts of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        crinkle : bool, default: False
            Crinkle the clip by extracting the entire cells along the
            clip. This adds the ``"cell_ids"`` array to the ``cell_data``
            attribute that tracks the original cell IDs of the original
            dataset.

        plane : PolyData, optional
            :func:`~pyvista.Plane` mesh to use for clipping. Use this as an
            alternative to setting ``origin`` and ``normal``. The mean of the
            plane's normal vectors is used for the ``normal`` parameter and
            the mean of the plane's points is used for the ``origin`` parameter.

            .. versionadded:: 0.47

        Returns
        -------
        output : DataSet | MultiBlock | tuple[DataSet | MultiBlock, DataSet | MultiBlock]
            Clipped mesh when ``return_clipped=False`` or a tuple containing the
            unclipped and clipped meshes. Output mesh type matches input type for
            :class:`~pyvista.PointSet`, :class:`~pyvista.PolyData`, and
            :class:`~pyvista.MultiBlock`; otherwise the output type is
            :class:`~pyvista.UnstructuredGrid`.

        Examples
        --------
        Clip a cube along the +X direction.  ``triangulate`` is used as
        the cube is initially composed of quadrilateral faces and
        subdivide only works on triangles.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip()
        >>> clipped_cube.plot()

        Clip a cube in the +Z direction.  This leaves half a cube
        below the XY plane.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip('z')
        >>> clipped_cube.plot()

        See :ref:`clip_with_surface_example` for more examples using this filter.

        """
        origin_, normal_ = _validate_plane_origin_and_normal(
            self, origin, normal, plane, default_normal='x'
        )
        # create the plane for clipping
        function = generate_plane(normal_, origin_)
        # run the clip
        result = self._clip_with_function(
            function,
            invert=invert,
            value=value,
            return_clipped=return_clipped,
            progress_bar=progress_bar,
            crinkle=crinkle,
        )

        # Post-process clip to fix output type and remove unused points
        input_bounds = self.bounds
        if isinstance(result, tuple):
            result = (
                _cast_output_to_match_input_type(result[0], self),
                _cast_output_to_match_input_type(result[1], self),
            )
            result = (
                _remove_unused_points_post_clip(result[0], input_bounds),
                _remove_unused_points_post_clip(result[1], input_bounds),
            )
        else:
            result = _cast_output_to_match_input_type(result, self)
            result = _remove_unused_points_post_clip(result, input_bounds)
        if inplace:
            if return_clipped:
                self.copy_from(result[0], deep=False)
                return self, result[1]
            else:
                self.copy_from(result, deep=False)
                return self
        return result

    @_deprecate_positional_args(allowed=['bounds'])
    def clip_box(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        bounds: float | VectorLike[float] | PolyData | None = None,
        invert: bool = True,  # noqa: FBT001, FBT002
        factor: float = 0.35,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        merge_points: bool = True,  # noqa: FBT001, FBT002
        crinkle: bool = False,  # noqa: FBT001, FBT002
    ):
        """Clip a dataset by a bounding box defined by the bounds.

        If no bounds are given, a corner of the dataset bounds will be removed.

        Parameters
        ----------
        bounds : sequence[float], optional
            Length 6 sequence of floats: ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
            Length 3 sequence of floats: distances from the min coordinate of
            of the input mesh. Single float value: uniform distance from the
            min coordinate. Length 12 sequence of length 3 sequence of floats:
            a plane collection (normal, center, ...).
            :class:`pyvista.PolyData`: if a poly mesh is passed that represents
            a box with 6 faces that all form a standard box, then planes will
            be extracted from the box to define the clipping region.

        invert : bool, default: True
            Flag on whether to flip/invert the clip.

        factor : float, default: 0.35
            If bounds are not given this is the factor along each axis to
            extract the default box.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        merge_points : bool, default: True
            If ``True``, coinciding points of independently defined mesh
            elements will be merged.

        crinkle : bool, default: False
            Crinkle the clip by extracting the entire cells along the
            clip. This adds the ``"cell_ids"`` array to the ``cell_data``
            attribute that tracks the original cell IDs of the original
            dataset.

        Returns
        -------
        pyvista.UnstructuredGrid
            Clipped dataset.

        Examples
        --------
        Clip a corner of a cube.  The bounds of a cube are normally
        ``[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]``, and this removes 1/8 of
        the cube's surface.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip_box([0, 1, 0, 1, 0, 1])
        >>> clipped_cube.plot()

        See :ref:`clip_with_plane_box_example` for more examples using this filter.

        """
        if bounds is None:

            def _get_quarter(dmin, dmax):
                """Get a section of the given range (internal helper)."""
                return dmax - ((dmax - dmin) * factor)

            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            xmin = _get_quarter(xmin, xmax)
            ymin = _get_quarter(ymin, ymax)
            zmin = _get_quarter(zmin, zmax)
            bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
        if isinstance(bounds, (float, int)):
            bounds = [bounds, bounds, bounds]
        elif isinstance(bounds, pv.PolyData):
            poly = bounds
            if poly.n_cells != 6:
                msg = 'The bounds mesh must have only 6 faces.'
                raise ValueError(msg)
            bounds = []
            poly.compute_normals(inplace=True)
            for cid in range(6):
                cell = poly.extract_cells(cid)
                normal = cell['Normals'][0]
                bounds.append(normal)
                bounds.append(cell.center)
        bounds_ = _validation.validate_array(
            bounds,  # type: ignore[arg-type]
            dtype_out=float,
            must_have_length=[3, 6, 12],
            name='bounds',
        )
        if len(bounds_) == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            bounds_ = np.array(
                (
                    xmin,
                    xmin + bounds_[0],
                    ymin,
                    ymin + bounds_[1],
                    zmin,
                    zmin + bounds_[2],
                )
            )
        if crinkle:
            active_scalars_info = _Crinkler.add_cell_ids(self)
        alg = _vtk.vtkBoxClipDataSet()
        if not merge_points:
            # vtkBoxClipDataSet uses vtkMergePoints by default
            alg.SetLocator(_vtk.vtkNonMergingPointLocator())
        alg.SetInputDataObject(self)
        alg.SetBoxClip(*bounds_)
        port = 0
        if invert:
            # invert the clip if needed
            port = 1
            alg.GenerateClippedOutputOn()
        _update_alg(alg, progress_bar=progress_bar, message='Clipping a Dataset by a Bounding Box')
        clipped = _get_output(alg, oport=port)
        if crinkle:
            clipped = _Crinkler.extract_crinkle_cells(self, clipped, None, active_scalars_info)
        return _remove_unused_points_post_clip(clipped, self.bounds)

    @_deprecate_positional_args(allowed=['implicit_function'])
    def slice_implicit(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        implicit_function: _vtk.vtkImplicitFunction,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Slice a dataset by a VTK implicit function.

        Parameters
        ----------
        implicit_function : :vtk:`vtkImplicitFunction`
            Specify the implicit function to perform the cutting.

        generate_triangles : bool, default: False
            If this is enabled (``False`` by default), the output will
            be triangles. Otherwise the output will be the intersection
            polygons. If the cutting function is not a plane, the
            output will be 3D polygons, which might be nice to look at
            but hard to compute with downstream.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice
        slice_orthogonal
        slice_along_axis
        slice_along_line
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the surface of a sphere.

        >>> import pyvista as pv
        >>> import vtk
        >>> sphere = vtk.vtkSphere()
        >>> sphere.SetRadius(10)
        >>> mesh = pv.Wavelet()
        >>> slice = mesh.slice_implicit(sphere)
        >>> slice.plot(show_edges=True, line_width=5)

        >>> cylinder = vtk.vtkCylinder()
        >>> cylinder.SetRadius(10)
        >>> mesh = pv.Wavelet()
        >>> slice = mesh.slice_implicit(cylinder)
        >>> slice.plot(show_edges=True, line_width=5)

        """
        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(self)  # Use the grid as the data we desire to cut
        alg.SetCutFunction(implicit_function)  # the cutter to use the function
        alg.SetGenerateTriangles(generate_triangles)
        _update_alg(alg, progress_bar=progress_bar, message='Slicing')
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output

    @_deprecate_positional_args(allowed=['normal'])
    def slice(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        normal: VectorLike[float] | _NormalsLiteral | None = None,
        origin: VectorLike[float] | None = None,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        plane: PolyData | None = None,
    ):
        """Slice a dataset by a plane at the specified origin and normal vector orientation.

        The origin and normal may be set explicitly or implicitly using a
        :func:`~pyvista.Plane`.

        If no parameters are given, the slice will occur in the center
        of the dataset along the x-axis.

        Parameters
        ----------
        normal : VectorLike[float] | str, optional
            Length-3 vector for the normal vector direction. Can also
            be specified as a string conventional direction such as
            ``'x'`` for ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.
            The ``'x'`` direction is used by default.

        origin : sequence[float], optional
            The center ``(x, y, z)`` coordinate of the plane on which
            the slice occurs. The default is the center of the dataset.

        generate_triangles : bool, default: False
            If this is enabled (``False`` by default), the output will
            be triangles. Otherwise the output will be the intersection
            polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        plane : PolyData, optional
            :func:`~pyvista.Plane` mesh to use for slicing. Use this as an
            alternative to setting ``origin`` and ``normal``. The mean of the
            plane's normal vectors is used for the ``normal`` parameter and
            the mean of the plane's points is used for the ``origin`` parameter.

            .. versionadded:: 0.47

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice_implicit
        slice_orthogonal
        slice_along_axis
        slice_along_line
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the surface of a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> slice_x = sphere.slice(normal='x')
        >>> slice_y = sphere.slice(normal='y')
        >>> slice_z = sphere.slice(normal='z')
        >>> slices = slice_x + slice_y + slice_z
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        origin_, normal_ = _validate_plane_origin_and_normal(
            self, origin, normal, plane, default_normal='x'
        )
        # create the plane for clipping
        implicit_function = generate_plane(normal_, origin_)
        return self.slice_implicit(
            implicit_function,
            generate_triangles=generate_triangles,
            contour=contour,
            progress_bar=progress_bar,
        )

    @_deprecate_positional_args
    def slice_orthogonal(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Create three orthogonal slices through the dataset on the three cartesian planes.

        Yields a MutliBlock dataset of the three slices.

        Parameters
        ----------
        x : float, optional
            The X location of the YZ slice.

        y : float, optional
            The Y location of the XZ slice.

        z : float, optional
            The Z location of the XY slice.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice
        slice_implicit
        slice_along_axis
        slice_along_line
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the random hills dataset with three orthogonal planes.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_orthogonal(contour=False)
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        # Create the three slices
        if x is None:
            x = self.center[0]
        if y is None:
            y = self.center[1]
        if z is None:
            z = self.center[2]
        output = pv.MultiBlock()
        if isinstance(self, pv.MultiBlock):
            for i in range(self.n_blocks):
                data = self[i]
                output.append(
                    data.slice_orthogonal(
                        x=x,
                        y=y,
                        z=z,
                        generate_triangles=generate_triangles,
                        contour=contour,
                    )
                    if data is not None
                    else data
                )
            return output
        output.append(
            self.slice(
                normal='x',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'YZ',
        )
        output.append(
            self.slice(
                normal='y',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'XZ',
        )
        output.append(
            self.slice(
                normal='z',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'XY',
        )
        return output

    @_deprecate_positional_args(allowed=['n', 'axis'])
    def slice_along_axis(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        n: int = 5,
        axis: Literal['x', 'y', 'z', 0, 1, 2] = 'x',
        tolerance: float | None = None,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        bounds=None,
        center=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Create many slices of the input dataset along a specified axis.

        Parameters
        ----------
        n : int, default: 5
            The number of slices to create.

        axis : str | int, default: 'x'
            The axis to generate the slices along. Perpendicular to the
            slices. Can be string name (``'x'``, ``'y'``, or ``'z'``) or
            axis index (``0``, ``1``, or ``2``).

        tolerance : float, optional
            The tolerance to the edge of the dataset bounds to create
            the slices. The ``n`` slices are placed equidistantly with
            an absolute padding of ``tolerance`` inside each side of the
            ``bounds`` along the specified axis. Defaults to 1% of the
            ``bounds`` along the specified axis.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        bounds : sequence[float], optional
            A 6-length sequence overriding the bounds of the mesh.
            The bounds along the specified axis define the extent
            where slices are taken.

        center : sequence[float], optional
            A 3-length sequence specifying the position of the line
            along which slices are taken. Defaults to the center of
            the mesh.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice
        slice_implicit
        slice_orthogonal
        slice_along_line
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the random hills dataset in the X direction.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_along_axis(n=10)
        >>> slices.plot(line_width=5)

        Slice the random hills dataset in the Z direction.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_along_axis(n=10, axis='z')
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        # parse axis input
        XYZLiteral = Literal['x', 'y', 'z']
        labels: list[XYZLiteral] = ['x', 'y', 'z']
        label_to_index: dict[Literal['x', 'y', 'z'], Literal[0, 1, 2]] = {
            'x': 0,
            'y': 1,
            'z': 2,
        }
        if isinstance(axis, int):
            ax_index = axis
            ax_label = labels[ax_index]
        elif isinstance(axis, str):
            ax_str = axis.lower()
            if ax_str in labels:
                ax_label = cast('XYZLiteral', ax_str)
                ax_index = label_to_index[ax_label]
            else:
                msg = f'Axis ({axis!r}) not understood. Choose one of {labels}.'
                raise ValueError(msg) from None
        # get the locations along that axis
        if bounds is None:
            bounds = self.bounds
        if center is None:
            center = self.center
        if tolerance is None:
            tolerance = (bounds[ax_index * 2 + 1] - bounds[ax_index * 2]) * 0.01
        rng = np.linspace(
            bounds[ax_index * 2] + tolerance, bounds[ax_index * 2 + 1] - tolerance, n
        )
        center = list(center)
        # Make each of the slices
        output = pv.MultiBlock()
        if isinstance(self, pv.MultiBlock):
            for i in range(self.n_blocks):
                data = self[i]
                output.append(
                    data.slice_along_axis(
                        n=n,
                        axis=ax_label,
                        tolerance=tolerance,
                        generate_triangles=generate_triangles,
                        contour=contour,
                        bounds=bounds,
                        center=center,
                    )
                    if data is not None
                    else data
                )
            return output
        for i in range(n):
            center[ax_index] = rng[i]
            slc = self.slice(
                normal=ax_label,
                origin=center,
                generate_triangles=generate_triangles,
                contour=contour,
                progress_bar=progress_bar,
            )
            output.append(slc, f'slice{i}')
        return output

    @_deprecate_positional_args(allowed=['line'])
    def slice_along_line(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        line: pv.PolyData,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Slice a dataset using a polyline/spline as the path.

        This also works for lines generated with :func:`pyvista.Line`.

        Parameters
        ----------
        line : pyvista.PolyData
            A PolyData object containing one single PolyLine cell.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice
        slice_implicit
        slice_orthogonal
        slice_along_axis
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the random hills dataset along a circular arc.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> center = np.array(hills.center)
        >>> point_a = center + np.array([5, 0, 0])
        >>> point_b = center + np.array([-5, 0, 0])
        >>> arc = pv.CircularArc(
        ...     pointa=point_a, pointb=point_b, center=center, resolution=100
        ... )
        >>> line_slice = hills.slice_along_line(arc)

        Plot the circular arc and the hills mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(hills, smooth_shading=True, style='wireframe')
        >>> _ = pl.add_mesh(
        ...     line_slice,
        ...     line_width=10,
        ...     render_lines_as_tubes=True,
        ...     color='k',
        ... )
        >>> _ = pl.add_mesh(arc, line_width=10, color='grey')
        >>> pl.show()

        See :ref:`slice_example` for more examples using this filter.

        """
        # check that we have a PolyLine cell in the input line
        if line.GetNumberOfCells() != 1:
            msg = 'Input line must have only one cell.'
            raise ValueError(msg)
        polyline = line.GetCell(0)
        if not isinstance(polyline, _vtk.vtkPolyLine):
            msg = f'Input line must have a PolyLine cell, not ({type(polyline)})'
            raise TypeError(msg)
        # Generate PolyPlane
        polyplane = _vtk.vtkPolyPlane()
        polyplane.SetPolyLine(polyline)
        # Create slice
        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(self)  # Use the grid as the data we desire to cut
        alg.SetCutFunction(polyplane)  # the cutter to use the poly planes
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        _update_alg(alg, progress_bar=progress_bar, message='Slicing along Line')
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output

    @_deprecate_positional_args
    def extract_all_edges(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        use_all_points: bool | None = None,  # noqa: FBT001
        clear_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Extract all the internal/external edges of the dataset as PolyData.

        This produces a full wireframe representation of the input dataset.

        Parameters
        ----------
        use_all_points : bool, optional
            .. deprecated:: 0.44.0
               Parameter ``use_all_points`` is deprecated since VTK < 9.2 is no
               longer supported. This parameter has no effect and is always ``True``.

        clear_data : bool, default: False
            Clear any point, cell, or field data. This is useful
            if wanting to strictly extract the edges.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Edges extracted from the dataset.

        Examples
        --------
        Extract the edges of a sample unstructured grid and plot the edges.
        Note how it plots interior edges.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hex_beam = pv.read(examples.hexbeamfile)
        >>> edges = hex_beam.extract_all_edges()
        >>> edges.plot(line_width=5, color='k')

        See :ref:`cell_centers_example` for more examples using this filter.

        """
        if use_all_points is not None:
            warn_external(
                "Parameter 'use_all_points' is deprecated since VTK < 9.2 is no longer "
                'supported. This parameter has no effect and is always `True`.',
                PyVistaDeprecationWarning,
            )

        alg = _vtk.vtkExtractEdges()
        alg.SetInputDataObject(self)
        # Always use all points since VTK >= 9.2 is required
        alg.SetUseAllPoints(True)
        # Suppress improperly used INFO for debugging messages in vtkExtractEdges
        with pv.vtk_verbosity('off'):
            _update_alg(alg, progress_bar=progress_bar, message='Extracting All Edges')
        output = _get_output(alg)
        if clear_data:
            output.clear_data()
        return output

    @_deprecate_positional_args
    def extract_surface(  # type: ignore[misc]  # noqa: PLR0917
        self: DataSet | MultiBlock,
        pass_pointid: bool = True,  # noqa: FBT001, FBT002
        pass_cellid: bool = True,  # noqa: FBT001, FBT002
        nonlinear_subdivision: int | None = None,
        algorithm: _ExtractSurfaceOptions | type[_SENTINEL] = _SENTINEL,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> PolyData:
        """Extract surface geometry of the mesh as :class:`~pyvista.PolyData`.

        .. note::
            The underlying VTK algorithm can be selected for surface extraction.
            Using ``algorithm=None`` is recommended, and the appropriate algorithm
            will automatically be selected. The current default is ``'dataset_surface'``,
            but this will change to ``None`` in a future version.

        .. versionchanged:: 0.47
            This filter is now generalized to also work with :class:`~pyvista.MultiBlock`.

        Parameters
        ----------
        pass_pointid : bool, default: True
            Adds a point array ``"vtkOriginalPointIds"`` that identifies which original points
            these surface points correspond to.

        pass_cellid : bool, default: True
            Adds a cell array ``"vtkOriginalCellIds"`` that identifies which original cells these
            surface cells correspond to.

        nonlinear_subdivision : int, default: 1
            Determines how many times the faces of non-linear cells are subdivided into linear
            faces. This option is only relevant when the input is an
            :class:`~pyvista.UnstructuredGrid` with non-linear cells, and cannot be used with
            the ``'geometry'`` algorithm.

            If ``0``, the output is the equivalent to its linear counterpart (and the midpoints
            determining the non-linear interpolation are discarded). If ``1`` (the default), the
            non-linear face is triangulated based on the midpoints. If greater than ``1``, the
            triangulated pieces are recursively subdivided to reach the desired subdivision.
            Setting the value to greater than ``1`` may cause some point data to not be passed even
            if no nonlinear faces exist.

        algorithm : 'auto' | 'geometry' | 'dataset_surface'
            VTK algorithm to use internally.

            - ``'geometry'``: use :vtk:`vtkGeometryFilter`.
            - ``'dataset_surface'``: use :vtk:`vtkDataSetSurfaceFilter`.
            - ``None``: The algorithm is automatically selected based on the input. For most
              cases, the ``'geometry'`` algorithm is selected by default. The ``'dataset_surface'``
              algorithm is only selected for cases where the input is an
              :class:`~pyvista.UnstructuredGrid` with non-linear cells.

            Using ``algorithm=None`` is recommended. The current default is ``'dataset_surface'``,
            but this will change to ``None`` in a future version.

            Both algorithms produce similar surfaces, but ``'geometry'`` is more performant.
            The ``'geometry'`` algorithm also

            - merges points by default,
            - tends to preserve the original mesh's point order and connectivity, and
            - generates closed surfaces where closed surfaces would normally be expected.

            See :ref:`compare_surface_extract_algorithms` for some examples of differences.

            In general, users should not need to select the specific algorithm. This option is
            mostly provided for backwards-compatibility or specific use cases. For example, if
            working with both linear and non-linear meshes, it may be preferable to use
            ``'dataset_surface'`` explicitly so that the generated surfaces may be more directly
            comparable.

            .. versionadded:: 0.47

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Surface mesh of the grid.

        Warnings
        --------
        Both ``"vtkOriginalPointIds"`` and ``"vtkOriginalCellIds"`` may be
        affected by other VTK operations. See `issue 1164
        <https://github.com/pyvista/pyvista/issues/1164>`_ for
        recommendations on tracking indices across operations.

        Examples
        --------
        Extract the surface of an UnstructuredGrid.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_hexbeam()
        >>> surf = grid.extract_surface(algorithm=None)
        >>> type(surf)
        <class 'pyvista.core.pointset.PolyData'>
        >>> surf['vtkOriginalPointIds']
        pyvista_ndarray([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                         28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                         42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                         56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                         70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                         84, 85, 86, 87, 88, 89])
        >>> surf['vtkOriginalCellIds']
        pyvista_ndarray([ 0,  0,  0,  1,  1,  1,  3,  3,  3,  2,  2,  2, 36, 36,
                         36, 37, 37, 37, 39, 39, 39, 38, 38, 38,  5,  5,  9,  9,
                         13, 13, 17, 17, 21, 21, 25, 25, 29, 29, 33, 33,  4,  4,
                          8,  8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28, 32, 32,
                          7,  7, 11, 11, 15, 15, 19, 19, 23, 23, 27, 27, 31, 31,
                         35, 35,  6,  6, 10, 10, 14, 14, 18, 18, 22, 22, 26, 26,
                         30, 30, 34, 34])

        Note that in the "vtkOriginalCellIds" array, the same original cells
        appears multiple times since this array represents the original cell of
        each surface cell extracted.

        See the :ref:`extract_surface_example` and :ref:`surface_smoothing_example`
        for more examples using this filter.

        """

        def warn_future():
            # Deprecated v0.47, convert to error in v0.50, remove v0.51
            if pv.version_info >= (0, 50):  # pragma: no cover
                msg = (
                    'Convert this future warning into an error '
                    'and update the docstring default value to None.'
                )
                raise RuntimeError(msg)
            if pv.version_info >= (0, 51):  # pragma: no cover
                msg = (
                    'Remove this future warning. _SENTINEL should be removed and the default '
                    'value in the function signature should be `algorithm=None`.'
                )
                raise RuntimeError(msg)

            msg = (
                f'The default value of `algorithm` for the filter\n'
                f'`{self.__class__.__name__}.extract_surface` will change in the future. '
                "It currently defaults to\n`'dataset_surface'`, but will change to `None`. "
                'Explicitly set the `algorithm` keyword to\nsilence this warning.'
            )
            warn_external(msg, pv.PyVistaFutureWarning)

        if algorithm is _SENTINEL:
            # Warn about future change in default alg
            warn_future()
            # The old default is 'dataset_surface', will be None in the future
            algorithm = 'dataset_surface'
        else:
            _validation.check_contains(
                get_args(_ExtractSurfaceOptions), must_contain=algorithm, name='algorithm'
            )

        if nonlinear_subdivision is None:
            nonlinear_subdivision = 1
        elif algorithm == 'geometry':
            msg = (
                'geometry algorithm cannot process non-linear cells and therefore '
                'cannot be used to control non-linear subdivision.'
            )
            raise ValueError(msg)

        if isinstance(self, pv.MultiBlock):
            # Extract surface from each block separately and combine into a single PolyData
            multi_polys = self.generic_filter(
                '_extract_surface',
                pass_pointid=pass_pointid,
                pass_cellid=pass_cellid,
                nonlinear_subdivision=nonlinear_subdivision,
                algorithm=algorithm,
                progress_bar=progress_bar,
            )
            append = _vtk.vtkAppendPolyData()
            for poly in multi_polys.recursive_iterator(skip_empty=True, skip_none=True):
                append.AddInputData(poly)
            _update_alg(append, progress_bar=progress_bar, message='Appending PolyData')
            return _get_output(append)

        return self._extract_surface(
            pass_pointid=pass_pointid,
            pass_cellid=pass_cellid,
            nonlinear_subdivision=nonlinear_subdivision,
            algorithm=algorithm,  # type: ignore[arg-type]
            progress_bar=progress_bar,
        )

    @_deprecate_positional_args
    def elevation(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        low_point: VectorLike[float] | None = None,
        high_point: VectorLike[float] | None = None,
        scalar_range: str | VectorLike[float] | None = None,
        preference: Literal['point', 'cell'] = 'point',
        set_active: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Generate scalar values on a dataset.

        The scalar values lie within a user specified range, and are
        generated by computing a projection of each dataset point onto
        a line.  The line can be oriented arbitrarily.  A typical
        example is to generate scalars based on elevation or height
        above a plane.

        .. warning::
           This will create a scalars array named ``'Elevation'`` on the
           point data of the input dataset and overwrite the array
           named ``'Elevation'`` if present.

        Parameters
        ----------
        low_point : sequence[float], optional
            The low point of the projection line in 3D space. Default is bottom
            center of the dataset. Otherwise pass a length 3 sequence.

        high_point : sequence[float], optional
            The high point of the projection line in 3D space. Default is top
            center of the dataset. Otherwise pass a length 3 sequence.

        scalar_range : str | sequence[float], optional
            The scalar range to project to the low and high points on the line
            that will be mapped to the dataset. If None given, the values will
            be computed from the elevation (Z component) range between the
            high and low points. Min and max of a range can be given as a length
            2 sequence. If ``str``, name of scalar array present in the
            dataset given, the valid range of that array will be used.

        preference : str, default: "point"
            When an array name is specified for ``scalar_range``, this is the
            preferred array type to search for in the dataset.
            Must be either ``'point'`` or ``'cell'``.

        set_active : bool, default: True
            A boolean flag on whether or not to set the new
            ``'Elevation'`` scalar as the active scalars array on the
            output dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset containing elevation scalars in the
            ``"Elevation"`` array in ``point_data``.

        Examples
        --------
        Generate the "elevation" scalars for a sphere mesh.  This is
        simply the height in Z from the XY plane.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere_elv = sphere.elevation()
        >>> sphere_elv.plot(smooth_shading=True)

        Access the first 4 elevation scalars.  This is a point-wise
        array containing the "elevation" of each point.

        >>> sphere_elv['Elevation'][:4]  # doctest:+SKIP
        array([-0.5       ,  0.5       , -0.49706897, -0.48831028], dtype=float32)

        See :ref:`using_filters_example` for more examples using this filter.

        """
        # Fix the projection line:
        if low_point is None:
            low_point_ = list(self.center)
            low_point_[2] = self.bounds.z_min
        else:
            low_point_ = _validation.validate_array3(low_point)
        if high_point is None:
            high_point_ = list(self.center)
            high_point_[2] = self.bounds.z_max
        else:
            high_point_ = _validation.validate_array3(high_point)
        # Fix scalar_range:
        if scalar_range is None:
            scalar_range_ = (low_point_[2], high_point_[2])
        elif isinstance(scalar_range, str):
            scalar_range_ = self.get_data_range(scalar_range, preference=preference)
        else:
            scalar_range_ = _validation.validate_data_range(scalar_range)

        # Construct the filter
        alg = _vtk.vtkElevationFilter()
        alg.SetInputDataObject(self)
        # Set the parameters
        alg.SetScalarRange(scalar_range_)
        alg.SetLowPoint(low_point_)
        alg.SetHighPoint(high_point_)
        _update_alg(alg, progress_bar=progress_bar, message='Computing Elevation')
        # Decide on updating active scalars array
        output = _get_output(alg)
        if not set_active:
            # 'Elevation' is automatically made active by the VTK filter
            output.point_data.active_scalars_name = self.point_data.active_scalars_name
        return output

    @_deprecate_positional_args
    def compute_cell_sizes(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        length: bool = True,  # noqa: FBT001, FBT002
        area: bool = True,  # noqa: FBT001, FBT002
        volume: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        vertex_count: bool = False,  # noqa: FBT001, FBT002
    ):
        """Compute sizes for 0D (vertex count), 1D (length), 2D (area) and 3D (volume) cells.

        Parameters
        ----------
        length : bool, default: True
            Specify whether or not to compute the length of 1D cells.

        area : bool, default: True
            Specify whether or not to compute the area of 2D cells.

        volume : bool, default: True
            Specify whether or not to compute the volume of 3D cells.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        vertex_count : bool, default: False
            Specify whether or not to compute sizes for vertex and polyvertex cells (0D cells).
            The computed value is the number of points in the cell.

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset with `cell_data` containing the ``"VertexCount"``,
            ``"Length"``, ``"Area"``, and ``"Volume"`` arrays if set
            in the parameters.  Return type matches input.

        Notes
        -----
        If cells do not have a dimension (for example, the length of
        hexahedral cells), the corresponding array will be all zeros.

        Examples
        --------
        Compute the face area of the example airplane mesh.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf.plot(show_edges=True, scalars='Area')

        """
        alg = _vtk.vtkCellSizeFilter()
        alg.SetInputDataObject(self)
        alg.SetComputeArea(area)
        alg.SetComputeVolume(volume)
        alg.SetComputeLength(length)
        alg.SetComputeVertexCount(vertex_count)
        _update_alg(alg, progress_bar=progress_bar, message='Computing Cell Sizes')
        return _get_output(alg)

    @_deprecate_positional_args
    def cell_centers(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        vertex: bool = True,  # noqa: FBT001, FBT002
        pass_cell_data: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Generate points at the center of the cells in this dataset.

        These points can be used for placing glyphs or vectors.

        Parameters
        ----------
        vertex : bool, default: True
            Enable or disable the generation of vertex cells.

        pass_cell_data : bool, default: True
            If enabled, pass the input cell data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Polydata where the points are the cell centers of the
            original dataset.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Plane()
        >>> mesh.point_data.clear()
        >>> centers = mesh.cell_centers()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, show_edges=True)
        >>> actor = pl.add_points(
        ...     centers,
        ...     render_points_as_spheres=True,
        ...     color='red',
        ...     point_size=20,
        ... )
        >>> pl.show()

        See :ref:`cell_centers_example` for more examples using this filter.

        """
        input_mesh = self.cast_to_poly_points() if isinstance(self, pv.PointSet) else self
        alg = _vtk.vtkCellCenters()
        alg.SetInputDataObject(input_mesh)
        alg.SetVertexCells(vertex)
        alg.SetCopyArrays(pass_cell_data)
        _update_alg(
            alg, progress_bar=progress_bar, message='Generating Points at the Center of the Cells'
        )
        return _get_output(alg)

    @_deprecate_positional_args
    def cell_data_to_point_data(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        pass_cell_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        The method of transformation is based on averaging the data
        values of all cells using a particular point. Optionally, the
        input cell data can be passed through to the output as well.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            If enabled, pass the input cell data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset with the point data transformed into cell data.
            Return type matches input.

        See Also
        --------
        point_data_to_cell_data
            Similar transformation applied to point data.
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
            Re-mesh :class:`~pyvista.ImageData` to a points-based representation.

        Examples
        --------
        First compute the face area of the example airplane mesh and
        show the cell values.  This is to show discrete cell data.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf.plot(scalars='Area')

        These cell scalars can be applied to individual points to
        effectively smooth out the cell data onto the points.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf = surf.cell_data_to_point_data()
        >>> surf.plot(scalars='Area')

        """
        alg = _vtk.vtkCellDataToPointData()
        alg.SetInputDataObject(self)
        alg.SetPassCellData(pass_cell_data)
        _update_alg(
            alg, progress_bar=progress_bar, message='Transforming cell data into point data.'
        )
        active_scalars = None
        if not isinstance(self, pv.MultiBlock):
            active_scalars = self.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    @_deprecate_positional_args
    def ctp(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        pass_cell_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        This method is an alias for :func:`cell_data_to_point_data`.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            If enabled, pass the input cell data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Deprecated keyword argument ``pass_cell_arrays``.

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset with the cell data transformed into point data.
            Return type matches input.

        """
        return self.cell_data_to_point_data(
            pass_cell_data=pass_cell_data,
            progress_bar=progress_bar,
            **kwargs,
        )

    @_deprecate_positional_args
    def point_data_to_cell_data(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        pass_point_data: bool = False,  # noqa: FBT001, FBT002
        categorical: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified within cells.
        Optionally, the input point data can be passed through to the output.

        Parameters
        ----------
        pass_point_data : bool, default: False
            If enabled, pass the input point data through to the output.

        categorical : bool, default: False
            Control whether the source point data is to be treated as
            categorical. If ``True``,  histograming is used to assign the
            cell data. Specifically, a histogram is populated for each cell
            from the scalar values at each point, and the bin with the most
            elements is selected. In case of a tie, the smaller value is selected.

            .. note::

                If the point data is continuous, values that are almost equal (within
                ``1e-6``) are merged into a single bin. Otherwise, for discrete data
                the number of bins equals the number of unique values.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset with the point data transformed into cell data.
            Return type matches input.

        See Also
        --------
        cell_data_to_point_data
            Similar transformation applied to cell data.
        :meth:`~pyvista.ImageDataFilters.points_to_cells`
            Re-mesh :class:`~pyvista.ImageData` to a cells-based representation.

        Examples
        --------
        Color cells by their z coordinates.  First, create point
        scalars based on z-coordinates of a sample sphere mesh.  Then
        convert this point data to cell data.  Use a low resolution
        sphere for emphasis of cell valued data.

        First, plot these values as point values to show the
        difference between point and cell data.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> sphere['Z Coordinates'] = sphere.points[:, 2]
        >>> sphere.plot()

        Now, convert these values to cell data and then plot it.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> sphere['Z Coordinates'] = sphere.points[:, 2]
        >>> sphere = sphere.point_data_to_cell_data()
        >>> sphere.plot()

        """
        alg = _vtk.vtkPointDataToCellData()
        alg.SetInputDataObject(self)
        alg.SetPassPointData(pass_point_data)
        alg.SetCategoricalData(categorical)
        _update_alg(
            alg, progress_bar=progress_bar, message='Transforming point data into cell data'
        )
        active_scalars = None
        if not isinstance(self, pv.MultiBlock):
            active_scalars = self.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    @_deprecate_positional_args
    def ptc(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        pass_point_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        This method is an alias for :func:`point_data_to_cell_data`.

        Parameters
        ----------
        pass_point_data : bool, default: False
            If enabled, pass the input point data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Deprecated keyword argument ``pass_point_arrays``.

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset with the point data transformed into cell data.
            Return type matches input.

        """
        return self.point_data_to_cell_data(
            pass_point_data=pass_point_data,
            progress_bar=progress_bar,
            **kwargs,
        )

    @_deprecate_positional_args
    def triangulate(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Return an all triangle mesh.

        More complex polygons will be broken down into triangles.

        Parameters
        ----------
        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing only triangles.

        Examples
        --------
        Generate a mesh with quadrilateral faces.

        >>> import pyvista as pv
        >>> plane = pv.Plane()
        >>> plane.point_data.clear()
        >>> plane.plot(show_edges=True, line_width=5)

        Convert it to an all triangle mesh.

        >>> mesh = plane.triangulate()
        >>> mesh.plot(show_edges=True, line_width=5)

        """
        alg = _vtk.vtkDataSetTriangleFilter()
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Converting to triangle mesh')

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)
            return self
        return mesh

    @_deprecate_positional_args(allowed=['target'])
    def sample(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        target: DataSet | _vtk.vtkDataSet,
        tolerance: float | None = None,
        pass_cell_data: bool = True,  # noqa: FBT001, FBT002
        pass_point_data: bool = True,  # noqa: FBT001, FBT002
        categorical: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        locator: Literal['cell', 'cell_tree', 'obb_tree', 'static_cell']
        | _vtk.vtkAbstractCellLocator
        | None = 'static_cell',
        pass_field_data: bool = True,  # noqa: FBT001, FBT002
        mark_blank: bool = True,  # noqa: FBT001, FBT002
        snap_to_closest_point: bool = False,  # noqa: FBT001, FBT002
    ):
        """Resample array data from a passed mesh onto this mesh.

        For `mesh1.sample(mesh2)`, the arrays from `mesh2` are sampled onto
        the points of `mesh1`.  This function interpolates within an
        enclosing cell.  This contrasts with
        :func:`pyvista.DataSetFilters.interpolate` that uses a distance
        weighting for nearby points.  If there is cell topology, `sample` is
        usually preferred.

        The point data 'vtkValidPointMask' stores whether the point could be sampled
        with a value of 1 meaning successful sampling. And a value of 0 means
        unsuccessful.

        This uses :vtk:`vtkResampleWithDataSet`.

        Parameters
        ----------
        target : pyvista.DataSet
            The vtk data object to sample from - point and cell arrays from
            this object are sampled onto the nodes of the ``dataset`` mesh.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        pass_cell_data : bool, default: True
            Preserve source mesh's original cell data arrays.

        pass_point_data : bool, default: True
            Preserve source mesh's original point data arrays.

        categorical : bool, default: False
            Control whether the source point data is to be treated as
            categorical. If the data is categorical, then the resultant data
            will be determined by a nearest neighbor interpolation scheme.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        locator : :vtk:`vtkAbstractCellLocator` or str or None, default: 'static_cell'
            Prototype cell locator to perform the ``FindCell()``
            operation.  If ``None``, uses the DataSet ``FindCell`` method.
            Valid strings with mapping to vtk cell locators are

                * 'cell' - :vtk:`vtkCellLocator`
                * 'cell_tree' - :vtk:`vtkCellTreeLocator`
                * 'obb_tree' - :vtk:`vtkOBBTree`
                * 'static_cell' - :vtk:`vtkStaticCellLocator`

        pass_field_data : bool, default: True
            Preserve source mesh's original field data arrays.

        mark_blank : bool, default: True
            Whether to mark blank points and cells in "vtkGhostType".

        snap_to_closest_point : bool, default: False
            Whether to snap to cell with closest point if no cell is found. Useful
            when sampling from data with vertex cells. Requires vtk >=9.3.0.

            .. versionadded:: 0.43

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset containing resampled data.

        See Also
        --------
        pyvista.DataSetFilters.interpolate
            Interpolate values from one mesh onto another.

        pyvista.ImageDataFilters.resample
            Resample image data to modify its dimensions and spacing.

        Examples
        --------
        Resample data from another dataset onto a sphere.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
        >>> data_to_probe = examples.load_uniform()
        >>> result = mesh.sample(data_to_probe)
        >>> result.plot(scalars='Spatial Point Data')

        If sampling from a set of points represented by a ``(n, 3)``
        shaped ``numpy.ndarray``, they need to be converted to a
        PyVista DataSet, e.g. :class:`pyvista.PolyData`, first.

        >>> import numpy as np
        >>> points = np.array([[1.5, 5.0, 6.2], [6.7, 4.2, 8.0]])
        >>> mesh = pv.PolyData(points)
        >>> result = mesh.sample(data_to_probe)
        >>> result['Spatial Point Data']
        pyvista_ndarray([ 46.5 , 225.12])

        See :ref:`resampling_example` and :ref:`interpolate_sample_example`
        for more examples using this filter.

        """
        alg = _vtk.vtkResampleWithDataSet()  # Construct the ResampleWithDataSet object
        alg.SetInputData(
            self
        )  # Set the Input data (actually the source i.e. where to sample from)
        # Set the Source data (actually the target, i.e. where to sample to)
        alg.SetSourceData(wrap(target))
        alg.SetPassCellArrays(pass_cell_data)
        alg.SetPassPointArrays(pass_point_data)
        alg.SetPassFieldArrays(pass_field_data)

        alg.SetMarkBlankPointsAndCells(mark_blank)
        alg.SetCategoricalData(categorical)

        if tolerance is not None:
            alg.SetComputeTolerance(False)
            alg.SetTolerance(tolerance)
        if locator:
            if isinstance(locator, str):
                locator_map = {
                    'cell': _vtk.vtkCellLocator(),
                    'cell_tree': _vtk.vtkCellTreeLocator(),
                    'obb_tree': _vtk.vtkOBBTree(),
                    'static_cell': _vtk.vtkStaticCellLocator(),
                }
                try:
                    locator = locator_map[locator]
                except KeyError as err:
                    msg = f'locator must be a string from {locator_map.keys()}, got {locator}'
                    raise ValueError(msg) from err
            alg.SetCellLocatorPrototype(locator)

        if snap_to_closest_point:
            try:
                alg.SnapToCellWithClosestPointOn()
            except AttributeError:  # pragma: no cover
                msg = '`snap_to_closest_point=True` requires vtk 9.3.0 or newer'
                raise VTKVersionError(msg)
        _update_alg(
            alg,
            progress_bar=progress_bar,
            message='Resampling array Data from a Passed Mesh onto Mesh',
        )
        return _get_output(alg)

    def cell_quality(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        quality_measure: Literal['all', 'all_valid']
        | _CellQualityLiteral
        | Sequence[_CellQualityLiteral] = 'scaled_jacobian',
        *,
        null_value: float = -1.0,
        progress_bar: bool = False,
    ) -> _DataSetOrMultiBlockType:
        r"""Compute a function of (geometric) quality for each cell of a mesh.

        The per-cell quality is added to the mesh's cell data, in an array with
        the same name as the quality measure. Cell types not supported by this
        filter or undefined quality of supported cell types will have an
        entry of ``-1``.

        See the :ref:`cell_quality_measures_table` below for all measures and the
        :class:`~pyvista.CellType` supported by each one.
        Defaults to computing the ``scaled_jacobian`` quality measure.

        .. _cell_quality_measures_table:

        .. include:: /api/core/cell_quality/cell_quality_measures_table.rst

        .. note::

            Refer to the `Verdict Library Reference Manual <https://github.com/sandialabs/verdict/raw/master/SAND2007-2853p.pdf>`_
            for low-level technical information about how each metric is computed.

        .. versionadded:: 0.45

        Parameters
        ----------
        quality_measure : str | sequence[str], default: 'scaled_jacobian'
            The cell quality measure(s) to use. May be either:

            - A single measure or a sequence of measures listed in
              :ref:`cell_quality_measures_table`.
            - ``'all'`` to compute all measures.
            - ``'all_valid'`` to only keep quality measures that are valid for the mesh's
              cell type(s).

            A separate array is created for each measure.

        null_value : float, default: -1.0
            Float value for undefined quality. Undefined quality are qualities
            that could be addressed by this filter but is not well defined for
            the particular geometry of cell in question, e.g. a volume query
            for a triangle. Undefined quality will always be undefined.
            The default value is -1.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        output : DataSet | MultiBlock
            Dataset with the computed mesh quality. Return type matches input.
            Cell data array(s) with the computed quality measure(s) are included.

        See Also
        --------
        :func:`~pyvista.cell_quality_info`
            Return information about a cell's quality measure, e.g. acceptable range.
        :meth:`~pyvista.DataObjectFilters.cell_validator`
        :meth:`~pyvista.DataObjectFilters.validate_mesh`

        Examples
        --------
        Compute and plot the minimum angle of a sample sphere mesh.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
        >>> cqual = sphere.cell_quality('min_angle')
        >>> cqual.plot(show_edges=True)

        Quality measures like ``'volume'`` do not apply to 2D cells, and a null value
        of ``-1`` is returned.

        >>> qual = sphere.cell_quality('volume')
        >>> qual.get_data_range('volume')
        (np.float64(-1.0), np.float64(-1.0))

        Compute all valid quality measures for the sphere. These measures all return
        non-null values for :attr:`~pyvista.CellType.TRIANGLE` cells.

        >>> cqual = sphere.cell_quality('all_valid')
        >>> valid_measures = cqual.cell_data.keys()
        >>> valid_measures  # doctest: +NORMALIZE_WHITESPACE
        ['area',
         'aspect_frobenius',
         'aspect_ratio',
         'condition',
         'distortion',
         'max_angle',
         'min_angle',
         'radius_ratio',
         'relative_size_squared',
         'scaled_jacobian',
         'shape',
         'shape_and_size']

        See :ref:`mesh_quality_example` for more examples using this filter.

        """
        # Validate measures
        _validation.check_instance(quality_measure, (str, list, tuple), name='quality_measure')
        keep_valid_only = quality_measure == 'all_valid'
        measures_available = _get_cell_quality_measures()
        measures_available_names = cast(
            'list[_CellQualityLiteral]', list(measures_available.keys())
        )
        if quality_measure in ['all', 'all_valid']:
            measures_requested = measures_available_names
        else:
            measures = [quality_measure] if isinstance(quality_measure, str) else quality_measure
            for measure in measures:
                _validation.check_contains(
                    measures_available_names, must_contain=measure, name='quality_measure'
                )
            measures_requested = cast('list[_CellQualityLiteral]', measures)

        cell_quality = functools.partial(
            DataObjectFilters._dataset_cell_quality,
            measures_requested=measures_requested,
            measures_available=measures_available,
            keep_valid_only=keep_valid_only,
            null_value=null_value,
            progress_bar=progress_bar,
        )
        return (
            self.generic_filter(cell_quality)  # type: ignore[return-value]
            if isinstance(self, pv.MultiBlock)
            else cell_quality(self)
        )

    def _dataset_cell_quality(  # type: ignore[misc]
        self: _DataSetType,
        *,
        measures_requested,
        measures_available,
        keep_valid_only,
        null_value,
        progress_bar,
    ) -> _DataSetType:
        """Compute cell quality of a DataSet (internal method)."""
        CELL_QUALITY = 'CellQuality'

        alg = _vtk.vtkCellQuality()
        alg.SetUndefinedQuality(null_value)

        if 'size' in ''.join(measures_requested):
            # Need to compute mesh quality statistics to get average cell size.
            # We only need to do this once. This will create field data arrays:
            # 'TriArea', 'QuadArea', 'TetVolume', 'PyrVolume', 'WedgeVolume', 'HexVolume'
            # which are used later by vtkCellQuality
            mesh_quality = _vtk.vtkMeshQuality()
            mesh_quality.SaveCellQualityOff()
            mesh_quality.SetInputData(self)
            # Setting any 'Size' measure for any cell (tri, quad, etc.) is sufficient to
            # ensure all necessary base stats are computed for all cell types and for
            # all 'Size' measures
            mesh_quality.SetTriangleQualityMeasureToShapeAndSize()
            _update_alg(mesh_quality)

            alg.SetInputDataObject(mesh_quality.GetOutput())
        else:
            alg.SetInputDataObject(self)

        output = self.copy()

        # Compute all measures
        for measure in measures_requested:
            # Set measure and update
            getattr(alg, measures_available[measure])()
            _update_alg(
                alg, progress_bar=progress_bar, message=f"Computing Cell Quality '{measure}'"
            )

            # Store the cell quality array with the output
            cell_quality_array = _get_output(alg).cell_data[CELL_QUALITY]
            if keep_valid_only and (
                np.max(cell_quality_array) == np.min(cell_quality_array) == null_value
            ):
                continue
            output.cell_data[measure] = cell_quality_array
        return output


def _get_cell_quality_measures() -> dict[str, str]:
    """Return snake case quality measure keys and vtkCellQuality attribute setter names."""
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


def _remove_unused_points_post_clip(clip_output, input_bounds):
    # VTK clip filters are buggy and sometimes retain unused points from the input, e.g.:
    # https://github.com/pyvista/pyvista/issues/6511
    # https://github.com/pyvista/pyvista/issues/7738

    def maybe_remove_unused_points(mesh: DataSet):
        # Unused points are correctly removed sometimes, so for performance we only
        # remove points when the clipped bounds match input bounds
        if np.allclose(clip_output.bounds, input_bounds) and hasattr(mesh, 'remove_unused_points'):
            return mesh.remove_unused_points()
        return mesh

    return (
        clip_output.generic_filter(maybe_remove_unused_points)
        if isinstance(clip_output, pv.MultiBlock)
        else maybe_remove_unused_points(clip_output)
    )


def _cast_output_to_match_input_type(
    output_mesh: DataSet | MultiBlock, input_mesh: DataSet | MultiBlock
):
    # Ensure output type matches input type

    def cast_output(mesh_out: DataSet, mesh_in: DataSet):
        if isinstance(mesh_in, pv.PolyData) and not isinstance(mesh_out, pv.PolyData):
            return mesh_out.extract_surface(algorithm=None, pass_cellid=False, pass_pointid=False)
        elif isinstance(mesh_in, pv.PointSet) and not isinstance(mesh_out, pv.PointSet):
            return mesh_out.cast_to_pointset()
        return mesh_out

    def cast_output_blocks(mesh_out: MultiBlock, mesh_in: MultiBlock):
        # Replace all blocks in the output mesh with cast versions that match the input
        for (ids, _, block_out), block_in in zip(
            mesh_out.recursive_iterator('all', skip_none=True),
            mesh_in.recursive_iterator(skip_none=True),
            strict=True,
        ):
            mesh_out.replace(ids, cast_output(block_out, block_in))
        return mesh_out

    return (
        cast_output_blocks(output_mesh, input_mesh)  # type: ignore[arg-type]
        if isinstance(output_mesh, pv.MultiBlock)
        else cast_output(output_mesh, input_mesh)  # type: ignore[arg-type]
    )


class _Crinkler:
    CELL_IDS = 'cell_ids'
    INT_DTYPE = np.int64
    ITER_KWARGS: ClassVar = dict(skip_none=True)

    @staticmethod
    def extract_cells(dataset, ids, active_scalars_info_):
        # Extract cells and remove arrays, and restore active scalars
        output = dataset.extract_cells(ids, pass_cell_ids=False, pass_point_ids=False)
        association, name = active_scalars_info_
        if not dataset.is_empty:
            dataset.set_active_scalars(name, preference=association)
        if not output.is_empty:
            output.set_active_scalars(name, preference=association)
        return output

    @staticmethod
    def extract_crinkle_cells(dataset, a_, b_, active_scalars_info):  # noqa: PLR0917
        if b_ is None:
            # Extract cells when `return_clipped=False`
            def extract_cells_from_block(block_, clipped_a, _, active_scalars_info_):
                if _Crinkler.CELL_IDS in clipped_a.cell_data.keys():
                    return _Crinkler.extract_cells(
                        block_,
                        np.unique(clipped_a.cell_data[_Crinkler.CELL_IDS]),
                        active_scalars_info_,
                    )
                return clipped_a
        else:
            # Extract cells when `return_clipped=True`
            def extract_cells_from_block(  # noqa: PLR0917
                block_, clipped_a, clipped_b, active_scalars_info_
            ):
                set_a = (
                    set(clipped_a.cell_data[_Crinkler.CELL_IDS])
                    if _Crinkler.CELL_IDS in clipped_a.cell_data.keys()
                    else set()
                )
                set_b = (
                    set(clipped_b.cell_data[_Crinkler.CELL_IDS])
                    if _Crinkler.CELL_IDS in clipped_b.cell_data.keys()
                    else set()
                )
                set_b = set_b - set_a

                # Need to cast as int dtype explicitly to ensure empty arrays have
                # the right type required by extract_cells
                array_a = np.array(list(set_a), dtype=_Crinkler.INT_DTYPE)
                array_b = np.array(list(set_b), dtype=_Crinkler.INT_DTYPE)

                clipped_a = _Crinkler.extract_cells(block_, array_a, active_scalars_info_)
                clipped_b = _Crinkler.extract_cells(block_, array_b, active_scalars_info_)
                return clipped_a, clipped_b

        def extract_cells_from_multiblock(  # noqa: PLR0917
            multi_in, multi_a, multi_b, active_scalars_info_
        ):
            # Iterate though input and output multiblocks
            # `multi_b` may be None depending on `return_clipped`
            self_iter = multi_in.recursive_iterator('all', **_Crinkler.ITER_KWARGS)
            a_iter = multi_a.recursive_iterator(**_Crinkler.ITER_KWARGS)
            b_iter = (
                multi_b.recursive_iterator(**_Crinkler.ITER_KWARGS)
                if multi_b is not None
                else itertools.repeat(None)
            )

            for (ids, _, block_self), block_a, block_b, scalars_info in zip(
                self_iter, a_iter, b_iter, active_scalars_info_, strict=False
            ):
                crinkled = extract_cells_from_block(block_self, block_a, block_b, scalars_info)
                # Replace blocks with crinkled ones
                if block_b is None:
                    # Only need to replace one block
                    multi_a.replace(ids, crinkled)
                else:
                    multi_a.replace(ids, crinkled[0])
                    multi_b.replace(ids, crinkled[1])
            return multi_a if multi_b is None else (multi_a, multi_b)

        if isinstance(dataset, pv.MultiBlock):
            return extract_cells_from_multiblock(dataset, a_, b_, active_scalars_info)
        return extract_cells_from_block(dataset, a_, b_, active_scalars_info[0])

    @staticmethod
    def add_cell_ids(dataset: DataSet | MultiBlock):
        # Add Cell IDs to all blocks and keep track of scalars to restore later
        active_scalars_info = []
        if isinstance(dataset, pv.MultiBlock):
            blocks: Iterable[DataSet] = dataset.recursive_iterator(
                'blocks',
                **_Crinkler.ITER_KWARGS,  # type: ignore[call-overload]
            )
        else:
            blocks = [dataset]
        for block in blocks:
            active_scalars_info.append(block.active_scalars_info)
            block.cell_data[_Crinkler.CELL_IDS] = np.arange(
                block.n_cells, dtype=_Crinkler.INT_DTYPE
            )
        return active_scalars_info
