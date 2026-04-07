"""CalculiX FRD file parser for PyVista."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import IntEnum
import pathlib
import re
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

import pyvista as pv
from pyvista.core.celltype import CellType

if TYPE_CHECKING:
    from pyvista import UnstructuredGrid


class FRDBlock(Enum):
    """CalculiX FRD block types."""

    NODES = '2C'
    ELEMENTS = '3C'
    RESULTS = '100'


class CGXRecord(IntEnum):
    """CGX Record Types.

    Source https://www.dhondt.de/cgx_2.23.pdf (11 Result Format, page 144).
    """

    NODAL_VALUES = -1
    ELEMENT_FACES = -2
    END_OF_BLOCK = -3
    ATTRIBUTE_HEADER = -4
    COMPONENT_DEFINITION = -5


class FRDElementType(IntEnum):
    """CalculiX FRD element types."""

    HE8 = 1
    PE6 = 2
    TE4 = 3
    HE20 = 4
    PE15 = 5
    TE10 = 6
    TR3 = 7
    TR6 = 8
    QU4 = 9
    QU8 = 10
    BE2 = 11
    BE3 = 12


# CalculiX element type -> VTK cell type
CCX_TO_VTK_TYPE: dict[FRDElementType, CellType] = {
    FRDElementType.HE8: CellType.HEXAHEDRON,
    FRDElementType.PE6: CellType.WEDGE,
    FRDElementType.TE4: CellType.TETRA,
    FRDElementType.HE20: CellType.QUADRATIC_HEXAHEDRON,
    FRDElementType.PE15: CellType.QUADRATIC_WEDGE,
    FRDElementType.TE10: CellType.QUADRATIC_TETRA,
    FRDElementType.TR3: CellType.TRIANGLE,
    FRDElementType.TR6: CellType.QUADRATIC_TRIANGLE,
    FRDElementType.QU4: CellType.QUAD,
    FRDElementType.QU8: CellType.QUADRATIC_QUAD,
    FRDElementType.BE2: CellType.LINE,
    FRDElementType.BE3: CellType.QUADRATIC_EDGE,
}

# Results hierarchy: step_time -> result_name -> node_id -> values
NodeResultData = dict[int, list[float]]
StepBucket = dict[str, NodeResultData]
ResultsByStep = dict[float, StepBucket]


class _LineTrackingStream:
    """Wrap a file-like iterator to track line numbers automatically."""

    def __init__(self, lines: Any) -> None:
        self._lines = iter(lines)
        self.line_number = 0

    def __iter__(self) -> _LineTrackingStream:
        return self

    def __next__(self) -> str:
        line = next(self._lines)
        self.line_number += 1
        return line


@dataclass
class _InvalidElement:
    """Dataclass to store invalid elements detected when parsing."""

    line_number: int
    element_type: int
    n_nodes_expected: int | None = None
    n_nodes_actual: int | None = None

    def __str__(self) -> str:
        items_to_print: list[str] = [f'line {self.line_number}']
        etype = self.element_type
        content = (
            f'{etype.value} ({etype.name})' if isinstance(etype, FRDElementType) else str(etype)
        )
        items_to_print.append(f'element type {content}')

        actual = self.n_nodes_actual
        expected = self.n_nodes_expected
        if actual is not None and expected is not None:
            items_to_print.append(f'num nodes {actual} (expected {expected})')
        return ', '.join(items_to_print)


@dataclass
class _FRDData:
    # Parsed data
    nodes: dict[int, list[float]] = field(default_factory=dict)
    elements: list[list[int]] = field(default_factory=list)
    cell_types: list[int] = field(default_factory=list)
    results_by_step: ResultsByStep = field(default_factory=dict)

    # Diagnostic data
    _has_too_many_points: list[_InvalidElement] = field(default_factory=list)
    _has_too_few_points: list[_InvalidElement] = field(default_factory=list)
    _has_unsupported_element: list[_InvalidElement] = field(default_factory=list)

    # Format detection
    is_long_format: bool = False
    _format_detected: bool = False


class _FRDParser:
    """Parses a CalculiX FRD file into an FRDData object."""

    # Compiled regex to fix scientific notation formatting issues
    _SCIENTIFIC_RE = re.compile(r'(?<![EeDd])-')

    def __init__(self, filename: str) -> None:
        self._filename = filename

    def parse(self) -> _FRDData:
        frd_data = _FRDData()
        with pathlib.Path(self._filename).open(errors='replace') as file_stream:
            lines = _LineTrackingStream(file_stream)
            for line in lines:
                s = line.strip()
                if s.startswith(FRDBlock.NODES.value):
                    self._parse_nodes(lines, frd_data)
                elif s.startswith(FRDBlock.ELEMENTS.value):
                    self._parse_elements(lines, frd_data)
                elif s.startswith(FRDBlock.RESULTS.value):
                    _step_id, step_time = self._parse_100cl_header(s)
                    frd_data.results_by_step.setdefault(step_time, {})
                    self._parse_results(lines, frd_data.results_by_step[step_time], frd_data)
        return frd_data

    @staticmethod
    def _fix_scientific(line: str) -> str:
        return _FRDParser._SCIENTIFIC_RE.sub(' -', line)

    @staticmethod
    def _parse_100cl_header(line: str) -> tuple[int, float]:
        parts = line.split()
        try:
            return int(parts[1]), float(parts[2])
        except (IndexError, ValueError):
            return -1, 0.0

    @staticmethod
    def _permute_nodes(node_ids: list[int], etype: FRDElementType) -> list[int]:
        """Reorder node IDs from CalculiX to VTK conventions."""
        if etype == FRDElementType.HE20:
            return node_ids[:8] + node_ids[8:12] + node_ids[16:20] + node_ids[12:16]
        if etype == FRDElementType.PE6:
            return [node_ids[0], node_ids[2], node_ids[1], node_ids[3], node_ids[5], node_ids[4]]
        if etype == FRDElementType.PE15:
            return node_ids[:9] + node_ids[12:15] + node_ids[9:12]
        return node_ids

    @staticmethod
    def _parse_nodes(file_stream: Any, frd_data: _FRDData) -> None:
        end_block = str(CGXRecord.END_OF_BLOCK.value)
        for line in file_stream:
            s = line.strip()
            if s.startswith(end_block):
                return
            if not s.startswith('-1'):
                continue

            idx = line.find('-1')
            if idx == -1:
                continue

            # _fix_scientific handles negative coordinates glued to ID (if any)
            # split() is completely immune to varying space widths in both tests and real files
            data_str = _FRDParser._fix_scientific(line[idx + 2 :].rstrip('\n\r'))
            parts = data_str.split()
            
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    frd_data.nodes[nid] = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError:
                    pass

    @staticmethod
    def _parse_elements(file_stream: Any, frd_data: _FRDData) -> None:
        end_block = str(CGXRecord.END_OF_BLOCK.value)
        elem_def = str(CGXRecord.NODAL_VALUES.value)
        elem_faces = str(CGXRecord.ELEMENT_FACES.value)

        needed = 0
        node_ids: list[int] = []
        etype = None
        vtk_type = None
        elem_line_number = -1

        for line in file_stream:
            s = line.strip()

            # Prevent overlapping errors - if the line is not consecutive nodes
            if etype is not None and not s.startswith(elem_faces):
                invalid = _InvalidElement(
                    line_number=elem_line_number,
                    element_type=etype,
                    n_nodes_expected=needed,
                    n_nodes_actual=len(node_ids),
                )
                frd_data._has_too_few_points.append(invalid)
                etype = None  # Forces state refresh

            if s.startswith(end_block):
                return

            if s.startswith(elem_def):
                elem_line_number = file_stream.line_number

                # Key: remove "-1" before split so that a glued ID does not break the parser
                idx = line.find('-1')
                if idx == -1:
                    continue
                data_str = line[idx + 2 :]
                parts = data_str.split()

                try:
                    # parts[0] = element ID, parts[1] = element type (e.g., 1 for HE8)
                    etype_val = int(parts[1])
                except (ValueError, IndexError):
                    etype = None
                    continue

                try:
                    etype = FRDElementType(etype_val)
                except ValueError:
                    invalid = _InvalidElement(line_number=elem_line_number, element_type=etype_val)
                    frd_data._has_unsupported_element.append(invalid)
                    etype = None
                    continue

                vtk_type = CCX_TO_VTK_TYPE[etype]
                needed = vtk_type.n_points
                node_ids = []

            elif s.startswith(elem_faces) and etype is not None and vtk_type is not None:
                idx = line.find('-2')
                if idx == -1:
                    continue
                data_str = line[idx + 2 :].rstrip('\n\r')

                if not frd_data._format_detected:
                    frd_data.is_long_format = len(data_str.rstrip()) > 50
                    frd_data._format_detected = True

                width = 10 if frd_data.is_long_format else 5

                new_nodes = []
                is_fixed = True
                
                # Hybrid check: try fixed-width chunking first for standard CalculiX files
                for i in range(0, len(data_str), width):
                    chunk = data_str[i : i + width]
                    clean_chunk = chunk.strip()
                    if not clean_chunk:
                        continue
                    # If there's an internal space, it's a test mock file, not a fixed-width string
                    if ' ' in clean_chunk:
                        is_fixed = False
                        break
                    try:
                        new_nodes.append(int(clean_chunk))
                    except ValueError:
                        is_fixed = False
                        break

                if is_fixed and new_nodes:
                    node_ids.extend(new_nodes)
                else:
                    # Fallback to space-separated values for PyVista's mock test suite files
                    for p in data_str.split():
                        try:
                            node_ids.append(int(p))
                        except ValueError:
                            pass

                if (n_nodes := len(node_ids)) < needed:
                    continue

                if n_nodes > needed:
                    invalid = _InvalidElement(
                        line_number=elem_line_number,
                        element_type=etype,
                        n_nodes_expected=needed,
                        n_nodes_actual=n_nodes,
                    )
                    frd_data._has_too_many_points.append(invalid)

                final_nodes = _FRDParser._permute_nodes(node_ids, etype)
                frd_data.elements.append(final_nodes[:needed])
                frd_data.cell_types.append(vtk_type.value)

                # Clean start for the next element
                etype = None
                node_ids = []

    @staticmethod
    def _parse_results(file_stream: Any, step_bucket: StepBucket, frd_data: _FRDData) -> None:
        name = 'Unknown'
        attr_header = str(CGXRecord.ATTRIBUTE_HEADER.value)
        comp_def = str(CGXRecord.COMPONENT_DEFINITION.value)
        nodal_vals = str(CGXRecord.NODAL_VALUES.value)
        end_block = str(CGXRecord.END_OF_BLOCK.value)

        for line in file_stream:
            s = line.strip()
            if s.startswith(attr_header):
                parts = s.split()
                if len(parts) >= 2:
                    name = parts[1]
            elif s.startswith(comp_def):
                continue
            elif s.startswith(nodal_vals):
                _FRDParser._parse_result_data(line, file_stream, name, step_bucket, frd_data)
                return
            elif s.startswith(end_block):
                return

    @staticmethod
    def _parse_result_data(  # noqa: PLR0917
        first_line: str, file_stream: Any, name: str, step_bucket: StepBucket, frd_data: _FRDData
    ) -> None:
        data: dict[int, list[float]] = {}
        end_block = str(CGXRecord.END_OF_BLOCK.value)
        nodal_vals = str(CGXRecord.NODAL_VALUES.value)

        def parse_line(line_str: str) -> None:
            s = line_str.strip()
            if not s.startswith(nodal_vals):
                return
            idx = line_str.find('-1')
            if idx == -1:
                return

            data_str = _FRDParser._fix_scientific(line_str[idx + 2 :].rstrip('\n\r'))
            parts = data_str.split()
            
            if len(parts) >= 2:
                try:
                    nid = int(parts[0])
                    vals = [float(x) for x in parts[1:]]
                    if vals:
                        data[nid] = vals
                except ValueError:
                    pass

        parse_line(first_line)

        for line in file_stream:
            s = line.strip()
            if s.startswith(end_block):
                break
            parse_line(line)

        if data:
            step_bucket[name] = data

    @staticmethod
    def _compute_derived_stress(
        grid: UnstructuredGrid, base_name: str, tensor: np.ndarray
    ) -> None:
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        vmises = np.sqrt(
            0.5
            * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2 + 6.0 * (xy**2 + yz**2 + zx**2))
        )
        trace = xx + yy + zz
        grid.point_data[f'{base_name}_Mises'] = vmises
        grid.point_data[f'{base_name}_sgMises'] = np.where(
            trace != 0, np.sign(trace) * vmises, vmises
        )
        _FRDParser._compute_principals(grid, base_name, tensor)

    @staticmethod
    def _compute_derived_strain(
        grid: UnstructuredGrid, base_name: str, tensor: np.ndarray
    ) -> None:
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        k = np.sqrt(2.0) / 3.0
        vmises_strain = k * np.sqrt(
            (xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2 + 6.0 * (xy**2 + yz**2 + zx**2)
        )
        volumetric = xx + yy + zz
        grid.point_data[f'{base_name}_Mises'] = vmises_strain
        grid.point_data[f'{base_name}_sgMises'] = np.where(
            volumetric != 0, np.sign(volumetric) * vmises_strain, vmises_strain
        )
        _FRDParser._compute_principals(grid, base_name, tensor)

    @staticmethod
    def _compute_principals(grid: UnstructuredGrid, base_name: str, tensor: np.ndarray) -> None:
        n = tensor.shape[0]
        mat = np.zeros((n, 3, 3))
        mat[:, 0, 0] = tensor[:, 0]
        mat[:, 1, 1] = tensor[:, 1]
        mat[:, 2, 2] = tensor[:, 2]
        mat[:, 0, 1] = mat[:, 1, 0] = tensor[:, 3]
        mat[:, 1, 2] = mat[:, 2, 1] = tensor[:, 4]
        mat[:, 0, 2] = mat[:, 2, 0] = tensor[:, 5]

        eigvals = np.linalg.eigvalsh(mat)
        grid.point_data[f'{base_name}_PS3'] = eigvals[:, 0]
        grid.point_data[f'{base_name}_PS2'] = eigvals[:, 1]
        grid.point_data[f'{base_name}_PS1'] = eigvals[:, 2]

    @staticmethod
    def _build_grid(frd_data: _FRDData, step_data: StepBucket) -> UnstructuredGrid:

        if not frd_data.nodes:
            msg = 'No nodes found in FRD file -- cannot build grid.'
            raise ValueError(msg)

        sorted_ids = sorted(frd_data.nodes)
        node_map = {nid: idx for idx, nid in enumerate(sorted_ids)}
        points = np.array([frd_data.nodes[n] for n in sorted_ids], dtype=float)

        cells: list[int] = []
        types: list[int] = []
        for conn, ctype in zip(frd_data.elements, frd_data.cell_types, strict=True):
            try:
                vtk_ids = [node_map[n] for n in conn]
            except KeyError:
                continue
            cells.append(len(vtk_ids))
            cells.extend(vtk_ids)
            types.append(ctype)

        grid = pv.UnstructuredGrid(
            np.array(cells, dtype=np.int64),
            np.array(types, dtype=np.uint8),
            points,
        )

        grid.point_data['original_node_ids'] = np.array([str(nid) for nid in sorted_ids])

        n_points = len(points)
        for name, data in step_data.items():
            if not data:
                continue
            n_components = len(next(iter(data.values())))

            if n_components == 1:
                arr = np.zeros(n_points)
                for nid, vals in data.items():
                    if nid in node_map:
                        arr[node_map[nid]] = vals[0]
                grid.point_data[name] = arr
            else:
                arr = np.zeros((n_points, n_components))
                for nid, vals in data.items():
                    if nid in node_map:
                        arr[node_map[nid]] = vals
                grid.point_data[name] = arr

                upper = name.upper()
                if 'STRESS' in upper and n_components == 6:
                    _FRDParser._compute_derived_stress(grid, name, arr)
                elif 'STRAIN' in upper and n_components == 6:
                    _FRDParser._compute_derived_strain(grid, name, arr)

        return grid
