"""CalculiX FRD file reader for PyVista."""

from __future__ import annotations

from enum import Enum
from enum import IntEnum
import pathlib
import re
from typing import Any
from typing import ClassVar

import numpy as np

from pyvista.core.celltype import _CELL_TYPE_TO_NUM_POINTS
from pyvista.core.celltype import CellType
from pyvista.core.pointset import UnstructuredGrid
from pyvista.core.utilities.reader import BaseVTKReader


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


# ---------------------------------------------------------------------------
# Low-level, VTK-style reader
# ---------------------------------------------------------------------------


class _FRDVTKReader(BaseVTKReader):
    """VTK-style reader for CalculiX FRD ASCII result files.

    This class mirrors the interface expected by :class:`BaseVTKReader` so
    that :class:`FRDReader` can delegate to it in exactly the same way as
    other de-novo readers (e.g. ``SeriesReader``).
    """

    # Compiled regex to fix scientific notation formatting issues
    _SCIENTIFIC_RE = re.compile(r'(?<![EeDd])-')

    # CalculiX element type -> VTK cell type
    CCX_TO_VTK_TYPE: ClassVar[dict[FRDElementType, CellType]] = {
        FRDElementType.HE8: CellType.HEXAHEDRON,
        FRDElementType.PE6: CellType.WEDGE,
        FRDElementType.TE4: CellType.TETRA,
        FRDElementType.HE20: CellType.QUADRATIC_HEXAHEDRON,
        FRDElementType.PE15: CellType.WEDGE,  # degraded; pe15 not native to base VTK
        FRDElementType.TE10: CellType.QUADRATIC_TETRA,
        FRDElementType.TR3: CellType.TRIANGLE,
        FRDElementType.TR6: CellType.QUADRATIC_TRIANGLE,
        FRDElementType.QU4: CellType.QUAD,
        FRDElementType.QU8: CellType.QUADRATIC_QUAD,
        FRDElementType.BE2: CellType.LINE,
        FRDElementType.BE3: CellType.QUADRATIC_EDGE,
    }

    NODES_PER_ELEM: ClassVar[dict[FRDElementType, int]] = {
        FRDElementType.HE8: _CELL_TYPE_TO_NUM_POINTS[CellType.HEXAHEDRON],
        FRDElementType.PE6: _CELL_TYPE_TO_NUM_POINTS[CellType.WEDGE],
        FRDElementType.TE4: _CELL_TYPE_TO_NUM_POINTS[CellType.TETRA],
        FRDElementType.HE20: _CELL_TYPE_TO_NUM_POINTS[CellType.QUADRATIC_HEXAHEDRON],
        FRDElementType.PE15: 15,  # Needs to read 15 nodes from file even if degrading to WEDGE
        FRDElementType.TE10: _CELL_TYPE_TO_NUM_POINTS[CellType.QUADRATIC_TETRA],
        FRDElementType.TR3: _CELL_TYPE_TO_NUM_POINTS[CellType.TRIANGLE],
        FRDElementType.TR6: _CELL_TYPE_TO_NUM_POINTS[CellType.QUADRATIC_TRIANGLE],
        FRDElementType.QU4: _CELL_TYPE_TO_NUM_POINTS[CellType.QUAD],
        FRDElementType.QU8: _CELL_TYPE_TO_NUM_POINTS[CellType.QUADRATIC_QUAD],
        FRDElementType.BE2: _CELL_TYPE_TO_NUM_POINTS[CellType.LINE],
        FRDElementType.BE3: _CELL_TYPE_TO_NUM_POINTS[CellType.QUADRATIC_EDGE],
    }

    def __init__(self) -> None:
        super().__init__()
        # Geometry
        self._nodes: dict[int, list[float]] = {}
        self._elements: list[list[int]] = []
        self._cell_types: list[int] = []

        # Time-step data: time_value -> { result_name -> { node_id -> [values] } }
        self._results_by_step: dict[float, dict[str, dict[int, list[float]]]] = {}

        # Time steps
        self._time_steps: list[float] = []
        self._active_time_point: int = 0
        self._output_time: object = object()
        self._output: UnstructuredGrid | None = None

    def SetFileName(self, filename: str) -> None:  # noqa: N802
        self._filename = filename

    def UpdateInformation(self) -> None:  # noqa: N802
        pass

    def Update(self) -> None:  # noqa: N802
        """Parse the file using a memory-efficient iterator approach."""
        self._nodes.clear()
        self._elements.clear()
        self._cell_types.clear()
        self._results_by_step.clear()
        self._time_steps.clear()
        self._active_time_point = 0
        self._output = None
        self._output_time = object()

        with pathlib.Path(self._filename).open(errors='replace') as file_stream:
            self._parse_lines(file_stream)

        self._time_steps = sorted(self._results_by_step.keys())

    def GetOutput(self) -> UnstructuredGrid:  # noqa: N802
        """Return an UnstructuredGrid for the currently active time step."""
        target_time = self._time_steps[self._active_time_point] if self._time_steps else None

        if self._output is None or self._output_time != target_time:
            if not self._nodes:
                self.Update()
            step_data = (
                self._results_by_step.get(target_time, {}) if target_time is not None else {}
            )
            self._output = self._build_grid(step_data)
            self._output_time = target_time

        return self._output

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @classmethod
    def _fix_scientific(cls, line: str) -> str:
        """Insert a space before a bare ``-`` sign adjacent to a number."""
        return cls._SCIENTIFIC_RE.sub(' -', line)

    @staticmethod
    def _parse_100cl_header(line: str) -> tuple[int, float]:
        parts = line.split()
        try:
            return int(parts[1]), float(parts[2])
        except (IndexError, ValueError):
            return -1, 0.0

    def _permute_nodes(self, node_ids: list[int], etype: FRDElementType) -> list[int]:
        """Reorder node IDs from CalculiX to VTK conventions."""
        if etype == FRDElementType.HE20 and len(node_ids) == 20:
            return [
                node_ids[0],
                node_ids[1],
                node_ids[2],
                node_ids[3],
                node_ids[4],
                node_ids[5],
                node_ids[6],
                node_ids[7],
                node_ids[8],
                node_ids[9],
                node_ids[10],
                node_ids[11],
                node_ids[16],
                node_ids[17],
                node_ids[18],
                node_ids[19],
                node_ids[12],
                node_ids[13],
                node_ids[14],
                node_ids[15],
            ]

        if etype == FRDElementType.TR3 and len(node_ids) == 3:
            p = [self._nodes[n] for n in node_ids]
            area = (
                p[0][0] * (p[1][1] - p[2][1])
                + p[1][0] * (p[2][1] - p[0][1])
                + p[2][0] * (p[0][1] - p[1][1])
            )
            if area < 0:
                return [node_ids[0], node_ids[2], node_ids[1]]
            return node_ids

        if etype in (FRDElementType.QU4, FRDElementType.QU8):
            corners = node_ids[:4]
            mids = node_ids[4:]
            p = [self._nodes[n] for n in corners]
            area = sum(p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1] for i in range(4))
            if area < 0:
                corners = [corners[0], corners[3], corners[2], corners[1]]
                if mids:
                    mids = [mids[0], mids[3], mids[2], mids[1]]
            return corners + mids

        return node_ids

    # ------------------------------------------------------------------
    # Block parsers (Memory efficient iterators)
    # ------------------------------------------------------------------

    def _parse_lines(self, file_stream: Any) -> None:
        """Dispatch lines to block-specific parsers."""
        for line in file_stream:
            s = line.strip()
            if s.startswith(FRDBlock.NODES.value):
                self._parse_nodes(file_stream)
            elif s.startswith(FRDBlock.ELEMENTS.value):
                self._parse_elements(file_stream)
            elif s.startswith(FRDBlock.RESULTS.value):
                _step_id, step_time = self._parse_100cl_header(s)
                self._parse_results(file_stream, step_time)

    def _parse_nodes(self, file_stream: Any) -> None:
        """Parse 2C node block."""
        end_block = str(CGXRecord.END_OF_BLOCK.value)
        for line in file_stream:
            s = line.strip()
            if s.startswith(end_block):
                return
            try:
                parts = self._fix_scientific(s).split()
                nid = int(parts[1])
                self._nodes[nid] = [float(parts[2]), float(parts[3]), float(parts[4])]
            except (ValueError, IndexError):
                pass

    def _parse_elements(self, file_stream: Any) -> None:
        """Parse 3C element block."""
        needed = 0
        node_ids: list[int] = []
        etype: FRDElementType | None = None
        vtk_type: CellType | None = None

        end_block = str(CGXRecord.END_OF_BLOCK.value)
        elem_def = str(CGXRecord.NODAL_VALUES.value)
        elem_faces = str(CGXRecord.ELEMENT_FACES.value)

        for line in file_stream:
            s = line.strip()
            if s.startswith(end_block):
                return

            if s.startswith(elem_def):
                parts = s.split()
                try:
                    etype_val = int(parts[2])
                    etype = FRDElementType(etype_val)
                except (ValueError, IndexError):
                    etype = None
                    continue

                if etype not in self.CCX_TO_VTK_TYPE:
                    etype = None
                    continue

                needed = self.NODES_PER_ELEM[etype]
                vtk_type = self.CCX_TO_VTK_TYPE[etype]
                node_ids = []

            elif s.startswith(elem_faces) and etype is not None and vtk_type is not None:
                node_ids.extend(int(x) for x in s.split()[1:])
                if len(node_ids) >= needed:
                    final_nodes = node_ids[:needed]
                    final_nodes = self._permute_nodes(final_nodes, etype)
                    self._elements.append(final_nodes)
                    self._cell_types.append(vtk_type.value)
                    etype = None  # Wait for the next -1 record

    def _parse_results(self, file_stream: Any, step_time: float) -> None:
        """Parse 100CL result block."""
        if step_time not in self._results_by_step:
            self._results_by_step[step_time] = {}

        step_bucket = self._results_by_step[step_time]
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
                # Pass the first data line and iterator down to collect data
                self._parse_result_data(s, file_stream, name, step_bucket)
                return
            elif s.startswith(end_block):
                return

    def _parse_result_data(  # noqa: PLR0917
        self,
        first_line: str,
        file_stream: Any,
        name: str,
        step_bucket: dict[str, dict[int, list[float]]],
    ) -> None:
        """Parse data records until sentinel is hit."""
        data: dict[int, list[float]] = {}

        end_block = str(CGXRecord.END_OF_BLOCK.value)
        nodal_vals = str(CGXRecord.NODAL_VALUES.value)

        # Process the very first line passed from _parse_results
        try:
            parts = self._fix_scientific(first_line).split()
            data[int(parts[1])] = [float(x) for x in parts[2:]]
        except (ValueError, IndexError):
            pass

        # Continue with the rest of the file iterator
        for line in file_stream:
            s = line.strip()
            if s.startswith(end_block):
                break
            if s.startswith(nodal_vals):
                try:
                    parts = self._fix_scientific(s).split()
                    data[int(parts[1])] = [float(x) for x in parts[2:]]
                except (ValueError, IndexError):
                    pass

        if data:
            step_bucket[name] = data

    # ------------------------------------------------------------------
    # Derived field computation
    # ------------------------------------------------------------------

    def _compute_derived_stress(
        self, grid: UnstructuredGrid, base_name: str, tensor: np.ndarray
    ) -> None:
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        vmises = np.sqrt(
            0.5
            * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2 + 6.0 * (xy**2 + yz**2 + zx**2))
        )
        trace = xx + yy + zz
        grid.point_data[f'{base_name}_vMises'] = vmises
        grid.point_data[f'{base_name}_sgMises'] = np.where(
            trace != 0, np.sign(trace) * vmises, vmises
        )
        self._compute_principals(grid, base_name, tensor)

    def _compute_derived_strain(
        self, grid: UnstructuredGrid, base_name: str, tensor: np.ndarray
    ) -> None:
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        k = np.sqrt(2.0) / 3.0
        vmises_strain = k * np.sqrt(
            (xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2 + 6.0 * (xy**2 + yz**2 + zx**2)
        )
        volumetric = xx + yy + zz
        grid.point_data[f'{base_name}_vMises'] = vmises_strain
        grid.point_data[f'{base_name}_sgMises'] = np.where(
            volumetric != 0, np.sign(volumetric) * vmises_strain, vmises_strain
        )
        self._compute_principals(grid, base_name, tensor)

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

    # ------------------------------------------------------------------
    # Grid assembly
    # ------------------------------------------------------------------

    def _build_grid(self, step_data: dict[str, dict[int, list[float]]]) -> UnstructuredGrid:
        if not self._nodes:
            msg = 'No nodes found in FRD file -- cannot build grid.'
            raise ValueError(msg)

        sorted_ids = sorted(self._nodes)
        node_map = {nid: idx for idx, nid in enumerate(sorted_ids)}
        points = np.array([self._nodes[n] for n in sorted_ids], dtype=float)

        cells: list[int] = []
        types: list[int] = []
        for conn, ctype in zip(self._elements, self._cell_types, strict=True):
            try:
                vtk_ids = [node_map[n] for n in conn]
            except KeyError:
                continue
            cells.append(len(vtk_ids))
            cells.extend(vtk_ids)
            types.append(ctype)

        grid = UnstructuredGrid(
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
                    self._compute_derived_stress(grid, name, arr)
                elif 'STRAIN' in upper and n_components == 6:
                    self._compute_derived_strain(grid, name, arr)

        return grid
