from __future__ import annotations

import pathlib
import re
from typing import Any
from typing import ClassVar

import numpy as np

from pyvista.core.pointset import UnstructuredGrid
from pyvista.core.utilities.reader import BaseReader
from pyvista.core.utilities.reader import BaseVTKReader
from pyvista.core.utilities.reader import TimeReader

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
    CCX_TO_VTK_TYPE: ClassVar[dict[int, int]] = {
        1: 12,  # he8  -> VTK_HEXAHEDRON
        2: 13,  # pe6  -> VTK_WEDGE
        3: 10,  # te4  -> VTK_TETRA
        4: 25,  # he20 -> VTK_QUADRATIC_HEXAHEDRON
        5: 13,  # pe15 -> VTK_WEDGE (degraded; pe15 not native to base VTK)
        6: 24,  # te10 -> VTK_QUADRATIC_TETRA
        7: 5,  # tr3  -> VTK_TRIANGLE
        8: 22,  # tr6  -> VTK_QUADRATIC_TRIANGLE
        9: 9,  # qu4  -> VTK_QUAD
        10: 23,  # qu8  -> VTK_QUADRATIC_QUAD
        11: 3,  # be2  -> VTK_LINE
        12: 21,  # be3  -> VTK_QUADRATIC_EDGE
    }

    NODES_PER_ELEM: ClassVar[dict[int, int]] = {
        1: 8,
        2: 6,
        3: 4,
        4: 20,
        5: 15,
        6: 10,
        7: 3,
        8: 6,
        9: 4,
        10: 8,
        11: 2,
        12: 3,
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

        with pathlib.Path(self._filename).open(errors='replace') as fh:
            self._parse_lines(fh)

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

        assert self._output is not None
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

    def _permute_nodes(self, node_ids: list[int], etype: int) -> list[int]:
        """Reorder node IDs from CalculiX to VTK conventions."""
        if etype == 4 and len(node_ids) == 20:
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

        if etype == 7 and len(node_ids) == 3:
            p = [self._nodes[n] for n in node_ids]
            area = (
                p[0][0] * (p[1][1] - p[2][1])
                + p[1][0] * (p[2][1] - p[0][1])
                + p[2][0] * (p[0][1] - p[1][1])
            )
            if area < 0:
                return [node_ids[0], node_ids[2], node_ids[1]]
            return node_ids

        if etype in (9, 10):
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

    def _parse_lines(self, fh: Any) -> None:
        """Main loop dispatching to block-specific parsers."""
        for line in fh:
            s = line.strip()
            if s.startswith('2C'):
                self._parse_nodes(fh)
            elif s.startswith('3C'):
                self._parse_elements(fh)
            elif s.startswith('100'):
                _step_id, step_time = self._parse_100cl_header(s)
                self._parse_results(fh, step_time)

    def _parse_nodes(self, fh: Any) -> None:
        """Parse 2C node block."""
        for line in fh:
            s = line.strip()
            if s.startswith('-3'):
                return
            try:
                parts = self._fix_scientific(s).split()
                nid = int(parts[1])
                self._nodes[nid] = [float(parts[2]), float(parts[3]), float(parts[4])]
            except (ValueError, IndexError):
                pass

    def _parse_elements(self, fh: Any) -> None:
        """Parse 3C element block."""
        needed = 0
        node_ids: list[int] = []
        etype: int | None = None
        vtk_type: int | None = None

        for line in fh:
            s = line.strip()
            if s.startswith('-3'):
                return

            if s.startswith('-1'):
                parts = s.split()
                try:
                    etype = int(parts[2])
                except (ValueError, IndexError):
                    etype = None
                    continue

                if etype not in self.CCX_TO_VTK_TYPE:
                    etype = None
                    continue

                needed = self.NODES_PER_ELEM[etype]
                vtk_type = self.CCX_TO_VTK_TYPE[etype]
                node_ids = []

            elif s.startswith('-2') and etype is not None and vtk_type is not None:
                node_ids.extend(int(x) for x in s.split()[1:])
                if len(node_ids) >= needed:
                    final_nodes = node_ids[:needed]
                    final_nodes = self._permute_nodes(final_nodes, etype)
                    self._elements.append(final_nodes)
                    self._cell_types.append(vtk_type)
                    etype = None  # Wait for the next -1 record

    def _parse_results(self, fh: Any, step_time: float) -> None:
        """Parse 100CL result block."""
        if step_time not in self._results_by_step:
            self._results_by_step[step_time] = {}

        step_bucket = self._results_by_step[step_time]
        name = 'Unknown'

        for line in fh:
            s = line.strip()
            if s.startswith('-4'):
                parts = s.split()
                if len(parts) >= 2:
                    name = parts[1]
            elif s.startswith('-5'):
                continue
            elif s.startswith('-1'):
                # Pass the first data line and iterator down to collect data
                self._parse_result_data(s, fh, name, step_bucket)
                return
            elif s.startswith('-3'):
                return

    def _parse_result_data(self, first_line: str, fh: Any, name: str, step_bucket: dict[str, dict[int, list[float]]]) -> None:
        """Parse -1 records until -3 sentinel is hit."""
        data: dict[int, list[float]] = {}

        # Process the very first line passed from _parse_results
        try:
            parts = self._fix_scientific(first_line).split()
            data[int(parts[1])] = [float(x) for x in parts[2:]]
        except (ValueError, IndexError):
            pass

        # Continue with the rest of the file iterator
        for line in fh:
            s = line.strip()
            if s.startswith('-3'):
                break
            if s.startswith('-1'):
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
        for conn, ctype in zip(self._elements, self._cell_types, strict=False):
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

        grid.point_data['Original_Node_ID'] = np.array([str(nid) for nid in sorted_ids])

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


# ---------------------------------------------------------------------------
# Public PyVista reader
# ---------------------------------------------------------------------------


class FRDReader(BaseReader, TimeReader):
    """Reader for CalculiX FRD ASCII result files (``.frd``)."""

    _class_reader = _FRDVTKReader

    def __init__(self, path: str | pathlib.Path) -> None:
        super().__init__(path)
        self._reader: _FRDVTKReader = self._class_reader()
        self._reader.SetFileName(self.path)
        self._reader.Update()

    @property
    def reader(self) -> _FRDVTKReader:
        return self._reader

    @property
    def number_time_points(self) -> int:
        return len(self.reader._time_steps)

    def time_point_value(self, time_point: int) -> float:
        return self.reader._time_steps[time_point]

    @property
    def time_values(self) -> list[float]:
        return list(self.reader._time_steps)

    def set_active_time_point(self, time_point: int) -> None:
        n = self.number_time_points
        if not 0 <= time_point < n:
            msg = f'time_point {time_point} is out of range (file has {n} time point(s)).'
            raise IndexError(msg)
        self.reader._active_time_point = time_point

    def set_active_time_value(self, time_value: float) -> None:
        steps = self.reader._time_steps
        if not steps:
            msg = 'No time steps found in the FRD file.'
            raise RuntimeError(msg)
        idx = int(np.argmin(np.abs(np.array(steps) - time_value)))
        self.reader._active_time_point = idx

    @property
    def active_time_value(self) -> float:
        steps = self.reader._time_steps
        if not steps:
            return 0.0
        return steps[self.reader._active_time_point]

    @active_time_value.setter
    def active_time_value(self, value: float) -> None:
        self.set_active_time_value(value)

    def read(self) -> UnstructuredGrid:
        return self.reader.GetOutput()
