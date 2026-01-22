from __future__ import annotations

import pathlib
import re

import numpy as np

from pyvista.core.pointset import UnstructuredGrid


class FRDReader:
    """FRD Reader for CalculiX files.
    Supports 1D, 2D and 3D elements, including HEX20 with correct VTK node permutation.
    Calculates derived fields: vMises, Signed vMises, and Principal Stresses/Strains.
    """

    # Mapping CalculiX element indices to VTK cell types
    CCX_TO_VTK_TYPE = {
        1: 12,  # he8
        2: 13,  # pe6
        3: 10,  # te4
        4: 25,  # he20 (Quadratic Hexahedron)
        5: 13,  # pe15 (Mapped to wedge)
        6: 24,  # te10 (Quadratic Tetra)
        7: 5,  # tr3
        8: 22,  # tr6
        9: 9,  # qu4
        10: 23,  # qu8
        11: 3,  # be2
        12: 21,  # be3
    }

    # Number of nodes expected for each CalculiX element type
    NODES_PER_ELEM = {1: 8, 2: 6, 3: 4, 4: 20, 5: 15, 6: 10, 7: 3, 8: 6, 9: 4, 10: 8, 11: 2, 12: 3}

    def __init__(self, filename):
        self.filename = filename
        self.nodes = {}
        self.elements = []
        self.cell_types = []
        self.raw_results = {}
        self.result_counter = {}

    def read(self):
        """Read the file and return a grid with attached results."""
        with pathlib.Path(self.filename).open(errors='replace') as f:
            lines = f.readlines()
        self._parse_lines(lines)
        return self._build_grid()

    def _fix_line(self, line):
        """Fixes CalculiX scientific notation (e.g., adds space before negative exponents if missing)."""
        return re.sub(r'(?<![EeDd])-', ' -', line)

    def _permute_nodes(self, node_ids, etype):
        """Reorders node IDs to match VTK standards.
        Crucial for HEX20 and surface elements (winding order).
        """
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

        # TR3: Ensure Counter-Clockwise
        if etype == 7 and len(node_ids) == 3:
            p = [self.nodes[n] for n in node_ids]
            A = (
                p[0][0] * (p[1][1] - p[2][1])
                + p[1][0] * (p[2][1] - p[0][1])
                + p[2][0] * (p[0][1] - p[1][1])
            )
            if A < 0:
                return [node_ids[0], node_ids[2], node_ids[1]]
            return node_ids

        # QU4: Ensure Counter-Clockwise
        if etype == 9 and len(node_ids) == 4:
            p = [self.nodes[n] for n in node_ids]
            A = 0
            for i in range(4):
                x1, y1 = p[i][0], p[i][1]
                x2, y2 = p[(i + 1) % 4][0], p[(i + 1) % 4][1]
                A += x1 * y2 - x2 * y1
            if A < 0:
                return [node_ids[0], node_ids[3], node_ids[2], node_ids[1]]
            return node_ids

        # QU8: Reorder if winding is reversed
        if etype == 10 and len(node_ids) == 8:
            corners = node_ids[0:4]
            mids = node_ids[4:8]
            p = [self.nodes[n] for n in corners]
            A = 0
            for i in range(4):
                x1, y1 = p[i][0], p[i][1]
                x2, y2 = p[(i + 1) % 4][0], p[(i + 1) % 4][1]
                A += x1 * y2 - x2 * y1
            if A < 0:
                corners = [corners[0], corners[3], corners[2], corners[1]]
                mids = [mids[0], mids[3], mids[2], mids[1]]
            return corners + mids

        return node_ids

    def _parse_lines(self, lines):
        i = 0
        total = len(lines)
        while i < total:
            line = lines[i].strip()
            if line.startswith('2C'):
                i = self._parse_nodes(lines, i)
            elif line.startswith('3C'):
                i = self._parse_elements(lines, i)
            elif line.startswith('100'):
                i = self._parse_results(lines, i)
            else:
                i += 1

    def _parse_nodes(self, lines, i):
        i += 1
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith('-3'):
                return i
            try:
                p = self._fix_line(s).split()
                nid = int(p[1])
                self.nodes[nid] = [float(p[2]), float(p[3]), float(p[4])]
            except:
                pass
            i += 1
        return i

    def _parse_elements(self, lines, i):
        i += 1
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith('-3'):
                return i
            if s.startswith('-1'):
                parts = s.split()
                try:
                    etype = int(parts[2])
                except:
                    i += 1
                    continue

                if etype not in self.CCX_TO_VTK_TYPE:
                    i += 1
                    continue

                needed = self.NODES_PER_ELEM[etype]
                vtk_type = self.CCX_TO_VTK_TYPE[etype]
                node_ids = []
                j = i + 1

                while j < len(lines):
                    t = lines[j].strip()
                    if t.startswith('-3'):
                        break
                    if t.startswith('-2'):
                        p = t.split()
                        node_ids.extend(int(x) for x in p[1:])
                    j += 1
                    if len(node_ids) >= needed:
                        break

                node_ids = node_ids[:needed]
                if len(node_ids) == needed:
                    node_ids = self._permute_nodes(node_ids, etype)
                    self.elements.append(node_ids)
                    self.cell_types.append(vtk_type)
                i = j
                continue
            i += 1
        return i

    def _parse_results(self, lines, i):
        name = 'Unknown'
        t = i + 1
        found = False
        while t < len(lines):
            s = lines[t].strip()
            if s.startswith('-1'):
                break
            if not s.startswith('100'):
                parts = s.split()
                for c in parts:
                    try:
                        float(c)
                        isnum = True
                    except:
                        isnum = False
                    if not isnum and len(c) > 1:
                        name = c
                        found = True
                        break
            if found:
                break
            t += 1

        if name not in self.result_counter:
            self.result_counter[name] = 0
            final = name
        else:
            self.result_counter[name] += 1
            final = f'{name}_{self.result_counter[name]}'

        data = {}
        i = t
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith('-3'):
                break
            if s.startswith('-1'):
                try:
                    p = self._fix_line(s).split()
                    nid = int(p[1])
                    vals = [float(x) for x in p[2:]]
                    data[nid] = vals
                except:
                    pass
            i += 1
        self.raw_results[final] = data
        return i

    # -------------------------------------------------------------------------
    # Calculation Methods (vMises, Principal, etc.)
    # -------------------------------------------------------------------------
    def _compute_derived_stress(self, grid, base_name, tensor):
        """Compute vMises, sgMises, PS1-3 for Stress."""
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        # Von Mises Stress
        vmises = np.sqrt(
            0.5
            * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2 + 6.0 * (xy**2 + yz**2 + zx**2))
        )
        grid.point_data[f'{base_name}_vMises'] = vmises

        # Signed Von Mises (Sign of the first invariant / trace)
        trace = xx + yy + zz
        sg_mises = np.sign(trace) * vmises
        # Handle zero trace case to preserve magnitude
        sg_mises[trace == 0] = vmises[trace == 0]
        grid.point_data[f'{base_name}_sgMises'] = sg_mises

        self._compute_principals(grid, base_name, tensor)

    def _compute_derived_strain(self, grid, base_name, tensor):
        """Compute vMises Strain (equivalent) & Signed Strain."""
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        # Coefficient for effective strain (CalculiX convention)
        k = np.sqrt(2.0) / 3.0

        vmises_strain = k * np.sqrt(
            (xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2 + 6.0 * (xy**2 + yz**2 + zx**2)
        )
        grid.point_data[f'{base_name}_vMises'] = vmises_strain

        # Signed based on volumetric strain (trace)
        volumetric = xx + yy + zz
        sg_vmises_strain = np.sign(volumetric) * vmises_strain
        sg_vmises_strain[volumetric == 0] = vmises_strain[volumetric == 0]

        grid.point_data[f'{base_name}_sgMises'] = sg_vmises_strain

        self._compute_principals(grid, base_name, tensor)

    def _compute_principals(self, grid, base_name, tensor):
        """Helper to compute Principal Values (PS1, PS2, PS3)."""
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        n_points = tensor.shape[0]
        # Construct symmetric tensor matrices for all points
        mat = np.zeros((n_points, 3, 3))
        mat[:, 0, 0] = xx
        mat[:, 1, 1] = yy
        mat[:, 2, 2] = zz
        mat[:, 0, 1] = xy
        mat[:, 1, 0] = xy
        mat[:, 1, 2] = yz
        mat[:, 2, 1] = yz
        mat[:, 0, 2] = zx
        mat[:, 2, 0] = zx

        # Calculate eigenvalues (eigh is for Hermitian/Symmetric matrices)
        eigvals = np.linalg.eigvalsh(mat)

        # Sort is usually ascending in numpy: PS3(min), PS2, PS1(max)
        grid.point_data[f'{base_name}_PS3'] = eigvals[:, 0]  # Min
        grid.point_data[f'{base_name}_PS2'] = eigvals[:, 1]  # Mid
        grid.point_data[f'{base_name}_PS1'] = eigvals[:, 2]  # Max

    # -------------------------------------------------------------------------
    # Grid Construction
    # -------------------------------------------------------------------------
    def _build_grid(self):
        if not self.nodes:
            msg = 'No nodes found.'
            raise ValueError(msg)

        sorted_ids = sorted(self.nodes.keys())
        node_map = {nid: idx for idx, nid in enumerate(sorted_ids)}
        points = np.array([self.nodes[n] for n in sorted_ids])

        cells = []
        types = []

        for conn, t in zip(self.elements, self.cell_types):
            try:
                idx = [node_map[n] for n in conn]
                cells.append(len(idx))
                cells.extend(idx)
                types.append(t)
            except:
                continue

        grid = UnstructuredGrid(
            np.array(cells, dtype=np.int64), np.array(types, dtype=np.uint8), points
        )

        # IMPORTANT: Persist original Node IDs for labeling/verification
        original_ids_str = np.array([str(nid) for nid in sorted_ids])
        grid.point_data['Original_Node_ID'] = original_ids_str

        # Attach results and calculate derived fields
        for name, d in self.raw_results.items():
            if not d:
                continue
            first = next(iter(d.values()))
            nc = len(first)

            if nc == 1:
                arr = np.zeros(len(points))
                for nid, v in d.items():
                    if nid in node_map:
                        arr[node_map[nid]] = v[0]
                grid.point_data[name] = arr
            else:
                arr = np.zeros((len(points), nc))
                for nid, v in d.items():
                    if nid in node_map:
                        arr[node_map[nid]] = v
                grid.point_data[name] = arr

                # Automatically calculate Stress/Strain invariants
                if 'STRESS' in name.upper() and nc == 6:
                    self._compute_derived_stress(grid, name, arr)

                if 'STRAIN' in name.upper() and nc == 6:
                    self._compute_derived_strain(grid, name, arr)

        return grid
