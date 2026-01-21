"""FRD (CalculiX) Reader module."""
import numpy as np

from pyvista.core.pointset import UnstructuredGrid
from pyvista.core.utilities.fileio import _get_vtk_id_type


class FRDReader:
    """FRD Reader for CalculiX files handling geometry, results, and derived fields."""

    CCX_TO_VTK_TYPE = {
        1: 12,  # C3D8 -> VTK_HEXAHEDRON
        4: 10,  # C3D4 -> VTK_TETRA
        6: 13,  # C3D6 -> VTK_WEDGE
        10: 24, # C3D10 -> VTK_QUADRATIC_TETRA
        20: 25, # C3D20 -> VTK_QUADRATIC_HEXAHEDRON
    }

    def __init__(self, filename):
        """Initialize the reader."""
        self.filename = filename
        self.nodes = {}
        self.elements = []
        self.cell_types = []
        self.raw_results = {} 
        self.result_counter = {}

    def read(self):
        """Read the file and return a grid with attached results."""
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self._parse_lines(lines)
        return self._build_grid()

    def _parse_lines(self, lines):
        """Parse lines including geometry and result blocks."""
        i = 0
        total_lines = len(lines)

        while i < total_lines:
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
            line = lines[i]
            if line.startswith('-3'):
                return i
            try:
                parts = line.split()
                nid = int(parts[1])
                coords = [float(parts[2]), float(parts[3]), float(parts[4])]
                self.nodes[nid] = coords
            except (ValueError, IndexError):
                pass
            i += 1
        return i

    def _parse_elements(self, lines, i):
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.startswith('-3'):
                return i
            try:
                parts = line.split()
                etype = int(parts[2])
                vtk_type = self.CCX_TO_VTK_TYPE.get(etype)
                if vtk_type:
                    node_ids = [int(x) for x in parts[3:]]
                    self.elements.append(node_ids)
                    self.cell_types.append(vtk_type)
            except (ValueError, IndexError):
                pass
            i += 1
        return i

    def _parse_results(self, lines, i):
        result_name = "Unknown"
        temp_idx = i + 1
        found_name = False
        
        while temp_idx < len(lines):
            line = lines[temp_idx].strip()
            if line.startswith('-1'):
                break
            if len(line) > 2 and not line.startswith('100'):
                 parts = line.split()
                 if parts:
                    candidate = parts[0]
                    if not candidate.replace('.', '', 1).isdigit():
                        result_name = candidate
                        found_name = True
            temp_idx += 1
        
        if not found_name:
             result_name = f"Result_Block_{i}"

        # Handle Time Steps
        if result_name not in self.result_counter:
            self.result_counter[result_name] = 0
            final_name = result_name 
        else:
            self.result_counter[result_name] += 1
            count = self.result_counter[result_name]
            final_name = f"{result_name}_{count}"

        current_data = {}
        i = temp_idx 
        
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('-3'):
                break
            if line.startswith('-1'):
                try:
                    parts = line.split()
                    nid = int(parts[1])
                    values = [float(val) for val in parts[2:]]
                    current_data[nid] = values
                except (ValueError, IndexError):
                    pass
            i += 1
            
        self.raw_results[final_name] = current_data
        return i

    def _build_grid(self):
        """Construct the Grid and attach results."""
        if not self.nodes:
            raise ValueError("No nodes found.")

        sorted_node_ids = sorted(self.nodes.keys())
        node_map = {nid: idx for idx, nid in enumerate(sorted_node_ids)}
        
        points = np.array([self.nodes[nid] for nid in sorted_node_ids])

        cells_vtk = []
        cell_types_vtk = []
        for node_ids, vtk_type in zip(self.elements, self.cell_types):
            try:
                current_cell_indices = [node_map[nid] for nid in node_ids]
                cells_vtk.append(len(current_cell_indices))
                cells_vtk.extend(current_cell_indices)
                cell_types_vtk.append(vtk_type)
            except KeyError:
                continue

        cells_vtk = np.array(cells_vtk, dtype=_get_vtk_id_type())
        cell_types_vtk = np.array(cell_types_vtk, dtype=np.uint8)
        
        grid = UnstructuredGrid(cells_vtk, cell_types_vtk, points)

        # Attach Results and Compute Derived Fields
        n_points = len(points)
        
        for res_name, data_dict in self.raw_results.items():
            if not data_dict:
                continue
            
            first_val = next(iter(data_dict.values()))
            n_comp = len(first_val)
            
            if n_comp == 1:
                result_array = np.zeros(n_points)
            else:
                result_array = np.zeros((n_points, n_comp))

            for nid, val in data_dict.items():
                if nid in node_map:
                    idx = node_map[nid]
                    result_array[idx] = val
            
            grid.point_data[res_name] = result_array
            
            # --- AUTO-CALCULATE DERIVED FIELDS ---
            # 1. Stress Processing
            if "STRESS" in res_name.upper() and n_comp == 6:
                self._compute_derived_stress(grid, res_name, result_array)
            
            # 2. Strain Processing (e.g. TOSTRAIN, ELSTRAIN)
            # Looks for 'STRAIN' in name but avoids 'PSTRAIN' (Plastic Strain) if it is scalar
            if "STRAIN" in res_name.upper() and "PSTRAIN" not in res_name.upper() and n_comp == 6:
                self._compute_derived_strain(grid, res_name, result_array)

        return grid

    def _compute_derived_stress(self, grid, base_name, tensor):
        """Compute vMises, sgMises, PS1-3 for Stress."""
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        # Von Mises Stress
        vmises = np.sqrt(0.5 * (
            (xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2 +
            6.0 * (xy**2 + yz**2 + zx**2)
        ))
        grid.point_data[f"{base_name}_vMises"] = vmises

        # Signed Von Mises (Sign based on Trace/Hydrostatic)
        trace = xx + yy + zz
        sg_mises = np.sign(trace) * vmises
        sg_mises[trace == 0] = vmises[trace == 0]
        grid.point_data[f"{base_name}_sgMises"] = sg_mises

        # Principal Stresses
        self._compute_principals(grid, base_name, tensor)

    def _compute_derived_strain(self, grid, base_name, tensor):
        """
        Compute Equivalent Strain and Signed Equivalent Strain.
        Based on CGX 2.23 Manual formulas.
        """
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        # Von Mises Equivalent Strain
        # Formula: (sqrt(2)/3) * sqrt( (ex-ey)^2 + ... + 6*(exy^2+...) )
        # Note: 6*shear because CalculiX stores tensor shear (epsilon), not engineering (gamma)
        k = np.sqrt(2.0) / 3.0
        
        vmises_strain = k * np.sqrt(
            (xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2 +
            6.0 * (xy**2 + yz**2 + zx**2)
        )
        grid.point_data[f"{base_name}_vMises"] = vmises_strain

        # Signed Von Mises Equivalent Strain
        # Sign is based on the volumetric strain (Trace of strain tensor)
        volumetric = xx + yy + zz
        sg_vmises_strain = np.sign(volumetric) * vmises_strain
        sg_vmises_strain[volumetric == 0] = vmises_strain[volumetric == 0]
        
        grid.point_data[f"{base_name}_sgMises"] = sg_vmises_strain
        
        # Also compute Principal Strains
        self._compute_principals(grid, base_name, tensor)

    def _compute_principals(self, grid, base_name, tensor):
        """Helper to compute Principal Values (PS1, PS2, PS3) for any symmetric tensor."""
        xx, yy, zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        xy, yz, zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        n_points = tensor.shape[0]
        mat = np.zeros((n_points, 3, 3))
        
        mat[:, 0, 0] = xx; mat[:, 1, 1] = yy; mat[:, 2, 2] = zz
        mat[:, 0, 1] = xy; mat[:, 1, 0] = xy
        mat[:, 1, 2] = yz; mat[:, 2, 1] = yz
        mat[:, 0, 2] = zx; mat[:, 2, 0] = zx

        # Calculate eigenvalues (ascending: min, mid, max)
        eigvals = np.linalg.eigvalsh(mat)
        
        # PS1 (Max), PS2 (Mid), PS3 (Min)
        grid.point_data[f"{base_name}_PS1"] = eigvals[:, 2]
        grid.point_data[f"{base_name}_PS2"] = eigvals[:, 1]
        grid.point_data[f"{base_name}_PS3"] = eigvals[:, 0]
