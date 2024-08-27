from __future__ import annotations

import pytest

import pyvista as pv


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 1, 0),
    reason="Requires VTK>=9.1.0 for a vtkIOChemistry.vtkCMLMoleculeReader",
)
def test_molecule():
    mol = pv.Molecule()
    assert mol.n_atoms == 0
    assert mol.n_bonds == 0
    h1 = mol.append_atom([0.0, 0.0, -0.5])
    h2 = mol.append_atom([0.0, 0.0, 0.5])
    _ = mol.append_bond(h1, h2)
    assert mol.n_atoms == 2
    assert mol.n_bonds == 1
