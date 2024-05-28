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
    h1 = mol.AppendAtom(1, 0.0, 0.0, -0.5)
    h2 = mol.AppendAtom(1, 0.0, 0.0, 0.5)
    _ = mol.AppendBond(h1, h2, 1)
    assert mol.n_atoms == 2
    assert mol.n_bonds == 1
