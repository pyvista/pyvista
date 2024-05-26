import pyvista as pv


def test_molecule():
    mol = pv.Molecule()
    assert mol.GetNumberOfAtoms() == 0
    assert mol.GetNumberOfBonds() == 0
    h1 = mol.AppendAtom(1, 0.0, 0.0, -0.5)
    h2 = mol.AppendAtom(1, 0.0, 0.0, 0.5)
    _ = mol.AppendBond(h1, h2, 1)
    assert mol.GetNumberOfAtoms() == 2
    assert mol.GetNumberOfBonds() == 1
