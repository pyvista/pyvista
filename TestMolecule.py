"""This file tests vtkMolecule, and verifies that atoms/bonds are added."""

from vtkmodules.vtkCommonDataModel import vtkMolecule

mol = vtkMolecule()
assert mol.GetNumberOfAtoms() == 0
assert mol.GetNumberOfBonds() == 0
h1 = mol.AppendAtom(1, 0.0, 0.0, -0.5)
h2 = mol.AppendAtom(1, 0.0, 0.0, 0.5)
_ = mol.AppendBond(h1, h2, 1)
assert mol.GetNumberOfAtoms() == 2
assert mol.GetNumberOfBonds() == 1
