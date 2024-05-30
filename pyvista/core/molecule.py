"""Contains the pyvista.Molecule class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _vtk_core as _vtk

if TYPE_CHECKING:
    from ._typing_core import NumpyArray


class Molecule(_vtk.vtkMolecule):
    """Class describing a molecule."""

    def __init__(self):
        super().__init__()

    @property
    def n_atoms(self) -> int:
        """Return the number of atoms.

        Returns
        -------
        int
            Number of atoms in the molecule.
        """
        return self.GetNumberOfAtoms()

    @property
    def n_bonds(self):
        """Return the number of bonds.

        Returns
        -------
        int
            Number of bonds in the molecule.
        """
        return self.GetNumberOfBonds()

    def append_atom(self, position: NumpyArray[float]) -> int:
        """Append an atom to the molecule.

        Parameters
        ----------
        position : NumpyArray[float]
            Position of the atom.

        Returns
        -------
        int
            Index of the atom.
        """
        return self.AppendAtom(self.n_atoms + 1, position[0], position[1], position[2])

    def append_bond(self, atom1: int, atom2: int) -> int:
        """Append a bond to the molecule.

        Parameters
        ----------
        atom1 : int
            Index of the first atom.

        atom2 : int
            Index of the second atom.

        Returns
        -------
        int
            Index of the bond.
        """
        return self.AppendBond(atom1, atom2, 1)
