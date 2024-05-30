"""Contains the pyvista.Molecule class."""

from __future__ import annotations

from . import _vtk_core as _vtk


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
