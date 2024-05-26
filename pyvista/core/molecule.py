"""Contains the pyvista.Molecule class."""

from . import _vtk_core as _vtk


class Molecule(_vtk.vtkMolecule):
    """Class describing a molecule."""

    def __init__(self):
        super().__init__()
