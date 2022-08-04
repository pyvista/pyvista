"""Filters module with class to manage filters/algorithms for rectilinear grid datasets."""

from pyvista import _vtk, abstract_class
from pyvista.core.filters import _get_output, _update_alg


@abstract_class
class RectilinearGridFilters:
    """An internal class to manage filters/algorithms for rectilinear grid datasets."""

    def to_tetrahedra(self, tetra_per_cell=5, mixed=False, pass_cell_ids=False, progress_bar=False):
        """Create a Tetrahedral mesh from a RectilinearGrid.

        Parameters
        ----------
        tetra_per_cell : int, default: 5
            The number of tetrahedrals to divide each cell in the into. Can be
            either ``5``, ``6``, or ``12``. If ``mixed=True``, this value is
            overridden.

        mixed : bool, default: False
            Set this to ``True`` to to subdivide some cells into 5 and some
            cells into 12. Set the active cell scalars of the
            :class:`pyvista.RectilinearGrid` to be either 5 or 12 to
            determinine the number of tetrahedra to generate per cell.

        pass_cell_ids : bool, default: False
            Set to ``True`` to make the tetrahedra have scalar data indicating which cell they
            came from in the original :class:`pyvista.RectilinearGrid`.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the tetrahedral cells.

        Examples
        --------
        Divide a rectangular grid into tetrahedrals. Each cell contains by
        default 5 tetrahedrals.

        First, create and plot the grid.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> xrng = np.linspace(0, 1, 2)
        >>> yrng = np.linspace(0, 1, 2)
        >>> zrng = np.linspace(0, 2, 3)
        >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.plot()

        Now, generate the tetrahedra plot the exploded view of the cell.

        >>> tet_grid = grid.to_tetrahedra()
        >>> tet_grid.explode(factor=0.5).plot(show_edges=True)

        Take the same grid but divide the first cell into 5 cells and the other
        cell into 12 tetra per cell.

        >>> grid['data'] = [5, 12]
        >>> tet_grid = grid.to_tetrahedra(mixed=True)
        >>> tet_grid.explode(factor=0.5).plot(show_edges=True)

        """
        alg = _vtk.vtkRectilinearGridToTetrahedra()
        alg.SetRememberVoxelId(pass_cell_ids)
        if mixed:
            alg.SetTetraPerCellTo5And12()
        else:
            if tetra_per_cell not in [5, 6, 12]:
                raise ValueError(
                    f'`tetra_per_cell` should be either 5, 6, or 12, not {tetra_per_cell}'
                )
            alg.SetTetraPerCell(tetra_per_cell)

        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Converting to tetrahedra')
        return _get_output(alg)
