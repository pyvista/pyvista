"""Filters module with the class to manage filters/algorithms for rectilinear grid datasets."""

import collections
from typing import Sequence, Union

import numpy as np

from pyvista.core import _vtk_core as _vtk
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.utilities.misc import abstract_class


@abstract_class
class RectilinearGridFilters:
    """An internal class to manage filters/algorithms for rectilinear grid datasets."""

    def to_tetrahedra(
        self,
        tetra_per_cell: int = 5,
        mixed: Union[Sequence[int], bool] = False,
        pass_cell_ids: bool = True,
        pass_data: bool = True,
        progress_bar: bool = False,
    ):
        """Create a tetrahedral mesh structured grid.

        Parameters
        ----------
        tetra_per_cell : int, default: 5
            The number of tetrahedrons to divide each cell into. Can be
            either ``5``, ``6``, or ``12``. If ``mixed=True``, this value is
            overridden.

        mixed : str, bool, sequence, default: False
            When set, subdivides some cells into 5 and some cells into 12. Set
            to ``True`` to use the active cell scalars of the
            :class:`pyvista.RectilinearGrid` to be either 5 or 12 to
            determining the number of tetrahedra to generate per cell.

            When a sequence, uses these values to subdivide the cells. When a
            string uses a cell array rather than the active array to determine
            the number of tetrahedra to generate per cell.

        pass_cell_ids : bool, default: True
            Set to ``True`` to make the tetrahedra have scalar data indicating
            which cell they came from in the original
            :class:`pyvista.RectilinearGrid`. The name of this array is
            ``'vtkOriginalCellIds'`` within the ``cell_data``.

        pass_data : bool, default: True
            Set to ``True`` to make the tetrahedra mesh have the cell data from
            the original :class:`pyvista.RectilinearGrid`.  This uses
            ``pass_cell_ids=True`` internally. If ``True``, ``pass_cell_ids``
            will also be set to ``True``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the tetrahedral cells.

        Examples
        --------
        Divide a rectangular grid into tetrahedrons. Each cell contains by
        default 5 tetrahedrons.

        First, create and plot the grid.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> xrng = np.linspace(0, 1, 2)
        >>> yrng = np.linspace(0, 1, 2)
        >>> zrng = np.linspace(0, 2, 3)
        >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.plot()

        Now, generate the tetrahedra plot in the exploded view of the cell.

        >>> tet_grid = grid.to_tetrahedra()
        >>> tet_grid.explode(factor=0.5).plot(show_edges=True)

        Take the same grid but divide the first cell into 5 cells and the other
        cell into 12 tetrahedrons per cell.

        >>> tet_grid = grid.to_tetrahedra(mixed=[5, 12])
        >>> tet_grid.explode(factor=0.5).plot(show_edges=True)

        """
        alg = _vtk.vtkRectilinearGridToTetrahedra()
        alg.SetRememberVoxelId(pass_cell_ids or pass_data)
        if mixed is not False:
            if isinstance(mixed, str):
                self.cell_data.active_scalars_name = mixed
            elif isinstance(mixed, (np.ndarray, collections.abc.Sequence)):
                self.cell_data['_MIXED_CELLS_'] = mixed  # type: ignore
            elif not isinstance(mixed, bool):
                raise TypeError('`mixed` must be either a sequence of ints or bool')
            alg.SetTetraPerCellTo5And12()
        else:
            if tetra_per_cell not in [5, 6, 12]:
                raise ValueError(
                    f'`tetra_per_cell` should be either 5, 6, or 12, not {tetra_per_cell}'
                )

            # Edge case causing a seg-fault where grid is flat in one dimension
            # See: https://gitlab.kitware.com/vtk/vtk/-/issues/18650
            if 1 in self.dimensions and tetra_per_cell == 12:  # type: ignore
                raise RuntimeError(
                    'Cannot split cells into 12 tetrahedrals when at least '
                    f'one dimension is 1. Dimensions are {self.dimensions}.'  # type: ignore[attr-defined]
                )

            alg.SetTetraPerCell(tetra_per_cell)

        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Converting to tetrahedra')
        out = _get_output(alg)

        if pass_data:
            # algorithm stores original cell ids in active scalars
            # this does not preserve active scalars, but we need to
            # keep active scalars until they are renamed
            for name in self.cell_data:  # type: ignore
                if name != out.cell_data.active_scalars_name:
                    out[name] = self.cell_data[name][out.cell_data.active_scalars]  # type: ignore

            for name in self.point_data:  # type: ignore
                out[name] = self.point_data[name]  # type: ignore

        if alg.GetRememberVoxelId():
            # original cell_ids are not named and are the active scalars
            out.cell_data.set_array(
                out.cell_data.pop(out.cell_data.active_scalars_name), 'vtkOriginalCellIds'
            )

        if pass_data:
            # Now reset active scalars in cast the original mesh had data with active scalars
            association, name = self.active_scalars_info  # type: ignore
            out.set_active_scalars(name, preference=association)

        return out
