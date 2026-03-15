"""Filters module with the class to manage filters/algorithms for rectilinear grid datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.utilities.misc import abstract_class


@abstract_class
class RectilinearGridFilters:
    """An internal class to manage filters/algorithms for rectilinear grid datasets."""

    @_deprecate_positional_args(allowed=['tetra_per_cell'])
    def to_tetrahedra(  # noqa: PLR0917
        self,
        tetra_per_cell: int = 5,
        mixed: str | Sequence[int] | bool = False,  # noqa: FBT001, FBT002
        pass_cell_ids: bool = True,  # noqa: FBT001, FBT002
        pass_data: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
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
                self.cell_data.active_scalars_name = mixed  # type: ignore[attr-defined]
            elif isinstance(mixed, (np.ndarray, Sequence)):
                self.cell_data['_MIXED_CELLS_'] = mixed  # type: ignore[attr-defined]
            elif not isinstance(mixed, bool):
                msg = '`mixed` must be either a sequence of ints or bool'  # type: ignore[unreachable]
                raise TypeError(msg)
            alg.SetTetraPerCellTo5And12()
        else:
            if tetra_per_cell not in [5, 6, 12]:
                msg = f'`tetra_per_cell` should be either 5, 6, or 12, not {tetra_per_cell}'
                raise ValueError(msg)

            # Edge case causing a seg-fault where grid is flat in one dimension
            # See: https://gitlab.kitware.com/vtk/vtk/-/issues/18650
            if 1 in self.dimensions and tetra_per_cell == 12:  # type: ignore[attr-defined]
                msg = (
                    'Cannot split cells into 12 tetrahedrals when at least '
                    f'one dimension is 1. Dimensions are {self.dimensions}.'  # type: ignore[attr-defined]
                )
                raise RuntimeError(msg)

            alg.SetTetraPerCell(tetra_per_cell)

        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Converting to tetrahedra')
        out = _get_output(alg)

        if pass_data:
            # algorithm stores original cell ids in active scalars
            # this does not preserve active scalars, but we need to
            # keep active scalars until they are renamed
            for name in self.cell_data:  # type: ignore[attr-defined]
                if name != out.cell_data.active_scalars_name:
                    out[name] = self.cell_data[name][out.cell_data.active_scalars]  # type: ignore[attr-defined]

            for name in self.point_data:  # type: ignore[attr-defined]
                out[name] = self.point_data[name]  # type: ignore[attr-defined]

        if alg.GetRememberVoxelId():
            # original cell_ids are not named and are the active scalars
            out.cell_data.set_array(
                out.cell_data.pop(out.cell_data.active_scalars_name),
                'vtkOriginalCellIds',
            )

        if pass_data:
            # Now reset active scalars in cast the original mesh had data with active scalars
            association, name = self.active_scalars_info  # type: ignore[attr-defined]
            out.set_active_scalars(name, preference=association)

        return out
