"""Generate the pyvista logo.

Logos generated with:
plot_logo(screenshot='pyvista_logo.png', window_size=(1920, 1080))
plot_logo(screenshot='pyvista_logo_sm.png', window_size=(960, 400), off_screen=True)

# different camera angle for square plot
cpos = [(-0.3654543687422538, 1.1098808905156292, 9.073223697728247),
        (2.553950615449191, 0.34145688392081264, 0.06127122762851659),
        (0.019308531920309947, 0.996708840795678, -0.07873161547192065)]

plot_logo(screenshot='pyvista_logo_sm_sq.png', window_size=(960, 960), cpos=cpos,
          off_screen=True)

"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import pyvista as pv
from pyvista import examples
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.features import _voxelize_legacy

THIS_PATH = str(Path(os.path.realpath(__file__)).parent)

LOGO_TITLE = 'PyVista'


def atomize(grid, shift_fac=0.1, scale=0.9):
    """Break apart and shrink and/or scale the individual cells of a mesh.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The input mesh to atomize.
    shift_fac : float, default: 0.1
        Factor by which to shift the individual cells apart.
    scale : float, default: 0.9
        Factor by which to scale the individual cells.

    Returns
    -------
    pyvista.UnstructuredGrid
        The atomized mesh with individually shifted and scaled cells.

    """
    cent = grid.center
    cells = []
    for i in range(grid.n_cells):
        cell = grid.extract_cells(i)
        ccent = np.array(cell.center)
        cell.points[:] = (cell.points - ccent) * scale + ccent
        cell.points += (ccent - np.array(cent)) * shift_fac
        cells.append(cell)

    return cells[0].merge(cells[1:])


def text_3d(string, depth=0.5):
    """Create 3D text from a given string.

    Parameters
    ----------
    string : str
        The string of text to convert into 3D text.

    depth : float, default: 0.5
        The depth of the extrusion used to create the 3D text.

    Returns
    -------
    pyvista.DataSet
        The 3D text in the form of a PyVista DataSet.

    See Also
    --------
    :ref:`create_pixel_art_example`

    """
    from vtkmodules.vtkRenderingFreeType import vtkVectorText  # noqa: PLC0415

    vec_text = vtkVectorText()
    vec_text.SetText(string)

    extrude = _vtk.vtkLinearExtrusionFilter()
    extrude.SetInputConnection(vec_text.GetOutputPort())
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0, 0, 1)
    extrude.SetScaleFactor(depth)

    tri_filter = _vtk.vtkTriangleFilter()
    tri_filter.SetInputConnection(extrude.GetOutputPort())
    tri_filter.Update()
    return pv.wrap(tri_filter.GetOutput())


@_deprecate_positional_args
def logo_letters(merge=False, depth=0.3):  # noqa: FBT002
    """Generate a mesh for each letter in "PyVista".

    Parameters
    ----------
    merge : bool, optional
        If ``True``, merge the meshes of the individual letters into a single
        mesh.  If ``False``, return a dictionary where the keys are the letters
        and the values are the respective meshes.
    depth : float, optional
        The depth of the extrusion for each letter in the mesh.

    Returns
    -------
    output : pyvista.PolyData or dict[str, pyvista.PolyData]
        If merge is ``True``, returns a single merged mesh containing all the
        letters in "PyVista". If merge is ``False``, returns a dictionary where
        the keys are the letters and the values are the respective meshes.

    """
    mesh_letters = pv.PolyData() if merge else {}  # type: ignore[var-annotated]

    # spacing between letters
    space_factor = 0.9
    width = 0
    for letter in LOGO_TITLE:
        mesh_letter = text_3d(letter, depth=depth)
        this_letter_width = mesh_letter.points[:, 0].max()
        mesh_letter.translate([width * space_factor, 0, 0.0], inplace=True)
        width += this_letter_width
        if merge:
            mesh_letters += mesh_letter
        else:
            mesh_letters[letter] = mesh_letter

    return mesh_letters


def logo_voxel(density=0.03):
    """Create a voxelized PyVista logo.

    Parameters
    ----------
    density : float, default: 0.03
        Density of the voxelization.

    Returns
    -------
    pyvista.UnstructuredGrid
        Voxelized PyVista logo as an unstructured grid.

    """
    return _voxelize_legacy(text_3d(LOGO_TITLE, depth=0.3), density=density)


def logo_basic():
    """Create a basic pyvista logo.

    Returns
    -------
    pyvista.UnstructuredGrid
        Grid containing the pyvista letters.

    Examples
    --------
    Plot the basic pyvista logo.

    >>> from pyvista import demos
    >>> logo = demos.logo_basic()
    >>> cpos = logo.plot(smooth_shading=True)

    Add scalars and plot the logo.

    >>> logo['x_coord'] = logo.points[:, 0]
    >>> cpos = logo.plot(
    ...     scalars='x_coord',
    ...     cmap='Spectral',
    ...     smooth_shading=True,
    ...     cpos='xy',
    ... )

    """
    return logo_letters(merge=True).compute_normals(split_vertices=True)


@_deprecate_positional_args
def plot_logo(  # noqa: PLR0917
    window_size=None,
    off_screen=None,
    screenshot=None,
    cpos=None,
    just_return_plotter=False,  # noqa: FBT002
    show_note=False,  # noqa: FBT002
    **kwargs,
):
    """Plot the stylized PyVista logo.

    Parameters
    ----------
    window_size : sequence[int], optional
        Size of the window in the format ``[width, height]``.
    off_screen : bool, optional
        Renders off screen when ``True``.
    screenshot : str, optional
        Save screenshot to path when specified.
    cpos : list or str, optional
        Camera position to use.
    just_return_plotter : bool, default: False
        Return the plotter instance without rendering.
    show_note : bool, default: False
        Show a text in the plot when ``True``.
    **kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    output : Plotter or camera position
        Returns the plotter instance if ``just_return_plotter`` is ``True``,
        otherwise returns the camera position if ``screenshot`` is specified,
        otherwise shows the plot.

    Examples
    --------
    >>> from pyvista import demos
    >>> cpos = demos.plot_logo()

    """
    # initialize plotter
    if window_size is None:
        window_size = [960, 400]
    pl = pv.Plotter(window_size=window_size, off_screen=off_screen)

    mesh_letters = logo_letters()

    # letter 'P'
    p_mesh = mesh_letters['P'].compute_normals(split_vertices=True)
    pl.add_mesh(p_mesh, color='#376fa0', smooth_shading=True)

    # letter 'y'
    y_mesh = mesh_letters['y'].compute_normals(split_vertices=True)
    pl.add_mesh(y_mesh, color='#ffd040', smooth_shading=True)

    # letter 'V'
    v_grid = _voxelize_legacy(mesh_letters['V'], density=0.08)
    v_grid_atom = atomize(v_grid)
    v_grid_atom['scalars'] = v_grid_atom.points[:, 0]
    v_grid_atom_surf = v_grid_atom.extract_surface()
    faces = v_grid_atom_surf.faces.reshape(-1, 5).copy()
    faces[:, 1:] = faces[:, 1:][:, ::-1]
    v_grid_atom_surf.faces = faces
    pl.add_mesh(
        v_grid_atom_surf,
        scalars='scalars',
        show_edges=True,
        cmap='winter',
        show_scalar_bar=False,
    )

    # letter 'i'
    i_grid = _voxelize_legacy(mesh_letters['i'], density=0.1)

    pl.add_mesh(
        i_grid.extract_surface(),
        style='points',
        color='r',
        render_points_as_spheres=True,
        point_size=14,
    )
    pl.add_mesh(i_grid, style='wireframe', color='k', line_width=4)

    # letter 's'
    mesh = mesh_letters['s']
    mesh['scalars'] = mesh.points[:, 0]
    pl.add_mesh(
        mesh,
        scalars='scalars',
        style='wireframe',
        line_width=2,
        cmap='gist_heat',
        backface_culling=True,
        render_lines_as_tubes=True,
        show_scalar_bar=False,
    )

    # letter 't'
    mesh = mesh_letters['t'].clean().compute_normals()
    scalars = mesh.points[:, 0]
    pl.add_mesh(mesh, scalars=scalars, show_edges=True, cmap='autumn', show_scalar_bar=False)

    # letter 'a'
    grid = examples.download_letter_a()
    grid.points[:, 0] += mesh_letters['a'].center[0] - grid.center[0]

    # select some cells from grid
    cells = grid.cells.reshape(-1, 5)
    mask = grid.points[cells[:, 1:], 2] < 0.2
    mask = mask.all(1)

    a_part = grid.extract_cells(mask)

    cells = a_part.cells.reshape(-1, 5)
    scalars = grid.points[cells[:, 1], 1]
    pl.add_mesh(a_part, scalars=scalars, show_edges=True, cmap='Greens', show_scalar_bar=False)

    if show_note:
        text = text_3d('You can move me!', depth=0.1)
        text.points *= 0.1
        text.translate([4.0, -0.3, 0], inplace=True)
        pl.add_mesh(text, color='black')

    # finalize plot and show it
    pl.set_background(kwargs.pop('background', 'white'))
    pl.camera_position = 'xy'
    if 'zoom' in kwargs:
        pl.camera.zoom(kwargs.pop('zoom'))

    # pl.remove_scalar_bar()
    pl.enable_anti_aliasing()

    if just_return_plotter:
        return pl

    if screenshot:  # pragma: no cover
        pl.show(cpos=cpos, auto_close=False)
        pl.screenshot(screenshot, True)
        cpos_final = pl.camera_position
        pl.close()
        return cpos_final
    else:
        return pl.show(cpos=cpos, **kwargs)


def logo_atomized(density=0.05, scale=0.6, depth=0.05):
    """Generate a voxelized pyvista logo with intra-cell spacing.

    Parameters
    ----------
    density : float, default: 0.05
        The spacing between voxels in the generated PyVista logo.
    scale : float, default: 0.6
        The scaling factor for the generated PyVista logo.
    depth : float, default: 0.05
        The depth of the generated PyVista logo.

    Returns
    -------
    pyvista.UnstructuredGrid
        A merged UnstructuredGrid representing the voxelized PyVista logo.

    """
    mesh_letters = logo_letters(depth=depth)
    grids = []
    for letter in mesh_letters.values():
        grid = _voxelize_legacy(letter, density=density)
        grids.append(atomize(grid, scale=scale))

    return grids[0].merge(grids[1:])
