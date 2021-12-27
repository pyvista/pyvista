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
import os

import numpy as np

import pyvista
from pyvista import _vtk, examples

THIS_PATH = os.path.dirname(os.path.realpath(__file__))

LOGO_TITLE = 'PyVista'


def atomize(grid, shift_fac=0.1, scale=0.9):
    """Break apart and shrink and/or scale the individual cells of a mesh."""
    cent = grid.center
    cells = []
    for i in range(grid.n_cells):
        cell = grid.extract_cells(i)
        ccent = np.array(cell.center)
        cell.points[:] = (cell.points - ccent)*scale + ccent
        cell.points += (ccent - np.array(cent))*shift_fac
        cells.append(cell)

    return cells[0].merge(cells[1:])


def text_3d(string, depth=0.5):
    """Create 3D text."""
    vec_text = _vtk.vtkVectorText()
    vec_text.SetText(string)

    extrude = _vtk.vtkLinearExtrusionFilter()
    extrude.SetInputConnection(vec_text.GetOutputPort())
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0, 0, 1)
    extrude.SetScaleFactor(depth)

    tri_filter = _vtk.vtkTriangleFilter()
    tri_filter.SetInputConnection(extrude.GetOutputPort())
    tri_filter.Update()
    return pyvista.wrap(tri_filter.GetOutput())


def logo_letters(merge=False, depth=0.3):
    """Generate a mesh for each letter in "PyVista"."""
    if merge:
        mesh_letters = pyvista.PolyData()
    else:
        mesh_letters = {}

    # spacing between letters
    space_factor = 0.9
    width = 0
    for letter in LOGO_TITLE:
        mesh_letter = text_3d(letter, depth=depth)
        this_letter_width = mesh_letter.points[:, 0].max()
        mesh_letter.translate([width*space_factor, 0, 0.0], inplace=True)
        width += this_letter_width
        if merge:
            mesh_letters += mesh_letter
        else:
            mesh_letters[letter] = mesh_letter

    return mesh_letters


def logo_voxel(density=0.03):
    """Create a voxelized PyVista logo."""
    return pyvista.voxelize(text_3d(LOGO_TITLE, depth=0.3), density)


def logo_basic():
    """Create a basic pyvista logo.

    Examples
    --------
    Plot the basic pyvista logo.

    >>> from pyvista import demos
    >>> logo = demos.logo_basic()
    >>> cpos = logo.plot(smooth_shading=True)

    Add scalars and plot the logo.

    >>> logo['x_coord'] = logo.points[:, 0]
    >>> cpos = logo.plot(scalars='x_coord', cmap='Spectral',
    ...                  smooth_shading=True, cpos='xy')

    """
    return logo_letters(merge=True).compute_normals(split_vertices=True)


def plot_logo(window_size=None, off_screen=None, screenshot=None,
              cpos=None, just_return_plotter=False, show_note=False, **kwargs):
    """Plot the stylized PyVista logo.

    Examples
    --------
    >>> from pyvista import demos
    >>> cpos = demos.plot_logo()

    """
    # initialize plotter
    if window_size is None:
        window_size = [960, 400]
    plotter = pyvista.Plotter(window_size=window_size, off_screen=off_screen)

    mesh_letters = logo_letters()

    # letter 'P'
    p_mesh = mesh_letters['P'].compute_normals(split_vertices=True)
    p_mesh.flip_normals()
    plotter.add_mesh(p_mesh, color='#376fa0', smooth_shading=True)

    # letter 'y'
    y_mesh = mesh_letters['y'].compute_normals(split_vertices=True)
    y_mesh.flip_normals()
    plotter.add_mesh(y_mesh, color='#ffd040', smooth_shading=True)

    # letter 'V'
    v_grid = pyvista.voxelize(mesh_letters['V'], density=0.08)
    v_grid_atom = atomize(v_grid)
    v_grid_atom['scalars'] = v_grid_atom.points[:, 0]
    v_grid_atom_surf = v_grid_atom.extract_surface()
    faces = v_grid_atom_surf.faces.reshape(-1, 5)
    faces[:, 1:] = faces[:, 1:][:, ::-1]
    v_grid_atom_surf.faces = faces
    plotter.add_mesh(v_grid_atom_surf, scalars='scalars', show_edges=True,
                     cmap='winter', show_scalar_bar=False)

    # letter 'i'
    i_grid = pyvista.voxelize(mesh_letters['i'], density=0.1)

    plotter.add_mesh(i_grid.extract_surface(),
                     style='points', color='r',
                     render_points_as_spheres=True, point_size=14)
    plotter.add_mesh(i_grid, style='wireframe', color='k', line_width=4)

    # letter 's'
    mesh = mesh_letters['s']
    mesh['scalars'] = mesh.points[:, 0]
    plotter.add_mesh(mesh, scalars='scalars', style='wireframe',
                     show_edges=True, line_width=2, cmap='gist_heat',
                     backface_culling=True, render_lines_as_tubes=True,
                     show_scalar_bar=False)

    # letter 't'
    mesh = mesh_letters['t'].clean().compute_normals()
    # strange behavior with pythreejs
    if pyvista.global_theme.jupyter_backend == 'pythreejs':
        mesh.flip_normals()
    scalars = mesh.points[:, 0]
    plotter.add_mesh(mesh, scalars=scalars, show_edges=True,
                     cmap='autumn', show_scalar_bar=False)

    # letter 'a'
    grid = examples.download_letter_a()
    grid.points[:, 0] += (mesh_letters['a'].center[0] - grid.center[0])

    # select some cells from grid
    cells = grid.cells.reshape(-1, 5)
    mask = grid.points[cells[:, 1:], 2] < 0.2
    mask = mask.all(1)

    a_part = grid.extract_cells(mask)

    cells = a_part.cells.reshape(-1, 5)
    scalars = grid.points[cells[:, 1], 1]
    plotter.add_mesh(a_part, scalars=scalars, show_edges=True, cmap='Greens',
                     show_scalar_bar=False)

    if show_note:
        text = text_3d("You can move me!", depth=0.1)
        text.points *= 0.1
        text.translate([4.0, -0.3, 0], inplace=True)
        plotter.add_mesh(text, color='black')

    # finalize plot and show it
    plotter.set_background(kwargs.pop('background', 'white'))
    plotter.camera_position = 'xy'
    if 'zoom' in kwargs:
        plotter.camera.zoom(kwargs.pop('zoom'))

    # plotter.remove_scalar_bar()
    plotter.enable_anti_aliasing()

    if just_return_plotter:
        return plotter

    if screenshot:  # pragma: no cover
        plotter.show(cpos=cpos, auto_close=False)
        plotter.screenshot(screenshot, True)
        cpos_final = plotter.camera_position
        plotter.close()
        return cpos_final
    else:
        return plotter.show(cpos=cpos, **kwargs)


def logo_atomized(density=0.05, scale=0.6, depth=0.05):
    """Generate a voxelized pyvista logo with intra-cell spacing."""
    mesh_letters = logo_letters(depth=depth)
    grids = []
    for letter in mesh_letters.values():
        grid = pyvista.voxelize(letter, density=density)
        grids.append(atomize(grid, scale=scale))

    return grids[0].merge(grids[1:])
