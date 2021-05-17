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
from pyvista import examples
import pyvista
from pyvista import _vtk

import numpy as np

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
        mesh_letter.translate([width*space_factor, 0, 0.0])
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


def plot_logo(window_size=None, off_screen=None, screenshot=None, cpos=None, **kwargs):
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
    plotter.add_mesh(p_mesh, color='#376fa0', smooth_shading=True)

    # letter 'y'
    p_mesh = mesh_letters['y'].compute_normals(split_vertices=True)
    plotter.add_mesh(p_mesh, color='#ffd040', smooth_shading=True)

    # letter 'V'
    v_grid = pyvista.voxelize(mesh_letters['V'], density=0.08)
    v_grid_atom = atomize(v_grid)
    v_grid_atom['scalars'] = v_grid_atom.points[:, 0]
    plotter.add_mesh(v_grid_atom, scalars='scalars', show_edges=True,
                     cmap='winter', show_scalar_bar=False)

    # letter 'i'
    i_grid = pyvista.voxelize(mesh_letters['i'], density=0.1)

    plotter.add_mesh(i_grid.extract_surface(),
                     style='points', color='r',
                     render_points_as_spheres=True, point_size=8)
    plotter.add_mesh(i_grid, style='wireframe', color='k', line_width=4)

    # letter 's'
    mesh = mesh_letters['s']
    scalars = mesh.points[:, 0]
    plotter.add_mesh(mesh, scalars=scalars, style='wireframe', color='w',
                     show_edges=True, line_width=2, cmap='gist_heat',
                     backface_culling=True, render_lines_as_tubes=True)

    # letter 't'
    mesh = mesh_letters['t']
    scalars = mesh.points[:, 0]
    plotter.add_mesh(mesh, scalars=scalars, show_edges=True,
                     cmap='autumn', lighting=True)

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
    plotter.add_mesh(a_part, scalars=scalars, show_edges=True, cmap='Greens')

    # finalize plot and show it
    plotter.set_background(kwargs.pop('background', 'white'))
    if cpos is None:
        cpos = [(0.9060226106040606, 0.7752122028710583, 5.148283455883558),
                (2.553950615449191, 0.34145688392081264, 0.06127122762851659),
                (0.019308531920309943, 0.9967088407956779, -0.07873161547192063)]
    plotter.camera_position = cpos

    plotter.remove_scalar_bar()
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


def _for_landing_page(jupyter_backend='ipygany', **kwargs):
    """Plot the stylized PyVista logo for ipygany.

    To be shown on the landing page at index.rst

    """
    mesh_letters = logo_letters()

    # letter 'P'
    p_mesh = mesh_letters['P'].compute_normals(split_vertices=True)

    # letter 'y'
    y_mesh = mesh_letters['y'].compute_normals(split_vertices=True)

    # letter 'V'
    v_grid = pyvista.voxelize(mesh_letters['V'], density=0.08)
    v_grid_atom = atomize(v_grid)
    v_grid_atom['scalars'] = v_grid_atom.points[:, 0]

    i_grid = pyvista.voxelize(mesh_letters['i'], density=0.1)
    i_mesh = i_grid.extract_surface().triangulate().subdivide(2)
    i_mesh = i_mesh.smooth(500)
    old_center = np.array(i_mesh.center)
    i_mesh.points *= 1.07
    i_mesh.points += old_center - np.array(i_mesh.center)

    # letter 's'
    s_vox = pyvista.voxelize(mesh_letters['s'], density=0.04)
    s_cent = s_vox.cell_centers()
    pd = pyvista.PolyData(s_cent.points)

    sphere = pyvista.Sphere(theta_resolution=9, phi_resolution=9)
    s_grid = pd.glyph(factor=0.04, geom=sphere)

    # letter 't'
    # t_mesh = mesh_letters['t'].subdivide(5)
    # t_mesh.flip_normals()
    # t_mesh = t_mesh.compute_normals(consistent_normals=True)
    # import pyacvd
    # clus = pyacvd.Clustering(t_mesh)
    # clus.cluster(140)
    # t_cmesh = clus.create_mesh()

    # import _ as fe
    # src = fe.Surface(t_cmesh)
    # tgt = fe.Surface(t_mesh)
    # src.morph(tgt, settings={'local_with_centroid': True, 'local_steps': 300})
    # src.morph(tgt, settings={'local_with_centroid': True, 'local_steps': 300})

    # t_mesh = t_cmesh.extract_all_edges().tube(radius=0.005, n_sides=4)
    # t_mesh.extract_surface().save(...)

    t_mesh_filename = os.path.join(THIS_PATH, 't_mesh.ply')
    t_mesh = pyvista.read(t_mesh_filename)

    # letter 'a'
    grid = examples.download_letter_a()
    grid.points[:, 0] += (mesh_letters['a'].center[0] - grid.center[0])

    # select some cells from grid
    cells = grid.cells.reshape(-1, 5)
    mask = grid.points[cells[:, 1:], 2] < 0.2
    mask = mask.all(1)

    a_part = grid.extract_cells(mask)

    plotter = pyvista.Plotter()
    plotter.add_mesh(p_mesh, color='#376fa0')
    plotter.add_mesh(y_mesh, color='#ffd040')
    vista = v_grid_atom.merge([i_mesh, s_grid, t_mesh, a_part])
    vista['xdist'] = vista.points[:, 0]
    plotter.add_mesh(vista, cmap='viridis')

    # cpos = None
    # cpos = [(-0.9785294154224577, 1.2712499319005408, 10.965733716449193),
    #         (2.553950615449191, 0.34145688392081264, 0.06127122762851659),
    #         (0.019308531920309947, 0.996708840795678, -0.07873161547192065)]

    # cpos = [(0.9060226106040606, 0.7752122028710583, 5.148283455883558),
    #         (2.553950615449191, 0.34145688392081264, 0.06127122762851659),
    #         (0.019308531920309943, 0.9967088407956779, -0.07873161547192063)]

    # cpos = [(0.6861237002108157, 0.7572283207509382, 5.078581054505883),
    #         (2.334051705055946, 0.3234730018006926, -0.008431173749159387),
    #         (0.019308531920309947, 0.996708840795678, -0.07873161547192065)]

    if jupyter_backend == 'ipygany':
        x = 2.7
        cpos = [(x, 0.306, 5),
                (x, 0.306, 0.15),
                (0.0, 1.0, 0.0)]

        text = text_3d("I'm interactive!", depth=0.1)
        text.points *= 0.15
        text.translate([4, -0.4, 0])

        plotter.add_mesh(text, color='black')

    else:
        cpos = [(0.9060226106040606, 0.7752122028710583, 5.148283455883558),
                (2.553950615449191, 0.34145688392081264, 0.06127122762851659),
                (0.019308531920309943, 0.9967088407956779, -0.07873161547192063)]

    plotter.background_color = 'white'
    plotter.remove_scalar_bar()
    return plotter.show(cpos=cpos, jupyter_backend=jupyter_backend,
                        jupyter_kwargs=kwargs)


# _for_landing_page()
