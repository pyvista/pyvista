"""Support for the ipygany plotter."""

from array import array
import warnings

import numpy as np
from IPython import display


# not to be imported at the init level
try:
    import ipygany
except ImportError:  # pragma: no cover
    raise ImportError('Install ``ipygany`` to use this feature.')

try:
    from ipywidgets import (FloatSlider, FloatRangeSlider, Dropdown, Layout,
                            Select, VBox, HBox, AppLayout, jslink, HTML)
except ImportError:  # pragma: no cover
    raise ImportError('Install ``ipywidgets`` to use this feature.')


from ipygany.vtk_loader import get_ugrid_data
from ipygany.ipygany import _grid_data_to_data_widget
from ipygany import Scene, PolyMesh, Component, IsoColor, PointCloud
from ipygany.colormaps import colormaps

import pyvista as pv


def pyvista_polydata_to_polymesh(obj):
    """Import a mesh from ``pyvista`` or ``vtk``.

    Copies over the active scalars and only the active scalars.

    Parameters
    ----------
    obj : pyvista compatible object
        Any object compatible with pyvista.  Includes most ``vtk``
        objects.

    Returns
    -------
    PolyMesh
        ``ipygany.PolyMesh`` object.
    """
    # attempt to wrap non-pyvista objects
    if not pv.is_pyvista_dataset(obj):  # pragma: no cover
        mesh = pv.wrap(obj)
        if not pv.is_pyvista_dataset(mesh):
            raise TypeError(f'Object type ({type(mesh)}) cannot be converted to '
                            'a pyvista dataset')
    else:
        mesh = obj

    # PolyMesh requires vertices and triangles, so we need to
    # convert the mesh to an all triangle polydata
    if not isinstance(obj, pv.PolyData):
        # unlikely case that mesh does not have extract_surface
        if not hasattr(mesh, 'extract_surface'):  # pragma: no cover
            mesh = mesh.cast_to_unstructured_grid()
        surf = mesh.extract_surface()
    else:
        surf = mesh

    # convert to an all-triangular surface
    if surf.is_all_triangles():
        trimesh = surf
    else:
        trimesh = surf.triangulate()

    # finally, pass the triangle vertices to PolyMesh
    triangle_indices = trimesh.faces.reshape(-1, 4)[:, 1:]

    if not triangle_indices.size:
        warnings.warn('Unable to convert mesh to triangular PolyMesh')

    # only copy active scalars
    data = []
    if trimesh.active_scalars is not None:
        arr = array('f', trimesh.active_scalars)
        components = [ipygany.Component('X1', arr)]
        data = [ipygany.Data(trimesh.active_scalars_name, components)]

    # for speed, only convert the active scalars later
    return PolyMesh(
        vertices=trimesh.points,
        triangle_indices=triangle_indices,
        data=data
    )


def pyvista_object_to_pointcloud(pv_object):
    """Convert any pyvista object into a ``ipygany.PointCloud``."""
    pc = PointCloud(vertices=pv_object.points,
                    data=_grid_data_to_data_widget(get_ugrid_data(pv_object)))
    return pc


def color_float_to_hex(r, g, b):
    """Convert RGB to hex."""
    def clamp(x):
        x = round(x*255)
        return max(0, min(x, 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))



def check_colormap(cmap):
    """Attempt to convert a colormap to ``ipygany``."""
    if cmap not in colormaps:
        # attempt to matplotlib cmaps to ipygany
        if cmap.capitalize() in colormaps:
            cmap = cmap.capitalize()

    if cmap not in colormaps:
        allowed = ', '.join([f"'{clmp}'" for clmp in colormaps.keys()])
        raise ValueError(f'``cmap`` "{cmap} is not supported by ``ipygany``\n'
                         'Pick from one of the following:\n'
                         + allowed)
    return cmap


def ipygany_block_from_actor(actor):
    """Convert a vtk actor to a ipygany Block."""
    mapper = actor.GetMapper()
    if mapper is None:
        return
    dataset = mapper.GetInputAsDataSet()

    prop = actor.GetProperty()
    rep_type = prop.GetRepresentationAsString()

    # check if missing faces as a polydata
    if rep_type != 'Points' and isinstance(dataset, pv.PolyData):
        if not dataset.faces.size and dataset.n_points:
            rep_type = 'Points'

    if rep_type == 'Points':
        pmesh = pyvista_object_to_pointcloud(dataset)
    elif rep_type == 'Wireframe':
        warnings.warn('Wireframe style is not supported in ipygany')
        return
    else:
        pmesh = pyvista_polydata_to_polymesh(dataset)
    pmesh.default_color = color_float_to_hex(*prop.GetColor())

    # determine if there are active scalars
    valid_mode = mapper.GetScalarModeAsString() in ['UsePointData', 'UseCellData']
    if valid_mode:
        # verify dataset is in pmesh
        names = [dataset.name for dataset in pmesh.data]
        if dataset.active_scalars_name in names:
            mn, mx = mapper.GetScalarRange()
            cmesh = IsoColor(pmesh, input=dataset.active_scalars_name, min=mn, max=mx)
            if hasattr(mapper, 'cmap'):
                cmap = check_colormap(mapper.cmap)
                cmesh.colormap = colormaps[cmap]
            return cmesh

    return pmesh


def ipygany_camera_from_plotter(plotter):
    """Return an ipygany camera dict from a ``pyvista.Plotter`` object."""
    position, target, up = plotter.camera_position
    # TODO: camera position appears twice as far within ipygany, adjust:

    position = np.array(position, copy=True)
    position -= (position - np.array(target))/2

    return {'position': position.tolist(),
            'target': target,
            'up': up}


def show_ipygany(plotter, return_viewer, height=None, width=None):
    """Show an ipygany scene."""
    # convert each mesh in the plotter to an ipygany scene
    actors = plotter.renderer._actors
    meshes = []
    for actor in actors.values():
        ipygany_obj = ipygany_block_from_actor(actor)
        if ipygany_obj is not None:
            meshes.append(ipygany_obj)

    bc_color = color_float_to_hex(*plotter.background_color)
    scene = Scene(meshes,
                  background_color=bc_color,
                  camera=ipygany_camera_from_plotter(plotter))

    # optionally size of the plotter
    if height is not None:
        scene.layout.height = f'{height}'
    if width is not None:
        scene.layout.width = f'{width}'

    cbar = None
    if len(plotter.scalar_bars):
        for mesh in meshes:
            if isinstance(mesh, ipygany.IsoColor):
                cbar = ipygany.ColorBar(mesh)
                colored_mesh = mesh
                break

    # Simply return the scene
    if return_viewer:
        return scene

    if cbar is not None:
        # Colormap choice widget
        colormap_dd = Dropdown(
            options=colormaps,
            description='Colormap:'
        )
        jslink((colored_mesh, 'colormap'), (colormap_dd, 'index'))

        # sensible colorbar maximum width, or else it looks bad when
        # window is large.
        cbar.layout.max_width = '500px'
        cbar.layout.min_height = '50px'  # stop from getting squished
        # cbar.layout.height = '20%'  # stop from getting squished
        # cbar.layout.max_height = ''

        # Create a slider that will dynamically change the boundaries of the colormap
        # colormap_slider_range = FloatRangeSlider(value=[height_min, height_max],
        #                                          min=height_min, max=height_max,
        #                                          step=(height_max - height_min) / 100.)

        # jslink((colored_mesh, 'range'), (colormap_slider_range, 'value'))

        # create app
        title = HTML(value=f'<h3>{list(plotter.scalar_bars.keys())[0]}</h3>')
        legend = VBox((title, colormap_dd, cbar))
        scene = AppLayout(center=scene,
                          footer=legend,
                          pane_heights=[0, 0, '150px'])

    display.display_html(scene)
