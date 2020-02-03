"""PyVista-like ITKwidgets plotter.

This is super experimental: use with caution.
"""
import numpy as np

import pyvista as pv

HAS_ITK = False
try:
    from itkwidgets import Viewer
    from itkwidgets._transform_types import to_geometry
    HAS_ITK = True
except ImportError:
    pass

class PlotterITK():
    """An EXPERIMENTAL interface for plotting in Jupyter notebooks.

    Use with caution, this is an experimental/demo feature. This creates an
    interface for 3D rendering with ``itkwidgets`` just like the
    :class:`pyvista.Plotter` class.
    """

    def __init__(self, **kwargs):
        """Initialize the itkwidgets plotter."""
        if not HAS_ITK:
            raise ImportError("Please install `itkwidgets>=0.25.2`.")
        self._actors = []
        self._point_sets = []
        self._geometries = []
        self._geometry_colors = []
        self._geometry_opacities = []
        self._cmap = 'Viridis (matplotlib)'
        self._point_set_colors = []

    def add_actor(self, actor):
        """Append internal list of actors."""
        self._actors.append(actor)

    def add_points(self, points, color=None):
        """Add XYZ points to the scene."""
        if pv.is_pyvista_dataset(points):
            point_array = points.points
        else:
            point_array = points

        self._point_set_colors.append(pv.parse_color(color))
        self._point_sets.append(point_array)

    def add_mesh(self, mesh, color=None, scalars=None, clim=None,
                 opacity=1.0, n_colors=256, smooth_shading=False,
                 **kwargs):
        """Add mesh to the scene."""
        if not pv.is_pyvista_dataset(mesh):
            mesh = pv.wrap(mesh)
        # mesh = mesh.copy()
        # if scalars is None and color is None:
        #     scalars = mesh.active_scalars_name

        if smooth_shading:
            # extract surface if mesh is exterior
            if not isinstance(mesh, pv.PolyData):
                grid = mesh
                mesh = grid.extract_surface()
                ind = mesh.point_arrays['vtkOriginalPointIds']
                # remap scalars
                if isinstance(scalars, np.ndarray):
                    scalars = scalars[ind]

            mesh.compute_normals(cell_normals=False, inplace=True)
        elif 'Normals' in mesh.point_arrays:
            # if 'normals' in mesh.point_arrays:
            mesh.point_arrays.pop('Normals')

        # copy this mesh as we're going to be clearing out scalars
        # mesh = mesh.copy()

        # if scalars is None and color is None:
        #     scalars = mesh.active_scalars_name

        if isinstance(scalars, str):
            if scalars in mesh:
                array = mesh[scalars].copy()
            else:
                raise ValueError('Scalars %s not in mesh' % scalars)
            mesh[scalars] = array
            mesh.active_scalars_name = scalars
        elif isinstance(scalars, np.ndarray):
            array = scalars
            scalar_name = '_scalars'
            mesh[scalar_name] = array
            mesh.active_scalars_name = scalar_name
        elif color is not None:
            mesh.active_scalars_name = None

        mesh = to_geometry(mesh)
        self._geometries.append(mesh)
        self._geometry_colors.append(pv.parse_color(color))
        self._geometry_opacities.append(opacity)
        # self._cmap = cmap

    def show(self, ui_collapsed=False):
        """Show in cell output."""
        plotter = Viewer(geometries=self._geometries,
                         geometry_colors=self._geometry_colors,
                         geometry_opacities=self._geometry_opacities,
                         point_set_colors=self._point_set_colors,
                         point_sets=self._point_sets,
                         ui_collapsed=ui_collapsed,
                         actors=self._actors,
                         cmap=self._cmap)
        return plotter
