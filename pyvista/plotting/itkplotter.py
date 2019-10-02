"""
This is super experimental: use with caution
"""

import pyvista as pv
from pyvista import examples
import numpy as np

HAS_ITK = False
try:
    from itkwidgets import view, Viewer
    from itkwidgets._transform_types import to_geometry
    HAS_ITK = True
except ImportError:
    pass

class PlotterITK():
    def __init__(self, **kwargs):
        if not HAS_ITK:
            raise ImportError("Please install `itkwidgets`.")
        self._geometries = []
        self._geometry_colors = []
        self._geometry_opacities = []
        self._cmap = 'Viridis (matplotlib)'


    def add_mesh(self, mesh, color=None, scalars=None, clim=None,
                 opacity=1.0, n_colors=256, cmap='Viridis (matplotlib)',
                 **kwargs):
        """Adds mesh to the scene"""
        if not pv.is_pyvista_dataset(mesh):
            mesh = pv.wrap(mesh)
        mesh = mesh.copy()
        if scalars is None and color is None:
            scalars = mesh.active_scalar_name

        if scalars is not None:
            array = mesh[scalars].copy()
            mesh.clear_arrays()
            mesh[scalars] = array
            mesh.active_scalar_name = scalars
        elif color is not None:
            mesh.clear_arrays()


        mesh = to_geometry(mesh)
        self._geometries.append(mesh)
        self._geometry_colors.append(pv.parse_color(color))
        self._geometry_opacities.append(opacity)
        self._cmap = cmap

        return


    def show(self, ui_collapsed=False):
        """Show in cell output"""
        plotter = Viewer(geometries=self._geometries,
                         geometry_colors=self._geometry_colors,
                         geometry_opacities=self._geometry_opacities,
                         ui_collapsed=ui_collapsed,
                         cmap=self._cmap)
        return plotter
