"""PyVista-like ITKwidgets plotter."""
import numpy as np
from scooby import meets_version

import pyvista
import pyvista as pv
from .theme import parse_color

HAS_ITK = False
ITK_EXCEPTION = None
try:
    from itkwidgets import Viewer
    from itkwidgets._transform_types import to_geometry
    HAS_ITK = True
except ImportError as e:  # pragma: no cover
    ITK_EXCEPTION = e


class PlotterITK():
    """ITKwidgets plotter.

    Used for plotting interactively within a jupyter notebook.
    Requires ``itkwidgets>=0.25.2``.  For installation see:

    https://github.com/InsightSoftwareConsortium/itkwidgets#installation

    Examples
    --------
    >>> import pyvista
    >>> mesh = pyvista.Sphere()
    >>> pl = pyvista.PlotterITK()  # doctest:+SKIP
    >>> pl.add_mesh(mesh, color='w')  # doctest:+SKIP
    >>> pl.background_color = 'k'  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP
    """

    def __init__(self, **kwargs):
        """Initialize the itkwidgets plotter."""
        if not HAS_ITK:  # pragma: no cover
            itk_import_err = ImportError("Please install `itkwidgets>=0.25.2`:\n%s" %
                                         str(ITK_EXCEPTION))
            raise itk_import_err

        from itkwidgets import __version__
        if not meets_version(__version__, "0.25.2"):  # pragma: no cover
            raise itk_import_err

        self._point_sets = []
        self._geometries = []
        self._geometry_colors = []
        self._geometry_opacities = []
        self._point_set_colors = []
        self._background_color = None
        self._camera_position = None
        self._point_set_sizes = []
        self._point_set_representations = []

    def add_points(self, points, color=None, point_size=3.0):
        """Add points to plotter.

        Parameters
        ----------
        points : np.ndarray or pyvista.Common
            n x 3 numpy array of points or pyvista dataset with points.

        color : string or 3 item list, optional. Color of points (if visible).
            Either a string, rgb list, or hex color string.  For example:

            ``color='white'``
            ``color='w'``
            ``color=[1, 1, 1]``
            ``color='#FFFFFF'``

        point_size : float, optional
            Point size of any nodes in the dataset plotted. Also applicable
            when style='points'. Default ``3.0``

        Examples
        --------
        Add 10 random points to the plotter

        >>> add_points(np.random.random((10, 3)), 'r', 10)  # doctest:+SKIP
        """
        if pv.is_pyvista_dataset(points):
            point_array = points.points
        else:
            point_array = points


        # style : str, optional
        #     How to represent the point set. One of ``'hidden'``,
        #     ``'points'``, or ``'spheres'``.


        # if style not in ['hidden', 'points', 'spheres']:
        #     raise ValueError("``style`` must be either 'hidden', 'points', or"
        #                      "'spheres'")

        if not isinstance(point_size, (int, float)):
            raise TypeError('``point_size`` parameter must be a float')

        self._point_set_sizes.append(point_size)
        self._point_set_colors.append(pv.parse_color(color))
        self._point_sets.append(point_array)
        # self._point_set_representations.append(style)

    def add_mesh(self, mesh, color=None, scalars=None,
                 opacity=1.0, smooth_shading=False):
        """Add a PyVista/VTK mesh or dataset.

        Adds any PyVista/VTK mesh that itkwidgets can wrap to the
        scene.

        Parameters
        ----------
        mesh : pyvista.Common or pyvista.MultiBlock
            Any PyVista or VTK mesh is supported. Also, any dataset
            that :func:`pyvista.wrap` can handle including NumPy arrays of XYZ
            points.

        color : string or 3 item list, optional, defaults to white
            Use to make the entire mesh have a single solid color.
            Either a string, RGB list, or hex color string.  For example:
            ``color='white'``, ``color='w'``, ``color=[1, 1, 1]``, or
            ``color='#FFFFFF'``. Color will be overridden if scalars are
            specified.

        scalars : str or numpy.ndarray, optional
            Scalars used to "color" the mesh.  Accepts a string name of an
            array that is present on the mesh or an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If both
            ``color`` and ``scalars`` are ``None``, then the active scalars are
            used.

        opacity : float, optional
            Opacity of the mesh. If a single float value is given, it will be
            the global opacity of the mesh and uniformly applied everywhere -
            should be between 0 and 1.  Default 1.0

        smooth_shading : bool, optional
            Smooth mesh surface mesh by taking into account surface
            normals.  Surface will appear smoother while sharp edges
            will still look sharp.  Default False.

        """
        if not pv.is_pyvista_dataset(mesh):
            mesh = pv.wrap(mesh)

        # smooth shading requires point normals to be freshly computed
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

        # make the scalars active
        if isinstance(scalars, str):
            if scalars in mesh.point_arrays or scalars in mesh.cell_arrays:
                array = mesh[scalars].copy()
            else:
                raise ValueError(f'Scalars ({scalars}) not in mesh')
            mesh[scalars] = array
            mesh.active_scalars_name = scalars
        elif isinstance(scalars, np.ndarray):
            array = scalars
            scalar_name = '_scalars'
            mesh[scalar_name] = array
            mesh.active_scalars_name = scalar_name
        elif color is not None:
            mesh.active_scalars_name = None

        # itkwidgets does not support VTK_ID_TYPE
        if 'vtkOriginalPointIds' in mesh.point_arrays:
            mesh.point_arrays.pop('vtkOriginalPointIds')

        if 'vtkOriginalCellIds' in mesh.cell_arrays:
            mesh.cell_arrays.pop('vtkOriginalCellIds')

        mesh = to_geometry(mesh)
        self._geometries.append(mesh)
        self._geometry_colors.append(pv.parse_color(color))
        self._geometry_opacities.append(opacity)

    @property
    def background_color(self):
        """Return the background color of the plotter."""
        return self._background_color

    @background_color.setter
    def background_color(self, color):
        """Set the background color of the plotter.

        Examples
        --------
        Set the background color to black

        >>> plotter.background_color = 'k'
        """
        self._background_color = parse_color(color)

    @property
    def camera_position(self):
        """Return camera position of the plotter as a list."""
        if self._camera_position is not None:
            return self._camera_position

    @camera_position.setter
    def camera_position(self, camera_location):
        """Set camera position of the plotter."""
        if isinstance(camera_location, str):
            raise ValueError('String camera positions are not supported in PlotterITK')
        else:
            # check if a valid camera position
            if not len(camera_location) == 3:
                raise pyvista.core.errors.InvalidCameraError
            elif any([len(item) != 3 for item in camera_location]):
                raise pyvista.core.errors.InvalidCameraError

        self._camera_position = camera_location

    def show(self, ui_collapsed=True, rotate=False, show_bounds=False,
             **kwargs):
        """Show itkwidgets plotter in cell output.

        Parameters
        ----------
        ui_collapsed : bool, optional
            Plot with the user interface collapsed.  UI can be enabled
            when plotting.  Default ``False``.

        rotate : bool, optional
            Rotate the camera around the scene.  Default ``False``.
            Appears to be computationally intensive.

        show_bounds : bool, optional
            Show the bounding box.  Default False

        point_size : int, optional
            Size of the points displayed in the

        Returns
        --------
        viewer : itkwidgets.Viewer
            ``ITKwidgets`` viewer.
        """
        if self._background_color is not None:
            kwargs.setdefault('background', self._background_color)
        if self._camera_position is not None:
            kwargs.setdefault('camera', self._camera_position)

        viewer = Viewer(geometries=self._geometries,
                        geometry_colors=self._geometry_colors,
                        geometry_opacities=self._geometry_opacities,
                        point_set_colors=self._point_set_colors,
                        point_sets=self._point_sets,
                        point_set_sizes=self._point_set_sizes,
                        point_set_representations=self._point_set_representations,
                        ui_collapsed=ui_collapsed,
                        rotate=rotate,
                        axes=show_bounds,
                        **kwargs)

        # always show if iPython is installed
        try:
            from IPython import display
            display.display_html(viewer)
        except ImportError:
            pass

        return viewer
