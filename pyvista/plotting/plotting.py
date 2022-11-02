"""PyVista plotting module."""
import collections.abc
from copy import deepcopy
import ctypes
from functools import wraps
import io
import logging
import os
import pathlib
import platform
import textwrap
from threading import Thread
import time
from typing import Dict
import warnings
import weakref

import numpy as np
import scooby

import pyvista
from pyvista import _vtk
from pyvista.utilities import (
    FieldAssociation,
    abstract_class,
    assert_empty_kwargs,
    convert_array,
    get_array,
    get_array_association,
    is_pyvista_dataset,
    numpy_to_texture,
    raise_not_matching,
    wrap,
)
from pyvista.utilities.arrays import _coerce_pointslike_arg

from ..utilities.misc import PyVistaDeprecationWarning, has_module, uses_egl
from ..utilities.regression import image_from_window
from ._plotting import (
    USE_SCALAR_BAR_ARGS,
    _common_arg_parser,
    prepare_smooth_shading,
    process_opacity,
)
from ._property import Property
from .actor import Actor
from .colors import Color, get_cmap_safe
from .composite_mapper import CompositePolyDataMapper
from .export_vtkjs import export_plotter_vtkjs
from .mapper import (
    DataSetMapper,
    FixedPointVolumeRayCastMapper,
    GPUVolumeRayCastMapper,
    OpenGLGPUVolumeRayCastMapper,
    PointGaussianMapper,
    SmartVolumeMapper,
)
from .picking import PickingHelper
from .render_window_interactor import RenderWindowInteractor
from .renderer import Camera, Renderer
from .renderers import Renderers
from .scalar_bars import ScalarBars
from .tools import FONTS, normalize, opacity_transfer_function, parse_font_family  # noqa
from .widgets import WidgetHelper

SUPPORTED_FORMATS = [".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"]
VERY_FIRST_RENDER = True  # windows plotter helper

# EXPERIMENTAL: permit pyvista to kill the render window
KILL_DISPLAY = platform.system() == 'Linux' and os.environ.get('PYVISTA_KILL_DISPLAY')
if KILL_DISPLAY:  # pragma: no cover
    # this won't work under wayland
    try:
        X11 = ctypes.CDLL("libX11.so")
        X11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    except OSError:
        warnings.warn('PYVISTA_KILL_DISPLAY: Unable to load X11.\nProbably using wayland')
        KILL_DISPLAY = False


def close_all():
    """Close all open/active plotters and clean up memory.

    Returns
    -------
    bool
        ``True`` when all plotters have been closed.

    """
    for pl in list(_ALL_PLOTTERS.values()):
        if not pl._closed:
            pl.close()
    _ALL_PLOTTERS.clear()
    return True


log = logging.getLogger(__name__)
log.setLevel('CRITICAL')
log.addHandler(logging.StreamHandler())


def _warn_xserver():  # pragma: no cover
    """Check if plotting is supported and persist this state.

    Check once and cache this value between calls.  Warn the user if
    plotting is not supported.  Configured to check on Linux and Mac
    OS since the Windows check is not quick.

    """
    # disable windows check until we can get a fast way of verifying
    # if windows has a windows manager (which it generally does)
    if os.name == 'nt':
        return

    if not hasattr(_warn_xserver, 'has_support'):
        _warn_xserver.has_support = pyvista.system_supports_plotting()

    if not _warn_xserver.has_support:
        # check if a display has been set
        if 'DISPLAY' in os.environ:
            return

        # finally, check if using a backend that doesn't require an xserver
        if pyvista.global_theme.jupyter_backend in ['ipygany', 'pythreejs']:
            return

        # Check if VTK has EGL support
        if uses_egl():
            return

        warnings.warn(
            '\n'
            'This system does not appear to be running an xserver.\n'
            'PyVista will likely segfault when rendering.\n\n'
            'Try starting a virtual frame buffer with xvfb, or using\n '
            ' ``pyvista.start_xvfb()``\n'
        )


@abstract_class
class BasePlotter(PickingHelper, WidgetHelper):
    """To be used by the Plotter and pyvistaqt.QtInteractor classes.

    Parameters
    ----------
    shape : list or tuple, optional
        Number of sub-render windows inside of the main window.
        Specify two across with ``shape=(2, 1)`` and a two by two grid
        with ``shape=(2, 2)``.  By default there is only one renderer.
        Can also accept a string descriptor as shape. E.g.:

            * ``shape="3|1"`` means 3 plots on the left and 1 on the right,
            * ``shape="4/2"`` means 4 plots on top and 2 at the bottom.

    border : bool, optional
        Draw a border around each render window.  Default ``False``.

    border_color : color_like, optional
        Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

    border_width : float, optional
        Width of the border in pixels when enabled.

    title : str, optional
        Window title of the scalar bar

    lighting : str, optional
        What lighting to set up for the plotter.
        Accepted options:

            * ``'light_kit'``: a vtk Light Kit composed of 5 lights.
            * ``'three lights'``: illumination using 3 lights.
            * ``'none'``: no light sources at instantiation.

        The default is a Light Kit (to be precise, 5 separate lights
        that act like a Light Kit).

    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    """

    mouse_position = None
    click_position = None

    def __init__(
        self,
        shape=(1, 1),
        border=None,
        border_color='k',
        border_width=2.0,
        title=None,
        splitting_position=None,
        groups=None,
        row_weights=None,
        col_weights=None,
        lighting='light kit',
        theme=None,
        **kwargs,
    ):
        """Initialize base plotter."""
        super().__init__(**kwargs)  # cooperative multiple inheritance
        log.debug('BasePlotter init start')
        self._initialized = False

        self._theme = pyvista.themes.DefaultTheme()
        if theme is None:
            # copy global theme to ensure local plot theme is fixed
            # after creation.
            self._theme.load_theme(pyvista.global_theme)
        else:
            if not isinstance(theme, pyvista.themes.DefaultTheme):
                raise TypeError(
                    'Expected ``pyvista.themes.DefaultTheme`` for '
                    f'``theme``, not {type(theme).__name__}.'
                )
            self._theme.load_theme(theme)

        self.image_transparent_background = self._theme.transparent_background

        # optional function to be called prior to closing
        self.__before_close_callback = None
        self._store_image = False
        self.mesh = None
        if title is None:
            title = self._theme.title
        self.title = str(title)

        # add renderers
        self.renderers = Renderers(
            self,
            shape,
            splitting_position,
            row_weights,
            col_weights,
            groups,
            border,
            border_color,
            border_width,
        )

        # This keeps track of scalars names already plotted and their ranges
        self._scalar_bars = ScalarBars(self)

        # track if the camera has been set up
        self._first_time = True
        # Keep track of the scale

        # track if render window has ever been rendered
        self._rendered = False

        # this helps managing closed plotters
        self._closed = False

        # lighting style; be forgiving with input (accept underscores
        # and ignore case)
        lighting_normalized = str(lighting).replace('_', ' ').lower()
        if lighting_normalized == 'light kit':
            self.enable_lightkit()
        elif lighting_normalized == 'three lights':
            self.enable_3_lights()
        elif lighting_normalized != 'none':
            raise ValueError(f'Invalid lighting option "{lighting}".')

        # Track all active plotters. This has the side effect of ensuring that plotters are not
        # collected until `close()`. See https://github.com//pull/3216
        self._id_name = f"{hex(id(self))}-{len(_ALL_PLOTTERS)}"
        _ALL_PLOTTERS[self._id_name] = self

        # Key bindings
        self.reset_key_events()
        log.debug('BasePlotter init stop')

        self._image_depth_null = None
        self.last_image_depth = None
        self.last_image = None
        self._has_background_layer = False

        # set hidden line removal based on theme
        if self.theme.hidden_line_removal:
            self.enable_hidden_line_removal()

        self._initialized = True

    @property
    def theme(self):
        """Return or set the theme used for this plotter.

        Examples
        --------
        Use the dark theme for a plotter.

        >>> import pyvista
        >>> from pyvista import themes
        >>> pl = pyvista.Plotter()
        >>> pl.theme = themes.DarkTheme()
        >>> actor = pl.add_mesh(pyvista.Sphere())
        >>> pl.show()

        """
        return self._theme

    @theme.setter
    def theme(self, theme):
        if not isinstance(theme, pyvista.themes.DefaultTheme):
            raise TypeError(
                'Expected a pyvista theme like '
                '``pyvista.themes.DefaultTheme``, '
                f'not {type(theme).__name__}.'
            )
        self._theme.load_theme(theme)

    def import_gltf(self, filename, set_camera=True):
        """Import a glTF file into the plotter.

        See https://www.khronos.org/gltf/ for more information.

        Parameters
        ----------
        filename : str
            Path to the glTF file.

        set_camera : bool, optional
            Set the camera viewing angle to one compatible with the
            default three.js perspective (``'xy'``).

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> helmet_file = examples.gltf.download_damaged_helmet()  # doctest:+SKIP
        >>> texture = examples.hdr.download_dikhololo_night()  # doctest:+SKIP
        >>> pl = pyvista.Plotter()  # doctest:+SKIP
        >>> pl.import_gltf(helmet_file)  # doctest:+SKIP
        >>> pl.set_environment_texture(cubemap)  # doctest:+SKIP
        >>> pl.camera.zoom(1.8)  # doctest:+SKIP
        >>> pl.show()  # doctest:+SKIP

        See :ref:`load_gltf` for a full example using this method.

        """
        if not _vtk.VTK9:  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Support for glTF requires VTK v9 or newer')

        filename = os.path.abspath(os.path.expanduser(str(filename)))
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'Unable to locate {filename}')

        # lazy import here to avoid importing unused modules
        from vtkmodules.vtkIOImport import vtkGLTFImporter

        importer = vtkGLTFImporter()
        importer.SetFileName(filename)
        importer.SetRenderWindow(self.ren_win)
        importer.Update()

        # register last actor in actors
        actor = self.renderer.GetActors().GetLastItem()
        name = actor.GetAddressAsString("")
        self.renderer._actors[name] = actor

        # set camera position to a three.js viewing perspective
        if set_camera:
            self.camera_position = 'xy'

    def import_vrml(self, filename):
        """Import a VRML file into the plotter.

        Parameters
        ----------
        filename : str
            Path to the VRML file.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> sextant_file = examples.vrml.download_sextant()  # doctest:+SKIP
        >>> pl = pyvista.Plotter()  # doctest:+SKIP
        >>> pl.import_vrml(sextant_file)  # doctest:+SKIP
        >>> pl.show()  # doctest:+SKIP

        See :ref:`load_vrml_example` for a full example using this method.

        """
        filename = os.path.abspath(os.path.expanduser(str(filename)))
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'Unable to locate {filename}')

        # lazy import here to avoid importing unused modules
        importer = _vtk.lazy_vtkVRMLImporter()
        importer.SetFileName(filename)
        importer.SetRenderWindow(self.ren_win)
        importer.Update()

    def export_html(self, filename, backend='pythreejs'):
        """Export this plotter as an interactive scene to a HTML file.

        You have the option of exposing the scene using either vtk.js (using
        ``panel``) or three.js (using ``pythreejs``), both of which are
        excellent JavaScript libraries to visualize small to moderately complex
        scenes for scientific visualization.

        Parameters
        ----------
        filename : str
            Path to export the html file to.

        backend : str, optional
            One of the following:

            - ``'pythreejs'``
            - ``'panel'``

            For more details about the advantages and disadvantages of each
            backend, see :ref:`jupyter_plotting`.

        Notes
        -----
        You will need ``ipywidgets`` and ``pythreejs`` installed if you
        wish to export using the ``'pythreejs'`` backend, or ``'panel'``
        installed to export using ``'panel'``.

        Examples
        --------
        Export as a three.js scene using the pythreejs backend.

        >>> import pyvista
        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> pl = pyvista.Plotter(shape=(1,2))
        >>> _ = pl.add_mesh(mesh, scalars='Spatial Point Data', show_edges=True)
        >>> pl.subplot(0,1)
        >>> _ = pl.add_mesh(mesh, scalars='Spatial Cell Data', show_edges=True)
        >>> pl.export_html('pyvista.html')  # doctest:+SKIP

        Export as a vtk.js scene using the panel backend.

        >>> pl.export_html('pyvista_panel.html', backend='panel')  # doctest:+SKIP

        """
        if backend == 'pythreejs':
            widget = self.to_pythreejs()
        elif backend == 'panel':
            self._save_panel(filename)
            return
        else:
            raise ValueError(f"Invalid backend {backend}. Should be either 'panel' or 'pythreejs'")

        # import after converting as we check for pythreejs import first
        try:
            from ipywidgets.embed import dependency_state, embed_minimal_html
        except ImportError:  # pragma: no cover
            raise ImportError('Please install ipywidgets with:\n\n\tpip install ipywidgets')

        # Garbage collection for embedded html output:
        # https://github.com/jupyter-widgets/pythreejs/issues/217
        state = dependency_state(widget)

        # convert and write to file
        embed_minimal_html(filename, None, title=self.title, state=state)

    def _save_panel(self, filename):
        """Save the render window as a ``panel.pane.vtk`` html file.

        See https://panel.holoviz.org/api/panel.pane.vtk.html

        Parameters
        ----------
        filename : str
            Path to export the plotter as a panel scene to.

        """
        from ..jupyter.notebook import handle_plotter

        pane = handle_plotter(self, backend='panel', return_viewer=True, title=self.title)
        pane.save(filename)

    def to_pythreejs(self):
        """Convert this plotting scene to a pythreejs widget.

        Returns
        -------
        ipywidgets.Widget
            Widget containing pythreejs renderer.

        """
        self._on_first_render_request()  # set up camera
        from pyvista.jupyter.pv_pythreejs import convert_plotter

        return convert_plotter(self)

    def export_gltf(self, filename, inline_data=True, rotate_scene=True, save_normals=True):
        """Export the current rendering scene as a glTF file.

        Visit https://gltf-viewer.donmccurdy.com/ for an online viewer.

        See https://vtk.org/doc/nightly/html/classvtkGLTFExporter.html
        for limitations regarding the exporter.

        Parameters
        ----------
        filename : str
            Path to export the gltf file to.

        inline_data : bool, optional
            Sets if the binary data be included in the json file as a
            base64 string.  When ``True``, only one file is exported.

        rotate_scene : bool, optional
            Rotate scene to be compatible with the glTF specifications.

        save_normals : bool, optional
            Saves the point array ``'Normals'`` as ``'NORMAL'`` in
            the outputted scene.

        Notes
        -----
        The VTK exporter only supports :class:`pyvista.PolyData` datasets. If
        the plotter contains any non-PolyData datasets, these will be converted
        in the plotter, leading to a copy of the data internally.

        Examples
        --------
        Output a simple point cloud represented as balls.

        >>> import numpy as np
        >>> import pyvista
        >>> point_cloud = np.random.random((100, 3))
        >>> pdata = pyvista.PolyData(point_cloud)
        >>> pdata['orig_sphere'] = np.arange(100)
        >>> sphere = pyvista.Sphere(radius=0.02)
        >>> pc = pdata.glyph(scale=False, geom=sphere, orient=False)
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(pc, cmap='reds', smooth_shading=True,
        ...                 show_scalar_bar=False)
        >>> pl.export_gltf('balls.gltf')  # doctest:+SKIP
        >>> pl.show()

        Output the orientation plotter.

        >>> from pyvista import demos
        >>> pl = demos.orientation_plotter()
        >>> pl.export_gltf('orientation_plotter.gltf')  # doctest:+SKIP
        >>> pl.show()

        """
        if not _vtk.VTK9:  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Support for glTF requires VTK v9 or newer')

        if not hasattr(self, "ren_win"):
            raise RuntimeError('This plotter has been closed and is unable to export the scene.')

        from vtkmodules.vtkIOExport import vtkGLTFExporter

        # rotate scene to gltf compatible view
        renamed_arrays = []  # any renamed normal arrays
        if rotate_scene:
            for renderer in self.renderers:
                for actor in renderer.actors.values():
                    if hasattr(actor, 'RotateX'):
                        actor.RotateX(-90)
                        actor.RotateZ(-90)

                    if save_normals:
                        try:
                            mapper = actor.GetMapper()
                            if mapper is None:
                                continue
                            dataset = mapper.dataset
                            if not isinstance(dataset, pyvista.PolyData):
                                warnings.warn(
                                    'Plotter contains non-PolyData datasets. These have been '
                                    'overwritten with PolyData surfaces and are internally '
                                    'copies of the original datasets.'
                                )

                                try:
                                    dataset = dataset.extract_surface()
                                    mapper.SetInputData(dataset)
                                except:  # pragma: no cover
                                    warnings.warn(
                                        'During gLTF export, failed to convert some '
                                        'datasets to PolyData. Exported scene will not have '
                                        'all datasets.'
                                    )

                            if 'Normals' in dataset.point_data:
                                # By default VTK uses the 'Normals' point data for normals
                                # but gLTF uses NORMAL.
                                point_data = dataset.GetPointData()
                                array = point_data.GetArray('Normals')
                                array.SetName('NORMAL')
                                renamed_arrays.append(array)

                        except:  # noqa: E722
                            pass

        exporter = vtkGLTFExporter()
        exporter.SetRenderWindow(self.ren_win)
        exporter.SetFileName(filename)
        exporter.SetInlineData(inline_data)
        exporter.SetSaveNormal(save_normals)
        exporter.Update()

        # rotate back if applicable
        if rotate_scene:
            for renderer in self.renderers:
                for actor in renderer.actors.values():
                    if hasattr(actor, 'RotateX'):
                        actor.RotateZ(90)
                        actor.RotateX(90)

        # revert any renamed arrays
        for array in renamed_arrays:
            array.SetName('Normals')

    def export_vrml(self, filename):
        """Export the current rendering scene as a VRML file.

        See `vtk.VRMLExporter <https://vtk.org/doc/nightly/html/classvtkVRMLExporter.html>`_
        for limitations regarding the exporter.

        Parameters
        ----------
        filename : str
            Filename to export the scene to.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(examples.load_hexbeam())
        >>> pl.export_vrml("sample")  # doctest:+SKIP

        """
        if not hasattr(self, "ren_win"):
            raise RuntimeError("This plotter has been closed and cannot be shown.")

        exporter = _vtk.lazy_vtkVRMLExporter()
        exporter.SetFileName(filename)
        exporter.SetRenderWindow(self.ren_win)
        exporter.Write()

    def enable_hidden_line_removal(self, all_renderers=True):
        """Enable hidden line removal.

        Wireframe geometry will be drawn using hidden line removal if
        the rendering engine supports it.

        Disable this with :func:`disable_hidden_line_removal
        <BasePlotter.disable_hidden_line_removal>`

        Parameters
        ----------
        all_renderers : bool
            If ``True``, applies to all renderers in subplots. If
            ``False``, then only applies to the active renderer.

        Examples
        --------
        Create a side-by-side plotter and render a sphere in wireframe
        with hidden line removal enabled on the left and disabled on
        the right.

        >>> import pyvista
        >>> sphere = pyvista.Sphere(theta_resolution=20, phi_resolution=20)
        >>> pl = pyvista.Plotter(shape=(1, 2))
        >>> _ = pl.add_mesh(sphere, line_width=3, style='wireframe')
        >>> _ = pl.add_text("With hidden line removal")
        >>> pl.enable_hidden_line_removal(all_renderers=False)
        >>> pl.subplot(0, 1)
        >>> pl.disable_hidden_line_removal(all_renderers=False)
        >>> _ = pl.add_mesh(sphere, line_width=3, style='wireframe')
        >>> _ = pl.add_text("Without hidden line removal")
        >>> pl.show()

        """
        if all_renderers:
            for renderer in self.renderers:
                renderer.enable_hidden_line_removal()
        else:
            self.renderer.enable_hidden_line_removal()

    def disable_hidden_line_removal(self, all_renderers=True):
        """Disable hidden line removal.

        Enable again with :func:`enable_hidden_line_removal
        <BasePlotter.enable_hidden_line_removal>`

        Parameters
        ----------
        all_renderers : bool
            If ``True``, applies to all renderers in subplots. If
            ``False``, then only applies to the active renderer.

        Examples
        --------
        Enable and then disable hidden line removal.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.enable_hidden_line_removal()
        >>> pl.disable_hidden_line_removal()

        """
        if all_renderers:
            for renderer in self.renderers:
                renderer.disable_hidden_line_removal()
        else:
            self.renderer.disable_hidden_line_removal()

    @property
    def scalar_bar(self):
        """First scalar bar.  Kept for backwards compatibility."""
        return list(self.scalar_bars.values())[0]

    @property
    def scalar_bars(self):
        """Scalar bars.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere['Data'] = sphere.points[:, 2]
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(sphere)
        >>> plotter.scalar_bars
        Scalar Bar Title     Interactive
        "Data"               False

        Select a scalar bar actor based on the title of the bar.

        >>> plotter.scalar_bars['Data']  # doctest:+SKIP
        (vtkmodules.vtkRenderingAnnotation.vtkScalarBarActor)0x7fcd3567ca00

        """
        return self._scalar_bars

    @property
    def _before_close_callback(self):
        """Return the cached function (expecting a reference)."""
        if self.__before_close_callback is not None:
            return self.__before_close_callback()

    @_before_close_callback.setter
    def _before_close_callback(self, func):
        """Store a weakref.ref of the function being called."""
        if func is not None:
            self.__before_close_callback = weakref.ref(func)
        else:
            self.__before_close_callback = None

    @property
    def shape(self):
        """Shape of the plotter.

        Examples
        --------
        Return the plotter shape.

        >>> import pyvista
        >>> plotter = pyvista.Plotter(shape=(2, 2))
        >>> plotter.shape
        (2, 2)
        """
        return self.renderers._shape

    @property
    def renderer(self):
        """Return the active renderer.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.renderer  # doctest:+SKIP
        (Renderer)0x7f916129bfa0

        """
        return self.renderers.active_renderer

    @property
    def store_image(self):
        """Store last rendered frame on close.

        This is normally disabled to avoid caching the image, and is
        enabled by default by setting:

        ``pyvista.BUILDING_GALLERY = True``

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter(off_screen=True)
        >>> pl.store_image = True
        >>> _ = pl.add_mesh(pyvista.Cube())
        >>> pl.show()
        >>> image = pl.last_image
        >>> type(image)  # doctest:+SKIP
        <class 'numpy.ndarray'>

        """
        return self._store_image

    @store_image.setter
    def store_image(self, value):
        """Store last rendered frame on close."""
        self._store_image = bool(value)

    def subplot(self, index_row, index_column=None):
        """Set the active subplot.

        Parameters
        ----------
        index_row : int
            Index of the subplot to activate along the rows.

        index_column : int
            Index of the subplot to activate along the columns.

        Examples
        --------
        Create a 2 wide plot and set the background of right-hand plot
        to orange.  Add a cube to the left plot and a sphere to the
        right.

        >>> import pyvista
        >>> pl = pyvista.Plotter(shape=(1, 2))
        >>> actor = pl.add_mesh(pyvista.Cube())
        >>> pl.subplot(0, 1)
        >>> actor = pl.add_mesh(pyvista.Sphere())
        >>> pl.set_background('orange', all_renderers=False)
        >>> pl.show()

        """
        self.renderers.set_active_renderer(index_row, index_column)

    @wraps(Renderer.add_legend)
    def add_legend(self, *args, **kwargs):
        """Wrap ``Renderer.add_legend``."""
        return self.renderer.add_legend(*args, **kwargs)

    @wraps(Renderer.remove_legend)
    def remove_legend(self, *args, **kwargs):
        """Wrap ``Renderer.remove_legend``."""
        return self.renderer.remove_legend(*args, **kwargs)

    @property
    def legend(self):
        """Legend actor.

        There can only be one legend actor per renderer.  If
        ``legend`` is ``None``, there is no legend actor.

        """
        return self.renderer.legend

    @wraps(Renderer.add_floor)
    def add_floor(self, *args, **kwargs):
        """Wrap ``Renderer.add_floor``."""
        return self.renderer.add_floor(*args, **kwargs)

    @wraps(Renderer.remove_floors)
    def remove_floors(self, *args, **kwargs):
        """Wrap ``Renderer.remove_floors``."""
        return self.renderer.remove_floors(*args, **kwargs)

    def enable_3_lights(self, only_active=False):
        """Enable 3-lights illumination.

        This will replace all pre-existing lights in the scene.

        Parameters
        ----------
        only_active : bool
            If ``True``, only change the active renderer. The default
            is that every renderer is affected.

        Examples
        --------
        >>> from pyvista import demos
        >>> pl = demos.orientation_plotter()
        >>> pl.enable_3_lights()
        >>> pl.show()

        Note how this varies from the default plotting.

        >>> pl = demos.orientation_plotter()
        >>> pl.show()

        """

        def _to_pos(elevation, azimuth):
            theta = azimuth * np.pi / 180.0
            phi = (90.0 - elevation) * np.pi / 180.0
            x = np.sin(theta) * np.sin(phi)
            y = np.cos(phi)
            z = np.cos(theta) * np.sin(phi)
            return x, y, z

        renderers = [self.renderer] if only_active else self.renderers
        for renderer in renderers:
            renderer.remove_all_lights()

        # Inspired from Mayavi's version of Raymond Maple 3-lights illumination
        intensities = [1, 0.6, 0.5]
        all_angles = [(45.0, 45.0), (-30.0, -60.0), (-30.0, 60.0)]
        for intensity, angles in zip(intensities, all_angles):
            light = pyvista.Light(light_type='camera light')
            light.intensity = intensity
            light.position = _to_pos(*angles)
            for renderer in renderers:
                renderer.add_light(light)

    def disable_3_lights(self):
        """Please use ``enable_lightkit``, this method has been depreciated."""
        from pyvista.core.errors import DeprecationError

        raise DeprecationError('DEPRECATED: Please use ``enable_lightkit``')

    def enable_lightkit(self, only_active=False):
        """Enable the default light-kit lighting.

        See:
        https://www.researchgate.net/publication/2926068_LightKit_A_lighting_system_for_effective_visualization

        This will replace all pre-existing lights in the renderer.

        Parameters
        ----------
        only_active : bool
            If ``True``, only change the active renderer. The default is that
            every renderer is affected.

        Examples
        --------
        Create a plotter without any lights and then enable the
        default light kit.

        >>> import pyvista
        >>> pl = pyvista.Plotter(lighting=None)
        >>> pl.enable_lightkit()
        >>> actor = pl.add_mesh(pyvista.Cube(), show_edges=True)
        >>> pl.show()

        """
        renderers = [self.renderer] if only_active else self.renderers

        light_kit = _vtk.vtkLightKit()
        for renderer in renderers:
            renderer.remove_all_lights()
            # Use the renderer as a vtkLightKit parser.
            # Feed it the LightKit, pop off the vtkLights, put back
            # pyvista Lights. This is the price we must pay for using
            # inheritance rather than composition.
            light_kit.AddLightsToRenderer(renderer)
            vtk_lights = renderer.lights
            renderer.remove_all_lights()
            for vtk_light in vtk_lights:
                light = pyvista.Light.from_vtk(vtk_light)
                renderer.add_light(light)
            renderer.LightFollowCameraOn()

    def enable_anti_aliasing(self, aa_type='fxaa', multi_samples=None, all_renderers=True):
        """Enable anti-aliasing.

        This tends to make edges appear softer and less pixelated.

        Parameters
        ----------
        aa_type : str, optional
            Anti-aliasing type. See the notes below. One of the following:

            * ``"ssaa"`` - Super-Sample Anti-Aliasing
            * ``"msaa"`` - Multi-Sample Anti-Aliasing
            * ``"fxaa"`` - Fast Approximate Anti-Aliasing

        multi_samples : int, optional
            The number of multi-samples when ``aa_type`` is ``"msaa"``. Note
            that using this setting automatically enables this for all
            renderers. Defaults to the theme multi_samples.

        all_renderers : bool
            If ``True``, applies to all renderers in subplots. If
            ``False``, then only applies to the active renderer.

        Notes
        -----
        SSAA, or Super-Sample Anti-Aliasing is a brute force method of
        anti-aliasing. It results in the best image quality but comes at a
        tremendous resource cost. SSAA works by rendering the scene at a higher
        resolution. The final image is produced by downsampling the
        massive source image using an averaging filter. This acts as a low pass
        filter which removes the high frequency components that would cause
        jaggedness.

        MSAA, or Multi-Sample Anti-Aliasing is an optimization of SSAA that
        reduces the amount of pixel shader evaluations that need to be computed
        by focusing on overlapping regions of the scene. The result is
        anti-aliasing along edges that is on par with SSAA and less
        anti-aliasing along surfaces as these make up the bulk of SSAA
        computations. MSAA is substantially less computationally expensive than
        SSAA and results in comparable image quality.

        FXAA, or Fast Approximate Anti-Aliasing is an Anti-Aliasing technique
        that is performed entirely in post processing. FXAA operates on the
        rasterized image rather than the scene geometry. As a consequence,
        forcing FXAA or using FXAA incorrectly can result in the FXAA filter
        smoothing out parts of the visual overlay that are usually kept sharp
        for reasons of clarity as well as smoothing out textures. FXAA is
        inferior to MSAA but is almost free computationally and is thus
        desirable on low end platforms.

        Examples
        --------
        Enable super-sample anti-aliasing (SSAA).

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.enable_anti_aliasing('ssaa')
        >>> _ = pl.add_mesh(pyvista.Sphere(), show_edges=True)
        >>> pl.show()

        See :ref:`anti_aliasing_example` for a full example demonstrating
        VTK's anti-aliasing approaches.

        """
        # apply MSAA to entire render window
        if aa_type == 'msaa':
            if not hasattr(self, 'ren_win'):
                raise AttributeError('The render window has been closed.')
            if multi_samples is None:
                multi_samples = self._theme.multi_samples
            self.ren_win.SetMultiSamples(multi_samples)
            return
        elif aa_type not in ['ssaa', 'fxaa']:
            raise ValueError(
                f'Invalid `aa_type` "{aa_type}". Should be either "fxaa", "ssaa", or "msaa"'
            )

        if all_renderers:
            for renderer in self.renderers:
                renderer.enable_anti_aliasing(aa_type)
        else:
            self.renderer.enable_anti_aliasing(aa_type)

    def disable_anti_aliasing(self, all_renderers=True):
        """Disable anti-aliasing.

        Parameters
        ----------
        all_renderers : bool
            If ``True``, applies to all renderers in subplots. If ``False``,
            then only applies to the active renderer.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.disable_anti_aliasing()
        >>> _ = pl.add_mesh(pyvista.Sphere(), show_edges=True)
        >>> pl.show()

        See :ref:`anti_aliasing_example` for a full example demonstrating
        VTK's anti-aliasing approaches.

        """
        self.ren_win.SetMultiSamples(0)

        if all_renderers:
            for renderer in self.renderers:
                renderer.disable_anti_aliasing()
        else:
            self.renderer.disable_anti_aliasing()

    @wraps(Renderer.set_focus)
    def set_focus(self, *args, render=True, **kwargs):
        """Wrap ``Renderer.set_focus``."""
        log.debug('set_focus: %s, %s', str(args), str(kwargs))
        self.renderer.set_focus(*args, **kwargs)
        if render:
            self.render()

    @wraps(Renderer.set_position)
    def set_position(self, *args, render=True, **kwargs):
        """Wrap ``Renderer.set_position``."""
        self.renderer.set_position(*args, **kwargs)
        if render:
            self.render()

    @wraps(Renderer.set_viewup)
    def set_viewup(self, *args, render=True, **kwargs):
        """Wrap ``Renderer.set_viewup``."""
        self.renderer.set_viewup(*args, **kwargs)
        if render:
            self.render()

    @wraps(Renderer.add_orientation_widget)
    def add_orientation_widget(self, *args, **kwargs):
        """Wrap ``Renderer.add_orientation_widget``."""
        return self.renderer.add_orientation_widget(*args, **kwargs)

    @wraps(Renderer.add_axes)
    def add_axes(self, *args, **kwargs):
        """Wrap ``Renderer.add_axes``."""
        return self.renderer.add_axes(*args, **kwargs)

    @wraps(Renderer.hide_axes)
    def hide_axes(self, *args, **kwargs):
        """Wrap ``Renderer.hide_axes``."""
        return self.renderer.hide_axes(*args, **kwargs)

    @wraps(Renderer.show_axes)
    def show_axes(self, *args, **kwargs):
        """Wrap ``Renderer.show_axes``."""
        return self.renderer.show_axes(*args, **kwargs)

    @wraps(Renderer.update_bounds_axes)
    def update_bounds_axes(self, *args, **kwargs):
        """Wrap ``Renderer.update_bounds_axes``."""
        return self.renderer.update_bounds_axes(*args, **kwargs)

    @wraps(Renderer.add_chart)
    def add_chart(self, *args, **kwargs):
        """Wrap ``Renderer.add_chart``."""
        return self.renderer.add_chart(*args, **kwargs)

    @wraps(Renderer.remove_chart)
    def remove_chart(self, *args, **kwargs):
        """Wrap ``Renderer.remove_chart``."""
        return self.renderer.remove_chart(*args, **kwargs)

    @wraps(Renderer.add_actor)
    def add_actor(self, *args, **kwargs):
        """Wrap ``Renderer.add_actor``."""
        return self.renderer.add_actor(*args, **kwargs)

    @wraps(Renderer.enable_parallel_projection)
    def enable_parallel_projection(self, *args, **kwargs):
        """Wrap ``Renderer.enable_parallel_projection``."""
        return self.renderer.enable_parallel_projection(*args, **kwargs)

    @wraps(Renderer.disable_parallel_projection)
    def disable_parallel_projection(self, *args, **kwargs):
        """Wrap ``Renderer.disable_parallel_projection``."""
        return self.renderer.disable_parallel_projection(*args, **kwargs)

    @wraps(Renderer.enable_ssao)
    def enable_ssao(self, *args, **kwargs):
        """Wrap ``Renderer.enable_ssao``."""
        return self.renderer.enable_ssao(*args, **kwargs)

    @wraps(Renderer.disable_ssao)
    def disable_ssao(self, *args, **kwargs):
        """Wrap ``Renderer.disable_ssao``."""
        return self.renderer.disable_ssao(*args, **kwargs)

    @wraps(Renderer.enable_shadows)
    def enable_shadows(self, *args, **kwargs):
        """Wrap ``Renderer.enable_shadows``."""
        return self.renderer.enable_shadows(*args, **kwargs)

    @wraps(Renderer.disable_shadows)
    def disable_shadows(self, *args, **kwargs):
        """Wrap ``Renderer.disable_shadows``."""
        return self.renderer.disable_shadows(*args, **kwargs)

    @property
    def parallel_projection(self):
        """Return parallel projection state of active render window."""
        return self.renderer.parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, state):
        """Set parallel projection state of all active render windows."""
        self.renderer.parallel_projection = state

    @property
    def parallel_scale(self):
        """Return parallel scale of active render window."""
        return self.renderer.parallel_scale

    @parallel_scale.setter
    def parallel_scale(self, value):
        """Set parallel scale of all active render windows."""
        self.renderer.parallel_scale = value

    @wraps(Renderer.add_axes_at_origin)
    def add_axes_at_origin(self, *args, **kwargs):
        """Wrap ``Renderer.add_axes_at_origin``."""
        return self.renderer.add_axes_at_origin(*args, **kwargs)

    @wraps(Renderer.show_bounds)
    def show_bounds(self, *args, **kwargs):
        """Wrap ``Renderer.show_bounds``."""
        return self.renderer.show_bounds(*args, **kwargs)

    @wraps(Renderer.add_bounding_box)
    def add_bounding_box(self, *args, **kwargs):
        """Wrap ``Renderer.add_bounding_box``."""
        return self.renderer.add_bounding_box(*args, **kwargs)

    @wraps(Renderer.remove_bounding_box)
    def remove_bounding_box(self, *args, **kwargs):
        """Wrap ``Renderer.remove_bounding_box``."""
        return self.renderer.remove_bounding_box(*args, **kwargs)

    @wraps(Renderer.remove_bounds_axes)
    def remove_bounds_axes(self, *args, **kwargs):
        """Wrap ``Renderer.remove_bounds_axes``."""
        return self.renderer.remove_bounds_axes(*args, **kwargs)

    @wraps(Renderer.show_grid)
    def show_grid(self, *args, **kwargs):
        """Wrap ``Renderer.show_grid``."""
        return self.renderer.show_grid(*args, **kwargs)

    @wraps(Renderer.set_scale)
    def set_scale(self, *args, **kwargs):
        """Wrap ``Renderer.set_scale``."""
        return self.renderer.set_scale(*args, **kwargs)

    @wraps(Renderer.enable_depth_of_field)
    def enable_depth_of_field(self, *args, **kwargs):
        """Wrap ``Renderer.enable_depth_of_field``."""
        return self.renderer.enable_depth_of_field(*args, **kwargs)

    @wraps(Renderer.disable_depth_of_field)
    def disable_depth_of_field(self, *args, **kwargs):
        """Wrap ``Renderer.disable_depth_of_field``."""
        return self.renderer.disable_depth_of_field(*args, **kwargs)

    @wraps(Renderer.add_blurring)
    def add_blurring(self, *args, **kwargs):
        """Wrap ``Renderer.add_blurring``."""
        return self.renderer.add_blurring(*args, **kwargs)

    @wraps(Renderer.remove_blurring)
    def remove_blurring(self, *args, **kwargs):
        """Wrap ``Renderer.remove_blurring``."""
        return self.renderer.remove_blurring(*args, **kwargs)

    @wraps(Renderer.enable_eye_dome_lighting)
    def enable_eye_dome_lighting(self, *args, **kwargs):
        """Wrap ``Renderer.enable_eye_dome_lighting``."""
        return self.renderer.enable_eye_dome_lighting(*args, **kwargs)

    @wraps(Renderer.disable_eye_dome_lighting)
    def disable_eye_dome_lighting(self, *args, **kwargs):
        """Wrap ``Renderer.disable_eye_dome_lighting``."""
        self.renderer.disable_eye_dome_lighting(*args, **kwargs)

    @wraps(Renderer.reset_camera)
    def reset_camera(self, *args, **kwargs):
        """Wrap ``Renderer.reset_camera``."""
        self.renderer.reset_camera(*args, **kwargs)
        self.render()

    @wraps(Renderer.isometric_view)
    def isometric_view(self, *args, **kwargs):
        """Wrap ``Renderer.isometric_view``."""
        self.renderer.isometric_view(*args, **kwargs)

    @wraps(Renderer.view_isometric)
    def view_isometric(self, *args, **kwarg):
        """Wrap ``Renderer.view_isometric``."""
        self.renderer.view_isometric(*args, **kwarg)

    @wraps(Renderer.view_vector)
    def view_vector(self, *args, **kwarg):
        """Wrap ``Renderer.view_vector``."""
        self.renderer.view_vector(*args, **kwarg)

    @wraps(Renderer.view_xy)
    def view_xy(self, *args, **kwarg):
        """Wrap ``Renderer.view_xy``."""
        self.renderer.view_xy(*args, **kwarg)

    @wraps(Renderer.view_yx)
    def view_yx(self, *args, **kwarg):
        """Wrap ``Renderer.view_yx``."""
        self.renderer.view_yx(*args, **kwarg)

    @wraps(Renderer.view_xz)
    def view_xz(self, *args, **kwarg):
        """Wrap ``Renderer.view_xz``."""
        self.renderer.view_xz(*args, **kwarg)

    @wraps(Renderer.view_zx)
    def view_zx(self, *args, **kwarg):
        """Wrap ``Renderer.view_zx``."""
        self.renderer.view_zx(*args, **kwarg)

    @wraps(Renderer.view_yz)
    def view_yz(self, *args, **kwarg):
        """Wrap ``Renderer.view_yz``."""
        self.renderer.view_yz(*args, **kwarg)

    @wraps(Renderer.view_zy)
    def view_zy(self, *args, **kwarg):
        """Wrap ``Renderer.view_zy``."""
        self.renderer.view_zy(*args, **kwarg)

    @wraps(Renderer.disable)
    def disable(self, *args, **kwarg):
        """Wrap ``Renderer.disable``."""
        self.renderer.disable(*args, **kwarg)

    @wraps(Renderer.enable)
    def enable(self, *args, **kwarg):
        """Wrap ``Renderer.enable``."""
        self.renderer.enable(*args, **kwarg)

    @wraps(Renderer.enable_depth_peeling)
    def enable_depth_peeling(self, *args, **kwargs):
        """Wrap ``Renderer.enable_depth_peeling``."""
        if hasattr(self, 'ren_win'):
            result = self.renderer.enable_depth_peeling(*args, **kwargs)
            if result:
                self.ren_win.AlphaBitPlanesOn()
        return result

    @wraps(Renderer.disable_depth_peeling)
    def disable_depth_peeling(self):
        """Wrap ``Renderer.disable_depth_peeling``."""
        if hasattr(self, 'ren_win'):
            self.ren_win.AlphaBitPlanesOff()
            return self.renderer.disable_depth_peeling()

    @wraps(Renderer.get_default_cam_pos)
    def get_default_cam_pos(self, *args, **kwargs):
        """Wrap ``Renderer.get_default_cam_pos``."""
        return self.renderer.get_default_cam_pos(*args, **kwargs)

    @wraps(Renderer.remove_actor)
    def remove_actor(self, *args, **kwargs):
        """Wrap ``Renderer.remove_actor``."""
        for renderer in self.renderers:
            renderer.remove_actor(*args, **kwargs)
        return True

    @wraps(Renderer.set_environment_texture)
    def set_environment_texture(self, *args, **kwargs):
        """Wrap ``Renderer.set_environment_texture``."""
        return self.renderer.set_environment_texture(*args, **kwargs)

    @wraps(Renderer.remove_environment_texture)
    def remove_environment_texture(self, *args, **kwargs):
        """Wrap ``Renderer.remove_environment_texture``."""
        return self.renderer.remove_environment_texture(*args, **kwargs)

    #### Properties from Renderer ####

    @property
    def camera(self):
        """Return the active camera of the active renderer."""
        if not self.camera_set:
            self.camera_position = self.get_default_cam_pos()
            self.reset_camera()
            self.camera_set = True
        return self.renderer.camera

    @camera.setter
    def camera(self, camera):
        """Set the active camera for the rendering scene."""
        self.renderer.camera = camera

    @property
    def camera_set(self):
        """Return if the camera of the active renderer has been set."""
        return self.renderer.camera_set

    @camera_set.setter
    def camera_set(self, is_set):
        """Set if the camera has been set on the active renderer."""
        self.renderer.camera_set = is_set

    @property
    def bounds(self):
        """Return the bounds of the active renderer.

        Returns
        -------
        list
            Bounds of the active renderer.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(pyvista.Cube())
        >>> pl.bounds
        [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]

        """
        return self.renderer.bounds

    @property
    def length(self):
        """Return the length of the diagonal of the bounding box of the scene."""
        return self.renderer.length

    @property
    def center(self):
        """Return the center of the active renderer."""
        return self.renderer.center

    @property
    def _scalar_bar_slots(self):
        """Return the scalar bar slots of the active renderer."""
        return self.renderer._scalar_bar_slots

    @_scalar_bar_slots.setter
    def _scalar_bar_slots(self, value):
        """Set the scalar bar slots of the active renderer."""
        self.renderer._scalar_bar_slots = value

    @property
    def _scalar_bar_slot_lookup(self):
        """Return the scalar bar slot lookup of the active renderer."""
        return self.renderer._scalar_bar_slot_lookup

    @_scalar_bar_slot_lookup.setter
    def _scalar_bar_slot_lookup(self, value):
        """Set the scalar bar slot lookup of the active renderer."""
        self.renderer._scalar_bar_slot_lookup = value

    @property
    def scale(self):
        """Return the scaling of the active renderer."""
        return self.renderer.scale

    @scale.setter
    def scale(self, scale):
        """Set the scaling of the active renderer."""
        self.renderer.set_scale(*scale)

    @property
    def camera_position(self):
        """Return camera position of the active render window.

        Examples
        --------
        Return camera's position and then reposition it via a list of tuples.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_bunny_coarse()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True, reset_camera=True)
        >>> pl.camera_position
        [(0.02430, 0.0336, 0.9446),
         (0.02430, 0.0336, -0.02225),
         (0.0, 1.0, 0.0)]
        >>> pl.camera_position = [
        ...     (0.3914, 0.4542, 0.7670),
        ...     (0.0243, 0.0336, -0.0222),
        ...     (-0.2148, 0.8998, -0.3796),
        ... ]
        >>> pl.show()

        Set the camera position using a string and look at the ``'xy'`` plane.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>> pl.camera_position = 'xy'
        >>> pl.show()

        Set the camera position using a string and look at the ``'zy'`` plane.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>> pl.camera_position = 'zy'
        >>> pl.show()

        For more examples, see :ref:`cameras_api`.

        """
        return self.renderer.camera_position

    @camera_position.setter
    def camera_position(self, camera_location):
        """Set camera position of the active render window."""
        self.renderer.camera_position = camera_location

    @property
    def background_color(self):
        """Return the background color of the active render window.

        Examples
        --------
        Set the background color to ``"pink"`` and plot it.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Cube(), show_edges=True)
        >>> pl.background_color = "pink"
        >>> pl.background_color
        Color(name='pink', hex='#ffc0cbff')
        >>> pl.show()

        """
        return self.renderers.active_renderer.background_color

    @background_color.setter
    def background_color(self, color):
        """Set the background color of all the render windows."""
        self.set_background(color)

    @property
    def window_size(self):
        """Return the render window size in ``(width, height)``.

        Examples
        --------
        Change the window size from ``200 x 200`` to ``400 x 400``.

        >>> import pyvista
        >>> pl = pyvista.Plotter(window_size=[200, 200])
        >>> pl.window_size
        [200, 200]
        >>> pl.window_size = [400, 400]
        >>> pl.window_size
        [400, 400]

        """
        return list(self.ren_win.GetSize())

    @window_size.setter
    def window_size(self, window_size):
        """Set the render window size."""
        self.ren_win.SetSize(window_size[0], window_size[1])

    @property
    def image_depth(self):
        """Return a depth image representing current render window.

        Helper attribute for ``get_image_depth``.

        """
        return self.get_image_depth()

    def _check_rendered(self):
        """Check if the render window has been shown and raise an exception if not."""
        if not self._rendered:
            raise AttributeError(
                '\nThis plotter has not yet been set up and rendered '
                'with ``show()``.\n'
                'Consider setting ``off_screen=True`` '
                'for off screen rendering.\n'
            )

    def _check_has_ren_win(self):
        """Check if render window attribute exists and raise an exception if not."""
        if not hasattr(self, 'ren_win'):
            raise AttributeError(
                '\n\nTo retrieve an image after the render window '
                'has been closed, set:\n\n'
                ' ``plotter.store_image = True``\n\n'
                'before closing the plotter.'
            )

    @property
    def image(self):
        """Return an image array of current render window.

        To retrieve an image after the render window has been closed,
        set: ``plotter.store_image = True`` before closing the plotter.
        """
        if not hasattr(self, 'ren_win') and self.last_image is not None:
            return self.last_image

        self._check_rendered()
        self._check_has_ren_win()

        data = image_from_window(self.ren_win)
        if self.image_transparent_background:
            return data

        # ignore alpha channel
        return data[:, :, :-1]

    def render(self):
        """Render the main window.

        Does nothing until ``show`` has been called.
        """
        if hasattr(self, 'ren_win') and not self._first_time:
            log.debug('Rendering')
            self.ren_win.Render()
            self._rendered = True

    @wraps(RenderWindowInteractor.add_key_event)
    def add_key_event(self, *args, **kwargs):
        """Wrap RenderWindowInteractor.add_key_event."""
        if hasattr(self, 'iren'):
            self.iren.add_key_event(*args, **kwargs)

    def clear_events_for_key(self, key):
        """Remove the callbacks associated to the key.

        Parameters
        ----------
        key : str
            Key to clear events for.

        """
        self.iren.clear_events_for_key(key)

    def store_mouse_position(self, *args):
        """Store mouse position."""
        if not hasattr(self, "iren"):
            raise AttributeError("This plotting window is not interactive.")
        self.mouse_position = self.iren.get_event_position()

    def store_click_position(self, *args):
        """Store click position in viewport coordinates."""
        if not hasattr(self, "iren"):
            raise AttributeError("This plotting window is not interactive.")
        self.click_position = self.iren.get_event_position()
        self.mouse_position = self.click_position

    def track_mouse_position(self):
        """Keep track of the mouse position.

        This will potentially slow down the interactor. No callbacks
        supported here - use
        :func:`pyvista.BasePlotter.track_click_position` instead.

        """
        self.iren.track_mouse_position(self.store_mouse_position)

    def untrack_mouse_position(self):
        """Stop tracking the mouse position."""
        self.iren.untrack_mouse_position()

    @wraps(RenderWindowInteractor.track_click_position)
    def track_click_position(self, *args, **kwargs):
        """Wrap RenderWindowInteractor.track_click_position."""
        self.iren.track_click_position(*args, **kwargs)

    @wraps(RenderWindowInteractor.untrack_click_position)
    def untrack_click_position(self, *args, **kwargs):
        """Stop tracking the click position."""
        self.iren.untrack_click_position(*args, **kwargs)

    @property
    def pickable_actors(self):
        """Return or set the pickable actors.

        When setting, this will be the list of actors to make
        pickable. All actors not in the list will be made unpickable.
        If ``actors`` is ``None``, all actors will be made unpickable.

        Returns
        -------
        list of vtk.vtkActors

        Examples
        --------
        Add two actors to a :class:`pyvista.Plotter`, make one
        pickable, and then list the pickable actors.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> sphere_actor = pl.add_mesh(pv.Sphere())
        >>> cube_actor = pl.add_mesh(pv.Cube(), pickable=False, style='wireframe')
        >>> len(pl.pickable_actors)
        1

        Set the pickable actors to both actors.

        >>> pl.pickable_actors = [sphere_actor, cube_actor]
        >>> len(pl.pickable_actors)
        2

        Set the pickable actors to ``None``.

        >>> pl.pickable_actors = None
        >>> len(pl.pickable_actors)
        0

        """
        pickable = []
        for renderer in self.renderers:
            for actor in renderer.actors.values():
                if actor.GetPickable():
                    pickable.append(actor)
        return pickable

    @pickable_actors.setter
    def pickable_actors(self, actors=None):
        """Set the pickable actors."""
        actors = [] if actors is None else actors
        if isinstance(actors, _vtk.vtkActor):
            actors = [actors]

        if not all([isinstance(actor, _vtk.vtkActor) for actor in actors]):
            raise TypeError(
                f'Expected a vtkActor instance or a list of vtkActors, got '
                f'{[type(actor) for actor in actors]} instead.'
            )

        for renderer in self.renderers:
            for actor in renderer.actors.values():
                actor.SetPickable(actor in actors)

    def _prep_for_close(self):
        """Make sure a screenshot is acquired before closing.

        This doesn't actually close anything! It just preps the plotter for
        closing.
        """
        # Grab screenshot right before renderer closes
        self.last_image = self.screenshot(True, return_img=True)
        self.last_image_depth = self.get_image_depth()

    def increment_point_size_and_line_width(self, increment):
        """Increment point size and line width of all actors.

        For every actor in the scene, increment both its point size
        and line width by the given value.

        Parameters
        ----------
        increment : float
            Amount to increment point size and line width.

        """
        for renderer in self.renderers:
            for actor in renderer._actors.values():
                if hasattr(actor, "GetProperty"):
                    prop = actor.GetProperty()
                    if hasattr(prop, "SetPointSize"):
                        prop.SetPointSize(prop.GetPointSize() + increment)
                    if hasattr(prop, "SetLineWidth"):
                        prop.SetLineWidth(prop.GetLineWidth() + increment)
        self.render()
        return

    def reset_key_events(self):
        """Reset all of the key press events to their defaults."""
        if hasattr(self, 'iren'):
            self.iren.clear_key_event_callbacks()

        self.add_key_event('q', self._prep_for_close)  # Add no matter what
        b_left_down_callback = lambda: self.iren.add_observer(
            'LeftButtonPressEvent', self.left_button_down
        )
        self.add_key_event('b', b_left_down_callback)
        self.add_key_event('v', lambda: self.isometric_view_interactive())
        self.add_key_event('C', lambda: self.enable_cell_picking())
        self.add_key_event('Up', lambda: self.camera.Zoom(1.05))
        self.add_key_event('Down', lambda: self.camera.Zoom(0.95))
        self.add_key_event('plus', lambda: self.increment_point_size_and_line_width(1))
        self.add_key_event('minus', lambda: self.increment_point_size_and_line_width(-1))

    @wraps(RenderWindowInteractor.key_press_event)
    def key_press_event(self, *args, **kwargs):
        """Wrap RenderWindowInteractor.key_press_event."""
        self.iren.key_press_event(*args, **kwargs)

    def left_button_down(self, obj, event_type):
        """Register the event for a left button down click."""
        if hasattr(self.ren_win, 'GetOffScreenFramebuffer'):
            if not self.ren_win.GetOffScreenFramebuffer().GetFBOIndex():
                # must raise a runtime error as this causes a segfault on VTK9
                raise ValueError('Invoking helper with no framebuffer')
        # Get 2D click location on window
        click_pos = self.iren.get_event_position()

        # Get corresponding click location in the 3D plot
        picker = _vtk.vtkWorldPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        self.pickpoint = np.asarray(picker.GetPickPosition()).reshape((-1, 3))
        if np.any(np.isnan(self.pickpoint)):
            self.pickpoint[:] = 0

    @wraps(RenderWindowInteractor.enable_trackball_style)
    def enable_trackball_style(self):
        """Wrap RenderWindowInteractor.enable_trackball_style."""
        self.iren.enable_trackball_style()

    @wraps(RenderWindowInteractor.enable_trackball_actor_style)
    def enable_trackball_actor_style(self):
        """Wrap RenderWindowInteractor.enable_trackball_actor_style."""
        self.iren.enable_trackball_actor_style()

    @wraps(RenderWindowInteractor.enable_image_style)
    def enable_image_style(self):
        """Wrap RenderWindowInteractor.enable_image_style."""
        self.iren.enable_image_style()

    @wraps(RenderWindowInteractor.enable_joystick_style)
    def enable_joystick_style(self):
        """Wrap RenderWindowInteractor.enable_joystick_style."""
        self.iren.enable_joystick_style()

    @wraps(RenderWindowInteractor.enable_joystick_actor_style)
    def enable_joystick_actor_style(self):
        """Wrap RenderWindowInteractor.enable_joystick_actor_style."""
        self.iren.enable_joystick_actor_style()

    @wraps(RenderWindowInteractor.enable_zoom_style)
    def enable_zoom_style(self):
        """Wrap RenderWindowInteractor.enable_zoom_style."""
        self.iren.enable_zoom_style()

    @wraps(RenderWindowInteractor.enable_terrain_style)
    def enable_terrain_style(self, *args, **kwargs):
        """Wrap RenderWindowInteractor.enable_terrain_style."""
        self.iren.enable_terrain_style(*args, **kwargs)

    @wraps(RenderWindowInteractor.enable_rubber_band_style)
    def enable_rubber_band_style(self):
        """Wrap RenderWindowInteractor.enable_rubber_band_style."""
        self.iren.enable_rubber_band_style()

    @wraps(RenderWindowInteractor.enable_rubber_band_2d_style)
    def enable_rubber_band_2d_style(self):
        """Wrap RenderWindowInteractor.enable_rubber_band_2d_style."""
        self.iren.enable_rubber_band_2d_style()

    def enable_stereo_render(self):
        """Enable anaglyph stereo rendering.

        Disable this with :func:`disable_stereo_render
        <BasePlotter.disable_stereo_render>`

        Examples
        --------
        Enable stereo rendering to show a cube as an anaglyph image.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Cube())
        >>> pl.enable_stereo_render()
        >>> pl.show()

        """
        if hasattr(self, 'ren_win'):
            self.ren_win.SetStereoTypeToAnaglyph()
            self.ren_win.StereoRenderOn()

    def disable_stereo_render(self):
        """Disable anaglyph stereo rendering.

        Enable again with :func:`enable_stereo_render
        <BasePlotter.enable_stereo_render>`

        Examples
        --------
        Enable and then disable stereo rendering. It should show a simple cube.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Cube())
        >>> pl.enable_stereo_render()
        >>> pl.disable_stereo_render()
        >>> pl.show()

        """
        if hasattr(self, 'ren_win'):
            self.ren_win.StereoRenderOff()

    def hide_axes_all(self):
        """Hide the axes orientation widget in all renderers."""
        for renderer in self.renderers:
            renderer.hide_axes()

    def show_axes_all(self):
        """Show the axes orientation widget in all renderers.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>>
        >>> # create multi-window plot (1 row, 2 columns)
        >>> pl = pyvista.Plotter(shape=(1, 2))
        >>>
        >>> # activate subplot 1 and add a mesh
        >>> pl.subplot(0, 0)
        >>> _ = pl.add_mesh(examples.load_globe())
        >>>
        >>> # activate subplot 2 and add a mesh
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(examples.load_airplane())
        >>>
        >>> # show the axes orientation widget in all subplots
        >>> pl.show_axes_all()
        >>>
        >>> # display the window
        >>> pl.show()

        """
        for renderer in self.renderers:
            renderer.show_axes()

    def isometric_view_interactive(self):
        """Set the current interactive render window to isometric view."""
        interactor = self.iren.get_interactor_style()
        renderer = interactor.GetCurrentRenderer()
        if renderer is None:
            renderer = self.renderer
        renderer.view_isometric()

    def update(self, stime=1, force_redraw=True):
        """Update window, redraw, process messages query.

        Parameters
        ----------
        stime : int, optional
            Duration of timer that interrupt vtkRenderWindowInteractor
            in milliseconds.

        force_redraw : bool, optional
            Call ``render`` immediately.

        """
        if stime <= 0:
            stime = 1

        curr_time = time.time()
        if Plotter.last_update_time > curr_time:
            Plotter.last_update_time = curr_time

        if self.iren is not None:
            update_rate = self.iren.get_desired_update_rate()
            if (curr_time - Plotter.last_update_time) > (1.0 / update_rate):
                self.right_timer_id = self.iren.create_repeating_timer(stime)
                self.render()
                Plotter.last_update_time = curr_time
                return

        if force_redraw:
            self.render()

    def add_composite(
        self,
        dataset,
        color=None,
        style=None,
        scalars=None,
        clim=None,
        show_edges=None,
        edge_color=None,
        point_size=5.0,
        line_width=None,
        opacity=1.0,
        flip_scalars=False,
        lighting=None,
        n_colors=256,
        interpolate_before_map=True,
        cmap=None,
        label=None,
        reset_camera=None,
        scalar_bar_args=None,
        show_scalar_bar=None,
        multi_colors=False,
        name=None,
        render_points_as_spheres=None,
        render_lines_as_tubes=False,
        smooth_shading=None,
        split_sharp_edges=None,
        ambient=0.0,
        diffuse=1.0,
        specular=0.0,
        specular_power=100.0,
        nan_color=None,
        nan_opacity=1.0,
        culling=None,
        rgb=None,
        categories=None,
        below_color=None,
        above_color=None,
        annotations=None,
        pickable=True,
        preference="point",
        log_scale=False,
        pbr=False,
        metallic=0.0,
        roughness=0.5,
        render=True,
        component=None,
        color_missing_with_nan=False,
        copy_mesh=False,
        **kwargs,
    ):
        """Add a composite dataset to the plotter.

        Parameters
        ----------
        dataset : pyvista.MultiBlock
            A :class:`pyvista.MultiBlock` dataset.

        color : color_like, default: :attr:`pyvista.themes.DefaultTheme.color`
            Use to make the entire mesh have a single solid color.
            Either a string, RGB list, or hex color string.  For example:
            ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
            ``color='#FFFFFF'``. Color will be overridden if scalars are
            specified.

        style : str, default: 'wireframe'
            Visualization style of the mesh.  One of the following:
            ``style='surface'``, ``style='wireframe'``, ``style='points'``.
            Defaults to ``'surface'``. Note that ``'wireframe'`` only shows a
            wireframe of the outer geometry.

        scalars : str, optional
            Scalars used to "color" the points or cells of the dataset.
            Accepts only a string name of an array that is present on the
            composite dataset.

        clim : 2 item list, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng``
            is also an accepted alias for this.

        show_edges : bool, default: :attr:`pyvista.global_theme.show_edges`
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        edge_color : color_like, default: :attr:`pyvista.global_theme.edge_color`
            The solid color to give the edges when ``show_edges=True``.
            Either a string, RGB list, or hex color string.

            Defaults to :attr:`pyvista.global_theme.edge_color`
            <pyvista.themes.DefaultTheme.edge_color>`.

        point_size : float, default: 5.0
            Point size of any points in the dataset plotted. Also
            applicable when style='points'. Default ``5.0``.

        line_width : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.

        opacity : float, default: 1.0
            Opacity of the mesh. A single float value that will be applied
            globally opacity of the mesh and uniformly
            applied everywhere - should be between 0 and 1.

        flip_scalars : bool, default: False
            Flip direction of cmap. Most colormaps allow ``*_r``
            suffix to do this as well.

        lighting : bool, default: True
            Enable or disable view direction lighting.

        n_colors : int, default: 256
            Number of colors to use when displaying scalars.  The scalar bar
            will also have this many colors.

        interpolate_before_map : bool, default: True
            Enabling makes for a smoother scalars display.  When ``False``,
            OpenGL will interpolate the mapped colors which can result in
            showing colors that are not present in the color map.

        cmap : str, list, or pyvista.LookupTable, default: :attr:`pyvista.themes.DefaultTheme.cmap`
            If a string, this is the name of the ``matplotlib`` colormap to use
            when mapping the ``scalars``.  See available Matplotlib colormaps.
            Only applicable for when displaying ``scalars``. Requires
            Matplotlib to be installed.  ``colormap`` is also an accepted alias
            for this. If ``colorcet`` or ``cmocean`` are installed, their
            colormaps can be specified by name.

            You can also specify a list of colors to override an existing
            colormap with a custom one.  For example, to create a three color
            colormap you might specify ``['green', 'red', 'blue']``.

            This parameter also accepts a :class:`pyvista.LookupTable`. If this
            is set, all parameters controlling the color map like ``n_colors``
            will be ignored.

        label : str, optional
            String label to use when adding a legend to the scene with
            :func:`pyvista.BasePlotter.add_legend`.

        reset_camera : bool, optional
            Reset the camera after adding this mesh to the scene. The default
            setting is ``None``, where the camera is only reset if this plotter
            has already been shown. If ``False``, the camera is not reset
            regardless of the state of the ``Plotter``. When ``True``, the
            camera is always reset.

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the
            scalar bar to the scene. For options, see
            :func:`pyvista.BasePlotter.add_scalar_bar`.

        show_scalar_bar : bool
            If ``False``, a scalar bar will not be added to the
            scene. Defaults to ``True`` unless ``rgba=True``.

        multi_colors : bool, default: False
            Color each block by a solid color using matplotlib's color cycler.

        name : str, optional
            The name for the added mesh/actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        render_points_as_spheres : bool, default: False
            Render points as spheres rather than dots.

        render_lines_as_tubes : bool, default: False
            Show lines as thick tubes rather than flat lines.  Control
            the width with ``line_width``.

        smooth_shading : bool, default: :attr`pyvista.themes.DefaultTheme.smooth_shading`
            Enable smooth shading when ``True`` using the Phong shading
            algorithm.  When ``False``, uses flat shading.  Automatically
            enabled when ``pbr=True``.  See :ref:`shading_example`.

        split_sharp_edges : bool, default: False
            Split sharp edges exceeding 30 degrees when plotting with smooth
            shading.  Control the angle with the optional keyword argument
            ``feature_angle``.  By default this is ``False`` unless overridden
            by the global or plotter theme.  Note that enabling this will
            create a copy of the input mesh within the plotter.  See
            :ref:`shading_example`.

        ambient : float, default: 0.0
            When lighting is enabled, this is the amount of light in
            the range of 0 to 1 (default 0.0) that reaches the actor
            when not directed at the light source emitted from the
            viewer.

        diffuse : float, default: 1.0
            The diffuse lighting coefficient.

        specular : float, default: 0.0
            The specular lighting coefficient.

        specular_power : float, default: 1.0
            The specular power. Between 0.0 and 128.0.

        nan_color : color_like, default: :attr:`pyvista.themes.DefaultTheme.nan_color`
            The color to use for all ``NaN`` values in the plotted
            scalar array.

        nan_opacity : float, default: 1.0
            Opacity of ``NaN`` values.  Should be between 0 and 1.

        culling : str, bool, default: False
            Does not render faces that are culled. This can be helpful for
            dense surface meshes, especially when edges are visible, but can
            cause flat meshes to be partially displayed. One of the following:

            * ``True`` - Enable backface culling
            * ``"b"`` - Enable backface culling
            * ``"back"`` - Enable backface culling
            * ``"backface"`` - Enable backface culling
            * ``"f"`` - Enable frontface culling
            * ``"front"`` - Enable frontface culling
            * ``"frontface"`` - Enable frontface culling
            * ``False`` - Disable both backface and frontface culling

        rgb : bool, default: False
            If an 2 dimensional array is passed as the scalars, plot
            those values as RGB(A) colors. ``rgba`` is also an
            accepted alias for this.  Opacity (the A) is optional.  If
            a scalars array ending with ``"_rgba"`` is passed, the default
            becomes ``True``.  This can be overridden by setting this
            parameter to ``False``.

        categories : bool, optional
            If set to ``True``, then the number of unique values in
            the scalar array will be used as the ``n_colors``
            argument.

        below_color : color_like, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``below_label`` to ``'Below'``.

        above_color : color_like, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``above_label`` to ``'Above'``.

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float
            values in the scalars range to annotate on the scalar bar
            and the values are the the string annotations.

        pickable : bool, default: True
            Set whether this actor is pickable.

        preference : str, default: 'point'
            For each block, when ``block.n_points == block.n_cells`` and
            setting scalars, this parameter sets how the scalars will be mapped
            to the mesh.  For example, when ``'point'`` the scalars will be
            associated with the mesh points if available.  Can be either
            ``'point'`` or ``'cell'``.

        log_scale : bool, default: False
            Use log scale when mapping data to colors. Scalars less
            than zero are mapped to the smallest representable
            positive float.

        pbr : bool, default: False
            Enable physics based rendering (PBR) if the mesh is
            ``PolyData``.  Use the ``color`` argument to set the base
            color. This is only available in VTK>=9.

        metallic : float, default: 0.0
            Usually this value is either 0 or 1 for a real material
            but any value in between is valid. This parameter is only
            used by PBR interpolation.

        roughness : float, default: 0.5
            This value has to be between 0 (glossy) and 1 (rough). A
            glossy material has reflections and a high specular
            part. This parameter is only used by PBR
            interpolation.

        render : bool, default: True
            Force a render when ``True``.

        component : int, optional
            Set component of vector valued scalars to plot.  Must be
            nonnegative, if supplied. If ``None``, the magnitude of
            the vector is plotted.

        color_missing_with_nan : bool, default: False
            Color any missing values with the ``nan_color``. This is useful
            when not all blocks of the composite dataset have the specified
            ``scalars``.

        copy_mesh : bool, optional
            If ``True``, a copy of the mesh will be made before adding it to
            the plotter.  This is useful if e.g. you would like to add the same
            mesh to a plotter multiple times and display different
            scalars. Setting ``copy_mesh`` to ``False`` is necessary if you
            would like to update the mesh after adding it to the plotter and
            have these updates rendered, e.g. by changing the active scalars or
            through an interactive widget.  Defaults to ``False``.

        **kwargs : dict, optional
            Optional developer keyword arguments.

        Returns
        -------
        pyvista.Actor
            Actor of the composite dataset.

        pyvista.CompositePolyDataMapper
            Composite PolyData mapper.

        Examples
        --------
        Add a sphere and a cube as a multiblock dataset to a plotter and then
        change the visibility and color of the blocks.

        Note index ``1`` and ``2`` are used to access the individual blocks of
        the composite dataset. This is because the :class:`pyvista.MultiBlock`
        is the root node of the "tree" and is index ``0``. This allows you to
        access individual blocks or the entire composite dataset itself in the
        case of multiple nested composite datasets.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[1].color = 'b'
        >>> mapper.block_attr[1].opacity = 0.5
        >>> mapper.block_attr[2].color = 'r'
        >>> pl.show()

        """
        if not isinstance(dataset, _vtk.vtkCompositeDataSet):
            raise TypeError(f'Invalid type ({type(dataset)}). Must be a composite dataset.')
        # always convert
        dataset = dataset.as_polydata_blocks(copy_mesh)
        self.mesh = dataset  # for legacy behavior

        # Parse arguments
        (
            scalar_bar_args,
            split_sharp_edges,
            show_scalar_bar,
            feature_angle,
            render_points_as_spheres,
            smooth_shading,
            clim,
            cmap,
            culling,
            name,
            nan_color,
            color,
            texture,
            rgb,
            interpolation,
            remove_existing_actor,
        ) = _common_arg_parser(
            dataset,
            self._theme,
            n_colors,
            scalar_bar_args,
            split_sharp_edges,
            show_scalar_bar,
            render_points_as_spheres,
            smooth_shading,
            pbr,
            clim,
            cmap,
            culling,
            name,
            nan_color,
            nan_opacity,
            color,
            None,
            rgb,
            style,
            **kwargs,
        )

        # Compute surface normals if using smooth shading
        if smooth_shading:
            dataset = dataset._compute_normals(
                cell_normals=False,
                split_vertices=True,
                feature_angle=feature_angle,
            )

        self.mapper = CompositePolyDataMapper(
            dataset,
            theme=self._theme,
            color_missing_with_nan=color_missing_with_nan,
            interpolate_before_map=interpolate_before_map,
        )

        actor, _ = self.add_actor(self.mapper)

        prop = Property(
            self._theme,
            interpolation=interpolation,
            metallic=metallic,
            roughness=roughness,
            point_size=point_size,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=specular_power,
            show_edges=show_edges,
            color=color,
            style=style,
            edge_color=edge_color,
            render_points_as_spheres=render_points_as_spheres,
            render_lines_as_tubes=render_lines_as_tubes,
            lighting=lighting,
            line_width=line_width,
            opacity=opacity,
            culling=culling,
        )
        actor.SetProperty(prop)

        if label is not None:
            self._add_legend_label(actor, label, None, prop.color)

        # check if there are any consistent active scalars
        if color is not None:
            self.mapper.scalar_visibility = False
        elif multi_colors:
            self.mapper.set_unique_colors()
        else:
            if scalars is None:
                point_name, cell_name = dataset._get_consistent_active_scalars()
                if point_name and cell_name:
                    if preference == 'point':
                        scalars = point_name
                    else:
                        scalars = cell_name
                else:
                    scalars = point_name if point_name is not None else cell_name

            elif not isinstance(scalars, str):
                raise TypeError(
                    f'`scalars` must be a string for `add_composite`, not ({type(scalars)})'
                )

            if categories:
                if not isinstance(categories, int):
                    raise TypeError('Categories must be an integer for a composite dataset.')
                n_colors = categories

            if scalars is not None:
                scalar_bar_args = self.mapper.set_scalars(
                    scalars,
                    preference,
                    component,
                    annotations,
                    rgb,
                    scalar_bar_args,
                    n_colors,
                    nan_color,
                    above_color,
                    below_color,
                    clim,
                    cmap,
                    flip_scalars,
                    categories,
                    log_scale,
                )
            else:
                self.mapper.scalar_visibility = False

        # Only show scalar bar if there are scalars
        if show_scalar_bar and scalars is not None:
            self.add_scalar_bar(**scalar_bar_args)

        # by default reset the camera if the plotting window has been rendered
        if reset_camera is None:
            reset_camera = not self._first_time and not self.camera_set

        self.add_actor(
            actor,
            reset_camera=reset_camera,
            name=name,
            pickable=pickable,
            render=render,
            remove_existing_actor=remove_existing_actor,
        )

        return actor, self.mapper

    def add_mesh(
        self,
        mesh,
        color=None,
        style=None,
        scalars=None,
        clim=None,
        show_edges=None,
        edge_color=None,
        point_size=5.0,
        line_width=None,
        opacity=1.0,
        flip_scalars=False,
        lighting=None,
        n_colors=256,
        interpolate_before_map=True,
        cmap=None,
        label=None,
        reset_camera=None,
        scalar_bar_args=None,
        show_scalar_bar=None,
        multi_colors=False,
        name=None,
        texture=None,
        render_points_as_spheres=None,
        render_lines_as_tubes=False,
        smooth_shading=None,
        split_sharp_edges=None,
        ambient=0.0,
        diffuse=1.0,
        specular=0.0,
        specular_power=100.0,
        nan_color=None,
        nan_opacity=1.0,
        culling=None,
        rgb=None,
        categories=False,
        silhouette=False,
        use_transparency=False,
        below_color=None,
        above_color=None,
        annotations=None,
        pickable=True,
        preference="point",
        log_scale=False,
        pbr=False,
        metallic=0.0,
        roughness=0.5,
        render=True,
        component=None,
        emissive=False,
        copy_mesh=False,
        backface_params=None,
        **kwargs,
    ):
        """Add any PyVista/VTK mesh or dataset that PyVista can wrap to the scene.

        This method is using a mesh representation to view the surfaces
        and/or geometry of datasets. For volume rendering, see
        :func:`pyvista.BasePlotter.add_volume`.

        To see the what most of the following parameters look like in action,
        please refer to :class:`pyvista.Property`.

        Parameters
        ----------
        mesh : pyvista.DataSet or pyvista.MultiBlock
            Any PyVista or VTK mesh is supported. Also, any dataset
            that :func:`pyvista.wrap` can handle including NumPy
            arrays of XYZ points.

        color : color_like, optional
            Use to make the entire mesh have a single solid color.
            Either a string, RGB list, or hex color string.  For example:
            ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
            ``color='#FFFFFF'``. Color will be overridden if scalars are
            specified.

            Defaults to :attr:`pyvista.global_theme.color
            <pyvista.themes.DefaultTheme.color>`.

        style : str, optional
            Visualization style of the mesh.  One of the following:
            ``style='surface'``, ``style='wireframe'``, ``style='points'``,
            ``style='points_gaussian'``. Defaults to ``'surface'``. Note that
            ``'wireframe'`` only shows a wireframe of the outer geometry.
            ``'points_gaussian'`` can be modified with the ``emissive``,
            ``render_points_as_spheres`` options.

        scalars : str or numpy.ndarray, optional
            Scalars used to "color" the mesh.  Accepts a string name
            of an array that is present on the mesh or an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If both
            ``color`` and ``scalars`` are ``None``, then the active
            scalars are used.

        clim : 2 item list, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng``
            is also an accepted alias for this.

        show_edges : bool, optional
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        edge_color : color_like, optional
            The solid color to give the edges when ``show_edges=True``.
            Either a string, RGB list, or hex color string.

            Defaults to :attr:`pyvista.global_theme.edge_color
            <pyvista.themes.DefaultTheme.edge_color>`.

        point_size : float, optional
            Point size of any nodes in the dataset plotted. Also
            applicable when style='points'. Default ``5.0``.

        line_width : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default ``None``.

        opacity : float, str, array-like
            Opacity of the mesh. If a single float value is given, it
            will be the global opacity of the mesh and uniformly
            applied everywhere - should be between 0 and 1. A string
            can also be specified to map the scalars range to a
            predefined opacity transfer function (options include:
            ``'linear'``, ``'linear_r'``, ``'geom'``, ``'geom_r'``).
            A string could also be used to map a scalars array from
            the mesh to the opacity (must have same number of elements
            as the ``scalars`` argument). Or you can pass a custom
            made transfer function that is an array either
            ``n_colors`` in length or shorter.

        flip_scalars : bool, optional
            Flip direction of cmap. Most colormaps allow ``*_r``
            suffix to do this as well.

        lighting : bool, optional
            Enable or disable view direction lighting. Default ``False``.

        n_colors : int, optional
            Number of colors to use when displaying scalars. Defaults to 256.
            The scalar bar will also have this many colors.

        interpolate_before_map : bool, optional
            Enabling makes for a smoother scalars display.  Default is
            ``True``.  When ``False``, OpenGL will interpolate the
            mapped colors which can result is showing colors that are
            not present in the color map.

        cmap : str, list, or pyvista.LookupTable, default: :attr:`pyvista.themes.DefaultTheme.cmap`
            If a string, this is the name of the ``matplotlib`` colormap to use
            when mapping the ``scalars``.  See available Matplotlib colormaps.
            Only applicable for when displaying ``scalars``. Requires
            Matplotlib to be installed.  ``colormap`` is also an accepted alias
            for this. If ``colorcet`` or ``cmocean`` are installed, their
            colormaps can be specified by name.

            You can also specify a list of colors to override an existing
            colormap with a custom one.  For example, to create a three color
            colormap you might specify ``['green', 'red', 'blue']``.

            This parameter also accepts a :class:`pyvista.LookupTable`. If this
            is set, all parameters controlling the color map like ``n_colors``
            will be ignored.

        label : str, optional
            String label to use when adding a legend to the scene with
            :func:`pyvista.BasePlotter.add_legend`.

        reset_camera : bool, optional
            Reset the camera after adding this mesh to the scene. The default
            setting is ``None``, where the camera is only reset if this plotter
            has already been shown. If ``False``, the camera is not reset
            regardless of the state of the ``Plotter``. When ``True``, the
            camera is always reset.

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the
            scalar bar to the scene. For options, see
            :func:`pyvista.BasePlotter.add_scalar_bar`.

        show_scalar_bar : bool
            If ``False``, a scalar bar will not be added to the
            scene. Defaults to ``True``.

        multi_colors : bool, optional
            If a :class:`pyvista.MultiBlock` dataset is given this will color
            each block by a solid color using matplotlib's color cycler.

        name : str, optional
            The name for the added mesh/actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        texture : vtk.vtkTexture or np.ndarray or bool or str, optional
            A texture to apply if the input mesh has texture
            coordinates.  This will not work with MultiBlock
            datasets. If set to ``True``, the first available texture
            on the object will be used. If a string name is given, it
            will pull a texture with that name associated to the input
            mesh.

        render_points_as_spheres : bool, optional
            Render points as spheres rather than dots.

        render_lines_as_tubes : bool, optional
            Show lines as thick tubes rather than flat lines.  Control
            the width with ``line_width``.

        smooth_shading : bool, optional
            Enable smooth shading when ``True`` using the Phong
            shading algorithm.  When ``False``, use flat shading.
            Automatically enabled when ``pbr=True``.  See
            :ref:`shading_example`.

        split_sharp_edges : bool, optional
            Split sharp edges exceeding 30 degrees when plotting with smooth
            shading.  Control the angle with the optional keyword argument
            ``feature_angle``.  By default this is ``False`` unless overridden
            by the global or plotter theme.  Note that enabling this will
            create a copy of the input mesh within the plotter.  See
            :ref:`shading_example`.

        ambient : float, optional
            When lighting is enabled, this is the amount of light in
            the range of 0 to 1 (default 0.0) that reaches the actor
            when not directed at the light source emitted from the
            viewer.

        diffuse : float, optional
            The diffuse lighting coefficient. Default 1.0.

        specular : float, optional
            The specular lighting coefficient. Default 0.0.

        specular_power : float, optional
            The specular power. Between 0.0 and 128.0.

        nan_color : color_like, optional, defaults to gray
            The color to use for all ``NaN`` values in the plotted
            scalar array.

        nan_opacity : float, optional
            Opacity of ``NaN`` values.  Should be between 0 and 1.
            Default 1.0.

        culling : str, optional
            Does not render faces that are culled. Options are
            ``'front'`` or ``'back'``. This can be helpful for dense
            surface meshes, especially when edges are visible, but can
            cause flat meshes to be partially displayed.  Defaults to
            ``False``.

        rgb : bool, optional
            If an 2 dimensional array is passed as the scalars, plot
            those values as RGB(A) colors. ``rgba`` is also an
            accepted alias for this.  Opacity (the A) is optional.  If
            a scalars array ending with ``"_rgba"`` is passed, the default
            becomes ``True``.  This can be overridden by setting this
            parameter to ``False``.

        categories : bool, optional
            If set to ``True``, then the number of unique values in
            the scalar array will be used as the ``n_colors``
            argument.

        silhouette : dict, bool, optional
            If set to ``True``, plot a silhouette highlight for the
            mesh. This feature is only available for a triangulated
            ``PolyData``.  As a ``dict``, it contains the properties
            of the silhouette to display:

                * ``color``: ``color_like``, color of the silhouette
                * ``line_width``: ``float``, edge width
                * ``opacity``: ``float`` between 0 and 1, edge transparency
                * ``feature_angle``: If a ``float``, display sharp edges
                  exceeding that angle in degrees.
                * ``decimate``: ``float`` between 0 and 1, level of decimation

        use_transparency : bool, optional
            Invert the opacity mappings and make the values correspond
            to transparency.

        below_color : color_like, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``below_label`` to ``'Below'``.

        above_color : color_like, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``above_label`` to ``'Above'``.

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float
            values in the scalars range to annotate on the scalar bar
            and the values are the the string annotations.

        pickable : bool, optional
            Set whether this actor is pickable.

        preference : str, optional
            When ``mesh.n_points == mesh.n_cells`` and setting
            scalars, this parameter sets how the scalars will be
            mapped to the mesh.  Default ``'point'``, causes the
            scalars will be associated with the mesh points.  Can be
            either ``'point'`` or ``'cell'``.

        log_scale : bool, optional
            Use log scale when mapping data to colors. Scalars less
            than zero are mapped to the smallest representable
            positive float. Default ``False``.

        pbr : bool, optional
            Enable physics based rendering (PBR) if the mesh is
            ``PolyData``.  Use the ``color`` argument to set the base
            color. This is only available in VTK>=9.

        metallic : float, optional
            Usually this value is either 0 or 1 for a real material
            but any value in between is valid. This parameter is only
            used by PBR interpolation. Default value is 0.0.

        roughness : float, optional
            This value has to be between 0 (glossy) and 1 (rough). A
            glossy material has reflections and a high specular
            part. This parameter is only used by PBR
            interpolation. Default value is 0.5.

        render : bool, optional
            Force a render when ``True``.  Default ``True``.

        component : int, optional
            Set component of vector valued scalars to plot.  Must be
            nonnegative, if supplied. If ``None``, the magnitude of
            the vector is plotted.

        emissive : bool, default: False
            Treat the points/splats as emissive light sources. Only valid for
            ``style='points_gaussian'`` representation.

        copy_mesh : bool, optional
            If ``True``, a copy of the mesh will be made before adding it to
            the plotter.  This is useful if you would like to add the same
            mesh to a plotter multiple times and display different
            scalars. Setting ``copy_mesh`` to ``False`` is necessary if you
            would like to update the mesh after adding it to the plotter and
            have these updates rendered, e.g. by changing the active scalars or
            through an interactive widget. This should only be set to ``True``
            with caution. Defaults to ``False``.

        backface_params : dict or pyvista.Property, optional
            A :class:`pyvista.Property` or a dict of parameters to use for
            backface rendering. This is useful for instance when the inside of
            oriented surfaces has a different color than the outside. When a
            :class:`pyvista.Property`, this is directly used for backface
            rendering. When a dict, valid keys are :class:`pyvista.Property`
            attributes, and values are corresponding values to use for the
            given property. Omitted keys (or the default of
            ``backface_params=None``) default to the corresponding frontface
            properties.

        **kwargs : dict, optional
            Optional developer keyword arguments.

        Returns
        -------
        pyvista.plotting.actor.Actor
            Actor of the mesh.

        Examples
        --------
        Add a sphere to the plotter and show it with a custom scalar
        bar title.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere['Data'] = sphere.points[:, 2]
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(sphere,
        ...                      scalar_bar_args={'title': 'Z Position'})
        >>> plotter.show()

        Plot using RGB on a single cell.  Note that since the number of
        points and the number of cells are identical, we have to pass
        ``preference='cell'``.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [.5, .667, 0], [0.5, .33, 0.667]])
        >>> faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2], [3, 0, 1, 3], [3, 1, 2, 3]])
        >>> mesh = pv.PolyData(vertices, faces)
        >>> mesh.cell_data['colors'] = [[255, 255, 255],
        ...                               [0, 255, 0],
        ...                               [0, 0, 255],
        ...                               [255, 0, 0]]
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(mesh, scalars='colors', lighting=False,
        ...                      rgb=True, preference='cell')
        >>> plotter.camera_position='xy'
        >>> plotter.show()

        Note how this varies from ``preference=='point'``.  This is
        because each point is now being individually colored, versus
        in ``preference=='point'``, each cell face is individually
        colored.

        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(mesh, scalars='colors', lighting=False,
        ...                      rgb=True, preference='point')
        >>> plotter.camera_position='xy'
        >>> plotter.show()

        Plot a plane with a constant color and vary its opacity by point.

        >>> plane = pv.Plane()
        >>> plane.plot(color='b', opacity=np.linspace(0, 1, plane.n_points),
        ...            show_edges=True)

        Plot the points of a sphere with gaussian smoothing while coloring by z
        position.

        >>> mesh = pv.Sphere()
        >>> mesh.plot(
        ...     scalars=mesh.points[:, 2],
        ...     style='points_gaussian',
        ...     opacity=0.5,
        ...     point_size=10,
        ...     render_points_as_spheres=False,
        ...     show_scalar_bar=False,
        ... )

        """
        if style == 'points_gaussian':
            self.mapper = PointGaussianMapper(theme=self.theme)
        else:
            self.mapper = DataSetMapper(theme=self.theme)

        # Convert the VTK data object to a pyvista wrapped object if necessary
        if not is_pyvista_dataset(mesh):
            mesh = wrap(mesh)
            if not is_pyvista_dataset(mesh):
                raise TypeError(
                    f'Object type ({type(mesh)}) not supported for plotting in PyVista.'
                )
        if isinstance(mesh, pyvista.PointSet):
            # cast to PointSet to PolyData
            mesh = mesh.cast_to_polydata(deep=False)
        elif isinstance(mesh, pyvista.MultiBlock):
            return self.add_composite(
                mesh,
                color=color,
                style=style,
                scalars=scalars,
                clim=clim,
                show_edges=show_edges,
                edge_color=edge_color,
                point_size=point_size,
                line_width=line_width,
                opacity=opacity,
                flip_scalars=flip_scalars,
                lighting=lighting,
                n_colors=n_colors,
                interpolate_before_map=interpolate_before_map,
                cmap=cmap,
                label=label,
                reset_camera=reset_camera,
                scalar_bar_args=scalar_bar_args,
                show_scalar_bar=show_scalar_bar,
                multi_colors=multi_colors,
                name=name,
                render_points_as_spheres=render_points_as_spheres,
                render_lines_as_tubes=render_lines_as_tubes,
                smooth_shading=smooth_shading,
                split_sharp_edges=split_sharp_edges,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                specular_power=specular_power,
                nan_color=nan_color,
                nan_opacity=nan_opacity,
                culling=culling,
                rgb=rgb,
                categories=categories,
                below_color=below_color,
                above_color=above_color,
                pickable=pickable,
                preference=preference,
                log_scale=log_scale,
                pbr=pbr,
                metallic=metallic,
                roughness=roughness,
                render=render,
                **kwargs,
            )
        elif copy_mesh:
            # A shallow copy of `mesh` is made here so when we set (or add) scalars
            # active, it doesn't modify the original input mesh.
            mesh = mesh.copy(deep=False)

        # Parse arguments
        (
            scalar_bar_args,
            split_sharp_edges,
            show_scalar_bar,
            feature_angle,
            render_points_as_spheres,
            smooth_shading,
            clim,
            cmap,
            culling,
            name,
            nan_color,
            color,
            texture,
            rgb,
            interpolation,
            remove_existing_actor,
        ) = _common_arg_parser(
            mesh,
            self._theme,
            n_colors,
            scalar_bar_args,
            split_sharp_edges,
            show_scalar_bar,
            render_points_as_spheres,
            smooth_shading,
            pbr,
            clim,
            cmap,
            culling,
            name,
            nan_color,
            nan_opacity,
            color,
            texture,
            rgb,
            style,
            **kwargs,
        )

        if silhouette:
            if isinstance(silhouette, dict):
                self.add_silhouette(mesh, silhouette)
            else:
                self.add_silhouette(mesh)

        # Try to plot something if no preference given
        if scalars is None and color is None and texture is None:
            # Prefer texture first
            if len(list(mesh.textures.keys())) > 0:
                texture = True
            # If no texture, plot any active scalar
            else:
                # Make sure scalars components are not vectors/tuples
                scalars = mesh.active_scalars_name
                # Don't allow plotting of string arrays by default
                if scalars is not None:  # and np.issubdtype(mesh.active_scalars.dtype, np.number):
                    scalar_bar_args.setdefault('title', scalars)
                else:
                    scalars = None

        # Make sure scalars is a numpy array after this point
        original_scalar_name = None
        scalars_name = pyvista.DEFAULT_SCALARS_NAME
        if isinstance(scalars, str):
            self.mapper.array_name = scalars

            # enable rgb if the scalars name ends with rgb or rgba
            if rgb is None:
                if scalars.endswith('_rgb') or scalars.endswith('_rgba'):
                    rgb = True

            original_scalar_name = scalars
            scalars = get_array(mesh, scalars, preference=preference, err=True)
            scalar_bar_args.setdefault('title', original_scalar_name)
            scalars_name = original_scalar_name

            # Set the active scalars name here. If the name already exists in
            # the input mesh, it may not be set as the active scalars within
            # the mapper. This should be refactored by 0.36.0
            field = get_array_association(mesh, original_scalar_name, preference=preference)
            if field == FieldAssociation.POINT:
                mesh.point_data.active_scalars_name = original_scalar_name
                self.mapper.scalar_map_mode = 'point'
            elif field == FieldAssociation.CELL:
                mesh.cell_data.active_scalars_name = original_scalar_name
                self.mapper.scalar_map_mode = 'cell'

        # Compute surface normals if using smooth shading
        if smooth_shading:
            mesh, scalars = prepare_smooth_shading(
                mesh, scalars, texture, split_sharp_edges, feature_angle, preference
            )

        if rgb:
            show_scalar_bar = False
            if scalars.ndim != 2 or scalars.shape[1] < 3 or scalars.shape[1] > 4:
                raise ValueError('RGB array must be n_points/n_cells by 3/4 in shape.')

        if mesh.n_points < 1:
            raise ValueError('Empty meshes cannot be plotted. Input mesh has zero points.')

        # set main values
        self.mesh = mesh
        self.mapper.dataset = self.mesh
        self.mapper.interpolate_before_map = interpolate_before_map

        actor = Actor(mapper=self.mapper)

        if texture is True or isinstance(texture, (str, int)):
            texture = mesh._activate_texture(texture)

        if texture:

            if isinstance(texture, np.ndarray):
                texture = numpy_to_texture(texture)
            if not isinstance(texture, (_vtk.vtkTexture, _vtk.vtkOpenGLTexture)):
                raise TypeError(f'Invalid texture type ({type(texture)})')
            if mesh.GetPointData().GetTCoords() is None:
                raise ValueError(
                    'Input mesh does not have texture coordinates to support the texture.'
                )
            actor.texture = texture
            # Set color to white by default when using a texture
            if color is None:
                color = 'white'
            if scalars is None:
                show_scalar_bar = False
            self.mapper.scalar_visibility = False

            # see https://github.com/pyvista/pyvista/issues/950
            mesh.set_active_scalars(None)

        # Handle making opacity array
        custom_opac, opacity = process_opacity(
            mesh, opacity, preference, n_colors, scalars, use_transparency
        )

        # Scalars formatting ==================================================
        if scalars is not None:
            self.mapper.set_scalars(
                scalars,
                scalars_name,
                n_colors,
                scalar_bar_args,
                rgb,
                component,
                preference,
                custom_opac,
                annotations,
                log_scale,
                nan_color,
                above_color,
                below_color,
                cmap,
                flip_scalars,
                opacity,
                categories,
                clim,
            )
            self.mapper.scalar_visibility = True
        elif custom_opac:  # no scalars but custom opacity
            self.mapper.set_custom_opacity(
                opacity,
                color,
                n_colors,
                preference,
            )
            self.mapper.scalar_visibility = True
        else:
            self.mapper.scalar_visibility = False

        # Set actor properties ================================================
        prop_kwargs = dict(
            theme=self._theme,
            interpolation=interpolation,
            metallic=metallic,
            roughness=roughness,
            point_size=point_size,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=specular_power,
            show_edges=show_edges,
            color=color,
            style=style if style != 'points_gaussian' else 'points',
            edge_color=edge_color,
            render_lines_as_tubes=render_lines_as_tubes,
            lighting=lighting,
            line_width=line_width,
            culling=culling,
        )

        if style == 'points_gaussian':
            self.mapper.emissive = emissive
            self.mapper.scale_factor = point_size * self.mapper.dataset.length / 1300
            if not render_points_as_spheres and not emissive:
                if opacity >= 1.0:
                    opacity = 0.9999  # otherwise, weird triangles

        if isinstance(opacity, (float, int)):
            prop_kwargs['opacity'] = opacity
        prop = Property(**prop_kwargs)
        actor.SetProperty(prop)

        if render_points_as_spheres:
            if style == 'points_gaussian':
                self.mapper.use_circular_splat(opacity)
                prop.opacity = 1.0
            else:
                prop.render_points_as_spheres = render_points_as_spheres

        if backface_params is not None:
            if isinstance(backface_params, Property):
                backface_prop = backface_params
            elif isinstance(backface_params, dict):
                # preserve omitted kwargs from frontface
                backface_kwargs = deepcopy(prop_kwargs)
                backface_kwargs.update(backface_params)
                backface_prop = Property(**backface_kwargs)
            else:
                raise TypeError(
                    'Backface params must be a pyvista.Property or a dict, '
                    f'not {type(backface_params).__name__}.'
                )
            actor.backface_prop = backface_prop

        # legend label
        if label is not None:
            self._add_legend_label(actor, label, scalars, actor.prop.color)

        # by default reset the camera if the plotting window has been rendered
        if reset_camera is None:
            reset_camera = not self._first_time and not self.camera_set

        self.add_actor(
            actor,
            reset_camera=reset_camera,
            name=name,
            pickable=pickable,
            render=render,
            remove_existing_actor=remove_existing_actor,
        )

        # hide scalar bar if using special scalars
        if scalar_bar_args.get('title') == '__custom_rgba':
            show_scalar_bar = False

        # Only show scalar bar if there are scalars
        if show_scalar_bar and scalars is not None:
            self.add_scalar_bar(**scalar_bar_args)

        self.renderer.Modified()
        return actor

    def _add_legend_label(self, actor, label, scalars, color):
        """Add a legend label based on an actor and its scalars."""
        if not isinstance(label, str):
            raise TypeError('Label must be a string')
        geom = pyvista.Triangle()
        if scalars is not None:
            geom = pyvista.Box()
            color = Color('black')
        geom.points -= geom.center
        addr = actor.GetAddressAsString("")
        self.renderer._labels[addr] = [geom, label, color]

    def add_volume(
        self,
        volume,
        scalars=None,
        clim=None,
        resolution=None,
        opacity='linear',
        n_colors=256,
        cmap=None,
        flip_scalars=False,
        reset_camera=None,
        name=None,
        ambient=0.0,
        categories=False,
        culling=False,
        multi_colors=False,
        blending='composite',
        mapper=None,
        scalar_bar_args=None,
        show_scalar_bar=None,
        annotations=None,
        pickable=True,
        preference="point",
        opacity_unit_distance=None,
        shade=False,
        diffuse=0.7,
        specular=0.2,
        specular_power=10.0,
        render=True,
        **kwargs,
    ):
        """Add a volume, rendered using a smart mapper by default.

        Requires a 3D :class:`numpy.ndarray` or :class:`pyvista.UniformGrid`.

        Parameters
        ----------
        volume : 3D numpy.ndarray or pyvista.UniformGrid
            The input volume to visualize. 3D numpy arrays are accepted.

        scalars : str or numpy.ndarray, optional
            Scalars used to "color" the mesh.  Accepts a string name of an
            array that is present on the mesh or an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If ``scalars`` is
            ``None``, then the active scalars are used.

        clim : 2 item list, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng``
            is also an accepted alias for this.

        resolution : list, optional
            Block resolution.

        opacity : str or numpy.ndarray, optional
            Opacity mapping for the scalars array.
            A string can also be specified to map the scalars range to a
            predefined opacity transfer function (options include: 'linear',
            'linear_r', 'geom', 'geom_r'). Or you can pass a custom made
            transfer function that is an array either ``n_colors`` in length or
            shorter.

        n_colors : int, optional
            Number of colors to use when displaying scalars. Defaults to 256.
            The scalar bar will also have this many colors.

        cmap : str, list, or pyvista.LookupTable, default: :attr:`pyvista.themes.DefaultTheme.cmap`
            If a string, this is the name of the ``matplotlib`` colormap to use
            when mapping the ``scalars``.  See available Matplotlib colormaps.
            Only applicable for when displaying ``scalars``. Requires
            Matplotlib to be installed.  ``colormap`` is also an accepted alias
            for this. If ``colorcet`` or ``cmocean`` are installed, their
            colormaps can be specified by name.

            You can also specify a list of colors to override an existing
            colormap with a custom one.  For example, to create a three color
            colormap you might specify ``['green', 'red', 'blue']``.

            This parameter also accepts a :class:`pyvista.LookupTable`. If this
            is set, all parameters controlling the color map like ``n_colors``
            will be ignored.

        flip_scalars : bool, optional
            Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do
            this as well.

        reset_camera : bool, optional
            Reset the camera after adding this mesh to the scene.

        name : str, optional
            The name for the added actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        ambient : float, optional
            When lighting is enabled, this is the amount of light from
            0 to 1 that reaches the actor when not directed at the
            light source emitted from the viewer.  Default 0.0.

        categories : bool, optional
            If set to ``True``, then the number of unique values in the scalar
            array will be used as the ``n_colors`` argument.

        culling : str, optional
            Does not render faces that are culled. Options are ``'front'`` or
            ``'back'``. This can be helpful for dense surface meshes,
            especially when edges are visible, but can cause flat
            meshes to be partially displayed.  Defaults ``False``.

        multi_colors : bool, optional
            Whether or not to use multiple colors when plotting MultiBlock
            object. Blocks will be colored sequentially as 'Reds', 'Greens',
            'Blues', and 'Grays'.

        blending : str, optional
            Blending mode for visualisation of the input object(s). Can be
            one of 'additive', 'maximum', 'minimum', 'composite', or
            'average'. Defaults to 'additive'.

        mapper : str, optional
            Volume mapper to use given by name. Options include:
            ``'fixed_point'``, ``'gpu'``, ``'open_gl'``, and
            ``'smart'``.  If ``None`` the ``"volume_mapper"`` in the
            ``self._theme`` is used.

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the
            scalar bar to the scene. For options, see
            :func:`pyvista.BasePlotter.add_scalar_bar`.

        show_scalar_bar : bool
            If ``False``, a scalar bar will not be added to the
            scene. Defaults to ``True``.

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float
            values in the scalars range to annotate on the scalar bar
            and the values are the the string annotations.

        pickable : bool, optional
            Set whether this mesh is pickable.

        preference : str, optional
            When ``mesh.n_points == mesh.n_cells`` and setting
            scalars, this parameter sets how the scalars will be
            mapped to the mesh.  Default ``'point'``, causes the
            scalars will be associated with the mesh points.  Can be
            either ``'point'`` or ``'cell'``.

        opacity_unit_distance : float
            Set/Get the unit distance on which the scalar opacity
            transfer function is defined. Meaning that over that
            distance, a given opacity (from the transfer function) is
            accumulated. This is adjusted for the actual sampling
            distance during rendering. By default, this is the length
            of the diagonal of the bounding box of the volume divided
            by the dimensions.

        shade : bool
            Default off. If shading is turned on, the mapper may
            perform shading calculations - in some cases shading does
            not apply (for example, in a maximum intensity projection)
            and therefore shading will not be performed even if this
            flag is on.

        diffuse : float, optional
            The diffuse lighting coefficient. Default ``1.0``.

        specular : float, optional
            The specular lighting coefficient. Default ``0.0``.

        specular_power : float, optional
            The specular power. Between ``0.0`` and ``128.0``.

        render : bool, optional
            Force a render when True.  Default ``True``.

        **kwargs : dict, optional
            Optional keyword arguments.

        Returns
        -------
        pyvista.Actor
            Actor of the volume.

        Examples
        --------
        Show a built-in volume example with the coolwarm colormap.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> bolt_nut = examples.download_bolt_nut()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_volume(bolt_nut, cmap="coolwarm")
        >>> pl.show()

        """
        # Handle default arguments

        # Supported aliases
        clim = kwargs.pop('rng', clim)
        cmap = kwargs.pop('colormap', cmap)
        culling = kwargs.pop('backface_culling', culling)

        if "scalar" in kwargs:
            raise TypeError(
                "`scalar` is an invalid keyword argument for `add_mesh`. Perhaps you mean `scalars` with an s?"
            )
        assert_empty_kwargs(**kwargs)

        # Avoid mutating input
        if scalar_bar_args is None:
            scalar_bar_args = {}
        else:
            scalar_bar_args = scalar_bar_args.copy()
        # account for legacy behavior
        if 'stitle' in kwargs:  # pragma: no cover
            warnings.warn(USE_SCALAR_BAR_ARGS, PyVistaDeprecationWarning)
            scalar_bar_args.setdefault('title', kwargs.pop('stitle'))

        if show_scalar_bar is None:
            show_scalar_bar = self._theme.show_scalar_bar

        if culling is True:
            culling = 'backface'

        if mapper is None:
            mapper = self._theme.volume_mapper

        # only render when the plotter has already been shown
        if render is None:
            render = not self._first_time

        # Convert the VTK data object to a pyvista wrapped object if necessary
        if not is_pyvista_dataset(volume):
            if isinstance(volume, np.ndarray):
                volume = wrap(volume)
                if resolution is None:
                    resolution = [1, 1, 1]
                elif len(resolution) != 3:
                    raise ValueError('Invalid resolution dimensions.')
                volume.spacing = resolution
            else:
                volume = wrap(volume)
                if not is_pyvista_dataset(volume):
                    raise TypeError(
                        f'Object type ({type(volume)}) not supported for plotting in PyVista.'
                    )
        else:
            # HACK: Make a copy so the original object is not altered.
            #       Also, place all data on the nodes as issues arise when
            #       volume rendering on the cells.
            volume = volume.cell_data_to_point_data()

        if name is None:
            name = f'{type(volume).__name__}({volume.memory_address})'

        if isinstance(volume, pyvista.MultiBlock):
            from itertools import cycle

            cycler = cycle(['Reds', 'Greens', 'Blues', 'Greys', 'Oranges', 'Purples'])
            # Now iteratively plot each element of the multiblock dataset
            actors = []
            for idx in range(volume.GetNumberOfBlocks()):
                if volume[idx] is None:
                    continue
                # Get a good name to use
                next_name = f'{name}-{idx}'
                # Get the data object
                block = wrap(volume.GetBlock(idx))
                if resolution is None:
                    try:
                        block_resolution = block.GetSpacing()
                    except AttributeError:
                        block_resolution = resolution
                else:
                    block_resolution = resolution
                if multi_colors:
                    color = next(cycler)
                else:
                    color = cmap

                a = self.add_volume(
                    block,
                    resolution=block_resolution,
                    opacity=opacity,
                    n_colors=n_colors,
                    cmap=color,
                    flip_scalars=flip_scalars,
                    reset_camera=reset_camera,
                    name=next_name,
                    ambient=ambient,
                    categories=categories,
                    culling=culling,
                    clim=clim,
                    mapper=mapper,
                    pickable=pickable,
                    opacity_unit_distance=opacity_unit_distance,
                    shade=shade,
                    diffuse=diffuse,
                    specular=specular,
                    specular_power=specular_power,
                    render=render,
                )

                actors.append(a)
            return actors

        if not isinstance(volume, pyvista.UniformGrid):
            raise TypeError(
                f'Type {type(volume)} not supported for volume rendering at this time. Use `pyvista.UniformGrid`.'
            )

        if opacity_unit_distance is None:
            opacity_unit_distance = volume.length / (np.mean(volume.dimensions) - 1)

        if scalars is None:
            # Make sure scalars components are not vectors/tuples
            scalars = volume.active_scalars.copy()
            # Don't allow plotting of string arrays by default
            if scalars is not None and np.issubdtype(scalars.dtype, np.number):
                scalar_bar_args.setdefault('title', volume.active_scalars_info[1])
            else:
                raise ValueError('No scalars to use for volume rendering.')

        title = 'Data'
        if isinstance(scalars, str):
            title = scalars
            scalars = get_array(volume, scalars, preference=preference, err=True)
            scalar_bar_args.setdefault('title', title)

        if not isinstance(scalars, np.ndarray):
            scalars = np.asarray(scalars)

        if not np.issubdtype(scalars.dtype, np.number):
            raise TypeError('Non-numeric scalars are currently not supported for volume rendering.')
        if scalars.ndim != 1:
            scalars = scalars.ravel()

        # Define mapper, volume, and add the correct properties
        mappers_lookup = {
            'fixed_point': FixedPointVolumeRayCastMapper,
            'gpu': GPUVolumeRayCastMapper,
            'open_gl': OpenGLGPUVolumeRayCastMapper,
            'smart': SmartVolumeMapper,
        }
        if not isinstance(mapper, str) or mapper not in mappers_lookup.keys():
            raise TypeError(
                f"Mapper ({mapper}) unknown. Available volume mappers include: {', '.join(mappers_lookup.keys())}"
            )
        self.mapper = mappers_lookup[mapper](self._theme)

        # Scalars interpolation approach
        if scalars.shape[0] == volume.n_points:
            volume.point_data.set_array(scalars, title, True)
            self.mapper.scalar_mode = 'point'
        elif scalars.shape[0] == volume.n_cells:
            volume.cell_data.set_array(scalars, title, True)
            self.mapper.scalar_mode = 'cell'
        else:
            raise_not_matching(scalars, volume)

        # Set scalars range
        if clim is None:
            clim = [np.nanmin(scalars), np.nanmax(scalars)]
        elif isinstance(clim, float) or isinstance(clim, int):
            clim = [-clim, clim]

        # convert the scalars to np.uint8 and scale between 0 and 255 within clim
        clim = np.asarray(clim, dtype=scalars.dtype)
        scalars.clip(clim[0], clim[1], out=scalars)
        min_ = np.nanmin(scalars)
        max_ = np.nanmax(scalars)
        np.true_divide((scalars - min_), (max_ - min_) / 255, out=scalars, casting='unsafe')
        volume[title] = np.array(scalars, dtype=np.uint8)

        self.mapper.scalar_range = clim

        if isinstance(cmap, pyvista.LookupTable):
            self.mapper.lookup_table = cmap
        else:
            if cmap is None:
                if not has_module('matplotlib'):
                    raise ImportError('Please install matplotlib for color maps.')

                cmap = self._theme.cmap

            cmap = get_cmap_safe(cmap)
            if categories:
                if categories is True:
                    n_colors = len(np.unique(scalars))
                elif isinstance(categories, int):
                    n_colors = categories

            if flip_scalars:
                cmap = cmap.reversed()

            # Set colormap and build lookup table
            self.mapper.lookup_table.apply_cmap(cmap, n_colors)
            self.mapper.lookup_table.apply_opacity(opacity)
            self.mapper.lookup_table.scalar_range = clim
            if isinstance(annotations, dict):
                self.mapper.lookup_table.annotations = annotations

        self.mapper.dataset = volume

        blending = blending.lower()
        if blending in ['additive', 'add', 'sum']:
            self.mapper.SetBlendModeToAdditive()
        elif blending in ['average', 'avg', 'average_intensity']:
            self.mapper.SetBlendModeToAverageIntensity()
        elif blending in ['composite', 'comp']:
            self.mapper.SetBlendModeToComposite()
        elif blending in ['maximum', 'max', 'maximum_intensity']:
            self.mapper.SetBlendModeToMaximumIntensity()
        elif blending in ['minimum', 'min', 'minimum_intensity']:
            self.mapper.SetBlendModeToMinimumIntensity()
        else:
            raise ValueError(
                f'Blending mode {blending!r} invalid. '
                'Please choose either "additive", '
                '"composite", "minimum" or "maximum".'
            )
        self.mapper.update()

        self.volume = _vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)

        prop = _vtk.vtkVolumeProperty()
        prop.SetColor(self.mapper.lookup_table.to_color_tf())
        prop.SetScalarOpacity(self.mapper.lookup_table.to_opacity_tf())
        prop.SetAmbient(ambient)
        prop.SetScalarOpacityUnitDistance(opacity_unit_distance)
        prop.SetShade(shade)
        prop.SetDiffuse(diffuse)
        prop.SetSpecular(specular)
        prop.SetSpecularPower(specular_power)
        self.volume.SetProperty(prop)

        actor, prop = self.add_actor(
            self.volume,
            reset_camera=reset_camera,
            name=name,
            culling=culling,
            pickable=pickable,
            render=render,
        )

        # Add scalar bar if scalars are available
        if show_scalar_bar and scalars is not None:
            self.add_scalar_bar(**scalar_bar_args)

        self.renderer.Modified()

        return actor

    def add_silhouette(self, mesh, params=None):
        """Add a silhouette of a PyVista or VTK dataset to the scene.

        A silhouette can also be generated directly in
        :func:`add_mesh <pyvista.Plotter.add_mesh>`. See also
        :ref:`silhouette_example`.

        Parameters
        ----------
        mesh : pyvista.PolyData
            Mesh for generating silhouette to plot.

        params : dict, optional

            * If not supplied, the default theme values will be used.
            * ``color``: ``color_like``, color of the silhouette
            * ``line_width``: ``float``, edge width
            * ``opacity``: ``float`` between 0 and 1, edge transparency
            * ``feature_angle``: If a ``float``, display sharp edges
              exceeding that angle in degrees.
            * ``decimate``: ``float`` between 0 and 1, level of decimation

        Returns
        -------
        vtk.vtkActor
            VTK actor of the silhouette.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> bunny = examples.download_bunny()
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(bunny, color='tan')
        >>> _ = plotter.add_silhouette(bunny,
        ...     params={'color': 'red', 'line_width': 8.0})
        >>> plotter.view_xy()
        >>> plotter.show()

        """
        silhouette_params = self._theme.silhouette.to_dict()
        if params:
            silhouette_params.update(params)

        if not is_pyvista_dataset(mesh):
            mesh = wrap(mesh)
        if not isinstance(mesh, pyvista.PolyData):
            raise TypeError(f"Expected type is `PolyData` but {type(mesh)} was given.")

        if isinstance(silhouette_params["decimate"], float):
            silhouette_mesh = mesh.decimate(silhouette_params["decimate"])
        else:
            silhouette_mesh = mesh
        alg = _vtk.vtkPolyDataSilhouette()
        alg.SetInputData(silhouette_mesh)
        alg.SetCamera(self.renderer.camera)
        if silhouette_params["feature_angle"] is not None:
            alg.SetEnableFeatureAngle(True)
            alg.SetFeatureAngle(silhouette_params["feature_angle"])
        else:
            alg.SetEnableFeatureAngle(False)
        mapper = DataSetMapper()
        mapper.SetInputConnection(alg.GetOutputPort())
        actor, prop = self.add_actor(mapper)
        prop.SetColor(Color(silhouette_params["color"]).float_rgb)
        prop.SetOpacity(silhouette_params["opacity"])
        prop.SetLineWidth(silhouette_params["line_width"])

        return actor

    def update_scalar_bar_range(self, clim, name=None):
        """Update the value range of the active or named scalar bar.

        Parameters
        ----------
        clim : sequence
            The new range of scalar bar. Two item list (e.g. ``[-1, 2]``).

        name : str, optional
            The title of the scalar bar to update.

        """
        if isinstance(clim, float) or isinstance(clim, int):
            clim = [-clim, clim]
        if len(clim) != 2:
            raise TypeError('clim argument must be a length 2 iterable of values: (min, max).')
        if name is None:
            if not hasattr(self, 'mapper'):
                raise AttributeError('This plotter does not have an active mapper.')
            self.mapper.scalar_range = clim
            return

        # Use the name to find the desired actor
        def update_mapper(mapper_helper):
            mapper_helper.scalar_range = clim
            return

        try:
            for mh in self._scalar_bar_mappers[name]:
                update_mapper(mh)
        except KeyError:
            raise KeyError('Name ({}) not valid/not found in this plotter.')
        return

    def clear_actors(self):
        """Clear actors from all renderers."""
        self.renderers.clear_actors()

    def clear(self):
        """Clear plot by removing all actors and properties.

        Examples
        --------
        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> actor = plotter.add_mesh(pyvista.Sphere())
        >>> plotter.clear()
        >>> plotter.renderer.actors
        {}

        """
        self.renderers.clear()
        self.scalar_bars.clear()
        self.mesh = None
        self.mapper = None

    def link_views(self, views=0):
        """Link the views' cameras.

        Parameters
        ----------
        views : int | tuple or list
            If ``views`` is int, link the views to the given view
            index or if ``views`` is a tuple or a list, link the given
            views cameras.

        Examples
        --------
        Not linked view case.

        >>> import pyvista
        >>> from pyvista import demos
        >>> ocube = demos.orientation_cube()
        >>> pl = pyvista.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> _ = pl.add_mesh(ocube['cube'], show_edges=True)
        >>> _ = pl.add_mesh(ocube['x_p'], color='blue')
        >>> _ = pl.add_mesh(ocube['x_n'], color='blue')
        >>> _ = pl.add_mesh(ocube['y_p'], color='green')
        >>> _ = pl.add_mesh(ocube['y_n'], color='green')
        >>> _ = pl.add_mesh(ocube['z_p'], color='red')
        >>> _ = pl.add_mesh(ocube['z_n'], color='red')
        >>> pl.camera_position = 'yz'
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(ocube['cube'], show_edges=True)
        >>> _ = pl.add_mesh(ocube['x_p'], color='blue')
        >>> _ = pl.add_mesh(ocube['x_n'], color='blue')
        >>> _ = pl.add_mesh(ocube['y_p'], color='green')
        >>> _ = pl.add_mesh(ocube['y_n'], color='green')
        >>> _ = pl.add_mesh(ocube['z_p'], color='red')
        >>> _ = pl.add_mesh(ocube['z_n'], color='red')
        >>> pl.show_axes()
        >>> pl.show()

        Linked view case.

        >>> pl = pyvista.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> _ = pl.add_mesh(ocube['cube'], show_edges=True)
        >>> _ = pl.add_mesh(ocube['x_p'], color='blue')
        >>> _ = pl.add_mesh(ocube['x_n'], color='blue')
        >>> _ = pl.add_mesh(ocube['y_p'], color='green')
        >>> _ = pl.add_mesh(ocube['y_n'], color='green')
        >>> _ = pl.add_mesh(ocube['z_p'], color='red')
        >>> _ = pl.add_mesh(ocube['z_n'], color='red')
        >>> pl.camera_position = 'yz'
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(ocube['cube'], show_edges=True)
        >>> _ = pl.add_mesh(ocube['x_p'], color='blue')
        >>> _ = pl.add_mesh(ocube['x_n'], color='blue')
        >>> _ = pl.add_mesh(ocube['y_p'], color='green')
        >>> _ = pl.add_mesh(ocube['y_n'], color='green')
        >>> _ = pl.add_mesh(ocube['z_p'], color='red')
        >>> _ = pl.add_mesh(ocube['z_n'], color='red')
        >>> pl.show_axes()
        >>> pl.link_views()
        >>> pl.show()

        """
        if isinstance(views, (int, np.integer)):
            for renderer in self.renderers:
                renderer.camera = self.renderers[views].camera
            return
        views = np.asarray(views)
        if np.issubdtype(views.dtype, np.integer):
            for view_index in views:
                self.renderers[view_index].camera = self.renderers[views[0]].camera
        else:
            raise TypeError(f'Expected type is int, list or tuple: {type(views)} is given')

    def unlink_views(self, views=None):
        """Unlink the views' cameras.

        Parameters
        ----------
        views : None, int, tuple or list
            If ``views`` is None unlink all the views, if ``views``
            is int unlink the selected view's camera or if ``views``
            is a tuple or a list, unlink the given views cameras.

        """
        if views is None:
            for renderer in self.renderers:
                renderer.camera = Camera()
                renderer.reset_camera()
        elif isinstance(views, int):
            self.renderers[views].camera = Camera()
            self.renderers[views].reset_camera()
        elif isinstance(views, collections.abc.Iterable):
            for view_index in views:
                self.renderers[view_index].camera = Camera()
                self.renderers[view_index].reset_camera()
        else:
            raise TypeError(f'Expected type is None, int, list or tuple: {type(views)} is given')

    @wraps(ScalarBars.add_scalar_bar)
    def add_scalar_bar(self, *args, **kwargs):
        """Wrap for ``ScalarBars.add_scalar_bar``."""
        # only render when the plotter has already been shown
        render = kwargs.get('render', None)
        if render is None:
            kwargs['render'] = not self._first_time

        # check if maper exists
        mapper = kwargs.get('mapper', None)
        if mapper is None:
            if not hasattr(self, 'mapper') or self.mapper is None:
                raise AttributeError('Mapper does not exist.  Add a mesh with scalars first.')
            kwargs['mapper'] = self.mapper

        # title can be the first and only arg
        if len(args):
            title = args[0]
        else:
            title = kwargs.get('title', '')
        if title is None:
            title = ''
        kwargs['title'] = title

        interactive = kwargs.get('interactive', None)
        if interactive is None:
            interactive = self._theme.interactive
            if self.shape != (1, 1):
                interactive = False
        elif interactive and self.shape != (1, 1):
            raise ValueError('Interactive scalar bars disabled for multi-renderer plots')
        # by default, use the plotter local theme
        kwargs.setdefault('theme', self._theme)
        return self.scalar_bars.add_scalar_bar(**kwargs)

    def update_scalars(self, scalars, mesh=None, render=True):
        """Update scalars of an object in the plotter.

        Parameters
        ----------
        scalars : np.ndarray
            Scalars to replace existing scalars.

        mesh : vtk.PolyData or vtk.UnstructuredGrid, optional
            Object that has already been added to the Plotter.  If
            None, uses last added mesh.

        render : bool, optional
            Force a render when True.  Default ``True``.
        """
        if mesh is None:
            mesh = self.mesh

        if isinstance(mesh, (collections.abc.Iterable, pyvista.MultiBlock)):
            # Recursive if need to update scalars on many meshes
            for m in mesh:
                self.update_scalars(scalars, mesh=m, render=False)
            if render:
                self.render()
            return

        if isinstance(scalars, str):
            # Grab scalars array if name given
            scalars = get_array(mesh, scalars)

        if scalars is None:
            if render:
                self.render()
            return

        if scalars.shape[0] == mesh.GetNumberOfPoints():
            data = mesh.GetPointData()
        elif scalars.shape[0] == mesh.GetNumberOfCells():
            data = mesh.GetCellData()
        else:
            raise_not_matching(scalars, mesh)

        vtk_scalars = data.GetScalars()
        if vtk_scalars is None:
            raise ValueError('No active scalars')
        s = convert_array(vtk_scalars)
        s[:] = scalars
        data.Modified()
        try:
            # Why are the points updated here? Not all datasets have points
            # and only the scalars array is modified by this function...
            mesh.GetPoints().Modified()
        except:
            pass

        if render:
            self.render()

    def update_coordinates(self, points, mesh=None, render=True):
        """Update the points of an object in the plotter.

        Parameters
        ----------
        points : np.ndarray
            Points to replace existing points.

        mesh : vtk.PolyData or vtk.UnstructuredGrid, optional
            Object that has already been added to the Plotter.  If
            None, uses last added mesh.

        render : bool, optional
            Force a render when True.  Default ``True``.
        """
        if mesh is None:
            mesh = self.mesh

        mesh.points = points

        # only render when the plotter has already been shown
        if render is None:
            render = not self._first_time

        if render:
            self.render()

    def _clear_ren_win(self):
        """Clear the render window."""
        if hasattr(self, 'ren_win'):
            self.ren_win.Finalize()
            del self.ren_win

    def close(self, render=False):
        """Close the render window.

        Parameters
        ----------
        render : bool
            Unused argument.

        """
        # optionally run just prior to exiting the plotter
        if self._before_close_callback is not None:
            self._before_close_callback(self)
            self._before_close_callback = None

        # must close out widgets first
        super().close()
        # Renderer has an axes widget, so close it
        self.renderers.close()
        self.renderers.remove_all_lights()

        # Grab screenshots of last render
        if self._store_image:
            self.last_image = self.screenshot(None, return_img=True)
            self.last_image_depth = self.get_image_depth()

        # reset scalar bars
        self.scalar_bars.clear()
        self.mesh = None
        self.mapper = None

        # grab the display id before clearing the window
        # this is an experimental feature
        if KILL_DISPLAY:  # pragma: no cover
            disp_id = None
            if hasattr(self, 'ren_win'):
                disp_id = self.ren_win.GetGenericDisplayId()
        self._clear_ren_win()

        if self.iren is not None:
            self.iren.close()
            if KILL_DISPLAY:  # pragma: no cover
                _kill_display(disp_id)
            self.iren = None

        if hasattr(self, 'textActor'):
            del self.textActor

        # end movie
        if hasattr(self, 'mwriter'):
            try:
                self.mwriter.close()
            except BaseException:
                pass

        # Remove the global reference to this plotter unless building the
        # gallery to allow it to collect.
        if not pyvista.BUILDING_GALLERY:
            if _ALL_PLOTTERS is not None:
                _ALL_PLOTTERS.pop(self._id_name, None)

        # this helps managing closed plotters
        self._closed = True

    def deep_clean(self):
        """Clean the plotter of the memory."""
        self.disable_picking()
        if hasattr(self, 'renderers'):
            self.renderers.deep_clean()
        self.mesh = None
        self.mapper = None
        self.volume = None
        self.textActor = None

    def add_text(
        self,
        text,
        position='upper_left',
        font_size=18,
        color=None,
        font=None,
        shadow=False,
        name=None,
        viewport=False,
        orientation=0.0,
        *,
        render=True,
    ):
        """Add text to plot object in the top left corner by default.

        Parameters
        ----------
        text : str
            The text to add the rendering.

        position : str, tuple(float), optional
            Position to place the bottom left corner of the text box.
            If tuple is used, the position of the text uses the pixel
            coordinate system (default). In this case,
            it returns a more general `vtkOpenGLTextActor`.
            If string name is used, it returns a `vtkCornerAnnotation`
            object normally used for fixed labels (like title or xlabel).
            Default is to find the top left corner of the rendering window
            and place text box up there. Available position: ``'lower_left'``,
            ``'lower_right'``, ``'upper_left'``, ``'upper_right'``,
            ``'lower_edge'``, ``'upper_edge'``, ``'right_edge'``, and
            ``'left_edge'``.

        font_size : float, optional
            Sets the size of the title font.  Defaults to 18.

        color : color_like, optional
            Either a string, RGB list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

            Defaults to :attr:`pyvista.global_theme.font.color <pyvista.themes._Font.color>`.

        font : str, optional
            Font name may be ``'courier'``, ``'times'``, or ``'arial'``.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to ``False``.

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        viewport : bool, optional
            If ``True`` and position is a tuple of float, uses the
            normalized viewport coordinate system (values between 0.0
            and 1.0 and support for HiDPI).

        orientation : float, optional
            Angle orientation of text counterclockwise in degrees.  The text
            is rotated around an anchor point that may be on the edge or
            corner of the text.  The default is 0 degrees, which is horizontal.

        render : bool, optional
            Force a render when ``True`` (default).

        Returns
        -------
        vtk.vtkTextActor
            Text actor added to plot.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_text('Sample Text', position='upper_right', color='blue',
        ...                     shadow=True, font_size=26)
        >>> pl.show()

        """
        if font is None:
            font = self._theme.font.family
        if font_size is None:
            font_size = self._theme.font.size
        if position is None:
            # Set the position of the text to the top left corner
            window_size = self.window_size
            x = (window_size[0] * 0.02) / self.shape[0]
            y = (window_size[1] * 0.85) / self.shape[0]
            position = [x, y]

        corner_mappings = {
            'lower_left': _vtk.vtkCornerAnnotation.LowerLeft,
            'lower_right': _vtk.vtkCornerAnnotation.LowerRight,
            'upper_left': _vtk.vtkCornerAnnotation.UpperLeft,
            'upper_right': _vtk.vtkCornerAnnotation.UpperRight,
            'lower_edge': _vtk.vtkCornerAnnotation.LowerEdge,
            'upper_edge': _vtk.vtkCornerAnnotation.UpperEdge,
            'left_edge': _vtk.vtkCornerAnnotation.LeftEdge,
            'right_edge': _vtk.vtkCornerAnnotation.RightEdge,
        }
        corner_mappings['ll'] = corner_mappings['lower_left']
        corner_mappings['lr'] = corner_mappings['lower_right']
        corner_mappings['ul'] = corner_mappings['upper_left']
        corner_mappings['ur'] = corner_mappings['upper_right']
        corner_mappings['top'] = corner_mappings['upper_edge']
        corner_mappings['bottom'] = corner_mappings['lower_edge']
        corner_mappings['right'] = corner_mappings['right_edge']
        corner_mappings['r'] = corner_mappings['right_edge']
        corner_mappings['left'] = corner_mappings['left_edge']
        corner_mappings['l'] = corner_mappings['left_edge']

        if isinstance(position, (int, str, bool)):
            if isinstance(position, str):
                position = corner_mappings[position]
            elif position is True:
                position = corner_mappings['upper_left']
            self.textActor = _vtk.vtkCornerAnnotation()
            # This is how you set the font size with this actor
            self.textActor.SetLinearFontScaleFactor(font_size // 2)
            self.textActor.SetText(position, text)
        else:
            self.textActor = _vtk.vtkTextActor()
            self.textActor.SetInput(text)
            self.textActor.SetPosition(position)
            if viewport:
                self.textActor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                self.textActor.GetActualPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
            self.textActor.GetTextProperty().SetFontSize(int(font_size * 2))

        text_prop = self.textActor.GetTextProperty()
        text_prop.SetColor(Color(color, default_color=self._theme.font.color).float_rgb)
        text_prop.SetFontFamily(FONTS[font].value)
        text_prop.SetShadow(shadow)
        text_prop.SetOrientation(orientation)

        self.add_actor(self.textActor, reset_camera=False, name=name, pickable=False, render=render)
        return self.textActor

    def open_movie(self, filename, framerate=24, quality=5, **kwargs):
        """Establish a connection to the ffmpeg writer.

        Parameters
        ----------
        filename : str
            Filename of the movie to open.  Filename should end in mp4,
            but other filetypes may be supported.  See :func:`imageio.get_writer()
            <imageio.v2.get_writer>`.

        framerate : int, optional
            Frames per second.

        quality : int, optional
            Quality 10 is the top possible quality for any codec. The
            range is ``0 - 10``.  Higher quality leads to a larger file.

        **kwargs : dict, optional
            See the documentation for :func:`imageio.get_writer()
            <imageio.v2.get_writer>` for additional kwargs.

        Notes
        -----
        See the documentation for :func:`imageio.get_writer() <imageio.v2.get_writer>`.

        Examples
        --------
        Open a MP4 movie and set the quality to maximum.

        >>> import pyvista
        >>> pl = pyvista.Plotter
        >>> pl.open_movie('movie.mp4', quality=10)  # doctest:+SKIP

        """
        from imageio import get_writer

        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        self.mwriter = get_writer(filename, fps=framerate, quality=quality, **kwargs)

    def open_gif(self, filename, loop=0, fps=10, palettesize=256, subrectangles=False, **kwargs):
        """Open a gif file.

        Parameters
        ----------
        filename : str
            Filename of the gif to open.  Filename must end in ``"gif"``.

        loop : int, optional
            The number of iterations. Default 0 (meaning loop indefinitely).

        fps : float, optional
            The number of frames per second. If duration is not given, the
            duration for each frame is set to 1/fps. Default 10.

        palettesize : int, optional
            The number of colors to quantize the image to. Is rounded to the
            nearest power of two. Must be between 2 and 256. Default 256.

        subrectangles : bool, optional
            If ``True``, will try and optimize the GIF by storing only the rectangular
            parts of each frame that change with respect to the previous. Default
            ``False``.

            .. note::
               Setting this to ``True`` may help reduce jitter in colorbars.

        **kwargs : dict, optional
            See the documentation for :func:`imageio.get_writer() <imageio.v2.get_writer>`
            for additional kwargs.

        Notes
        -----
        Consider using `pygifsicle
        <https://github.com/LucaCappelletti94/pygifsicle>`_ to reduce the final
        size of the gif. See `Optimizing a GIF using pygifsicle
        <https://imageio.readthedocs.io/en/stable/examples.html#optimizing-a-gif-using-pygifsicle>`_.

        Examples
        --------
        Open a gif file, setting the framerate to 8 frames per second and
        reducing the colorspace to 64.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.open_gif('movie.gif', fps=8, palettesize=64)  # doctest:+SKIP

        See :ref:`gif_movie_example` for a full example using this method.

        """
        from imageio import get_writer

        if filename[-3:] != 'gif':
            raise ValueError('Unsupported filetype.  Must end in .gif')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        self._gif_filename = os.path.abspath(filename)
        self.mwriter = get_writer(
            filename,
            mode='I',
            loop=loop,
            fps=fps,
            palettesize=palettesize,
            subrectangles=subrectangles,
            **kwargs,
        )

    def write_frame(self):
        """Write a single frame to the movie file.

        Examples
        --------
        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> plotter.open_movie(filename)  # doctest:+SKIP
        >>> plotter.add_mesh(pyvista.Sphere())  # doctest:+SKIP
        >>> plotter.write_frame()  # doctest:+SKIP

        See :ref:`movie_example` for a full example using this method.

        """
        # if off screen, show has not been called and we must render
        # before extracting an image
        if self._first_time:
            self._on_first_render_request()
            self.render()

        if not hasattr(self, 'mwriter'):
            raise RuntimeError('This plotter has not opened a movie or GIF file.')
        self.update()
        self.mwriter.append_data(self.image)

    def _run_image_filter(self, ifilter):
        # Update filter and grab pixels
        ifilter.Modified()
        ifilter.Update()
        image = pyvista.wrap(ifilter.GetOutput())
        img_size = image.dimensions
        img_array = pyvista.utilities.point_array(image, 'ImageScalars')

        # Reshape and write
        tgt_size = (img_size[1], img_size[0], -1)
        return img_array.reshape(tgt_size)[::-1]

    def get_image_depth(self, fill_value=np.nan, reset_camera_clipping_range=True):
        """Return a depth image representing current render window.

        Parameters
        ----------
        fill_value : float, optional
            Fill value for points in image that do not include objects
            in scene.  To not use a fill value, pass ``None``.

        reset_camera_clipping_range : bool, optional
            Reset the camera clipping range to include data in view.

        Returns
        -------
        numpy.ndarray
            Image of depth values from camera orthogonal to image
            plane.

        Notes
        -----
        Values in image_depth are negative to adhere to a
        right-handed coordinate system.

        Examples
        --------
        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> actor = plotter.add_mesh(pyvista.Sphere())
        >>> plotter.store_image = True
        >>> plotter.show()
        >>> zval = plotter.get_image_depth()

        """
        # allow no render window
        if not hasattr(self, 'ren_win') and self.last_image_depth is not None:
            zval = self.last_image_depth.copy()
            if fill_value is not None:
                zval[self._image_depth_null] = fill_value
            return zval

        self._check_rendered()
        self._check_has_ren_win()

        # Ensure points in view are within clipping range of renderer?
        if reset_camera_clipping_range:
            self.renderer.ResetCameraClippingRange()

        # Get the z-buffer image
        ifilter = _vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.ren_win)
        ifilter.ReadFrontBufferOff()
        ifilter.SetInputBufferTypeToZBuffer()
        zbuff = self._run_image_filter(ifilter)[:, :, 0]

        # Convert z-buffer values to depth from camera
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            near, far = self.camera.clipping_range
            if self.camera.parallel_projection:
                zval = (zbuff - near) / (far - near)
            else:
                zval = 2 * near * far / ((zbuff - 0.5) * 2 * (far - near) - near - far)

            # Consider image values outside clipping range as nans
            self._image_depth_null = np.logical_or(zval < -far, np.isclose(zval, -far))

        if fill_value is not None:
            zval[self._image_depth_null] = fill_value

        return zval

    def add_lines(self, lines, color='w', width=5, label=None, name=None):
        """Add lines to the plotting object.

        Parameters
        ----------
        lines : np.ndarray or pyvista.PolyData
            Points representing line segments.  For example, two line
            segments would be represented as ``np.array([[0, 0, 0],
            [1, 0, 0], [1, 0, 0], [1, 1, 0]])``.

        color : color_like, optional
            Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        width : float, optional
            Thickness of lines.

        label : str, optional
            String label to use when adding a legend to the scene with
            :func:`pyvista.BasePlotter.add_legend`.

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        Returns
        -------
        vtk.vtkActor
            Lines actor.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> points = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]])
        >>> actor = pl.add_lines(points, color='yellow', width=3)
        >>> pl.camera_position = 'xy'
        >>> pl.show()

        """
        if not isinstance(lines, np.ndarray):
            raise TypeError('Input should be an array of point segments')

        lines = pyvista.lines_from_points(lines)

        actor = Actor(mapper=DataSetMapper(lines))
        actor.prop.line_width = width
        actor.prop.show_edges = True
        actor.prop.edge_color = color
        actor.prop.color = color
        actor.prop.lighting = False

        # legend label
        if label:
            if not isinstance(label, str):
                raise TypeError('Label must be a string')
            addr = actor.GetAddressAsString("")
            self.renderer._labels[addr] = [lines, label, Color(color)]

        # Add to renderer
        self.add_actor(actor, reset_camera=False, name=name, pickable=False)
        return actor

    @wraps(ScalarBars.remove_scalar_bar)
    def remove_scalar_bar(self, *args, **kwargs):
        """Remove the active scalar bar."""
        self.scalar_bars.remove_scalar_bar(*args, **kwargs)

    def add_point_labels(
        self,
        points,
        labels,
        italic=False,
        bold=True,
        font_size=None,
        text_color=None,
        font_family=None,
        shadow=False,
        show_points=True,
        point_color=None,
        point_size=5,
        name=None,
        shape_color='grey',
        shape='rounded_rect',
        fill_shape=True,
        margin=3,
        shape_opacity=1.0,
        pickable=False,
        render_points_as_spheres=False,
        tolerance=0.001,
        reset_camera=None,
        always_visible=False,
        render=True,
    ):
        """Create a point actor with one label from list labels assigned to each point.

        Parameters
        ----------
        points : sequence or pyvista.DataSet
            An ``n x 3`` sequence points or pyvista dataset with points.

        labels : list or str
            List of labels.  Must be the same length as points. If a
            string name is given with a :class:`pyvista.DataSet` input for
            points, then these are fetched.

        italic : bool, optional
            Italicises title and bar labels.  Default ``False``.

        bold : bool, optional
            Bolds title and bar labels.  Default ``True``.

        font_size : float, optional
            Sets the size of the title font.  Defaults to 16.

        text_color : color_like, optional
            Color of text. Either a string, RGB sequence, or hex color string.

            * ``text_color='white'``
            * ``text_color='w'``
            * ``text_color=[1.0, 1.0, 1.0]``
            * ``text_color='#FFFFFF'``

        font_family : str, optional
            Font family.  Must be either ``'courier'``, ``'times'``,
            or ``'arial``.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to ``False``.

        show_points : bool, optional
            Controls if points are visible.  Default ``True``.

        point_color : color_like, optional
            Either a string, rgb list, or hex color string.  One of
            the following.

            * ``point_color='white'``
            * ``point_color='w'``
            * ``point_color=[1.0, 1.0, 1.0]``
            * ``point_color='#FFFFFF'``

        point_size : float, optional
            Size of points if visible.

        name : str, optional
            The name for the added actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        shape_color : color_like, optional
            Color of shape (if visible).  Either a string, rgb
            sequence, or hex color string.

        shape : str, optional
            The string name of the shape to use. Options are ``'rect'`` or
            ``'rounded_rect'``. If you want no shape, pass ``None``.

        fill_shape : bool, optional
            Fill the shape with the ``shape_color``. Outlines if ``False``.

        margin : int, optional
            The size of the margin on the label background shape. Default is 3.

        shape_opacity : float, optional
            The opacity of the shape in the range of ``[0, 1]``.

        pickable : bool, optional
            Set whether this actor is pickable.

        render_points_as_spheres : bool, optional
            Render points as spheres rather than dots.

        tolerance : float, optional
            A tolerance to use to determine whether a point label is
            visible.  A tolerance is usually required because the
            conversion from world space to display space during
            rendering introduces numerical round-off.

        reset_camera : bool, optional
            Reset the camera after adding the points to the scene.

        always_visible : bool, optional
            Skip adding the visibility filter. Default False.

        render : bool, optional
            Force a render when ``True`` (default).

        Returns
        -------
        vtk.vtkActor2D
            VTK label actor.  Can be used to change properties of the labels.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> points = np.array([[0.0, 0.0, 0.0],
        ...                    [1.0, 1.0, 0.0],
        ...                    [2.0, 0.0, 0.0]])
        >>> labels = ['Point A', 'Point B', 'Point C']
        >>> actor = pl.add_point_labels(points, labels, italic=True, font_size=20,
        ...                             point_color='red', point_size=20,
        ...                             render_points_as_spheres=True,
        ...                             always_visible=True, shadow=True)
        >>> pl.camera_position = 'xy'
        >>> pl.show()

        """
        if font_family is None:
            font_family = self._theme.font.family
        if font_size is None:
            font_size = self._theme.font.size
        point_color = Color(point_color, default_color=self._theme.color)

        if isinstance(points, (list, tuple)):
            points = np.array(points)

        if isinstance(points, np.ndarray):
            vtkpoints = pyvista.PolyData(points)  # Cast to poly data
        elif is_pyvista_dataset(points):
            vtkpoints = pyvista.PolyData(points.points)
            if isinstance(labels, str):
                labels = points.point_data[labels]
        else:
            raise TypeError(f'Points type not usable: {type(points)}')

        if len(vtkpoints.points) != len(labels):
            raise ValueError('There must be one label for each point')

        if name is None:
            name = f'{type(vtkpoints).__name__}({vtkpoints.memory_address})'

        vtklabels = _vtk.vtkStringArray()
        vtklabels.SetName('labels')
        for item in labels:
            vtklabels.InsertNextValue(str(item))
        vtkpoints.GetPointData().AddArray(vtklabels)

        # Create hierarchy
        hier = _vtk.vtkPointSetToLabelHierarchy()
        hier.SetLabelArrayName('labels')

        if always_visible:
            hier.SetInputData(vtkpoints)
        else:
            # Only show visible points
            vis_points = _vtk.vtkSelectVisiblePoints()
            vis_points.SetInputData(vtkpoints)
            vis_points.SetRenderer(self.renderer)
            vis_points.SetTolerance(tolerance)

            hier.SetInputConnection(vis_points.GetOutputPort())

        # create label mapper
        labelMapper = _vtk.vtkLabelPlacementMapper()
        labelMapper.SetInputConnection(hier.GetOutputPort())
        if not isinstance(shape, str):
            labelMapper.SetShapeToNone()
        elif shape.lower() in 'rect':
            labelMapper.SetShapeToRect()
        elif shape.lower() in 'rounded_rect':
            labelMapper.SetShapeToRoundedRect()
        else:
            raise ValueError(f'Shape ({shape}) not understood')
        if fill_shape:
            labelMapper.SetStyleToFilled()
        else:
            labelMapper.SetStyleToOutline()
        labelMapper.SetBackgroundColor(Color(shape_color).float_rgb)
        labelMapper.SetBackgroundOpacity(shape_opacity)
        labelMapper.SetMargin(margin)

        textprop = hier.GetTextProperty()
        textprop.SetItalic(italic)
        textprop.SetBold(bold)
        textprop.SetFontSize(font_size)
        textprop.SetFontFamily(parse_font_family(font_family))
        textprop.SetColor(Color(text_color, default_color=self._theme.font.color).float_rgb)
        textprop.SetShadow(shadow)

        self.remove_actor(f'{name}-points', reset_camera=False)
        self.remove_actor(f'{name}-labels', reset_camera=False)

        # add points
        if show_points:
            self.add_mesh(
                vtkpoints,
                color=point_color,
                point_size=point_size,
                name=f'{name}-points',
                pickable=pickable,
                render_points_as_spheres=render_points_as_spheres,
                reset_camera=reset_camera,
                render=render,
            )

        label_actor = _vtk.vtkActor2D()
        label_actor.SetMapper(labelMapper)
        self.add_actor(label_actor, reset_camera=False, name=f'{name}-labels', pickable=False)
        return label_actor

    def add_point_scalar_labels(self, points, labels, fmt=None, preamble='', **kwargs):
        """Label the points from a dataset with the values of their scalars.

        Wrapper for :func:`pyvista.BasePlotter.add_point_labels`.

        Parameters
        ----------
        points : Sequence(float) or np.ndarray or pyvista.DataSet
            An ``n x 3`` numpy.ndarray or pyvista dataset with points.

        labels : list or str
            List of scalars of labels.  Must be the same length as points. If a
            string name is given with a :class:`pyvista.DataSet` input for
            points, then these are fetched.

        fmt : str, optional
            String formatter used to format numerical data.

        preamble : str, optional
            Text before the start of each label.

        **kwargs : dict, optional
            Keyword arguments passed to
            :func:`pyvista.BasePlotter.add_point_labels`.

        Returns
        -------
        vtk.vtkActor2D
            VTK label actor.  Can be used to change properties of the labels.

        """
        if not is_pyvista_dataset(points):
            points, _ = _coerce_pointslike_arg(points, copy=False)
        if not isinstance(labels, (str, list)):
            raise TypeError(
                'labels must be a string name of the scalars array to use or list of scalars'
            )
        if fmt is None:
            fmt = self._theme.font.fmt
        if fmt is None:
            fmt = '%.6e'
        if isinstance(points, np.ndarray):
            scalars = labels
        elif is_pyvista_dataset(points):
            scalars = points.point_data[labels]
        phrase = f'{preamble} {fmt}'
        labels = [phrase % val for val in scalars]
        return self.add_point_labels(points, labels, **kwargs)

    def add_points(self, points, style='points', **kwargs):
        """Add points to a mesh.

        Parameters
        ----------
        points : numpy.ndarray or pyvista.DataSet
            Array of points or the points from a pyvista object.

        style : str, default: 'points'
            Visualization style of the mesh.  One of the following:
            ``style='points'``, ``style='points_gaussian'``.
            ``'points_gaussian'`` can be controlled with the ``emissive`` and
            ``render_points_as_spheres`` options.

        **kwargs : dict, optional
            See :func:`pyvista.BasePlotter.add_mesh` for optional
            keyword arguments.

        Returns
        -------
        pyvista.Actor
            Actor of the mesh.

        Examples
        --------
        Add a numpy array of points to a mesh.

        >>> import numpy as np
        >>> import pyvista
        >>> points = np.random.random((10, 3))
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_points(points, render_points_as_spheres=True,
        ...                       point_size=100.0)
        >>> pl.show()

        Plot using the ``'points_gaussian'`` style

        >>> points = np.random.random((10, 3))
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_points(points, style='points_gaussian')
        >>> pl.show()

        """
        if style not in ['points', 'points_gaussian']:
            raise ValueError(
                f'Invalid style {style} for add_points. Should be either "points" or '
                '"points_gaussian".'
            )
        return self.add_mesh(points, style=style, **kwargs)

    def add_arrows(self, cent, direction, mag=1, **kwargs):
        """Add arrows to the plotter.

        Parameters
        ----------
        cent : np.ndarray
            Array of centers.

        direction : np.ndarray
            Array of direction vectors.

        mag : float, optional
            Amount to scale the direction vectors.

        **kwargs : dict, optional
            See :func:`pyvista.BasePlotter.add_mesh` for optional
            keyword arguments.

        Returns
        -------
        pyvista.Actor
            Actor of the arrows.

        Examples
        --------
        Plot a random field of vectors and save a screenshot of it.

        >>> import numpy as np
        >>> import pyvista
        >>> cent = np.random.random((10, 3))
        >>> direction = np.random.random((10, 3))
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_arrows(cent, direction, mag=2)
        >>> plotter.show()

        """
        if cent.shape != direction.shape:  # pragma: no cover
            raise ValueError('center and direction arrays must have the same shape')

        direction = direction.copy()
        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        if mag != 1:
            direction = direction * mag

        pdata = pyvista.vector_poly_data(cent, direction)
        # Create arrow object
        arrow = _vtk.vtkArrowSource()
        arrow.Update()
        glyph3D = _vtk.vtkGlyph3D()
        glyph3D.SetSourceData(arrow.GetOutput())
        glyph3D.SetInputData(pdata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.Update()

        arrows = wrap(glyph3D.GetOutput())
        return self.add_mesh(arrows, **kwargs)

    @staticmethod
    def _save_image(image, filename, return_img):
        """Save to file and/or return a NumPy image array.

        This is an internal helper.

        """
        if not image.size:
            raise ValueError('Empty image. Have you run plot() first?')
        # write screenshot to file if requested
        if isinstance(filename, (str, pathlib.Path, io.BytesIO)):
            from PIL import Image

            if isinstance(filename, (str, pathlib.Path)):
                filename = pathlib.Path(filename)
                if isinstance(pyvista.FIGURE_PATH, str) and not filename.is_absolute():
                    filename = pathlib.Path(os.path.join(pyvista.FIGURE_PATH, filename))
                if not filename.suffix:
                    filename = filename.with_suffix('.png')
                elif filename.suffix not in SUPPORTED_FORMATS:
                    raise ValueError(
                        f'Unsupported extension {filename.suffix}\n'
                        f'Must be one of the following: {SUPPORTED_FORMATS}'
                    )
                filename = os.path.abspath(os.path.expanduser(str(filename)))
                Image.fromarray(image).save(filename)
            else:
                Image.fromarray(image).save(filename, format="PNG")
        # return image array if requested
        if return_img:
            return image

    def save_graphic(self, filename, title='PyVista Export', raster=True, painter=True):
        """Save a screenshot of the rendering window as a graphic file.

        This can be helpful for publication documents.

        The supported formats are:

        * ``'.svg'``
        * ``'.eps'``
        * ``'.ps'``
        * ``'.pdf'``
        * ``'.tex'``

        Parameters
        ----------
        filename : str
            Path to fsave the graphic file to.

        title : str, optional
            Title to use within the file properties.

        raster : bool, optional
            Attempt to write 3D properties as a raster image.

        painter : bool, optional
            Configure the exporter to expect a painter-ordered 2D
            rendering, that is, a rendering at a fixed depth where
            primitives are drawn from the bottom up.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(examples.load_airplane(), smooth_shading=True)
        >>> _ = pl.add_background_image(examples.mapfile)
        >>> pl.save_graphic("img.svg")  # doctest:+SKIP

        """
        if not hasattr(self, 'ren_win'):
            raise AttributeError('This plotter is closed and unable to save a screenshot.')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        filename = os.path.abspath(os.path.expanduser(filename))
        extension = pyvista.fileio.get_ext(filename)

        writer = _vtk.lazy_vtkGL2PSExporter()
        modes = {
            '.svg': writer.SetFileFormatToSVG,
            '.eps': writer.SetFileFormatToEPS,
            '.ps': writer.SetFileFormatToPS,
            '.pdf': writer.SetFileFormatToPDF,
            '.tex': writer.SetFileFormatToTeX,
        }
        if extension not in modes:
            raise ValueError(
                f"Extension ({extension}) is an invalid choice.\n\n"
                f"Valid options include: {', '.join(modes.keys())}"
            )
        writer.CompressOff()
        writer.SetFilePrefix(filename.replace(extension, ''))
        writer.SetInput(self.ren_win)
        modes[extension]()
        writer.SetTitle(title)
        writer.SetWrite3DPropsAsRasterImage(raster)
        if painter:
            writer.UsePainterSettings()
        writer.Update()

    def screenshot(
        self, filename=None, transparent_background=None, return_img=True, window_size=None
    ):
        """Take screenshot at current camera position.

        Parameters
        ----------
        filename : str, pathlib.Path, BytesIO, optional
            Location to write image to.  If ``None``, no image is written.

        transparent_background : bool, optional
            Whether to make the background transparent.  The default is
            looked up on the plotter's theme.

        return_img : bool, optional
            If ``True`` (the default), a NumPy array of the image will
            be returned.

        window_size : 2-length tuple, optional
            Set the plotter's size to this ``(width, height)`` before
            taking the screenshot.

        Returns
        -------
        numpy.ndarray
            Array containing pixel RGB and alpha.  Sized:

            * [Window height x Window width x 3] if
              ``transparent_background`` is set to ``False``.
            * [Window height x Window width x 4] if
              ``transparent_background`` is set to ``True``.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> plotter = pyvista.Plotter(off_screen=True)
        >>> actor = plotter.add_mesh(sphere)
        >>> plotter.screenshot('screenshot.png')  # doctest:+SKIP

        """
        if window_size is not None:
            self.window_size = window_size

        # configure image filter
        if transparent_background is None:
            transparent_background = self._theme.transparent_background
        self.image_transparent_background = transparent_background

        # This if statement allows you to save screenshots of closed plotters
        # This is needed for the sphinx-gallery to work
        if not hasattr(self, 'ren_win'):
            # If plotter has been closed...
            # check if last_image exists
            if self.last_image is not None:
                # Save last image
                return self._save_image(self.last_image, filename, return_img)
            # Plotter hasn't been rendered or was improperly closed
            raise RuntimeError('This plotter is closed and unable to save a screenshot.')

        if self._first_time and not self.off_screen:
            raise RuntimeError(
                "Nothing to screenshot - call .show first or use the off_screen argument"
            )

        # if off screen, show has not been called and we must render
        # before extracting an image
        if self._first_time:
            self._on_first_render_request()
            self.render()

        return self._save_image(self.image, filename, return_img)

    @wraps(Renderers.set_background)
    def set_background(self, *args, **kwargs):
        """Wrap ``Renderers.set_background``."""
        self.renderers.set_background(*args, **kwargs)

    def generate_orbital_path(self, factor=3.0, n_points=20, viewup=None, shift=0.0):
        """Generate an orbital path around the data scene.

        Parameters
        ----------
        factor : float, optional
            A scaling factor when building the orbital extent.

        n_points : int, optional
            Number of points on the orbital path.

        viewup : list(float), optional
            The normal to the orbital plane.

        shift : float, optional
            Shift the plane up/down from the center of the scene by
            this amount.

        Returns
        -------
        pyvista.PolyData
            PolyData containing the orbital path.

        Examples
        --------
        Generate an orbital path around a sphere.

        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(pyvista.Sphere())
        >>> viewup = [0, 0, 1]
        >>> orbit = plotter.generate_orbital_path(factor=2.0, n_points=50,
        ...                                       shift=0.0, viewup=viewup)

        See :ref:`orbiting_example` for a full example using this method.

        """
        if viewup is None:
            viewup = self._theme.camera['viewup']
        center = np.array(self.center)
        bnds = np.array(self.bounds)
        radius = (bnds[1] - bnds[0]) * factor
        y = (bnds[3] - bnds[2]) * factor
        if y > radius:
            radius = y
        center += np.array(viewup) * shift
        return pyvista.Polygon(center=center, radius=radius, normal=viewup, n_sides=n_points)

    def fly_to(self, point):
        """Move the current camera's focal point to a position point.

        The movement is animated over the number of frames specified in
        NumberOfFlyFrames. The LOD desired frame rate is used.

        Parameters
        ----------
        point : sequence
            Point to fly to in the form of ``(x, y, z)``.

        """
        self.iren.fly_to(self.renderer, point)

    def orbit_on_path(
        self,
        path=None,
        focus=None,
        step=0.5,
        viewup=None,
        write_frames=False,
        threaded=False,
        progress_bar=False,
    ):
        """Orbit on the given path focusing on the focus point.

        Parameters
        ----------
        path : pyvista.PolyData
            Path of orbital points. The order in the points is the order of
            travel.

        focus : list(float) of length 3, optional
            The point of focus the camera.

        step : float, optional
            The timestep between flying to each camera position. Ignored when
            the plotter run "off screen".

        viewup : list(float), optional
            The normal to the orbital plane.

        write_frames : bool, optional
            Assume a file is open and write a frame on each camera
            view during the orbit.

        threaded : bool, optional
            Run this as a background thread.  Generally used within a
            GUI (i.e. PyQt).

        progress_bar : bool, optional
            Show the progress bar when proceeding through the path.
            This can be helpful to show progress when generating
            movies with ``off_screen=True``.

        Examples
        --------
        Plot an orbit around the earth.  Save the gif as a temporary file.

        >>> import os
        >>> from tempfile import mkdtemp
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = os.path.join(mkdtemp(), 'orbit.gif')
        >>> plotter = pyvista.Plotter(window_size=[300, 300])
        >>> _ = plotter.add_mesh(examples.load_globe(), smooth_shading=True)
        >>> plotter.open_gif(filename)
        >>> viewup = [0, 0, 1]
        >>> orbit = plotter.generate_orbital_path(factor=2.0, n_points=24,
        ...                                       shift=0.0, viewup=viewup)
        >>> plotter.orbit_on_path(orbit, write_frames=True, viewup=viewup,
        ...                       step=0.02)

        See :ref:`orbiting_example` for a full example using this method.

        """
        if focus is None:
            focus = self.center
        if viewup is None:
            viewup = self._theme.camera['viewup']
        if path is None:
            path = self.generate_orbital_path(viewup=viewup)
        if not is_pyvista_dataset(path):
            path = pyvista.PolyData(path)
        points = path.points

        # Make sure the whole scene is visible
        self.camera.thickness = path.length

        if progress_bar:
            try:
                from tqdm import tqdm
            except ImportError:  # pragma: no cover
                raise ImportError("Please install `tqdm` to use ``progress_bar=True``")

        def orbit():
            """Define the internal thread for running the orbit."""
            if progress_bar:
                points_seq = tqdm(points)
            else:
                points_seq = points

            for point in points_seq:
                tstart = time.time()  # include the render time in the step time
                self.set_position(point, render=False)
                self.set_focus(focus, render=False)
                self.set_viewup(viewup, render=False)
                self.renderer.ResetCameraClippingRange()
                if write_frames:
                    self.write_frame()
                else:
                    self.render()
                sleep_time = step - (time.time() - tstart)
                if sleep_time > 0 and not self.off_screen:
                    time.sleep(sleep_time)
            if write_frames:
                self.mwriter.close()

        if threaded:
            thread = Thread(target=orbit)
            thread.start()
        else:
            orbit()

    def export_vtkjs(self, filename, compress_arrays=False):
        """Export the current rendering scene as a VTKjs scene.

        It can be used for rendering in a web browser.

        Parameters
        ----------
        filename : str
            Filename to export the scene to.  A filename extension of
            ``'.vtkjs'`` will be added.

        compress_arrays : bool, optional
            Enable array compression.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(examples.load_hexbeam())
        >>> pl.export_vtkjs("sample")  # doctest:+SKIP

        """
        if not hasattr(self, 'ren_win'):
            raise RuntimeError('Export must be called before showing/closing the scene.')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        else:
            filename = os.path.abspath(os.path.expanduser(filename))

        export_plotter_vtkjs(self, filename, compress_arrays=compress_arrays)

    def export_obj(self, filename):
        """Export scene to OBJ format.

        Parameters
        ----------
        filename : str
            Filename to export the scene to.  Must end in ``'.obj'``.

        Examples
        --------
        Export the scene to "scene.obj"

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.export_obj('scene.obj')  # doctest:+SKIP

        """
        if pyvista.vtk_version_info <= (8, 1, 2):
            raise pyvista.core.errors.VTKVersionError()

        if not hasattr(self, "ren_win"):
            raise RuntimeError("This plotter must still have a render window open.")
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        else:
            filename = os.path.abspath(os.path.expanduser(filename))

        if not filename.endswith('.obj'):
            raise ValueError('`filename` must end with ".obj"')

        exporter = _vtk.lazy_vtkOBJExporter()
        # remove the extension as VTK always adds it in
        exporter.SetFilePrefix(filename[:-4])
        exporter.SetRenderWindow(self.ren_win)
        exporter.Write()

    @property
    def _datasets(self):
        """Return a list of all datasets associated with this plotter."""
        datasets = []
        for renderer in self.renderers:
            for actor in renderer.actors.values():
                mapper = actor.GetMapper()

                # ignore any mappers whose inputs are not datasets
                if hasattr(mapper, 'GetInputAsDataSet'):
                    datasets.append(mapper.GetInputAsDataSet())

        return datasets

    def __del__(self):
        """Delete the plotter."""
        # We have to check here if the plotter was only partially initialized
        if self._initialized:
            if not self._closed:
                self.close()
        self.deep_clean()
        if self._initialized:
            del self.renderers

    def add_background_image(self, image_path, scale=1, auto_resize=True, as_global=True):
        """Add a background image to a plot.

        Parameters
        ----------
        image_path : str
            Path to an image file.

        scale : float, optional
            Scale the image larger or smaller relative to the size of
            the window.  For example, a scale size of 2 will make the
            largest dimension of the image twice as large as the
            largest dimension of the render window.  Defaults to 1.

        auto_resize : bool, optional
            Resize the background when the render window changes size.

        as_global : bool, optional
            When multiple render windows are present, setting
            ``as_global=False`` will cause the background to only
            appear in one window.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> plotter = pyvista.Plotter()
        >>> actor = plotter.add_mesh(pyvista.Sphere())
        >>> plotter.add_background_image(examples.mapfile)
        >>> plotter.show()

        """
        if self.renderers.has_active_background_renderer:
            raise RuntimeError(
                'A background image already exists.  '
                'Remove it with ``remove_background_image`` '
                'before adding one'
            )

        # Need to change the number of layers to support an additional
        # background layer
        if not self._has_background_layer:
            self.ren_win.SetNumberOfLayers(3)
        renderer = self.renderers.add_background_renderer(image_path, scale, as_global)
        self.ren_win.AddRenderer(renderer)

        # set up autoscaling of the image
        if auto_resize:  # pragma: no cover
            self.iren.add_observer('ModifiedEvent', renderer.resize)

    @wraps(Renderers.remove_background_image)
    def remove_background_image(self):
        """Wrap ``Renderers.remove_background_image``."""
        self.renderers.remove_background_image()

        # return the active renderer to the top, otherwise flat background
        # will not be rendered
        self.renderer.layer = 0

    def _on_first_render_request(self, cpos=None):
        """Once an image or render is officially requested, run this routine.

        For example on the show call or any screenshot producing code.
        """
        # reset unless camera for the first render unless camera is set
        if self._first_time:  # and not self.camera_set:
            for renderer in self.renderers:
                if not renderer.camera_set and cpos is None:
                    renderer.camera_position = renderer.get_default_cam_pos()
                    renderer.ResetCamera()
                elif cpos is not None:
                    renderer.camera_position = cpos
            self._first_time = False

    def reset_camera_clipping_range(self):
        """Reset camera clipping planes."""
        self.renderer.ResetCameraClippingRange()

    def add_light(self, light, only_active=False):
        """Add a Light to the scene.

        Parameters
        ----------
        light : Light or vtkLight
            The light to be added.

        only_active : bool, optional
            If ``True``, only add the light to the active
            renderer. The default is that every renderer adds the
            light. To add the light to an arbitrary renderer, see
            :func:`pyvista.plotting.renderer.Renderer.add_light`.

        Examples
        --------
        Create a plotter that we initialize with no lights, and add a
        cube and a single headlight to it.

        >>> import pyvista as pv
        >>> plotter = pv.Plotter(lighting='none')
        >>> _ = plotter.add_mesh(pv.Cube())
        >>> light = pv.Light(color='cyan', light_type='headlight')
        >>> plotter.add_light(light)
        >>> plotter.show()

        """
        renderers = [self.renderer] if only_active else self.renderers
        for renderer in renderers:
            renderer.add_light(light)

    def remove_all_lights(self, only_active=False):
        """Remove all lights from the scene.

        Parameters
        ----------
        only_active : bool
            If ``True``, only remove lights from the active
            renderer. The default is that lights are stripped from
            every renderer.

        Examples
        --------
        Create a plotter and remove all lights after initialization.
        Note how the mesh rendered is completely flat

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> plotter.remove_all_lights()
        >>> plotter.renderer.lights
        []
        >>> _ = plotter.add_mesh(pv.Sphere(), show_edges=True)
        >>> plotter.show()

        Note how this differs from a plot with default lighting

        >>> pv.Sphere().plot(show_edges=True, lighting=True)

        """
        renderers = [self.renderer] if only_active else self.renderers
        for renderer in renderers:
            renderer.remove_all_lights()

    def where_is(self, name):
        """Return the subplot coordinates of a given actor.

        Parameters
        ----------
        name : str
            Actor's name.

        Returns
        -------
        list(tuple(int))
            A list with the subplot coordinates of the actor.

        Examples
        --------
        >>> import pyvista as pv
        >>> plotter = pv.Plotter(shape=(2, 2))
        >>> plotter.subplot(0, 0)
        >>> _ = plotter.add_mesh(pv.Box(), name='box')
        >>> plotter.subplot(0, 1)
        >>> _ = plotter.add_mesh(pv.Sphere(), name='sphere')
        >>> plotter.subplot(1, 0)
        >>> _ = plotter.add_mesh(pv.Box(), name='box')
        >>> plotter.subplot(1, 1)
        >>> _ = plotter.add_mesh(pv.Cone(), name='cone')
        >>> plotter.where_is('box')
        [(0, 0), (1, 0)]

        >>> plotter.show()
        """
        places = []
        for index in range(len(self.renderers)):
            if name in self.renderers[index]._actors:
                places.append(tuple(self.renderers.index_to_loc(index)))
        return places

    def add_ruler(
        self,
        pointa,
        pointb,
        flip_range=False,
        number_labels=5,
        show_labels=True,
        font_size_factor=0.6,
        label_size_factor=1.0,
        label_format=None,
        title="Distance",
        number_minor_ticks=0,
        tick_length=5,
        minor_tick_length=3,
        show_ticks=True,
        tick_label_offset=2,
    ):
        """Add ruler.

        The ruler is a 2D object that is not occluded by 3D objects.
        To avoid issues with perspective, it is recommended to use
        parallel projection, i.e. :func:`Plotter.enable_parallel_projection`,
        and place the ruler orthogonal to the viewing direction.

        The title and labels are placed to the right of ruler moving from
        ``pointa`` to ``pointb``. Use ``flip_range`` to flip the ``0`` location,
        if needed.

        Since the ruler is placed in an overlay on the viewing scene, the camera
        does not automatically reset to include the ruler in the view.

        Parameters
        ----------
        pointa : Sequence
            Starting point for ruler.

        pointb : Sequence
            Ending point for ruler.

        flip_range : bool
            If ``True``, the distance range goes from ``pointb`` to ``pointa``.

        number_labels : int
            Number of labels to place on ruler.

        show_labels : bool, optional
            Whether to show labels.

        font_size_factor : float
            Factor to scale font size overall.

        label_size_factor : float
            Factor to scale label size relative to title size.

        label_format : str, optional
            A printf style format for labels, e.g. '%E'.

        title : str, optional
            The title to display.

        number_minor_ticks : int, optional
            Number of minor ticks between major ticks.

        tick_length : int
            Length of ticks in pixels.

        minor_tick_length : int
            Length of minor ticks in pixels.

        show_ticks : bool, optional
            Whether to show the ticks.

        tick_label_offset : int
            Offset between tick and label in pixels.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the ruler.

        Examples
        --------
        >>> import pyvista
        >>> cone = pyvista.Cone(height=2.0, radius=0.5)
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(cone)

        Measure x direction of cone and place ruler slightly below.

        >>> _ = plotter.add_ruler(
        ...     pointa=[cone.bounds[0], cone.bounds[2] - 0.1, 0.0],
        ...     pointb=[cone.bounds[1], cone.bounds[2] - 0.1, 0.0],
        ...     title="X Distance"
        ... )

        Measure y direction of cone and place ruler slightly to left.
        The title and labels are placed to the right of the ruler when
        traveling from ``pointa`` to ``pointb``.

        >>> _ = plotter.add_ruler(
        ...     pointa=[cone.bounds[0] - 0.1, cone.bounds[3], 0.0],
        ...     pointb=[cone.bounds[0] - 0.1, cone.bounds[2], 0.0],
        ...     flip_range=True,
        ...     title="Y Distance"
        ... )
        >>> plotter.enable_parallel_projection()
        >>> plotter.view_xy()
        >>> plotter.show()

        """
        ruler = _vtk.vtkAxisActor2D()

        ruler.GetPositionCoordinate().SetCoordinateSystemToWorld()
        ruler.GetPosition2Coordinate().SetCoordinateSystemToWorld()
        ruler.GetPositionCoordinate().SetReferenceCoordinate(None)
        ruler.GetPositionCoordinate().SetValue(pointa[0], pointa[1], pointa[2])
        ruler.GetPosition2Coordinate().SetValue(pointb[0], pointb[1], pointb[2])

        distance = np.linalg.norm(np.asarray(pointa) - np.asarray(pointb))
        if flip_range:
            ruler.SetRange(distance, 0)
        else:
            ruler.SetRange(0, distance)

        ruler.SetTitle(title)
        ruler.SetFontFactor(font_size_factor)
        ruler.SetLabelFactor(label_size_factor)
        ruler.SetNumberOfLabels(number_labels)
        ruler.SetLabelVisibility(show_labels)
        if label_format:
            ruler.SetLabelFormat(label_format)

        ruler.SetNumberOfMinorTicks(number_minor_ticks)
        ruler.SetTickVisibility(show_ticks)
        ruler.SetTickLength(tick_length)
        ruler.SetMinorTickLength(minor_tick_length)
        ruler.SetTickOffset(tick_label_offset)

        self.add_actor(ruler, reset_camera=True, pickable=False)
        return ruler


class Plotter(BasePlotter):
    """Plotting object to display vtk meshes or numpy arrays.

    Parameters
    ----------
    off_screen : bool, optional
        Renders off screen when ``True``.  Useful for automated
        screenshots.

    notebook : bool, optional
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.  Automatically
        enables ``off_screen``.

    shape : list or tuple, optional
        Number of sub-render windows inside of the main window.
        Specify two across with ``shape=(2, 1)`` and a two by two grid
        with ``shape=(2, 2)``.  By default there is only one render
        window.  Can also accept a string descriptor as shape. E.g.:

        * ``shape="3|1"`` means 3 plots on the left and 1 on the right,
        * ``shape="4/2"`` means 4 plots on top and 2 at the bottom.

    border : bool, optional
        Draw a border around each render window.  Default ``False``.

    border_color : color_like, optional
        Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

    window_size : list, optional
        Window size in pixels.  Defaults to ``[1024, 768]``, unless
        set differently in the relevant theme's ``window_size``
        property.

    multi_samples : int, optional
        The number of multi-samples used to mitigate aliasing. 4 is a
        good default but 8 will have better results with a potential
        impact on performance.

    line_smoothing : bool, optional
        If ``True``, enable line smoothing.

    polygon_smoothing : bool, optional
        If ``True``, enable polygon smoothing.

    lighting : str, optional
        What lighting to set up for the plotter.
        Accepted options:

            * ``'light_kit'``: a vtk Light Kit composed of 5 lights.
            * ``'three lights'``: illumination using 3 lights.
            * ``'none'``: no light sources at instantiation.

        The default is a ``'light_kit'`` (to be precise, 5 separate
        lights that act like a Light Kit).

    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> mesh = examples.load_hexbeam()
    >>> another_mesh = examples.load_uniform()
    >>> plotter = pyvista.Plotter()
    >>> actor = plotter.add_mesh(mesh, color='red')
    >>> actor = plotter.add_mesh(another_mesh, color='blue')
    >>> plotter.show()

    """

    last_update_time = 0.0
    right_timer_id = -1

    def __init__(
        self,
        off_screen=None,
        notebook=None,
        shape=(1, 1),
        groups=None,
        row_weights=None,
        col_weights=None,
        border=None,
        border_color='k',
        border_width=2.0,
        window_size=None,
        multi_samples=None,
        line_smoothing=False,
        point_smoothing=False,
        polygon_smoothing=False,
        splitting_position=None,
        title=None,
        lighting='light kit',
        theme=None,
    ):
        """Initialize a vtk plotting object."""
        super().__init__(
            shape=shape,
            border=border,
            border_color=border_color,
            border_width=border_width,
            groups=groups,
            row_weights=row_weights,
            col_weights=col_weights,
            splitting_position=splitting_position,
            title=title,
            lighting=lighting,
            theme=theme,
        )
        # reset partial initialization flag
        self._initialized = False

        log.debug('Plotter init start')

        # check if a plotting backend is enabled
        _warn_xserver()

        def on_timer(iren, event_id):
            """Exit application if interactive renderer stops."""
            if event_id == 'TimerEvent' and self.iren._style != "Context":
                self.iren.terminate_app()

        if off_screen is None:
            off_screen = pyvista.OFF_SCREEN

        if notebook is None:
            if self._theme.notebook is not None:
                notebook = self._theme.notebook
            else:
                notebook = scooby.in_ipykernel()

        self.notebook = notebook
        if self.notebook:
            off_screen = True
        self.off_screen = off_screen

        self._window_size_unset = False
        if window_size is None:
            self._window_size_unset = True
            window_size = self._theme.window_size
        self.__prior_window_size = window_size

        # initialize render window
        self.ren_win = _vtk.vtkRenderWindow()
        self.ren_win.SetMultiSamples(0)
        self.ren_win.SetBorders(True)
        if line_smoothing:
            self.ren_win.LineSmoothingOn()
        if point_smoothing:
            self.ren_win.PointSmoothingOn()
        if polygon_smoothing:
            self.ren_win.PolygonSmoothingOn()

        for renderer in self.renderers:
            self.ren_win.AddRenderer(renderer)

        # Add the shadow renderer to allow us to capture interactions within
        # a given viewport
        # https://vtk.org/pipermail/vtkusers/2018-June/102030.html
        number_or_layers = self.ren_win.GetNumberOfLayers()
        current_layer = self.renderer.GetLayer()
        self.ren_win.SetNumberOfLayers(number_or_layers + 1)
        self.ren_win.AddRenderer(self.renderers.shadow_renderer)
        self.renderers.shadow_renderer.SetLayer(current_layer + 1)
        self.renderers.shadow_renderer.SetInteractive(False)  # never needs to capture

        if self.off_screen:
            self.ren_win.SetOffScreenRendering(1)
            # vtkGenericRenderWindowInteractor has no event loop and
            # allows the display client to close on Linux when
            # off_screen.  We still want an interactor for off screen
            # plotting since there are some widgets (like the axes
            # widget) that need an interactor
            interactor = _vtk.vtkGenericRenderWindowInteractor()
        else:
            interactor = None

        # Add ren win and interactor
        self.iren = RenderWindowInteractor(self, light_follow_camera=False, interactor=interactor)
        self.iren.set_render_window(self.ren_win)
        self.enable_trackball_style()  # internally calls update_style()
        self.iren.add_observer("KeyPressEvent", self.key_press_event)

        # Set camera widget based on theme. This requires that an
        # interactor be present.
        if self.theme._enable_camera_orientation_widget:
            self.add_camera_orientation_widget()

        # Set background
        self.set_background(self._theme.background)

        # Set window size
        self.window_size = window_size

        # add timer event if interactive render exists
        self.iren.add_observer(_vtk.vtkCommand.TimerEvent, on_timer)

        if self._theme.depth_peeling.enabled:
            if self.enable_depth_peeling():
                for renderer in self.renderers:
                    renderer.enable_depth_peeling()

        # set anti_aliasing based on theme
        if self.theme.anti_aliasing:
            self.enable_anti_aliasing(self.theme.anti_aliasing)

        # some cleanup only necessary for fully initialized plotters
        self._initialized = True
        log.debug('Plotter init stop')

    def show(
        self,
        title=None,
        window_size=None,
        interactive=True,
        auto_close=None,
        interactive_update=False,
        full_screen=None,
        screenshot=False,
        return_img=False,
        cpos=None,
        use_ipyvtk=None,
        jupyter_backend=None,
        return_viewer=False,
        return_cpos=None,
        before_close_callback=None,
        **kwargs,
    ):
        """Display the plotting window.

        Parameters
        ----------
        title : str, optional
            Title of plotting window.  Defaults to
            :attr:`pyvista.global_theme.title <pyvista.themes.DefaultTheme.title>`.

        window_size : list, optional
            Window size in pixels.  Defaults to
            :attr:`pyvista.global_theme.window_size <pyvista.themes.DefaultTheme.window_size>`.

        interactive : bool, optional
            Enabled by default.  Allows user to pan and move figure.
            Defaults to
            :attr:`pyvista.global_theme.interactive <pyvista.themes.DefaultTheme.interactive>`.

        auto_close : bool, optional
            Exits plotting session when user closes the window when
            interactive is ``True``.  Defaults to
            :attr:`pyvista.global_theme.auto_close <pyvista.themes.DefaultTheme.auto_close>`.

        interactive_update : bool, optional
            Disabled by default.  Allows user to non-blocking draw,
            user should call :func:`BasePlotter.update` in each iteration.

        full_screen : bool, optional
            Opens window in full screen.  When enabled, ignores
            ``window_size``.  Defaults to
            :attr:`pyvista.global_theme.full_screen <pyvista.themes.DefaultTheme.full_screen>`.

        screenshot : str, pathlib.Path, BytesIO or bool, optional
            Take a screenshot of the initial state of the plot.
            If a string, it specifies the path to which the screenshot
            is saved. If ``True``, the screenshot is returned as an
            array. Defaults to ``False``. For interactive screenshots
            it's recommended to first call ``show()`` with
            ``auto_close=False`` to set the scene, then save the
            screenshot in a separate call to ``show()`` or
            :func:`Plotter.screenshot`.

        return_img : bool, default: False
            Returns a numpy array representing the last image along
            with the camera position.

        cpos : list(tuple(floats)), optional
            The camera position.  You can also set this with
            :attr:`Plotter.camera_position`.

        use_ipyvtk : bool, optional
            Deprecated.  Instead, set the backend either globally with
            ``pyvista.set_jupyter_backend('ipyvtklink')`` or with
            ``backend='ipyvtklink'``.

        jupyter_backend : str, optional
            Jupyter notebook plotting backend to use.  One of the
            following:

            * ``'none'`` : Do not display in the notebook.
            * ``'pythreejs'`` : Show a ``pythreejs`` widget
            * ``'static'`` : Display a static figure.
            * ``'ipygany'`` : Show a ``ipygany`` widget
            * ``'panel'`` : Show a ``panel`` widget.

            This can also be set globally with
            :func:`pyvista.set_jupyter_backend`.

        return_viewer : bool, optional
            Return the jupyterlab viewer, scene, or display object
            when plotting with jupyter notebook.

        return_cpos : bool, optional
            Return the last camera position from the render window
            when enabled.  Default based on theme setting.  See
            :attr:`pyvista.themes.DefaultTheme.return_cpos`.

        before_close_callback : Callable, optional
            Callback that is called before the plotter is closed.
            The function takes a single parameter, which is the plotter object
            before it closes. An example of use is to capture a screenshot after
            interaction::

                def fun(plotter):
                    plotter.screenshot('file.png')

        **kwargs : dict, optional
            Developer keyword arguments.

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up.
            Returned only when ``return_cpos=True`` or set in the
            default global or plot theme.  Not returned when in a
            jupyter notebook and ``return_viewer=True``.

        image : np.ndarray
            Numpy array of the last image when either ``return_img=True``
            or ``screenshot=True`` is set. Not returned when in a
            jupyter notebook with ``return_viewer=True``. Optionally
            contains alpha values. Sized:

            * [Window height x Window width x 3] if the theme sets
              ``transparent_background=False``.
            * [Window height x Window width x 4] if the theme sets
              ``transparent_background=True``.

        widget
            IPython widget when ``return_viewer=True``.

        Notes
        -----
        Please use the ``q``-key to close the plotter as some
        operating systems (namely Windows) will experience issues
        saving a screenshot if the exit button in the GUI is pressed.

        Examples
        --------
        Simply show the plot of a mesh.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Cube())
        >>> pl.show()

        Take a screenshot interactively.  Screenshot will be of the
        first image shown, so use the first call with
        ``auto_close=False`` to set the scene before taking the
        screenshot.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Cube())
        >>> pl.show(auto_close=False)  # doctest:+SKIP
        >>> pl.show(screenshot='my_image.png')  # doctest:+SKIP

        Display a ``pythreejs`` scene within a jupyter notebook

        >>> pl.show(jupyter_backend='pythreejs')  # doctest:+SKIP

        Return a ``pythreejs`` scene.

        >>> pl.show(jupyter_backend='pythreejs', return_viewer=True)  # doctest:+SKIP

        Obtain the camera position when using ``show``.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.show(return_cpos=True)   # doctest:+SKIP
        [(2.223005211686484, -0.3126909484828709, 2.4686209867735065),
        (0.0, 0.0, 0.0),
        (-0.6839951597283509, -0.47207319712073137, 0.5561452310578585)]

        """
        jupyter_kwargs = kwargs.pop('jupyter_kwargs', {})
        assert_empty_kwargs(**kwargs)

        if before_close_callback is None:
            before_close_callback = pyvista.global_theme._before_close_callback
        self._before_close_callback = before_close_callback

        if interactive_update and auto_close is None:
            auto_close = False
        elif interactive_update and auto_close:
            warnings.warn(
                textwrap.dedent(
                    """
                    The plotter will close immediately automatically since ``auto_close=True``.
                    Either, do not specify ``auto_close``, or set it to ``False`` if you want to
                    interact with the plotter interactively.
                    """
                ).strip()
            )
        elif auto_close is None:
            auto_close = self._theme.auto_close

        if use_ipyvtk:
            txt = textwrap.dedent(
                """
                use_ipyvtk is deprecated.  Set the backend
                globally with ``pyvista.set_jupyter_backend("ipyvtklink")
                or with ``backend="ipyvtklink"``
                """
            ).strip()
            from pyvista.core.errors import DeprecationError

            raise DeprecationError(txt)

        if not hasattr(self, "ren_win"):
            raise RuntimeError("This plotter has been closed and cannot be shown.")

        if full_screen is None:
            full_screen = self._theme.full_screen

        if full_screen:
            self.ren_win.SetFullScreen(True)
            self.ren_win.BordersOn()  # super buggy when disabled
        else:
            if window_size is None:
                window_size = self.window_size
            else:
                self._window_size_unset = False
            self.ren_win.SetSize(window_size[0], window_size[1])

        # reset unless camera for the first render unless camera is set
        self._on_first_render_request(cpos)

        # handle plotter notebook
        if jupyter_backend and not self.notebook:
            warnings.warn(
                'Not within a jupyter notebook environment.\nIgnoring ``jupyter_backend``.'
            )

        if self.notebook:
            from ..jupyter.notebook import handle_plotter

            if jupyter_backend is None:
                jupyter_backend = self._theme.jupyter_backend

            if jupyter_backend != 'none':
                if screenshot:
                    warnings.warn(
                        '\nSet `jupyter_backend` backend to `"none"` to take a screenshot'
                        ' within a notebook environment.'
                    )

                disp = handle_plotter(
                    self, backend=jupyter_backend, return_viewer=return_viewer, **jupyter_kwargs
                )
                return disp

        self.render()

        # initial double render needed for certain passes when offscreen
        if self.off_screen and 'vtkDepthOfFieldPass' in self.renderer._render_passes._passes:
            self.render()

        # This has to be after the first render for some reason
        if title is None:
            title = self.title
        if title:
            self.ren_win.SetWindowName(title)
            self.title = title

        # Keep track of image for sphinx-gallery
        if pyvista.BUILDING_GALLERY or screenshot:
            # always save screenshots for sphinx_gallery

            self.last_image = self.screenshot(screenshot, return_img=True)
            self.last_image_depth = self.get_image_depth()

        # See: https://github.com/pyvista/pyvista/issues/186#issuecomment-550993270
        if interactive and not self.off_screen:
            try:  # interrupts will be caught here
                log.debug('Starting iren')
                self.iren.update_style()
                if not interactive_update:

                    # Resolves #1260
                    if os.name == 'nt':
                        if _vtk.VTK9:
                            self.iren.process_events()
                        else:
                            global VERY_FIRST_RENDER
                            if not VERY_FIRST_RENDER:
                                self.iren.start()
                            VERY_FIRST_RENDER = False

                    self.iren.start()
                self.iren.initialize()
            except KeyboardInterrupt:
                log.debug('KeyboardInterrupt')
                self.close()
                raise KeyboardInterrupt
        # In the event that the user hits the exit-button on the GUI  (on
        # Windows OS) then it must be finalized and deleted as accessing it
        # will kill the kernel.
        # Here we check for that and clean it up before moving on to any of
        # the closing routines that might try to still access that
        # render window.
        if not self.ren_win.IsCurrent():
            self._clear_ren_win()  # The ren_win is deleted
            # proper screenshots cannot be saved if this happens
            if not auto_close:
                warnings.warn(
                    "`auto_close` ignored: by clicking the exit button, "
                    "you have destroyed the render window and we have to "
                    "close it out."
                )
                auto_close = True
        # NOTE: after this point, nothing from the render window can be accessed
        #       as if a user pressed the close button, then it destroys the
        #       the render view and a stream of errors will kill the Python
        #       kernel if code here tries to access that renderer.
        #       See issues #135 and #186 for insight before editing the
        #       remainder of this function.

        # Close the render window if requested
        if auto_close:
            self.close()

        # If user asked for screenshot, return as numpy array after camera
        # position
        if return_img or screenshot is True:
            if return_cpos:
                return self.camera_position, self.last_image

        if return_cpos:
            return self.camera_position

    def add_title(self, title, font_size=18, color=None, font=None, shadow=False):
        """Add text to the top center of the plot.

        This is merely a convenience method that calls ``add_text``
        with ``position='upper_edge'``.

        Parameters
        ----------
        title : str
            The text to add the rendering.

        font_size : float, optional
            Sets the size of the title font.  Defaults to 16 or the
            value of the global theme if set.

        color : color_like, optional,
            Either a string, rgb list, or hex color string.  Defaults
            to white or the value of the global theme if set.  For
            example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        font : str, optional
            Font name may be ``'courier'``, ``'times'``, or ``'arial'``.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to ``False``.

        Returns
        -------
        vtk.vtkTextActor
            Text actor added to plot.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.background_color = 'grey'
        >>> actor = pl.add_title('Plot Title', font='courier', color='k',
        ...                      font_size=40)
        >>> pl.show()

        """
        # add additional spacing from the top of the figure by default
        title = '\n' + title
        return self.add_text(
            title,
            position='upper_edge',
            font_size=font_size,
            color=color,
            font=font,
            shadow=shadow,
            name='title',
            viewport=False,
        )

    def add_cursor(
        self,
        bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        focal_point=(0.0, 0.0, 0.0),
        color=None,
    ):
        """Add a cursor of a PyVista or VTK dataset to the scene.

        Parameters
        ----------
        bounds : length 6 sequence, default: (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
            Specify the bounds in the format of:

            - ``(xmin, xmax, ymin, ymax, zmin, zmax)``

            Defaults to ``(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)``.

        focal_point : list or tuple, optional
            The focal point of the cursor.

            Defaults to ``(0.0, 0.0, 0.0)``.

        color : color_like, optional
            Either a string, RGB sequence, or hex color string.  For one
            of the following.

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        Returns
        -------
        vtk.vtkActor
            VTK actor of the 2D cursor.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(sphere)
        >>> _ = plotter.add_cursor()
        >>> plotter.show()

        """
        alg = _vtk.vtkCursor3D()
        alg.SetModelBounds(bounds)
        alg.SetFocalPoint(focal_point)
        alg.AllOn()
        mapper = DataSetMapper()
        mapper.SetInputConnection(alg.GetOutputPort())
        actor, prop = self.add_actor(mapper)
        prop.SetColor(Color(color).float_rgb)

        return actor


# Tracks created plotters.  This is the end of the module as we need to
# define ``BasePlotter`` before including it in the type definition.
#
# When pyvista.BUILDING_GALLERY = False, the objects will be ProxyType, and
# when True, BasePlotter.
_ALL_PLOTTERS: Dict[str, BasePlotter] = {}


def _kill_display(disp_id):  # pragma: no cover
    """Forcibly close the display on Linux.

    See: https://gitlab.kitware.com/vtk/vtk/-/issues/17917#note_783584

    And more details into why...
    https://stackoverflow.com/questions/64811503

    Notes
    -----
    This is to be used experimentally and is known to cause issues
    on `pyvistaqt`

    """
    if platform.system() != 'Linux':
        raise OSError('This method only works on Linux')

    if disp_id:
        cdisp_id = int(disp_id[1:].split('_')[0], 16)

        # this is unsafe as events might be queued, but sometimes the
        # window fails to close if we don't just close it
        Thread(target=X11.XCloseDisplay, args=(cdisp_id,)).start()
