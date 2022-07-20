"""This module contains some convenience helper functions."""
import time
from inspect import signature
import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import is_pyvista_dataset

from .plotting import Plotter


def plot(
    var_item,
    off_screen=None,
    full_screen=None,
    screenshot=None,
    interactive=True,
    cpos=None,
    window_size=None,
    show_bounds=False,
    show_axes=None,
    notebook=None,
    background=None,
    text='',
    return_img=False,
    eye_dome_lighting=False,
    volume=False,
    parallel_projection=False,
    use_ipyvtk=None,
    jupyter_backend=None,
    return_viewer=False,
    return_cpos=False,
    jupyter_kwargs=None,
    theme=None,
    hidden_line_removal=None,
    anti_aliasing=None,
    zoom=None,
    **kwargs,
):
    """Plot a PyVista, numpy, or vtk object.

    Parameters
    ----------
    var_item : pyvista.DataSet, vtk, or numpy object
        PyVista, VTK, or ``numpy`` object to be plotted.

    off_screen : bool, optional
        Plots off screen when ``True``.  Helpful for saving
        screenshots without a window popping up.  Defaults to the
        global setting ``pyvista.OFF_SCREEN``.

    full_screen : bool, optional
        Opens window in full screen.  When enabled, ignores
        ``window_size``.  Defaults to active theme setting in
        :attr:`pyvista.global_theme.full_screen
        <pyvista.themes.DefaultTheme.full_screen>`.

    screenshot : str or bool, optional
        Saves screenshot to file when enabled.  See:
        :func:`Plotter.screenshot() <pyvista.Plotter.screenshot>`.
        Default ``False``.

        When ``True``, takes screenshot and returns ``numpy`` array of
        image.

    interactive : bool, optional
        Allows user to pan and move figure.  Defaults to
        :attr:`pyvista.global_theme.interactive <pyvista.themes.DefaultTheme.interactive>`.

    cpos : list, optional
        List of camera position, focal point, and view up.

    window_size : list, optional
        Window size in pixels.  Defaults to global theme
        :attr:`pyvista.global_theme.window_size <pyvista.themes.DefaultTheme.window_size>`.

    show_bounds : bool, optional
        Shows mesh bounds when ``True``.  Default ``False``.

    show_axes : bool, optional
        Shows a vtk axes widget.  If ``None``, enabled according to
        :attr:`pyvista.global_theme.axes.show <pyvista.themes._AxesConfig.show>`.

    notebook : bool, optional
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.

    background : color_like, optional
        Color of the background.

    text : str, optional
        Adds text at the bottom of the plot.

    return_img : bool, optional
        Returns numpy array of the last image rendered.

    eye_dome_lighting : bool, optional
        Enables eye dome lighting.

    volume : bool, optional
        Use the :func:`Plotter.add_volume()
        <pyvista.Plotter.add_volume>` method for volume rendering.

    parallel_projection : bool, optional
        Enable parallel projection.

    use_ipyvtk : bool, optional
        Deprecated.  Instead, set the backend either globally with
        ``pyvista.set_jupyter_backend('ipyvtklink')`` or with
        ``backend='ipyvtklink'``.

    jupyter_backend : str, optional
        Jupyter notebook plotting backend to use.  One of the
        following:

        * ``'none'`` : Do not display in the notebook.
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
        when enabled.  Defaults to value in theme settings.

    jupyter_kwargs : dict, optional
        Keyword arguments for the Jupyter notebook plotting backend.

    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    hidden_line_removal : bool, optional
        Wireframe geometry will be drawn using hidden line removal if
        the rendering engine supports it.  See
        :func:`Plotter.enable_hidden_line_removal
        <Plotter.enable_hidden_line_removal>`.  Defaults to the
        theme setting :attr:`pyvista.global_theme.hidden_line_removal
        <pyvista.themes.DefaultTheme.hidden_line_removal>`.

    anti_aliasing : bool, optional
        Enable or disable anti-aliasing.  Defaults to the theme
        setting :attr:`pyvista.global_theme.antialiasing
        <pyvista.themes.DefaultTheme.antialiasing>`.

    zoom : float, optional
        Camera zoom.  A value greater than 1 is a zoom-in, a value
        less than 1 is a zoom-out.  Must be greater than 0.

    **kwargs : optional keyword arguments
        See :func:`pyvista.Plotter.add_mesh` for additional options.

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

    Examples
    --------
    Plot a simple sphere while showing its edges.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> mesh.plot(show_edges=True)

    Plot a volume mesh. Color by distance from the center of the
    UniformGrid. Note ``volume=True`` is passed.

    >>> import numpy as np
    >>> grid = pv.UniformGrid(dims=(32, 32, 32), spacing=(0.5, 0.5, 0.5))
    >>> grid['data'] = np.linalg.norm(grid.center - grid.points, axis=1)
    >>> grid['data'] = np.abs(grid['data'] - grid['data'].max())**3
    >>> grid.plot(volume=True)

    """
    if jupyter_kwargs is None:
        jupyter_kwargs = {}

    # undocumented kwarg used within pytest to run a function before closing
    before_close_callback = kwargs.pop('before_close_callback', None)

    # pop from kwargs here to avoid including them in add_mesh or add_volumex
    eye_dome_lighting = kwargs.pop("edl", eye_dome_lighting)
    show_grid = kwargs.pop('show_grid', False)
    auto_close = kwargs.get('auto_close')

    pl = Plotter(off_screen=off_screen, notebook=notebook, theme=theme)

    if show_axes is None:
        show_axes = pl.theme.axes.show
    if show_axes:
        pl.add_axes()
    if anti_aliasing:
        pl.enable_anti_aliasing()

    pl.set_background(background)

    if isinstance(var_item, list):
        if len(var_item) == 2:  # might be arrows
            isarr_0 = isinstance(var_item[0], np.ndarray)
            isarr_1 = isinstance(var_item[1], np.ndarray)
            if isarr_0 and isarr_1:
                pl.add_arrows(var_item[0], var_item[1])
            else:
                for item in var_item:
                    if volume or (isinstance(item, np.ndarray) and item.ndim == 3):
                        pl.add_volume(item, **kwargs)
                    else:
                        pl.add_mesh(item, **kwargs)
        else:
            for item in var_item:
                if volume or (isinstance(item, np.ndarray) and item.ndim == 3):
                    pl.add_volume(item, **kwargs)
                else:
                    pl.add_mesh(item, **kwargs)
    else:
        if volume or (isinstance(var_item, np.ndarray) and var_item.ndim == 3):
            pl.add_volume(var_item, **kwargs)
        else:
            pl.add_mesh(var_item, **kwargs)

    if text:
        pl.add_text(text)

    if show_grid:
        pl.show_grid()
    elif show_bounds:
        pl.show_bounds()

    if cpos is None:
        cpos = pl.get_default_cam_pos()
        pl.camera_position = cpos
        pl.camera_set = False
    else:
        pl.camera_position = cpos

    if zoom is not None:
        pl.camera.zoom(zoom)

    if eye_dome_lighting:
        pl.enable_eye_dome_lighting()

    if parallel_projection:
        pl.enable_parallel_projection()

    return pl.show(
        window_size=window_size,
        auto_close=auto_close,
        interactive=interactive,
        full_screen=full_screen,
        screenshot=screenshot,
        return_img=return_img,
        use_ipyvtk=use_ipyvtk,
        jupyter_backend=jupyter_backend,
        before_close_callback=before_close_callback,
        jupyter_kwargs=jupyter_kwargs,
        return_viewer=return_viewer,
        return_cpos=return_cpos,
    )


def plot_arrows(cent, direction, **kwargs):
    """Plot arrows as vectors.

    Parameters
    ----------
    cent : numpy.ndarray
        Accepts a single 3d point or array of 3d points.

    direction : numpy.ndarray
        Accepts a single 3d point or array of 3d vectors.
        Must contain the same number of items as ``cent``.

    **kwargs : dict, optional
        See :func:`pyvista.plot`.

    Returns
    -------
    tuple
        See the returns of :func:`pyvista.plot`.

    See Also
    --------
    :func:`pyvista.plot`

    Examples
    --------
    Plot a single random arrow.

    >>> import numpy as np
    >>> import pyvista
    >>> cent = np.random.random(3)
    >>> direction = np.random.random(3)
    >>> pyvista.plot_arrows(cent, direction)

    Plot 100 random arrows.

    >>> import numpy as np
    >>> import pyvista
    >>> cent = np.random.random((100, 3))
    >>> direction = np.random.random((100, 3))
    >>> pyvista.plot_arrows(cent, direction)

    """
    return plot([cent, direction], **kwargs)


def plot_compare_four(
    data_a,
    data_b,
    data_c,
    data_d,
    disply_kwargs=None,
    plotter_kwargs=None,
    show_kwargs=None,
    screenshot=None,
    camera_position=None,
    outline=None,
    outline_color='k',
    labels=('A', 'B', 'C', 'D'),
    link=True,
    notebook=None,
):
    """Plot a 2 by 2 comparison of data objects.

    Plotting parameters and camera positions will all be the same.

    """
    datasets = [[data_a, data_b], [data_c, data_d]]
    labels = [labels[0:2], labels[2:4]]

    if plotter_kwargs is None:
        plotter_kwargs = {}
    if disply_kwargs is None:
        disply_kwargs = {}
    if show_kwargs is None:
        show_kwargs = {}

    plotter_kwargs['notebook'] = notebook

    pl = pyvista.Plotter(shape=(2, 2), **plotter_kwargs)

    for i in range(2):
        for j in range(2):
            pl.subplot(i, j)
            pl.add_mesh(datasets[i][j], **disply_kwargs)
            pl.add_text(labels[i][j])
            if is_pyvista_dataset(outline):
                pl.add_mesh(outline, color=outline_color)
            if camera_position is not None:
                pl.camera_position = camera_position

    if link:
        pl.link_views()
        # when linked, camera must be reset such that the view range
        # of all subrender windows matches
        pl.reset_camera()

    return pl.show(screenshot=screenshot, **show_kwargs)


def plot_itk(mesh, color=None, scalars=None, opacity=1.0, smooth_shading=False):
    """Plot a PyVista/VTK mesh or dataset.

    Adds any PyVista/VTK mesh that itkwidgets can wrap to the
    scene.

    Parameters
    ----------
    mesh : pyvista.DataSet or pyvista.MultiBlock
        Any PyVista or VTK mesh is supported. Also, any dataset that
        :func:`pyvista.wrap` can handle including NumPy arrays of XYZ
        points.

    color : color_like, optional, defaults to white
        Use to make the entire mesh have a single solid color.  Either
        a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

    scalars : str or numpy.ndarray, optional
        Scalars used to "color" the mesh.  Accepts a string name of an
        array that is present on the mesh or an array equal to the
        number of cells or the number of points in the mesh.  Array
        should be sized as a single vector. If both ``color`` and
        ``scalars`` are ``None``, then the active scalars are used.

    opacity : float, optional
        Opacity of the mesh. If a single float value is given, it will
        be the global opacity of the mesh and uniformly applied
        everywhere - should be between 0 and 1.  Default 1.0

    smooth_shading : bool, optional
        Smooth mesh surface mesh by taking into account surface
        normals.  Surface will appear smoother while sharp edges will
        still look sharp.  Default ``False``.

    Returns
    --------
    itkwidgets.Viewer
        ITKwidgets viewer.
    """
    pl = pyvista.PlotterITK()
    if isinstance(mesh, np.ndarray):
        pl.add_points(mesh, color)
    else:
        pl.add_mesh(mesh, color, scalars, opacity, smooth_shading)
    return pl.show()


def _scalars_from_disp(values, scalar_axis):
    """Get scalars from values based on an axis."""

    if scalar_axis == 'norm':
        return np.linalg.norm(values, axis=1)
    elif scalar_axis == 'x':
        return values[:, 0]
    elif scalar_axis == 'y':
        return values[:, 0]
    elif scalar_axis == 'z':
        return values[:, 0]
    elif scalar_axis is not None:
        raise ValueError(
            f'invalid `scalar_axis` "{scalar_axis}". Should be "x", "y", "z", "norm"'
            ' or None'
        )


def animate_displacement(
        mesh, displacements, values='cyclic', n_values=360, func=None,
        scalar_axis='norm', fps=30, loop=None, mag=1.0,
        displace=True, filename=None, **kwargs
):
    """Animate the displacement of a dataset."""
    if isinstance(displacements, str):
        if displacements in mesh.point_data:
            displacements = mesh.point_data[displacements]
    elif isinstance(displacements, np.ndarray):
        if displacements.shape[0] != mesh.n_points:
            raise ValueError('Number of displacements must match the number of points')
    else:
        raise TypeError('`displacements` must be either a string or a numpy array')

    # verify displacements are 3D...    

    if isinstance(values, str):
        if values == 'cyclic':
            values = np.sin(np.linspace(0, 2*np.pi, n_values, endpoint=False))
        elif values == 'linear':
            values = np.linspace(0, 1, n_values)
        else:
            raise ValueError(
                'If `values` is a string, it should be either "cyclic" or "linear"'
            )
    elif isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError('`values` parameter should be a single dimension')
    else:
        raise TypeError('`values` must be either a string or a sequence.')

    if func is not None:
        if not callable(func):
            raise TypeError('`func` must be callable')

        if len(signature(func).parameters) != 2:
            raise ValueError(
                '`func` must contain exactly two parameters: `displacements` and `value`'
            )

        disp_modifier = func
    else:
        def disp_modifier(displacements, value):
            return displacements*value

    # Check mesh datatype and extract exterior surface to improve plotting
    # performance
    if not isinstance(mesh, pyvista.DataSet):
        raise TypeError('`mesh` must be a pyvista.DataSet')
    elif not isinstance(mesh, pyvista.PolyData):
        mesh = mesh.extract_surface(pass_cellid=False)
        displacements = displacements[mesh['vtkOriginalPointIds']]

    if displace:
        orig_points = mesh.points.copy()

    # reset the active displacements to avoid the plot flashing initially
    init_values = disp_modifier(displacements, values[0])
    scalars = _scalars_from_disp(init_values, scalar_axis)
    use_scalars = scalars is not None

    # start plotting
    theme = kwargs.pop('theme', pyvista.global_theme)
    off_screen = kwargs.pop('off_screen', None)
    lighting = kwargs.pop('lighting', 'light kit')
    n_colors = kwargs.pop('n_colors', 256)
    pl = pyvista.Plotter(
        off_screen=off_screen,
        theme=theme,
        lighting=lighting,
    )
    off_screen = pl.off_screen
    pl.add_mesh(mesh, scalars=scalars, n_colors=n_colors, **kwargs)

    if filename:
        if filename.endswith('gif'):
            palettesize = 256 if n_colors > 256 else n_colors
            pl.open_gif(filename, fps=fps, palettesize=palettesize)
        else:
            pl.open_movie(filename, framerage=fps)
        loop = False

    pl.show(auto_close=False, interactive_update=True, interactive=False)

    animating = [True]

    def q_callback():
        """Exit when user wants to leave"""
        animating[0] = False

    def exit_callback(*args):
        """Exit when user wants to leave"""
        animating[0] = False
        pl.close()

    pl.iren.add_observer(
        _vtk.vtkCommand.ExitEvent, lambda render, event: exit_callback(render, event)
    )

    mesh_scalars = pl.mesh.point_data.active_scalars

    # ideal render time
    ren_time_ide = 1/fps

    while animating[0]:
        for value in values:
            if pl._closed or not animating[0]:
                break

            if not off_screen:
                tstart = time.time()

            # displace the scalars and update the active scalars
            disp = disp_modifier(displacements, value)
            if displace:
                pl.mesh.points[:] = orig_points + disp * mag

            if use_scalars:
                mesh_scalars[:] = _scalars_from_disp(disp, scalar_axis)

            if filename:
                pl.write_frame()

            if not off_screen:
                # render and allow interaction
                pl.update(1, force_redraw=filename is None)

                # wait to start next render based on the last render time
                ren_time_act = time.time() - tstart
                sleeptime = ren_time_ide - ren_time_act
                if sleeptime > 0:
                    time.sleep(sleeptime)

        if not loop:
            break

    pl.close()

