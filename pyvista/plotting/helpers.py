"""This module contains some convenience helper functions."""

from typing import Tuple

import numpy as np

import pyvista
from pyvista.utilities import is_pyvista_dataset


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
    jupyter_backend=None,
    return_viewer=False,
    return_cpos=False,
    jupyter_kwargs=None,
    theme=None,
    hidden_line_removal=None,
    anti_aliasing=None,
    zoom=None,
    border=False,
    border_color='k',
    border_width=2.0,
    ssao=False,
    **kwargs,
):
    """Plot a PyVista, numpy, or vtk object.

    Parameters
    ----------
    var_item : pyvista.DataSet
        See :func:`Plotter.add_mesh <pyvista.Plotter.add_mesh>` for all
        supported types.

    off_screen : bool, optional
        Plots off screen when ``True``.  Helpful for saving
        screenshots without a window popping up.  Defaults to the
        global setting ``pyvista.OFF_SCREEN``.

    full_screen : bool, default: :attr:`pyvista.themes.DefaultTheme.full_screen`
        Opens window in full screen.  When enabled, ignores
        ``window_size``.

    screenshot : str or bool, optional
        Saves screenshot to file when enabled.  See:
        :func:`Plotter.screenshot() <pyvista.Plotter.screenshot>`.
        Default ``False``.

        When ``True``, takes screenshot and returns ``numpy`` array of
        image.

    interactive : bool, default: :attr:`pyvista.themes.DefaultTheme.interactive`
        Allows user to pan and move figure.

    cpos : list, optional
        List of camera position, focal point, and view up.

    window_size : sequence, default: :attr:`pyvista.themes.DefaultTheme.window_size`
        Window size in pixels.

    show_bounds : bool, default: False
        Shows mesh bounds when ``True``.

    show_axes : bool, default: :attr:`pyvista.themes._AxesConfig.show`
        Shows a vtk axes widget.

    notebook : bool, default: :attr:`pyvista.themes.DefaultTheme.notebook`
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.

    background : ColorLike, default: :attr:`pyvista.themes.DefaultTheme.background`
        Color of the background.

    text : str, optional
        Adds text at the bottom of the plot.

    return_img : bool, default: False
        Returns numpy array of the last image rendered.

    eye_dome_lighting : bool, optional
        Enables eye dome lighting.

    volume : bool, default: False
        Use the :func:`Plotter.add_volume()
        <pyvista.Plotter.add_volume>` method for volume rendering.

    parallel_projection : bool, default: False
        Enable parallel projection.

    jupyter_backend : str, default: :attr:`pyvista.themes.DefaultTheme.jupyter_backend`
        Jupyter notebook plotting backend to use.  One of the
        following:

        * ``'none'`` : Do not display in the notebook.
        * ``'static'`` : Display a static figure.
        * ``'ipygany'`` : Show a ``ipygany`` widget
        * ``'panel'`` : Show a ``panel`` widget.
        * ``'trame'`` : Display using ``trame``.

        This can also be set globally with
        :func:`pyvista.set_jupyter_backend`.

    return_viewer : bool, default: False
        Return the jupyterlab viewer, scene, or display object
        when plotting with jupyter notebook.

    return_cpos : bool, default: False
        Return the last camera position from the render window
        when enabled.  Defaults to value in theme settings.

    jupyter_kwargs : dict, optional
        Keyword arguments for the Jupyter notebook plotting backend.

    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    hidden_line_removal : bool, default: :attr:`pyvista.themes.DefaultTheme.hidden_line_removal`
        Wireframe geometry will be drawn using hidden line removal if
        the rendering engine supports it.  See
        :func:`Plotter.enable_hidden_line_removal
        <Plotter.enable_hidden_line_removal>`.

    anti_aliasing : bool, default: :attr:`pyvista.themes.DefaultTheme.anti_aliasing`
        Enable or disable anti-aliasing.

    zoom : float, str, optional
        Camera zoom.  Either ``'tight'`` or a float. A value greater than 1
        is a zoom-in, a value less than 1 is a zoom-out.  Must be greater
        than 0.

    border : bool, default: False
        Draw a border around each render window.

    border_color : ColorLike, default: "k"
        Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

    border_width : float, default: 2.0
        Width of the border in pixels when enabled.

    ssao : bool, optional
        Enable surface space ambient occlusion (SSAO). See
        :func:`Plotter.enable_ssao` for more details.

    **kwargs : dict, optional
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

    widget : ipywidgets.Widget
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
    >>> grid = pv.UniformGrid(
    ...     dimensions=(32, 32, 32), spacing=(0.5, 0.5, 0.5)
    ... )
    >>> grid['data'] = np.linalg.norm(grid.center - grid.points, axis=1)
    >>> grid['data'] = np.abs(grid['data'] - grid['data'].max()) ** 3
    >>> grid.plot(volume=True)

    """
    if jupyter_kwargs is None:
        jupyter_kwargs = {}

    # undocumented kwarg used within pytest to run a function before closing
    before_close_callback = kwargs.pop('before_close_callback', None)

    # pop from kwargs here to avoid including them in add_mesh or add_volume
    eye_dome_lighting = kwargs.pop("edl", eye_dome_lighting)
    show_grid = kwargs.pop('show_grid', False)
    auto_close = kwargs.get('auto_close')

    pl = pyvista.Plotter(
        window_size=window_size,
        off_screen=off_screen,
        notebook=notebook,
        theme=theme,
        border=border,
        border_color=border_color,
        border_width=border_width,
    )

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
        elif isinstance(var_item, pyvista.MultiBlock):
            pl.add_composite(var_item, **kwargs)
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

    if eye_dome_lighting:
        pl.enable_eye_dome_lighting()

    if parallel_projection:
        pl.enable_parallel_projection()

    if ssao:
        pl.enable_ssao()

    if zoom is not None:
        pl.camera.zoom(zoom)

    return pl.show(
        auto_close=auto_close,
        interactive=interactive,
        full_screen=full_screen,
        screenshot=screenshot,
        return_img=return_img,
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
    cent : array_like[float]
        Accepts a single 3d point or array of 3d points.

    direction : array_like[float]
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
    pyvista.plot

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


def view_vectors(view: str, negative: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Given a plane to view, return vectors for setting up camera.

    Parameters
    ----------
    view : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy'}
        Plane to return vectors for.

    negative : bool, default: False
        Whether to view from opposite direction.

    Returns
    -------
    vec : numpy.ndarray
        ``[x, y, z]`` vector that points in the viewing direction

    viewup : numpy.ndarray
        ``[x, y, z]`` vector that points to the viewup direction

    """
    if view == 'xy':
        vec = np.array([0, 0, 1])
        viewup = np.array([0, 1, 0])
    elif view == 'yx':
        vec = np.array([0, 0, -1])
        viewup = np.array([1, 0, 0])
    elif view == 'xz':
        vec = np.array([0, -1, 0])
        viewup = np.array([0, 0, 1])
    elif view == 'zx':
        vec = np.array([0, 1, 0])
        viewup = np.array([1, 0, 0])
    elif view == 'yz':
        vec = np.array([1, 0, 0])
        viewup = np.array([0, 0, 1])
    elif view == 'zy':
        vec = np.array([-1, 0, 0])
        viewup = np.array([0, 1, 0])
    else:
        raise ValueError(
            f"Unexpected value for direction {view}\n"
            "    Expected: 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'"
        )

    if negative:
        vec *= -1
    return vec, viewup
