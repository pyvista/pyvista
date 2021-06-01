"""This module contains some convenience helper functions."""

import numpy as np
import scooby

import pyvista
from pyvista.utilities import is_pyvista_dataset
from .plotting import Plotter


def plot(var_item, off_screen=None, full_screen=False, screenshot=None,
         interactive=True, cpos=None, window_size=None,
         show_bounds=False, show_axes=None, notebook=None, background=None,
         text='', return_img=False, eye_dome_lighting=False, volume=False,
         parallel_projection=False, use_ipyvtk=None, jupyter_backend=None,
         return_viewer=False, jupyter_kwargs={}, theme=None, **kwargs):
    """Plot a vtk or numpy object.

    Parameters
    ----------
    item : vtk or numpy object
        VTK object or ``numpy`` array to be plotted.

    off_screen : bool
        Plots off screen when ``True``.  Helpful for saving screenshots
        without a window popping up.

    full_screen : bool, optional
        Opens window in full screen.  When enabled, ignores
        ``window_size``.  Default ``False``.

    screenshot : str or bool, optional
        Saves screenshot to file when enabled.  See:
        ``help(pyvista.Plotter.screenshot)``.  Default ``False``.

        When ``True``, takes screenshot and returns ``numpy`` array of
        image.

    window_size : list, optional
        Window size in pixels.  Defaults to ``[1024, 768]``

    show_bounds : bool, optional
        Shows mesh bounds when ``True``.  Default ``False``. Alias
        ``show_grid`` also accepted.

    notebook : bool, optional
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.

    show_axes : bool, optional
        Shows a vtk axes widget.  If ``None``, enabled according to
        ``pyvista.global_theme.axes.show``.

    text : str, optional
        Adds text at the bottom of the plot.

    volume : bool, optional
        Use the ``add_volume`` method for volume rendering.

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
        ``pyvista.set_jupyter_backend``

    jupyter_kwargs : dict, optional
        Keyword arguments for the Jupyter notebook plotting backend.

    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    **kwargs : optional keyword arguments
        See ``help(pyvista.Plotter.add_mesh)`` for additional options.

    Returns
    -------
    cpos : list
        List of camera position, focal point, and view up.

    img : numpy.ndarray
        Array containing pixel RGB and optionally alpha values.
        Sized:

        * [Window height x Window width x 3] if the theme sets
          ``transparent_background=False``.
        * [Window height x Window width x 4] if the theme sets
          ``transparent_background=True``.

        Returned only when ``screenshot=True``.

    Examples
    --------
    Plot a simple sphere while showing its edges.

    >>> import pyvista
    >>> mesh = pyvista.Sphere()
    >>> mesh.plot(show_edges=True)  # doctest:+SKIP

    """
    if notebook is None:
        notebook = scooby.in_ipykernel()

    if theme is None:
        theme = pyvista.global_theme

    # undocumented kwarg used within pytest to run a function before closing
    before_close_callback = kwargs.pop('before_close_callback', None)

    eye_dome_lighting = kwargs.pop("edl", eye_dome_lighting)
    show_grid = kwargs.pop('show_grid', False)
    auto_close = kwargs.get('auto_close', theme.auto_close)

    if notebook:
        off_screen = notebook
    plotter = Plotter(off_screen=off_screen, notebook=notebook, theme=theme)

    if show_axes is None:
        show_axes = theme.axes.show
    if show_axes:
        plotter.add_axes()

    plotter.set_background(background)

    if isinstance(var_item, list):
        if len(var_item) == 2:  # might be arrows
            isarr_0 = isinstance(var_item[0], np.ndarray)
            isarr_1 = isinstance(var_item[1], np.ndarray)
            if isarr_0 and isarr_1:
                plotter.add_arrows(var_item[0], var_item[1])
            else:
                for item in var_item:
                    if volume or (isinstance(item, np.ndarray) and item.ndim == 3):
                        plotter.add_volume(item, **kwargs)
                    else:
                        plotter.add_mesh(item, **kwargs)
        else:
            for item in var_item:
                if volume or (isinstance(item, np.ndarray) and item.ndim == 3):
                    plotter.add_volume(item, **kwargs)
                else:
                    plotter.add_mesh(item, **kwargs)
    else:
        if volume or (isinstance(var_item, np.ndarray) and var_item.ndim == 3):
            plotter.add_volume(var_item, **kwargs)
        else:
            plotter.add_mesh(var_item, **kwargs)

    if text:
        plotter.add_text(text)

    if show_grid:
        plotter.show_grid()
    elif show_bounds:
        plotter.show_bounds()

    if cpos is None:
        cpos = plotter.get_default_cam_pos()
        plotter.camera_position = cpos
        plotter.camera_set = False
    else:
        plotter.camera_position = cpos

    if eye_dome_lighting:
        plotter.enable_eye_dome_lighting()

    if parallel_projection:
        plotter.enable_parallel_projection()

    result = plotter.show(window_size=window_size,
                          auto_close=auto_close,
                          interactive=interactive,
                          full_screen=full_screen,
                          screenshot=screenshot,
                          return_img=return_img,
                          use_ipyvtk=use_ipyvtk,
                          jupyter_backend=jupyter_backend,
                          before_close_callback=before_close_callback,
                          jupyter_kwargs=jupyter_kwargs,
                          return_viewer=return_viewer)

    # Result will be handled by plotter.show(): cpos or [cpos, img] or
    # the jupyterlab scene when return_viewer is True
    return result


def plot_arrows(cent, direction, **kwargs):
    """Plot arrows as vectors.

    Parameters
    ----------
    cent : np.ndarray
        Accepts a single 3d point or array of 3d points.

    directions : np.ndarray
        Accepts a single 3d point or array of 3d vectors.
        Must contain the same number of items as cent.

    **kwargs : additional arguments, optional
        See ``help(pyvista.plot)``.

    Returns
    -------
    Same as ``pyvista.plot``.  See ``help(pyvista.plot)``.

    Examples
    --------
    Plot a single random arrow.

    >>> import numpy as np
    >>> import pyvista
    >>> cent = np.random.random(3)
    >>> direction = np.random.random(3)
    >>> cpos = pyvista.plot_arrows(cent, direction)

    Plot 100 random arrows.

    >>> import numpy as np
    >>> import pyvista
    >>> cent = np.random.random((100, 3))
    >>> direction = np.random.random((100, 3))
    >>> cpos = pyvista.plot_arrows(cent, direction)

    """
    return plot([cent, direction], **kwargs)


def plot_compare_four(data_a, data_b, data_c, data_d, disply_kwargs=None,
                      plotter_kwargs=None, show_kwargs=None, screenshot=None,
                      camera_position=None, outline=None, outline_color='k',
                      labels=('A', 'B', 'C', 'D'), link=True, notebook=None):
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


def plot_itk(mesh, color=None, scalars=None, opacity=1.0,
             smooth_shading=False):
    """Plot a PyVista/VTK mesh or dataset.

    Adds any PyVista/VTK mesh that itkwidgets can wrap to the
    scene.

    Parameters
    ----------
    mesh : pyvista.DataSet or pyvista.MultiBlock
        Any PyVista or VTK mesh is supported. Also, any dataset that
        :func:`pyvista.wrap` can handle including NumPy arrays of XYZ
        points.

    color : string or 3 item list, optional, defaults to white
        Use to make the entire mesh have a single solid color.  Either
        a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1, 1, 1]``, or
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
    plotter : itkwidgets.Viewer
        ITKwidgets viewer.
    """
    pl = pyvista.PlotterITK()
    if isinstance(mesh, np.ndarray):
        pl.add_points(mesh, color)
    else:
        pl.add_mesh(mesh, color, scalars, opacity, smooth_shading)
    return pl.show()
