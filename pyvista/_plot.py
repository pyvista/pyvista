"""PyVista's famous ``plot()`` helper method.

This method is placed at the top-level to allow us to easily bind
the method to all of the core datatypes before importing the
``pyvista.plotting`` module and libGL dependent VTK modules.
This is necessary for future versions of PyVista that will fully
decouple the ``core`` and ``plotting`` APIs.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args

if TYPE_CHECKING:
    from pyvista.jupyter import JupyterBackendOptions
    from pyvista.plotting._typing import CameraPositionOptions
    from pyvista.plotting._typing import ColorLike
    from pyvista.plotting.themes import Theme


@_deprecate_positional_args(allowed=['var_item'])
def plot(  # noqa: PLR0917
    var_item,
    off_screen: bool | None = None,  # noqa: FBT001
    full_screen: bool | None = None,  # noqa: FBT001
    screenshot: str | bool | None = None,  # noqa: FBT001
    interactive: bool = True,  # noqa: FBT001, FBT002
    cpos: CameraPositionOptions | None = None,
    window_size: list[int] | None = None,
    show_bounds: bool = False,  # noqa: FBT001, FBT002
    show_axes: bool | None = None,  # noqa: FBT001
    notebook: bool | None = None,  # noqa: FBT001
    background: ColorLike | None = None,
    text: str = '',
    return_img: bool = False,  # noqa: FBT001, FBT002
    eye_dome_lighting: bool = False,  # noqa: FBT001, FBT002
    volume: bool = False,  # noqa: FBT001, FBT002
    parallel_projection: bool = False,  # noqa: FBT001, FBT002
    jupyter_backend: JupyterBackendOptions | None = None,
    return_viewer: bool = False,  # noqa: FBT001, FBT002
    return_cpos: bool = False,  # noqa: FBT001, FBT002
    jupyter_kwargs: dict | None = None,  # type: ignore[type-arg]
    theme: Theme | None = None,
    anti_aliasing: Literal['ssaa', 'msaa', 'fxaa'] | bool | None = None,  # noqa: FBT001
    zoom: str | float | None = None,
    border: bool = False,  # noqa: FBT001, FBT002
    border_color: ColorLike = 'k',
    border_width: float = 2.0,
    ssao: bool = False,  # noqa: FBT001, FBT002
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

    full_screen : bool, default: :attr:`pyvista.plotting.themes.Theme.full_screen`
        Opens window in full screen.  When enabled, ignores
        ``window_size``.

    screenshot : str | bool, optional
        Saves screenshot to file when enabled.  See:
        :func:`Plotter.screenshot() <pyvista.Plotter.screenshot>`.
        Default ``False``.

        When ``True``, takes screenshot and returns ``numpy`` array of
        image.

    interactive : bool, default: :attr:`pyvista.plotting.themes.Theme.interactive`
        Allows user to pan and move figure.

    cpos : CameraPositionOptions, optional
        List of camera position, focal point, and view up.

    window_size : list[int], default: :attr:`pyvista.plotting.themes.Theme.window_size`
        Window size in pixels.

    show_bounds : bool, default: False
        Shows mesh bounds when ``True``.

    show_axes : bool, default: :attr:`pyvista.plotting.themes._AxesConfig.show`
        Shows a vtk axes widget.

    notebook : bool, default: :attr:`pyvista.plotting.themes.Theme.notebook`
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.

    background : ColorLike, default: :attr:`pyvista.plotting.themes.Theme.background`
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

    jupyter_backend : JupyterBackendOptions, optional
        Jupyter notebook plotting backend to use.  One of the
        following:

        * ``'none'`` : Do not display in the notebook.
        * ``'static'`` : Display a static figure.
        * ``'trame'`` : Display using ``trame``.

        This can also be set globally with
        :func:`pyvista.set_jupyter_backend`.

        Defaults to :attr:`pyvista.plotting.themes.Theme.jupyter_backend`

    return_viewer : bool, default: False
        Return the jupyterlab viewer, scene, or display object
        when plotting with jupyter notebook.

    return_cpos : bool, default: False
        Return the last camera position from the render window
        when enabled.  Defaults to value in theme settings.

    jupyter_kwargs : dict, optional
        Keyword arguments for the Jupyter notebook plotting backend.
        See :ref:`customize_trame_toolbar_example` for an example
        using this keyword.

    theme : pyvista.plotting.themes.Theme, optional
        Plot-specific theme.

    anti_aliasing : Literal['ssaa', 'msaa', 'fxaa'] | bool, optional
        Enable or disable anti-aliasing. If ``True``, uses ``"msaa"``. If False,
        disables anti_aliasing. If a string, should be one of the following:

        * ``"ssaa"`` - Super-Sample Anti-Aliasing
        * ``"msaa"`` - Multi-Sample Anti-Aliasing
        * ``"fxaa"`` - Fast Approximate Anti-Aliasing

        Defaults to :attr:`pyvista.plotting.themes.Theme.anti_aliasing`

    zoom : float | str, optional
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
    ImageData. Note ``volume=True`` is passed.

    >>> import numpy as np
    >>> grid = pv.ImageData(dimensions=(32, 32, 32), spacing=(0.5, 0.5, 0.5))
    >>> grid['data'] = np.linalg.norm(grid.center - grid.points, axis=1)
    >>> grid['data'] = np.abs(grid['data'] - grid['data'].max()) ** 3
    >>> grid.plot(volume=True)

    """
    if jupyter_kwargs is None:
        jupyter_kwargs = {}

    # undocumented kwarg used within pytest to run a function before closing
    before_close_callback = kwargs.pop('before_close_callback', None)

    # pop from kwargs here to avoid including them in add_mesh or add_volume
    eye_dome_lighting = kwargs.pop('edl', eye_dome_lighting)
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
        if pl.theme.axes.box:
            pl.add_box_axes()
        else:
            pl.add_axes()

    if anti_aliasing:
        if anti_aliasing is True:
            pl.enable_anti_aliasing('msaa', multi_samples=pyvista.global_theme.multi_samples)
        else:
            pl.enable_anti_aliasing(anti_aliasing)
    elif anti_aliasing is False:
        pl.disable_anti_aliasing()

    try:
        pl.set_background(background)
    except (ValueError, TypeError):
        if isinstance(background, (str, Path)):
            path = Path(background)
            if path.is_file():
                pl.add_background_image(path)
        else:
            msg = f'Background must be color-like or a file path. Got {background} instead.'
            raise TypeError(msg)

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
    elif volume or (isinstance(var_item, np.ndarray) and var_item.ndim == 3):
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
