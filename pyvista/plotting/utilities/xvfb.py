"""Start xvfb from Python."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import warnings

from pyvista.core.errors import PyVistaDeprecationWarning

XVFB_INSTALL_NOTES = """Please install Xvfb with:

Debian
$ sudo apt install libgl1-mesa-glx xvfb

CentOS / RHL
$ sudo yum install libgl1-mesa-glx xvfb

"""


def start_xvfb(wait=3, window_size=None):
    """Start the virtual framebuffer Xvfb.

    Parameters
    ----------
    wait : float, optional
        Time to wait for the virtual framebuffer to start.  Set to 0
        to disable wait.

    window_size : list, optional
        Window size of the virtual frame buffer.  Defaults to
        :attr:`pyvista.global_theme.window_size
        <pyvista.plotting.themes.Theme.window_size>`.

    Notes
    -----
    Only available on Linux.  Be sure to install ``libgl1-mesa-glx
    xvfb`` in your package manager.

    Examples
    --------
    >>> import pyvista as pv
    >>> pv.start_xvfb()  # doctest:+SKIP

    """
    # Deprecated on 0.45.0, estimated removal on 0.48.0
    warnings.warn(
        'This function is deprecated and will be removed in future version of '
        'PyVista. Use vtk-osmesa instead.',
        PyVistaDeprecationWarning,
    )

    from pyvista import global_theme

    if os.name != 'posix':
        raise OSError('`start_xvfb` is only supported on Linux')

    # Ensure Xvfb is installed and get the full path
    xvfb_path = shutil.which('Xvfb')
    if not xvfb_path:
        raise OSError(XVFB_INSTALL_NOTES)

    # use current default window size
    if window_size is None:
        window_size = global_theme.window_size
    window_size_param = f'{window_size[0]:d}x{window_size[1]:d}x24'
    display_num = ':99'

    try:
        # Start Xvfb using subprocess.run with the full path
        subprocess.run(  # noqa: S603
            [xvfb_path, display_num, '-screen', '0', window_size_param],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Failed to start Xvfb: {e}') from e

    os.environ['DISPLAY'] = display_num
    if wait:
        time.sleep(wait)
