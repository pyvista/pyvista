"""Start xvfb from Python."""
import time
import os

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
        Window size of the virtual frame buffer.  Defaults to the
        default window size in ``rcParams``.

    """
    from pyvista import rcParams

    if os.name != 'posix':
        raise OSError('`start_xvfb` is only supported on Linux')

    if os.system('which Xvfb > /dev/null'):
        raise OSError(XVFB_INSTALL_NOTES)

    # use current default window size
    window_size_parm = '%dx%dx24' % tuple(rcParams['window_size'])
    display_num = ':99'
    os.system(f'Xvfb {display_num} -screen 0 {window_size_parm} > /dev/null 2>&1 &')
    os.environ['DISPLAY'] = display_num
    if wait:
        time.sleep(wait)
