"""

knowledge
=========

The knowledge base.

It contains, for instance, known odd locations of version information for
particular modules (``VERSION_ATTRIBUTES``, ``VERSION_METHODS``)

It also checks and stores mandatory additional information, if possible, such
as available RAM or MKL info.

"""


try:
    import psutil
except ImportError:
    psutil = False

try:
    import mkl
except ImportError:
    mkl = False

try:
    import numexpr
except ImportError:
    numexpr = False

# Get available RAM, if available
if psutil:
    tmem = psutil.virtual_memory().total
    TOTAL_RAM = '{:.1f} GB'.format(tmem / (1024.0 ** 3))
else:
    TOTAL_RAM = False

# Get mkl info from numexpr or mkl, if available
if mkl:
    MKL_INFO = mkl.get_version_string()
elif numexpr:
    MKL_INFO = numexpr.get_vml_version()
else:
    MKL_INFO = False

# Define unusual version locations
VERSION_ATTRIBUTES = {
    'vtk': 'VTK_VERSION',
    'vtkmodules.all': 'VTK_VERSION',
    'PyQt5': 'Qt.PYQT_VERSION_STR',
}


def get_pyqt5_version():
    """Returns the PyQt5 version"""
    from PyQt5.Qt import PYQT_VERSION_STR
    return PYQT_VERSION_STR


VERSION_METHODS = {
    'PyQt5': get_pyqt5_version,
}


# Check the environments
def in_ipython():
    """Mystery: are we in an IPython environment?

    Returns
    -------
    bool : True if in an IPython environment
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def in_ipykernel():
    """Mystery: are we in a ipykernel (most likely Jupyter) environment?

    Warning
    -------
    There is no way to tell if the code is being executed in a notebook
    (Jupyter Notebook or Jupyter Lab) or a kernel is used but executed in a
    QtConsole, or in an IPython console, or any other frontend GUI. However, if
    `in_ipykernel` returns True, you are most likely in a Jupyter Notebook/Lab,
    just keep it in mind that there are other possibilities.

    Returns
    -------
    bool : True if using an ipykernel
    """
    ipykernel = False
    if in_ipython():
        try:
            ipykernel = type(get_ipython()).__module__.startswith('ipykernel.')
        except NameError:
            pass
    return ipykernel
