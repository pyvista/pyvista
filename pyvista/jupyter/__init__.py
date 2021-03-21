"""Jupyter notebook plotting module."""

import os
from .. import rcParams
from .itkplotter import PlotterITK


# Set ipyvtk_vtk rcParam flag for interactive notebook rendering
def check_backend_env_var():
    if 'PYVISTA_JUPYTER_BACKEND' in os.environ:
        set_jupyter_backend(os.environ['PYVISTA_JUPYTER_BACKEND'])


def set_jupyter_backend(backend):
    """Set the plotting backend for a jupyter notebook"""
    # Must be a string
    if backend is None:
        backend = 'static'
    backend = backend.lower()

    try:
        import IPython
    except ImportError:
        raise ImportError('Install IPython to display with pyvista in a notebook.')

    allowed_backends = ['ipyvtk_simple', 'panel', 'ipygany', 'static', 'none']
    if backend not in allowed_backends:
        backend_list_str = ', '.join([f'"{item}"' for item in allowed_backends])
        raise ValueError(f'Invalid Jupyter notebook plotting backend "{backend}".\n'
                         f'Use one of the following:\n{backend_list_str}')

    # verify required packages are installed
    if backend == 'ipyvtk_simple':
        try:
            import ipyvtk_simple
        except ImportError:
            raise ImportError('Please install `ipyvtk_simple` to use this feature')

    if backend == 'panel':
        try:
            import panel
        except ImportError:
            raise ImportError('Please install `panel` to use this feature')
        panel.extension('vtk')

    if backend == 'ipygany':
        # raises an import error when fail
        import pyvista.jupyter.pv_ipygany

    if backend == 'none':
        backend = None

    rcParams['jupyter_backend'] = backend


# this will run on __init__ to set the backend
check_backend_env_var()
