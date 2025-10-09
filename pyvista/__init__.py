"""PyVista package for 3D plotting and mesh analysis."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING
from typing import Literal

from pyvista._plot import plot as plot
from pyvista._version import __version__ as __version__
from pyvista._version import version_info as version_info
from pyvista.core import *
from pyvista.core import _validation as _validation
from pyvista.core._typing_core._dataset_types import _DataObjectType as _DataObjectType
from pyvista.core._typing_core._dataset_types import (
    _DataSetOrMultiBlockType as _DataSetOrMultiBlockType,
)
from pyvista.core._typing_core._dataset_types import _DataSetType as _DataSetType
from pyvista.core._typing_core._dataset_types import _GridType as _GridType
from pyvista.core._typing_core._dataset_types import _PointGridType as _PointGridType
from pyvista.core._typing_core._dataset_types import _PointSetType as _PointSetType
from pyvista.core._vtk_core import _MIN_SUPPORTED_VTK_VERSION
from pyvista.core._vtk_core import VersionInfo
from pyvista.core._vtk_core import vtk_version_info as vtk_version_info
from pyvista.core.cell import _get_vtk_id_type
from pyvista.core.utilities.observers import send_errors_to_logging
from pyvista.core.wrappers import _wrappers as _wrappers
from pyvista.jupyter import JupyterBackendOptions as JupyterBackendOptions
from pyvista.jupyter import set_jupyter_backend as set_jupyter_backend
from pyvista.report import GPUInfo as GPUInfo
from pyvista.report import Report as Report
from pyvista.report import check_math_text_support as check_math_text_support
from pyvista.report import check_matplotlib_vtk_compatibility as check_matplotlib_vtk_compatibility
from pyvista.report import get_gpu_info as get_gpu_info

if TYPE_CHECKING:
    import numpy as np

# get the int type from vtk
ID_TYPE: type[np.int32 | np.int64] = _get_vtk_id_type()

if vtk_version_info < _MIN_SUPPORTED_VTK_VERSION:  # pragma: no cover
    from pyvista.core.errors import VTKVersionError

    msg = f'VTK version must be {VersionInfo._format(_MIN_SUPPORTED_VTK_VERSION)} or greater.'
    raise VTKVersionError(msg)

# A simple flag to set when generating the documentation
OFF_SCREEN = os.environ.get('PYVISTA_OFF_SCREEN', 'false').lower() == 'true'

# flag for when building the sphinx_gallery
BUILDING_GALLERY = os.environ.get('PYVISTA_BUILDING_GALLERY', 'false').lower() == 'true'

# A threshold for the max cells to compute a volume for when repr-ing
REPR_VOLUME_MAX_CELLS = 1e6

# Set where figures are saved
FIGURE_PATH = os.environ.get('PYVISTA_FIGURE_PATH', None)

ON_SCREENSHOT = os.environ.get('PYVISTA_ON_SCREENSHOT', 'false').lower() == 'true'

# Send VTK messages to the logging module:
send_errors_to_logging()

# theme to use by default for the plot directive
PLOT_DIRECTIVE_THEME = None

# Set a parameter to control default print format for floats outside
# of the plotter
FLOAT_FORMAT = '{:.3e}'

# Serialization format to be used when pickling `DataObject`
PICKLE_FORMAT: Literal['vtk', 'xml', 'legacy'] = 'vtk' if vtk_version_info >= (9, 3) else 'xml'

# Name used for unnamed scalars
DEFAULT_SCALARS_NAME = 'Data'

MAX_N_COLOR_BARS = 10

_VTK_SNAKE_CASE_STATE: Literal['allow', 'warning', 'error'] = 'error'

# Allow setting new private -- but not public -- attributes by default
_ALLOW_NEW_ATTRIBUTES_MODE: Literal['private', True, False] = 'private'


# Import all modules for type checkers and linters
if TYPE_CHECKING:
    from pyvista import demos as demos
    from pyvista import examples as examples
    from pyvista import ext as ext
    from pyvista import trame as trame
    from pyvista import utilities as utilities
    from pyvista.plotting import *


# Lazily import/access the plotting module
def __getattr__(name):
    """Fetch an attribute ``name`` from ``globals()`` or the ``pyvista.plotting`` module.

    This override is implemented to prevent importing all of the plotting module
    and GL-dependent VTK modules when importing PyVista.

    Raises
    ------
    AttributeError
        If the attribute is not found.

    """
    import importlib  # noqa: PLC0415
    import inspect  # noqa: PLC0415

    allow = {
        'demos',
        'examples',
        'ext',
        'trame',
        'utilities',
    }
    if name in allow:
        return importlib.import_module(f'pyvista.{name}')

    # avoid recursive import
    if 'pyvista.plotting' not in sys.modules:
        import pyvista.plotting  # noqa: F401, PLC0415

    try:
        feature = inspect.getattr_static(sys.modules['pyvista.plotting'], name)
    except AttributeError:
        msg = f"module 'pyvista' has no attribute '{name}'"
        raise AttributeError(msg) from None

    return feature
