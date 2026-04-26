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
from pyvista.core._vtk_utilities import _MIN_SUPPORTED_VTK_VERSION
from pyvista.core._vtk_utilities import VersionInfo
from pyvista.core._vtk_utilities import vtk_version_info as vtk_version_info
from pyvista.core.cell import _get_vtk_id_type
from pyvista.core.filters.data_object import MeshValidationFields as MeshValidationFields
from pyvista.core.utilities.accessor_registry import AccessorRegistration as AccessorRegistration
from pyvista.core.utilities.accessor_registry import DataSetAccessor as DataSetAccessor
from pyvista.core.utilities.accessor_registry import (
    register_dataset_accessor as register_dataset_accessor,
)
from pyvista.core.utilities.accessor_registry import registered_accessors as registered_accessors
from pyvista.core.utilities.accessor_registry import (
    unregister_dataset_accessor as unregister_dataset_accessor,
)
from pyvista.core.utilities.observers import send_errors_to_logging
from pyvista.core.utilities.reader_registry import LocalFileRequiredError as LocalFileRequiredError
from pyvista.core.utilities.reader_registry import ReaderRegistration as ReaderRegistration
from pyvista.core.utilities.reader_registry import has_scheme as has_scheme
from pyvista.core.utilities.reader_registry import register_reader as register_reader
from pyvista.core.utilities.reader_registry import registered_readers as registered_readers
from pyvista.core.utilities.writer_registry import WriterRegistration as WriterRegistration
from pyvista.core.utilities.writer_registry import register_writer as register_writer
from pyvista.core.utilities.writer_registry import registered_writers as registered_writers
from pyvista.core.wrappers import _wrappers as _wrappers
from pyvista.jupyter import JupyterBackendOptions as JupyterBackendOptions
from pyvista.jupyter import JupyterBackendRegistration as JupyterBackendRegistration
from pyvista.jupyter import register_jupyter_backend as register_jupyter_backend
from pyvista.jupyter import registered_jupyter_backends as registered_jupyter_backends
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


# Tracks whether the ``PYVISTA_PLOT_THEME`` environment variable has been
# applied yet. Applying a plugin theme runs arbitrary plugin code that can
# call back into ``pyvista`` before this module has finished the caller's
# original request; the flag keeps the apply single-shot.
_env_theme_applied: bool = False


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

    if name == 'hexcolors':
        from pyvista.plotting.colors import _get_deprecated_hexcolors  # noqa: PLC0415

        return _get_deprecated_hexcolors()

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

    # Apply ``PYVISTA_PLOT_THEME`` once, now that ``pyvista.plotting`` is fully
    # loaded and the caller's requested attribute is already resolved. Doing
    # this inside ``pyvista.plotting.__init__`` invites re-entrant access to a
    # partially-initialized module when an entry-point-registered plugin is
    # imported (Python 3.12 evaluates annotations like ``pv.Plotter`` eagerly
    # at plugin module load). The flag is set before the call to prevent
    # re-entrant double-application if a plugin's module body accesses
    # attributes on ``pyvista`` during the theme apply.
    global _env_theme_applied  # noqa: PLW0603
    if not _env_theme_applied:
        _env_theme_applied = True
        sys.modules['pyvista.plotting']._set_plot_theme_from_env()

    return feature
