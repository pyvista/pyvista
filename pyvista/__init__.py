"""PyVista package for 3D plotting and mesh analysis."""
import warnings
import os
import appdirs
from pyvista._version import __version__
from pyvista.plotting import *
from pyvista.utilities import *
from pyvista.core import *
# Per contract with Sphinx-Gallery, this method must be available at top level
from pyvista.utilities.sphinx_gallery import _get_sg_image_scraper

# get the int type from vtk
VTK_ID_TYPE_SIZE = vtk.vtkIdTypeArray().GetDataTypeSize()
ID_TYPE = np.int32
if VTK_ID_TYPE_SIZE == 4:
    ID_TYPE = np.int32
elif VTK_ID_TYPE_SIZE == 8:
    ID_TYPE = np.int64

# for additional error output for VTK segfaults
try:
    import faulthandler
    faulthandler.enable()
except Exception as e:  # pragma: no cover
    warnings.warn(f'Unable to enable faulthandler:\n{e}')


# determine if using vtk > 5
if vtk.vtkVersion().GetVTKMajorVersion() <= 5:
    raise RuntimeError('VTK version must be 5.0 or greater.')

# catch annoying numpy/vtk future warning:
warnings.simplefilter(action='ignore', category=FutureWarning)

# A simple flag to set when generating the documentation
OFF_SCREEN = False
try:
    if os.environ['PYVISTA_OFF_SCREEN'].lower() == 'true':
        OFF_SCREEN = True
except KeyError:
    pass

# flag for when building the sphinx_gallery
BUILDING_GALLERY = False
if 'PYVISTA_BUILDING_GALLERY' in os.environ:
    if os.environ['PYVISTA_BUILDING_GALLERY'].lower() == 'true':
        BUILDING_GALLERY = True

# Grab system flag for anti-aliasing
try:
    rcParams['multi_samples'] = int(os.environ['PYVISTA_MULTI_SAMPLES'])
except KeyError:
    pass

# Grab system flag for auto-closing
try:
    # This only sets to false if PYVISTA_AUTO_CLOSE is false
    rcParams['auto_close'] = not os.environ['PYVISTA_AUTO_CLOSE'].lower() == 'false'
except KeyError:
    pass

# A threshold for the max cells to compute a volume for when repr-ing
REPR_VOLUME_MAX_CELLS = 1e6

# Set where figures are saved
FIGURE_PATH = None

# Set up data directory
USER_DATA_PATH = appdirs.user_data_dir('pyvista')
if not os.path.exists(USER_DATA_PATH):
    os.makedirs(USER_DATA_PATH)


# allow user to override the examples path
if 'PYVISTA_USERDATA_PATH' in os.environ:
    USER_DATA_PATH = os.environ['PYVISTA_USERDATA_PATH']
    if not os.path.isdir(USER_DATA_PATH):
        raise FileNotFoundError(f'Invalid PYVISTA_USERDATA_PATH at {USER_DATA_PATH}')

try:
    EXAMPLES_PATH = os.path.join(USER_DATA_PATH, 'examples')
    if not os.path.exists(EXAMPLES_PATH):
        try:
            os.makedirs(EXAMPLES_PATH)
        except FileExistsError:  # Edge case due to IO race conditions
            pass
except Exception as e:
    warnings.warn(f'Unable to create `EXAMPLES_PATH` at "{EXAMPLES_PATH}"\nError: {e}\n\n'
                  'Override the default path by setting the environmental variable '
                  '`PYVISTA_USERDATA_PATH` to a writable path.')
    EXAMPLES_PATH = None

# Send VTK messages to the logging module:
send_errors_to_logging()

# Set ipyvtk_vtk rcParam flag for interactive notebook rendering
try:
    if os.environ['PYVISTA_USE_IPYVTK'].lower() == 'false':
        rcParams['use_ipyvtk'] = False
    elif os.environ['PYVISTA_USE_IPYVTK'].lower() == 'true':
        rcParams['use_ipyvtk'] = True
except KeyError:
    pass

# Set preferred plot theme
try:
    theme = os.environ['PYVISTA_PLOT_THEME'].lower()
    set_plot_theme(theme)
except KeyError:
    pass


# Set a parameter to control default print format for floats
FLOAT_FORMAT = "{:.3e}"
