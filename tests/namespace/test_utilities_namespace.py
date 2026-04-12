from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pyvista.core.errors import PyVistaDeprecationWarning

namespace_data = Path(__file__).parent / 'namespace-utilities.txt'
with namespace_data.open() as f:
    namespace = f.read().splitlines()
    # ignore commented data
    namespace = [n.split(', ')[0] for n in namespace if not n.startswith('#')]


@pytest.mark.parametrize('name', namespace)
def test_utilities_namespace(name):
    import pyvista.utilities as utilities  # noqa: PLR0402

    with pytest.warns(PyVistaDeprecationWarning):
        assert hasattr(utilities, name)


@pytest.mark.parametrize(
    'name',
    [
        'algorithms',
        'arrays',
        'cell_type_helper',
        'cells',
        'common',
        'docs',
        'errors',
        'features',
        'fileio',
        'geometric_objects',
        'helpers',
        'misc',
        'parametric_objects',
        'reader',
        'regression',
        'sphinx_gallery',
        'transformations',
        'wrappers',
        'xvfb',
    ],
)
def test_utilities_modules(name):
    # Smoke test to make sure same modules still exist
    importlib.import_module(f'pyvista.utilities.{name}')


def _import_all_utilities():
    """Import all utilities to test deprecated namespace imports."""
    import pyvista.utilities  # noqa: F401

    # Import specific items to ensure they exist
    from pyvista.utilities import abstract_class  # noqa: F401
    from pyvista.utilities import assert_empty_kwargs  # noqa: F401
    from pyvista.utilities import conditional_decorator  # noqa: F401
    from pyvista.utilities import convert_string_array  # noqa: F401
    from pyvista.utilities import generate_plane  # noqa: F401
    from pyvista.utilities import get_array  # noqa: F401
    from pyvista.utilities import get_array_association  # noqa: F401
    from pyvista.utilities import get_vtk_type  # noqa: F401
    from pyvista.utilities import threaded  # noqa: F401
    from pyvista.utilities import try_callback  # noqa: F401
    from pyvista.utilities import xvfb  # noqa: F401
    from pyvista.utilities.algorithms import add_ids_algorithm  # noqa: F401
    from pyvista.utilities.algorithms import algorithm_to_mesh_handler  # noqa: F401
    from pyvista.utilities.algorithms import crinkle_algorithm  # noqa: F401
    from pyvista.utilities.algorithms import outline_algorithm  # noqa: F401
    from pyvista.utilities.algorithms import pointset_to_polydata_algorithm  # noqa: F401
    from pyvista.utilities.algorithms import set_algorithm_input  # noqa: F401
    from pyvista.utilities.errors import GPUInfo  # noqa: F401
    from pyvista.utilities.geometric_objects import Arrow  # noqa: F401
    from pyvista.utilities.geometric_objects import Cylinder  # noqa: F401
    from pyvista.utilities.geometric_objects import PlatonicSolid  # noqa: F401
    from pyvista.utilities.helpers import vtk_id_list_to_array  # noqa: F401
    from pyvista.utilities.sphinx_gallery import _get_sg_image_scraper  # noqa: F401
    from pyvista.utilities.xvfb import start_xvfb  # noqa: F401


def test_common_utilities_import_paths():
    # These are `pyvista.utilities` imports found via search on GitHub
    # across multiple public repositories
    with pytest.warns(PyVistaDeprecationWarning):
        _import_all_utilities()


def test_failure_to_find():
    module = importlib.import_module('pyvista.utilities')
    with pytest.raises(
        AttributeError,
        match=r'Module `pyvista\.utilities` has been deprecated '
        r'and we could not automatically find',
    ):
        _ = module.this_does_not_exist
