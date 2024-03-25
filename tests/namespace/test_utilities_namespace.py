import importlib
import pathlib

import pytest

from pyvista.core.errors import PyVistaDeprecationWarning

namespace_data = pathlib.Path(__file__).parent / 'namespace-utilities.txt'
with namespace_data.open() as f:
    namespace = f.read().splitlines()
    # ignore commented data
    namespace = [n.split(', ')[0] for n in namespace if not n.startswith('#')]


@pytest.mark.parametrize('name', namespace)
def test_utilities_namespace(name):
    with pytest.warns(PyVistaDeprecationWarning):
        import pyvista.utilities as utilities

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


def test_common_utilities_import_paths():
    # These are `pyvista.utilities` imports found via search on GitHub
    # across multiple public repositories
    with pytest.warns(PyVistaDeprecationWarning):
        from pyvista.utilities import (  # noqa: F401
            NORMALS,
            abstract_class,
            assert_empty_kwargs,
            conditional_decorator,
            convert_string_array,
            generate_plane,
            get_array,
            get_array_association,
            get_vtk_type,
            threaded,
            try_callback,
            xvfb,
        )
        from pyvista.utilities.algorithms import (  # noqa: F401
            add_ids_algorithm,
            algorithm_to_mesh_handler,
            crinkle_algorithm,
            outline_algorithm,
            pointset_to_polydata_algorithm,
            set_algorithm_input,
        )
        from pyvista.utilities.errors import GPUInfo  # noqa: F401
        from pyvista.utilities.geometric_objects import Arrow, Cylinder, PlatonicSolid  # noqa: F401
        from pyvista.utilities.helpers import vtk_id_list_to_array  # noqa: F401
        from pyvista.utilities.sphinx_gallery import _get_sg_image_scraper  # noqa: F401
        from pyvista.utilities.xvfb import start_xvfb  # noqa: F401


def test_failure_to_find():
    module = importlib.import_module('pyvista.utilities')
    with pytest.raises(
        AttributeError,
        match=r'Module `pyvista\.utilities` has been deprecated and we could not automatically find',
    ):
        getattr(module, 'this_does_not_exist')
