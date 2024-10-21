from __future__ import annotations

import functools
from importlib import metadata
import re

import numpy as np
from numpy.random import default_rng
import pytest

import pyvista
from pyvista import examples
from pyvista.core._vtk_core import VersionInfo

pyvista.OFF_SCREEN = True

NUMPY_VERSION_INFO = VersionInfo(
    major=int(np.__version__.split('.')[0]),
    minor=int(np.__version__.split('.')[1]),
    micro=int(np.__version__.split('.')[2]),
)


def flaky_test(
    test_function=None, *, times: int = 3, exceptions: tuple[Exception, ...] = (AssertionError,)
):
    """Decorator for re-trying flaky tests.

    Parameters
    ----------
    test_function : optional
        Flaky test function. This parameter exists to allow using `@flaky_test`
        instead of `@flaky_test(). It should not be used when applying the decorator
        and can safely be ignored.

    times : int, default: 3
        Number of times to try to test.

    exceptions : tuple[Exception, ...], default: (AssertionError,)
        Exceptions which will cause the test to be re-tried. By default, tests are only
        retried for assertion errors. Customize this to retry for other exceptions
        depending on the cause(s) of the flaky test, e.g. `(ValueError, TypeError)`.

    """
    if test_function is None:
        # Allow decorator is called without parentheses
        return lambda func: flaky_test(func, times=times, exceptions=exceptions)

    @functools.wraps(test_function)
    def wrapper(*args, **kwargs):
        for i in range(times):
            try:
                test_function(*args, **kwargs)
            except exceptions as e:
                func_name = test_function.__name__
                module_name = test_function.__module__
                error_name = e.__class__.__name__
                msg = f'FLAKY TEST FAILED (Attempt {i + 1} of {times}) - {module_name}::{func_name} - {error_name}'
                if i == times - 1:
                    print(msg)
                    raise  # Re-raise the last failure if all retries fail
                else:
                    print(msg + ', retrying...')
            else:
                return  # Exit if the test passes

    return wrapper


@pytest.fixture
def global_variables_reset():  # noqa: PT004
    tmp_screenshots = pyvista.ON_SCREENSHOT
    tmp_figurepath = pyvista.FIGURE_PATH
    yield
    pyvista.ON_SCREENSHOT = tmp_screenshots
    pyvista.FIGURE_PATH = tmp_figurepath


@pytest.fixture(scope='session', autouse=True)
def set_mpl():  # noqa: PT004
    """Avoid matplotlib windows popping up."""
    try:
        import matplotlib as mpl
    except ImportError:
        pass
    else:
        mpl.rcdefaults()
        mpl.use('agg', force=True)


@pytest.fixture
def cube():
    return pyvista.Cube()


@pytest.fixture
def airplane():
    return examples.load_airplane()


@pytest.fixture
def rectilinear():
    return examples.load_rectilinear()


@pytest.fixture
def sphere():
    return examples.load_sphere()


@pytest.fixture
def uniform():
    return examples.load_uniform()


@pytest.fixture
def ant():
    return examples.load_ant()


@pytest.fixture
def globe():
    return examples.load_globe()


@pytest.fixture
def hexbeam():
    return examples.load_hexbeam()


@pytest.fixture
def tetbeam():
    return examples.load_tetbeam()


@pytest.fixture
def struct_grid():
    x, y, z = np.meshgrid(
        np.arange(-10, 10, 2, dtype=np.float32),
        np.arange(-10, 10, 2, dtype=np.float32),
        np.arange(-10, 10, 2, dtype=np.float32),
    )
    return pyvista.StructuredGrid(x, y, z)


@pytest.fixture
def plane():
    return pyvista.Plane(direction=(0, 0, -1))


@pytest.fixture
def spline():
    return examples.load_spline()


@pytest.fixture
def random_hills():
    return examples.load_random_hills()


@pytest.fixture
def tri_cylinder():
    """Triangulated cylinder"""
    return pyvista.Cylinder().triangulate()


@pytest.fixture
def datasets():
    return [
        examples.load_uniform(),  # ImageData
        examples.load_rectilinear(),  # RectilinearGrid
        examples.load_hexbeam(),  # UnstructuredGrid
        examples.load_airplane(),  # PolyData
        examples.load_structured(),  # StructuredGrid
    ]


@pytest.fixture
def multiblock_poly():
    # format and order of data (including missing) is intentional
    mesh_a = pyvista.Sphere(center=(0, 0, 0), direction=(0, 0, -1))
    mesh_a['data_a'] = mesh_a.points[:, 0] * 10
    mesh_a['data_b'] = mesh_a.points[:, 1] * 10
    mesh_a['cell_data'] = mesh_a.cell_centers().points[:, 0]
    mesh_a.point_data.set_array(mesh_a.points[:, 2] * 10, 'all_data')

    mesh_b = pyvista.Sphere(center=(1, 0, 0), direction=(0, 0, -1))
    mesh_b['data_a'] = mesh_b.points[:, 0] * 10
    mesh_b['data_b'] = mesh_b.points[:, 1] * 10
    mesh_b['cell_data'] = mesh_b.cell_centers().points[:, 0]
    mesh_b.point_data.set_array(mesh_b.points[:, 2] * 10, 'all_data')

    mesh_c = pyvista.Sphere(center=(2, 0, 0), direction=(0, 0, -1))
    mesh_c.point_data.set_array(mesh_c.points, 'multi-comp')
    mesh_c.point_data.set_array(mesh_c.points[:, 2] * 10, 'all_data')

    mblock = pyvista.MultiBlock()
    mblock.append(mesh_a)
    mblock.append(mesh_b)
    mblock.append(mesh_c)
    return mblock


@pytest.fixture
def datasets_vtk9():
    return [
        examples.load_explicit_structured(),
    ]


@pytest.fixture
def pointset():
    rng = default_rng(0)
    points = rng.random((10, 3))
    return pyvista.PointSet(points)


@pytest.fixture
def multiblock_all(datasets):
    """Return datasets fixture combined in a pyvista multiblock."""
    return pyvista.MultiBlock(datasets)


@pytest.fixture
def multiblock_all_with_nested_and_none(datasets, multiblock_all):
    """Return datasets fixture combined in a pyvista multiblock."""
    multiblock_all.append(None)
    return pyvista.MultiBlock([*datasets, None, multiblock_all])


@pytest.fixture
def noise_2d():
    freq = [10, 5, 0]
    noise = pyvista.perlin_noise(1, freq, (0, 0, 0))
    return pyvista.sample_function(noise, bounds=(0, 10, 0, 10, 0, 10), dim=(2**4, 2**4, 1))


@pytest.fixture
def texture():
    # create a basic texture by plotting a sphere and converting the image
    # buffer to a texture
    pl = pyvista.Plotter(window_size=(300, 200), lighting=None)
    mesh = pyvista.Sphere()
    pl.add_mesh(mesh, scalars=range(mesh.n_points), show_scalar_bar=False)
    pl.background_color = 'w'
    return pyvista.Texture(pl.screenshot())


@pytest.fixture
def image(texture):
    return texture.to_image()


def pytest_addoption(parser):
    parser.addoption('--test_downloads', action='store_true', default=False)


def marker_names(item):
    return [marker.name for marker in item.iter_markers()]


def pytest_collection_modifyitems(config, items):
    test_downloads = config.getoption('--test_downloads')

    # skip all tests that need downloads
    if not test_downloads:
        skip_downloads = pytest.mark.skip('Downloads not enabled with --test_downloads')
        for item in items:
            if 'needs_download' in marker_names(item):
                item.add_marker(skip_downloads)


def pytest_runtest_setup(item):
    """Custom setup to handle skips based on VTK version.

    See pytest.mark.needs_vtk_version in pyproject.toml.

    """
    for item_mark in item.iter_markers('needs_vtk_version'):
        # this test needs the given VTK version
        # allow both needs_vtk_version(9, 1) and needs_vtk_version((9, 1))
        args = item_mark.args
        version_needed = args[0] if len(args) == 1 and isinstance(args[0], tuple) else args
        if pyvista.vtk_version_info < version_needed:
            version_str = '.'.join(map(str, version_needed))
            pytest.skip(f'Test needs VTK {version_str} or newer.')


def pytest_report_header(config):
    """Header for pytest to show versions of required and optional packages."""
    required = []
    extra = {}
    for item in metadata.requires('pyvista'):
        pkg_name = re.findall(r'[a-z0-9_\-]+', item, re.IGNORECASE)[0]
        if pkg_name == 'pyvista':
            continue
        elif res := re.findall('extra == [\'"](.+)[\'"]', item):
            assert len(res) == 1, item
            pkg_extra = res[0]
            if pkg_extra not in extra:
                extra[pkg_extra] = []
            extra[pkg_extra].append(pkg_name)
        else:
            required.append(pkg_name)

    lines = []
    items = []
    for name in required:
        try:
            version = metadata.version(name)
            items.append(f'{name}-{version}')
        except metadata.PackageNotFoundError:
            items.append(f'{name} (not found)')
    lines.append('required packages: ' + ', '.join(items))

    not_found = []
    for pkg_extra in extra.keys():
        installed = []
        for name in extra[pkg_extra]:
            try:
                version = metadata.version(name)
                installed.append(f'{name}-{version}')
            except metadata.PackageNotFoundError:
                not_found.append(name)
        if installed:
            plrl = 's' if len(installed) != 1 else ''
            comma_lst = ', '.join(installed)
            lines.append(f'optional {pkg_extra!r} package{plrl}: {comma_lst}')
    if not_found:
        plrl = 's' if len(not_found) != 1 else ''
        comma_lst = ', '.join(not_found)
        lines.append(f'optional package{plrl} not found: {comma_lst}')
    return '\n'.join(lines)
