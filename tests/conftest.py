from __future__ import annotations

import faulthandler
import functools
from importlib import metadata
from inspect import BoundArguments
from inspect import Parameter
from inspect import Signature
import os
import platform
import re

import numpy as np
from numpy.random import default_rng
import PIL
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core._vtk_utilities import VersionInfo
from pyvista.plotting.utilities.gl_checks import uses_egl

pv.OFF_SCREEN = True

NUMPY_VERSION_INFO = VersionInfo(
    major=int(np.__version__.split('.')[0]),
    minor=int(np.__version__.split('.')[1]),
    micro=int(np.__version__.split('.')[2]),
)
PILLOW_VERSION_INFO = VersionInfo(
    major=int(PIL.__version__.split('.')[0]),
    minor=int(PIL.__version__.split('.')[1]),
    micro=int(PIL.__version__.split('.')[2]),
)

faulthandler.enable()


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
                msg = (
                    f'FLAKY TEST FAILED (Attempt {i + 1} of {times}) - '
                    f'{module_name}::{func_name} - {error_name}'
                )
                if i == times - 1:
                    print(msg)
                    raise  # Re-raise the last failure if all retries fail
                else:
                    print(msg + ', retrying...')
            else:
                return  # Exit if the test passes

    return wrapper


@pytest.fixture
def global_variables_reset():
    tmp_screenshots = pv.ON_SCREENSHOT
    tmp_figurepath = pv.FIGURE_PATH
    yield
    pv.ON_SCREENSHOT = tmp_screenshots
    pv.FIGURE_PATH = tmp_figurepath


@pytest.fixture(scope='session', autouse=True)
def set_mpl():
    """Avoid matplotlib windows popping up."""
    try:
        import matplotlib as mpl
    except ImportError:
        pass
    else:
        mpl.rcdefaults()
        mpl.use('agg', force=True)


@pytest.fixture(autouse=True)
def reset_global_state():
    # Default is to allow new 'private' attributes for downstream packages,
    # but for PyVista itself we enforce no new attributes
    pv.allow_new_attributes(False)
    assert pv.allow_new_attributes() is False

    yield

    pv.vtk_snake_case('error')
    assert pv.vtk_snake_case() == 'error'

    pv.vtk_verbosity('info')
    assert pv.vtk_verbosity() == 'info'

    pv.allow_new_attributes(False)
    assert pv.allow_new_attributes() is False

    pv.PICKLE_FORMAT = 'vtk'


@pytest.fixture
def cube():
    return pv.Cube()


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
    return pv.StructuredGrid(x, y, z)


@pytest.fixture
def plane():
    return pv.Plane(direction=(0, 0, -1))


@pytest.fixture
def spline():
    return examples.load_spline()


@pytest.fixture
def random_hills():
    return examples.load_random_hills()


@pytest.fixture
def tri_cylinder():
    """Triangulated cylinder"""
    return pv.Cylinder().triangulate()


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
def datasets_plus_pointset(datasets, ant):
    return [*datasets, ant.cast_to_pointset()]


@pytest.fixture
def multiblock_poly():
    # format and order of data (including missing) is intentional
    mesh_a = pv.Sphere(center=(0, 0, 0), direction=(0, 0, -1))
    mesh_a['data_a'] = mesh_a.points[:, 0] * 10
    mesh_a['data_b'] = mesh_a.points[:, 1] * 10
    mesh_a['cell_data'] = mesh_a.cell_centers().points[:, 0]
    mesh_a.point_data.set_array(mesh_a.points[:, 2] * 10, 'all_data')

    mesh_b = pv.Sphere(center=(1, 0, 0), direction=(0, 0, -1))
    mesh_b['data_a'] = mesh_b.points[:, 0] * 10
    mesh_b['data_b'] = mesh_b.points[:, 1] * 10
    mesh_b['cell_data'] = mesh_b.cell_centers().points[:, 0]
    mesh_b.point_data.set_array(mesh_b.points[:, 2] * 10, 'all_data')

    mesh_c = pv.Sphere(center=(2, 0, 0), direction=(0, 0, -1))
    mesh_c.point_data.set_array(mesh_c.points, 'multi-comp')
    mesh_c.point_data.set_array(mesh_c.points[:, 2] * 10, 'all_data')

    mblock = pv.MultiBlock()
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
    return pv.PointSet(points)


@pytest.fixture
def multiblock_all(datasets):
    """Return datasets fixture combined in a pyvista multiblock."""
    return pv.MultiBlock(datasets)


@pytest.fixture
def multiblock_all_with_nested_and_none(datasets, multiblock_all):
    """Return datasets fixture combined in a pyvista multiblock."""
    multiblock_all.append(None)
    return pv.MultiBlock([*datasets, None, multiblock_all.copy()])


@pytest.fixture
def noise_2d():
    freq = [10, 5, 0]
    noise = pv.perlin_noise(1, freq, (0, 0, 0))
    return pv.sample_function(noise, bounds=(0, 10, 0, 10, 0, 10), dim=(2**4, 2**4, 1))


@pytest.fixture
def texture():
    # create a basic texture by plotting a sphere and converting the image
    # buffer to a texture
    pl = pv.Plotter(window_size=(300, 200), lighting=None)
    mesh = pv.Sphere()
    pl.add_mesh(mesh, scalars=range(mesh.n_points), show_scalar_bar=False)
    pl.background_color = 'w'
    return pv.Texture(pl.screenshot())


@pytest.fixture
def image(texture):
    return texture.to_image()


def pytest_addoption(parser):
    parser.addoption('--test_downloads', action='store_true', default=False)


def _check_args_kwargs_marker(item_mark: pytest.Mark, sig: Signature):
    """Test for a given args and kwargs for a mark using its signature"""

    try:
        bounds = sig.bind(*item_mark.args, **item_mark.kwargs)
    except TypeError as e:
        msg = (
            f'Marker `{item_mark.name}` called with incorrect arguments.\n'
            f'Signature should be: @pytest.mark.{item_mark.name}{sig}'
        )
        raise ValueError(msg) from e
    else:
        bounds.apply_defaults()
        return bounds


def _get_min_max_vtk_version(
    item_mark: pytest.Mark,
    sig: Signature,
) -> tuple[tuple[int] | None, tuple[int] | None, BoundArguments]:
    bounds = _check_args_kwargs_marker(item_mark=item_mark, sig=sig)

    def _pad_version(val: tuple[int] | None):
        if val is None:
            return val

        if (l := len(val)) == (expected := 3):
            return val

        if l > expected:
            msg = f'Version tuple incorrect length (needs <= {expected})'
            raise ValueError(msg)

        return val + (0,) * (expected - l)

    # Distinguish scenarios from positional arguments
    if (len(args := bounds.arguments['args']) > 0) and (bounds.arguments['at_least'] is not None):
        msg = (
            f'Cannot specify both *args and `at_least` keyword argument to '
            f'`{item_mark.name}` marker.'
        )
        raise ValueError(msg)

    if len(args) > 0:
        min_version = args[0] if len(args) == 1 and isinstance(args[0], tuple) else args
        return _pad_version(min_version), _pad_version(bounds.arguments['less_than']), bounds

    _min = bounds.arguments['at_least']
    _max = bounds.arguments['less_than']

    if _max is None and _min is None:
        msg = (
            f'Need to specify either `at_least` or `less_than` keyword arguments to '
            f'`{item_mark.name}` marker.'
        )
        raise ValueError(msg)

    return _pad_version(_min), _pad_version(_max), bounds


def pytest_runtest_setup(item: pytest.Item):
    """Custom setup to handle skips based on VTK version.

    See custom marks in pyproject.toml.
    """
    needs_vtk_version = 'needs_vtk_version'
    # this test needs a given VTK version
    for item_mark in item.iter_markers(needs_vtk_version):
        sig = Signature(
            [
                Parameter(
                    'args',
                    kind=Parameter.VAR_POSITIONAL,
                    annotation=int | tuple[int],
                ),
                Parameter(
                    'at_least',
                    kind=Parameter.KEYWORD_ONLY,
                    annotation=tuple[int] | None,
                    default=None,
                ),
                Parameter(
                    'less_than',
                    kind=Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=tuple[int] | None,
                ),
                Parameter(
                    'reason',
                    kind=Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=str | None,
                ),
            ]
        )
        _min, _max, bounds = _get_min_max_vtk_version(item_mark=item_mark, sig=sig)
        _min = (_min,) if isinstance(_min, int) else _min
        _max = (_max,) if isinstance(_max, int) else _max

        if (_min is not None and _min <= pv._MIN_SUPPORTED_VTK_VERSION) or (
            _max is not None and _max <= pv._MIN_SUPPORTED_VTK_VERSION
        ):
            msg = (
                f'The {needs_vtk_version!r} marker is no longer necessary\n'
                f'and can be removed from test {item}.'
            )
            raise pv.VTKVersionError(msg)

        curr_version = pv.vtk_version_info

        if _max is None and curr_version < _min:
            reason = item_mark.kwargs.get(
                'reason', f'Test needs VTK version >= {_min}, current is {curr_version}.'
            )
            pytest.skip(reason=reason)

        if _min is None and curr_version >= _max:
            reason = item_mark.kwargs.get(
                'reason', f'Test needs VTK version < {_max}, current is {curr_version}.'
            )
            pytest.skip(reason=reason)

        if _min is not None and _max is not None:
            if _min > _max:
                msg = 'Cannot specify a minimum version greater than the maximum one.'
                raise ValueError(msg)

            if curr_version < _min or curr_version >= _max:
                reason = item_mark.kwargs.get(
                    'reason',
                    f'Test needs {_min} <= VTK version < {_max}, current is {curr_version}.',
                )
                pytest.skip(reason=reason)

    if item_mark := item.get_closest_marker('skip_egl'):
        sig = Signature(
            [
                Parameter(
                    r := 'reason',
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    default='Test fails when using OSMesa/EGL VTK build',
                    annotation=str,
                )
            ]
        )

        bounds = _check_args_kwargs_marker(item_mark=item_mark, sig=sig)
        if uses_egl():
            pytest.skip(bounds.arguments[r])

    if item_mark := item.get_closest_marker('skip_windows'):
        sig = Signature(
            [
                Parameter(
                    r := 'reason',
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    default='Test fails on Windows',
                    annotation=str,
                )
            ]
        )

        bounds = _check_args_kwargs_marker(item_mark=item_mark, sig=sig)
        if os.name == 'nt':
            pytest.skip(bounds.arguments[r])

    if item_mark := item.get_closest_marker('skip_mac'):
        sig = Signature(
            [
                Parameter(
                    r := 'reason',
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    default='Test fails on MacOS',
                    annotation=str,
                ),
                Parameter(
                    p := 'processor',
                    kind=Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=str | None,
                ),
                Parameter(
                    m := 'machine',
                    kind=Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=str | None,
                ),
            ]
        )

        bounds = _check_args_kwargs_marker(item_mark=item_mark, sig=sig)

        should_skip = platform.system() == 'Darwin'
        if (proc := bounds.arguments[p]) is not None:
            should_skip &= proc == platform.processor()

        if (machine := bounds.arguments[m]) is not None:
            should_skip &= machine == platform.machine()

        if should_skip:
            pytest.skip(bounds.arguments[r])

    test_downloads = item.config.getoption(flag := '--test_downloads')
    if item.get_closest_marker('needs_download') and not test_downloads:
        pytest.skip(f'Downloads not enabled with {flag}')


def pytest_report_header(config):  # noqa: ARG001
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
    for pkg_extra in extra.keys():  # noqa: PLC0206
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
