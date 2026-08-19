from __future__ import annotations

import faulthandler
import functools
from importlib.metadata import PackageNotFoundError
from importlib.metadata import requires
from importlib.metadata import version
import importlib.util
import inspect
import os
from pathlib import Path
import platform
import re

import numpy as np
from numpy.random import default_rng
import PIL
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista import examples
from pyvista.core._vtk_utilities import VersionInfo
from pyvista.core.utilities.accessor_registry import (
    _restore_registry_state as _restore_accessor_registry_state,
)
from pyvista.core.utilities.accessor_registry import (
    _save_registry_state as _save_accessor_registry_state,
)
from pyvista.core.utilities.reader_registry import _restore_registry_state
from pyvista.core.utilities.reader_registry import _save_registry_state
from pyvista.core.utilities.writer_registry import (
    _restore_registry_state as _restore_writer_registry_state,
)
from pyvista.core.utilities.writer_registry import (
    _save_registry_state as _save_writer_registry_state,
)

# Probe the active backend for a rendering module (precise, and no import-error control flow).
HAS_PLOTTING = importlib.util.find_spec(f'{_vtk._VTK_ROOT}.vtkRenderingCore') is not None

if HAS_PLOTTING:
    from pyvista.plotting.component_registry import (
        _restore_registry_state as _restore_component_registry_state,
    )
    from pyvista.plotting.component_registry import (
        _save_registry_state as _save_component_registry_state,
    )
    from pyvista.plotting.interactor_style_registry import (
        _restore_registry_state as _restore_style_registry_state,
    )
    from pyvista.plotting.interactor_style_registry import (
        _save_registry_state as _save_style_registry_state,
    )
    from pyvista.plotting.theme_registry import (
        _restore_registry_state as _restore_theme_registry_state,
    )
    from pyvista.plotting.theme_registry import _save_registry_state as _save_theme_registry_state
    from pyvista.plotting.utilities.gl_checks import uses_egl
else:  # core-only VTK backend: rendering modules are absent

    def _save_component_registry_state():
        return None

    def _restore_component_registry_state(state):  # noqa: ARG001
        return None

    def _save_style_registry_state():
        return None

    def _restore_style_registry_state(state):  # noqa: ARG001
        return None

    def _save_theme_registry_state():
        return None

    def _restore_theme_registry_state(state):  # noqa: ARG001
        return None

    def uses_egl():
        return False


def _has_vtk_module(module_name: str) -> bool:
    """Probe the active backend for an IO module a core-only build may omit.

    Lets ``pytest_collection_modifyitems`` auto-apply ``needs_io_extra`` so the
    core-only subset can deselect the readers/writers that wrap these lazily.
    """
    return importlib.util.find_spec(f'{_vtk._VTK_ROOT}.{module_name}') is not None


# IO-tier VTK modules a core-only build may omit; each backs lazily-wrapped readers/writers.
HAS_IO_HDF = _has_vtk_module('vtkIOHDF')  # HDFReader, .vtkhdf save
HAS_IO_ENSIGHT = _has_vtk_module('vtkIOEnSight')  # EnSightReader (.case)
HAS_IO_CHEMISTRY = _has_vtk_module('vtkIOChemistry')  # PDB / XYZ / GaussianCube
HAS_IO_EXTRA = HAS_IO_HDF and HAS_IO_ENSIGHT and HAS_IO_CHEMISTRY

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
    test_function=None,
    *,
    times: int = 3,
    exceptions: tuple[type[Exception], ...] = (AssertionError,),
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

    exceptions : tuple[type[Exception], ...], default: (AssertionError,)
        Exception types which will cause the test to be re-tried. By default, tests
        are only retried for assertion errors. Customize this to retry for other
        exceptions depending on the cause(s) of the flaky test, e.g.
        ``(ValueError, TypeError)``.

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
    import matplotlib as mpl

    mpl.rcdefaults()
    mpl.use('agg', force=True)


@pytest.fixture(autouse=True)
def reset_global_state():
    # Default is to allow new 'private' attributes for downstream packages,
    # but for PyVista itself we enforce no new attributes
    pv.allow_new_attributes(False)
    assert pv.allow_new_attributes() is False

    style_registry_state = _save_style_registry_state()
    reader_registry_state = _save_registry_state()
    writer_registry_state = _save_writer_registry_state()
    accessor_registry_state = _save_accessor_registry_state()
    component_registry_state = _save_component_registry_state()
    theme_registry_state = _save_theme_registry_state()

    yield

    _restore_style_registry_state(style_registry_state)
    _restore_registry_state(reader_registry_state)
    _restore_writer_registry_state(writer_registry_state)
    _restore_accessor_registry_state(accessor_registry_state)
    _restore_component_registry_state(component_registry_state)
    _restore_theme_registry_state(theme_registry_state)

    pv.vtk_snake_case('error')
    assert pv.vtk_snake_case() == 'error'

    pv.vtk_verbosity('info')
    # On VTK built with VTK_ENABLE_LOGGING=OFF the cutoff is fixed at the
    # disabled sentinel and setting verbosity is a no-op, so it stays 'off'.
    assert pv.vtk_verbosity() in ('info', 'off')

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


@pytest.hookimpl(trylast=True)
def pytest_sessionstart():
    """Register Hypothesis' global PRNG now, while the heap is still thawed.

    The leak check freezes the heap for the whole body of every test it covers (see
    :mod:`tests.gc_check`), and ``gc.get_referrers()`` does not look in the permanent
    generation. Hypothesis creates and registers its global PRNG lazily, on the first
    ``@given`` test in the process, and ``register_random`` sanity-checks that PRNG with
    ``gc.get_referrers()``. Run inside a frozen body, that comes back empty, so Hypothesis
    warns that the PRNG looks collectable -- and ``filterwarnings = ['error']`` turns the
    warning into a failure of that first test.

    Two conditions have to line up for that. On Python 3.14 ``threading.local()``
    allocates its per-thread dict at construction, so the dict holding the PRNG is built
    when ``hypothesis.core`` is imported and lands in the permanent generation; through
    3.13 it is created on first access, inside the frozen window, where it stays visible.
    And the lazy ``register_random`` only runs under ``derandomize=True``, which comes
    from Hypothesis' ``ci`` profile, auto-loaded when ``CI`` is set. A local run on 3.13,
    or without ``CI``, will not reproduce it; xdist only picks which test takes the hit.

    Running one throwaway example here does that registration once, at session start,
    where nothing is frozen. Deleting this brings the failures back.

    ``trylast`` so it runs after Hypothesis' own ``pytest_sessionstart``, which is where
    Hypothesis stops treating the process as still initializing; before that it warns
    about strategies being evaluated in a plugin.
    """
    # The docs-test dependency group installs no Hypothesis, and this conftest still has
    # to import under it, so probe for the package instead of importing it at module scope.
    if importlib.util.find_spec('hypothesis') is None:
        return

    from hypothesis import HealthCheck
    from hypothesis import given
    from hypothesis import settings
    from hypothesis import strategies as st

    @settings(
        max_examples=1, deadline=None, database=None, suppress_health_check=list(HealthCheck)
    )
    @given(_=st.none())
    def warm_up_prng(_):
        pass

    warm_up_prng()


def pytest_addoption(parser):
    parser.addoption('--test_downloads', action='store_true', default=False)
    parser.addoption(
        '--playwright',
        action='store_true',
        default=False,
        help='run Playwright-based tests',
    )


# --- core-only test selection --------------------------------------------------
# Auto-apply ``needs_rendering`` (by location / fixture / name) so `-m "not
# needs_rendering"` selects the subset that runs without any rendering module -- the
# offline use case served by the rendering-free ``cvista`` core wheel.

# Fixtures that build a real Plotter; requesting one marks the test needs_rendering.
_RENDERING_FIXTURES = frozenset({'texture', 'image'})

# Otherwise-core tests exercising formats implemented in rendering-only modules
# (Text3D, VRML/3DS/Facet readers, texture reads). Scattered inside core modules,
# so matched by test-name substring.
_RENDERING_NAME_KEYWORDS = (
    'text3d',
    'text_3d',
    'vrml_reader',
    'threeds_reader',
    'facetreader',
    'read_texture',
    'jpeg_reader',
)


def _name_needs_rendering(item) -> bool:
    name = (getattr(item, 'originalname', '') or item.name).lower()
    return any(kw in name for kw in _RENDERING_NAME_KEYWORDS)


# Non-``tests/plotting`` modules (relative to ``tests/``) whose tests require
# rendering because they construct a Plotter at runtime.
_RENDERING_MODULES = frozenset(
    {
        'test_attributes.py',
        'test_cli.py',
        'examples/test_gltf.py',
        'typing/test_return_type.py',
        # These also evaluate plotting symbols at module scope, so on a
        # rendering-free backend they are skipped at collection time (see
        # ``_RENDERING_ONLY_MODULES`` / ``pytest_ignore_collect``).
        'core/test_dataobject_filters.py',
        'core/test_dataset_filters.py',
        'core/test_helpers.py',
        'core/test_polydata.py',
        'core/test_utilities.py',
    }
)

# Subset of ``_RENDERING_MODULES`` importing rendering at module scope, so on a
# rendering-free backend they must be skipped at collection time (a marker is too late).
_RENDERING_ONLY_MODULES = frozenset(
    {
        'test_attributes.py',
        'core/test_dataobject_filters.py',
        'core/test_dataset_filters.py',
        'core/test_helpers.py',
        'core/test_polydata.py',
        'core/test_utilities.py',
    }
)


# Tests exercising an IO-tier format a core-only build omits, matched by name
# substring and gated on the matching ``HAS_IO_*`` probe (a no-op on a full build).
_IO_EXTRA_NAME_KEYWORDS = (
    ('hdf', HAS_IO_HDF),  # HDFReader, .vtkhdf save, download_can_crushed_hdf
    ('ensight', HAS_IO_ENSIGHT),  # EnSightReader (.case)
    ('pdbreader', HAS_IO_CHEMISTRY),  # PDBReader
    ('gaussian_cubes_reader', HAS_IO_CHEMISTRY),  # GaussianCubeReader (.cube)
)


def _name_needs_io_extra(item) -> bool:
    name = (getattr(item, 'originalname', '') or item.name).lower()
    return any(kw in name for kw, available in _IO_EXTRA_NAME_KEYWORDS if not available)


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    """Skip collecting modules that import rendering at module scope.

    Only relevant when the active VTK backend ships no rendering modules
    (``HAS_PLOTTING is False``); otherwise these modules collect and run
    normally (and carry the ``needs_rendering`` marker).
    """
    if HAS_PLOTTING:
        return None

    tests_root = Path(__file__).parent
    try:
        rel = Path(str(collection_path)).relative_to(tests_root).as_posix()
    except ValueError:
        return None
    return True if rel in _RENDERING_ONLY_MODULES else None


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-apply the ``needs_rendering`` / ``needs_io_extra`` markers.

    See ``_RENDERING_*`` and ``_IO_EXTRA_*`` above. The ``needs_io_extra``
    marking only fires when the relevant ``HAS_IO_*`` probe reports the module
    absent, so it is a no-op on a full VTK build.
    """
    tests_root = Path(__file__).parent
    plotting_dir = tests_root / 'plotting'
    mark = pytest.mark.needs_rendering
    io_extra_mark = pytest.mark.needs_io_extra
    for item in items:
        path = Path(str(getattr(item, 'fspath', item.nodeid)))
        rel = None
        try:
            rel = path.relative_to(tests_root).as_posix()
        except ValueError:
            rel = path.as_posix()

        needs = (
            plotting_dir in path.parents
            or rel in _RENDERING_MODULES
            or bool(_RENDERING_FIXTURES.intersection(getattr(item, 'fixturenames', ())))
            or _name_needs_rendering(item)
        )
        if needs:
            item.add_marker(mark)
        if _name_needs_io_extra(item):
            item.add_marker(io_extra_mark)


def _check_args_kwargs_marker(item_mark: pytest.Mark, sig: inspect.Signature):
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
    sig: inspect.Signature,
) -> tuple[tuple[int] | None, tuple[int] | None, inspect.BoundArguments]:
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
    if item_mark := item.get_closest_marker('skip_vtk_backend'):
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    b := 'backend',
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
                inspect.Parameter(
                    r := 'reason',
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default='Test diverges on this VTK backend',
                    annotation=str,
                ),
            ]
        )

        bounds = _check_args_kwargs_marker(item_mark=item_mark, sig=sig)
        backend = bounds.arguments[b]
        if pv.vtk_backend() == backend:
            pytest.skip(f'{backend} backend: {bounds.arguments[r]}')

    needs_vtk_version = 'needs_vtk_version'
    # this test needs a given VTK version
    for item_mark in item.iter_markers(needs_vtk_version):
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    'args',
                    kind=inspect.Parameter.VAR_POSITIONAL,
                    annotation=int | tuple[int],
                ),
                inspect.Parameter(
                    'at_least',
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=tuple[int] | None,
                    default=None,
                ),
                inspect.Parameter(
                    'less_than',
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=tuple[int] | None,
                ),
                inspect.Parameter(
                    'reason',
                    kind=inspect.Parameter.KEYWORD_ONLY,
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
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    r := 'reason',
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default='Test fails when using OSMesa/EGL VTK build',
                    annotation=str,
                )
            ]
        )

        bounds = _check_args_kwargs_marker(item_mark=item_mark, sig=sig)
        if uses_egl():
            pytest.skip(bounds.arguments[r])

    if item_mark := item.get_closest_marker('skip_windows'):
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    r := 'reason',
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default='Test fails on Windows',
                    annotation=str,
                )
            ]
        )

        bounds = _check_args_kwargs_marker(item_mark=item_mark, sig=sig)
        if os.name == 'nt':
            pytest.skip(bounds.arguments[r])

    if item_mark := item.get_closest_marker('skip_mac'):
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    r := 'reason',
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default='Test fails on MacOS',
                    annotation=str,
                ),
                inspect.Parameter(
                    p := 'processor',
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=str | None,
                ),
                inspect.Parameter(
                    m := 'machine',
                    kind=inspect.Parameter.KEYWORD_ONLY,
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

    playwright = item.config.getoption(flag := '--playwright')
    if item.get_closest_marker('needs_playwright') and not playwright:
        pytest.skip(f'Playwright test not enabled with {flag}')


def pytest_report_header(config):  # noqa: ARG001
    """Header for pytest to show versions of required and optional packages."""
    required = []
    extra = {}
    for item in requires('pyvista'):
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
            pkg_version = version(name)
            items.append(f'{name}-{pkg_version}')
        except PackageNotFoundError:
            items.append(f'{name} (not found)')
    lines.append('required packages: ' + ', '.join(items))

    not_found = []
    for pkg_extra in extra.keys():  # noqa: PLC0206
        installed = []
        for name in extra[pkg_extra]:
            try:
                pkg_version = version(name)
                installed.append(f'{name}-{pkg_version}')
            except PackageNotFoundError:
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
