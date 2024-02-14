from importlib import metadata
import re

import numpy as np
from numpy.random import default_rng
from pytest import fixture, mark, skip

import pyvista
from pyvista import examples

pyvista.OFF_SCREEN = True


@fixture()
def global_variables_reset():
    tmp_screenshots = pyvista.ON_SCREENSHOT
    tmp_figurepath = pyvista.FIGURE_PATH
    yield
    pyvista.ON_SCREENSHOT = tmp_screenshots
    pyvista.FIGURE_PATH = tmp_figurepath


@fixture(scope='session', autouse=True)
def set_mpl():
    """Avoid matplotlib windows popping up."""
    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.rcdefaults()
        matplotlib.use('agg', force=True)


@fixture()
def cube():
    return pyvista.Cube()


@fixture()
def airplane():
    return examples.load_airplane()


@fixture()
def rectilinear():
    return examples.load_rectilinear()


@fixture()
def sphere():
    return examples.load_sphere()


@fixture()
def uniform():
    return examples.load_uniform()


@fixture()
def ant():
    return examples.load_ant()


@fixture()
def globe():
    return examples.load_globe()


@fixture()
def hexbeam():
    return examples.load_hexbeam()


@fixture()
def tetbeam():
    return examples.load_tetbeam()


@fixture()
def struct_grid():
    x, y, z = np.meshgrid(
        np.arange(-10, 10, 2, dtype=np.float32),
        np.arange(-10, 10, 2, dtype=np.float32),
        np.arange(-10, 10, 2, dtype=np.float32),
    )
    return pyvista.StructuredGrid(x, y, z)


@fixture()
def plane():
    return pyvista.Plane(direction=(0, 0, -1))


@fixture()
def spline():
    return examples.load_spline()


@fixture()
def random_hills():
    return examples.load_random_hills()


@fixture()
def tri_cylinder():
    """Triangulated cylinder"""
    return pyvista.Cylinder().triangulate()


@fixture()
def datasets():
    return [
        examples.load_uniform(),  # ImageData
        examples.load_rectilinear(),  # RectilinearGrid
        examples.load_hexbeam(),  # UnstructuredGrid
        examples.load_airplane(),  # PolyData
        examples.load_structured(),  # StructuredGrid
    ]


@fixture()
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


@fixture()
def datasets_vtk9():
    return [
        examples.load_explicit_structured(),
    ]


@fixture()
def pointset():
    rng = default_rng(0)
    points = rng.random((10, 3))
    return pyvista.PointSet(points)


@fixture()
def multiblock_all(datasets):
    """Return datasets fixture combined in a pyvista multiblock."""
    return pyvista.MultiBlock(datasets)


@fixture()
def noise_2d():
    freq = [10, 5, 0]
    noise = pyvista.perlin_noise(1, freq, (0, 0, 0))
    return pyvista.sample_function(noise, bounds=(0, 10, 0, 10, 0, 10), dim=(2**4, 2**4, 1))


@fixture()
def texture():
    # create a basic texture by plotting a sphere and converting the image
    # buffer to a texture
    pl = pyvista.Plotter(window_size=(300, 200), lighting=None)
    mesh = pyvista.Sphere()
    pl.add_mesh(mesh, scalars=range(mesh.n_points), show_scalar_bar=False)
    pl.background_color = 'w'
    return pyvista.Texture(pl.screenshot())


@fixture()
def image(texture):
    return texture.to_image()


def pytest_addoption(parser):
    parser.addoption("--test_downloads", action='store_true', default=False)


def marker_names(item):
    return [marker.name for marker in item.iter_markers()]


def pytest_collection_modifyitems(config, items):
    test_downloads = config.getoption("--test_downloads")

    # skip all tests that need downloads
    if not test_downloads:
        skip_downloads = mark.skip("Downloads not enabled with --test_downloads")
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
        if len(args) == 1 and isinstance(args[0], tuple):
            version_needed = args[0]
        else:
            version_needed = args
        if pyvista.vtk_version_info < version_needed:
            version_str = '.'.join(map(str, version_needed))
            skip(f'Test needs VTK {version_str} or newer.')


def pytest_report_header(config):
    """Header for pytest to show versions of required and optional packages."""

    required = []
    extra = {}
    for item in metadata.requires("pyvista"):
        pkg_name = re.findall(r"[a-z0-9_\-]+", item, re.IGNORECASE)[0]
        if pkg_name == "pyvista":
            continue
        elif res := re.findall("extra == ['\"](.+)['\"]", item):
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
            items.append(f"{name}-{version}")
        except metadata.PackageNotFoundError:
            items.append(f"{name} (not found)")
    lines.append("required packages: " + ", ".join(items))

    not_found = []
    for pkg_extra in extra.keys():
        installed = []
        for name in extra[pkg_extra]:
            try:
                version = metadata.version(name)
                installed.append(f"{name}-{version}")
            except metadata.PackageNotFoundError:
                not_found.append(name)
        if installed:
            plrl = "s" if len(installed) != 1 else ""
            comma_lst = ", ".join(installed)
            lines.append(f"optional {pkg_extra!r} package{plrl}: {comma_lst}")
    if not_found:
        plrl = "s" if len(not_found) != 1 else ""
        comma_lst = ", ".join(not_found)
        lines.append(f"optional package{plrl} not found: {comma_lst}")
    return "\n".join(lines)
