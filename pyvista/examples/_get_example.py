"""Single entry point for accessing any example dataset."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import functools
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Literal
from typing import cast

from typing_extensions import TypeVar

# `typing.overload` registers with `get_overloads` only from 3.11
from typing_extensions import overload

import pyvista as pv
from pyvista.examples._dataset_loader import _DOWNLOADABLE_TYPES
from pyvista.examples._dataset_loader import _DatasetLoader
from pyvista.examples._dataset_loader import _FileProps

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from pyvista.examples._dataset_loader import DatasetObject

_DatasetT_co = TypeVar('_DatasetT_co', covariant=True, default='DatasetObject')
_ReadersT_co = TypeVar(
    '_ReadersT_co',
    bound='tuple[pv.BaseReader[Any], ...]',
    covariant=True,
    default='tuple[pv.BaseReader[Any], ...]',
)
_DatasetT = TypeVar('_DatasetT')


@dataclass(frozen=True)
class Example(Generic[_DatasetT_co, _ReadersT_co]):
    """A single example dataset: its files, where they come from, and how to read it.

    .. versionadded:: 0.49

    Call :func:`~pyvista.examples.get_example` to get an example; this class is not
    meant to be constructed directly. Every sequence-valued field is a tuple with one
    entry per path, in the same order, including for single-file examples. The class is generic over the dataset :meth:`load` returns and the
    tuple :attr:`readers` returns, which :func:`~pyvista.examples.get_example`
    resolves statically for every example name.

    Examples
    --------
    Look up an example. This resolves its files but does not read them.

    >>> from pyvista import examples
    >>> frog = examples.get_example('frog')
    >>> frog.name
    'frog'

    It is stored as two files, but only one of them is read.

    >>> len(frog.paths)
    2
    >>> len(frog.readers)
    1

    Sizes are in bytes, one per path, so examples compare directly.

    >>> bunny = examples.get_example('bunny')
    >>> sum(frog.file_sizes) > sum(bunny.file_sizes)
    True

    Read the dataset itself.

    >>> mesh = frog.load()
    >>> mesh.n_cells
    31594185

    """

    name: str
    """Name of the example, such as ``'frog'``."""

    function: Callable[..., _DatasetT_co]
    """Public function which returns this example's dataset, such as ``examples.download_frog``."""

    paths: tuple[str, ...]
    """Local path of every file or folder belonging to the example, in declaration order."""

    file_sizes: tuple[int, ...]
    """Size in bytes of each entry in ``paths``, one per path, folders counted in full."""

    source_urls: tuple[str, ...]
    """URL of each file which is downloaded, empty for an example which ships with PyVista."""

    @functools.cached_property
    def _loader(self) -> _DatasetLoader:
        """Return the loader backing this example, resolved from :attr:`function` once."""
        loader, _, _ = _get_dataset_loader(self.function)
        return loader

    @functools.cached_property
    def readers(self) -> _ReadersT_co:
        """Return a reader for each file which has one.

        The readers report which reader PyVista resolves for each file. They are not
        the objects :meth:`load` reads through, so configuring one does not change what
        :meth:`load` returns. They are resolved on first access and reused.

        Empty for examples read with a custom function or generated in memory, and
        shorter than :attr:`paths` when only some files are read directly.

        Returns
        -------
        tuple[pyvista.BaseReader, ...]
            One reader per file which has one.

        """
        loader = self._loader
        if not isinstance(loader, _FileProps):
            return cast('_ReadersT_co', ())
        return cast('_ReadersT_co', tuple(r for r in loader._readers if r is not None))

    def load(self) -> _DatasetT_co:
        """Read the example and return its dataset.

        Returns
        -------
        pyvista.DataObject | numpy.ndarray
            The dataset the example's own :attr:`function` returns, read from
            :attr:`paths`.

        """
        return cast('_DatasetT_co', self._loader.load())


def _supported_modules() -> tuple[ModuleType, ...]:
    """Return the modules which define example dataset loaders."""
    return (pv.examples.examples, pv.examples.downloads, pv.examples.planets)


def _example_loader(module: ModuleType, name: str) -> _DatasetLoader | None:
    """Return the loader ``module`` defines for an example, if it defines one."""
    loader = getattr(module, '_dataset_' + name, None)
    return loader if isinstance(loader, _DatasetLoader) else None


def _example_names(module: ModuleType) -> list[str]:
    """Return the name of every example defined by ``module``."""
    names = (
        attr.removeprefix('_dataset_') for attr in vars(module) if attr.startswith('_dataset_')
    )
    return [name for name in names if _example_loader(module, name) is not None]


def _public_function(module: ModuleType, name: str) -> Callable[..., Any]:
    """Return the public ``download_``/``load_`` function for an example name."""
    for prefix in ('download_', 'load_'):
        func = getattr(module, prefix + name, None)
        if func is not None:
            return func
    msg = f'Example {name!r} has no public function in {module.__name__!r}.'
    raise ValueError(msg)


def _get_dataset_loader(
    name: str | Callable[..., Any],
) -> tuple[_DatasetLoader, str, Callable[..., Any]]:
    """Return the loader, name, and public function for an example."""
    if callable(name):
        dataset_name = name.__name__.removeprefix('download_').removeprefix('load_')
        module = sys.modules[name.__module__]
        loader = _example_loader(module, dataset_name)
        if loader is None:
            msg = f'Function {name.__name__!r} does not have an example dataset.'
            raise ValueError(msg)
        # Several names can share a stem, and only one of them owns the dataset:
        # `planets` has both `download_saturn_rings` and a deprecated `load_saturn_rings`
        canonical = _public_function(module, dataset_name)
        if canonical is not name:
            msg = (
                f'Function {name.__name__!r} is not the function for example '
                f'{dataset_name!r}; that is {canonical.__name__!r}.'
            )
            raise ValueError(msg)
        return loader, dataset_name, name

    dataset_name = name.removeprefix('download_').removeprefix('load_')
    for module in _supported_modules():
        loader = _example_loader(module, dataset_name)
        if loader is not None:
            return loader, dataset_name, _public_function(module, dataset_name)

    available = sorted(
        example for module in _supported_modules() for example in _example_names(module)
    )
    msg = f'Example {dataset_name!r} does not exist.'
    if close := difflib.get_close_matches(dataset_name, available, n=3):
        msg += f' Did you mean: {", ".join(map(repr, close))}?'
    raise ValueError(msg)


def _resolve_paths(loader: _DatasetLoader, name: str, *, download: bool) -> tuple[str, ...]:
    """Return the example's file paths, downloading them first if allowed."""
    downloaded = False
    if download and isinstance(loader, _DOWNLOADABLE_TYPES):
        loader.download()
        downloaded = True
    if not isinstance(loader, _FileProps):
        return ()

    # A relative path is an archive member `download()` has not extracted yet; `Path`
    # would resolve it against the working directory and make it look present.
    paths = tuple(loader.paths)
    if missing := [p for p in paths if not (Path(p).is_absolute() and Path(p).exists())]:
        missing_str = '\n\t'.join(missing)
        if not download:
            reason = 'and download=False'
        elif downloaded:
            reason = 'even after downloading'
        else:
            reason = 'and cannot be downloaded'
        msg = (
            f'Example {name!r} is not available locally {reason}.\n'
            f'Missing:\n\t{missing_str}\n'
            f'Call get_example({name!r}) to download it.'
        )
        raise FileNotFoundError(msg)
    return paths


# `ExampleName` names every example, so an editor can complete the string, and one
# overload per example gives its dataset and reader types statically from the name.
# Both are generated from the example functions and their readers by
# ``tests/examples/test_get_example.py``; regenerate them with
#   pytest tests/examples/test_get_example.py -k overloads_current \
#       --test_downloads --regenerate_overloads
# and run pre-commit afterwards. Each stub is one line: the block is not formatted,
# and `E501` is ignored for this file in pyproject.toml.
# fmt: off
# --- generated overloads ---
ExampleName = Literal[
    '3gqp', 'action_figure', 'aero_bracket', 'airplane', 'angular_sector', 'ant',
    'antarctica_velocity', 'armadillo', 'avocado', 'backward_facing_step', 'beach',
    'biplane', 'bird', 'bird_bath', 'bird_texture', 'black_vase', 'blood_vessels', 'blow',
    'bolt_nut', 'brain', 'brain_atlas_with_sides', 'bunny', 'bunny_coarse', 'cad_model',
    'cad_model_case', 'caffeine', 'cake_easy', 'cake_easy_texture', 'can_crushed_hdf',
    'can_crushed_vtu', 'carburetor', 'carotid', 'cavity', 'cells_nd', 'cgns_multi',
    'cgns_structured', 'channels', 'chest', 'cloud_dark_matter',
    'cloud_dark_matter_dense', 'clown', 'coastlines', 'coil_magnetic_field', 'cow',
    'cow_head', 'crater_imagery', 'crater_topo', 'cubemap_park', 'cubemap_space_16k',
    'cubemap_space_4k', 'cylinder_crossflow', 'damaged_helmet', 'damavand_volcano',
    'delaunay_example', 'dicom_stack', 'dikhololo_night', 'disc_quads', 'dolfin',
    'doorman', 'dragon', 'drill', 'dual_sphere_animation', 'e07733s002i009',
    'electronics_cooling', 'embryo', 'emoji', 'emoji_texture', 'exodus',
    'explicit_structured', 'face', 'face2', 'faults', 'fea_bracket',
    'fea_hertzian_contact_cylinder', 'filled_contours', 'flamingo', 'foot_bones', 'frd',
    'frog', 'frog_tissues', 'full_head', 'gearbox', 'gears', 'gif_simple', 'globe',
    'globe_texture', 'gourds', 'gourds_pnm', 'gourds_texture', 'gpr_data_array',
    'gpr_path', 'grasshopper', 'great_white_shark', 'grey_nurse_shark', 'guitar', 'head',
    'head_2', 'headsq', 'hexbeam', 'honolulu', 'horse', 'horse_points', 'human',
    'hydrogen_orbital', 'iron_protein', 'ivan_angel', 'jupiter_surface', 'kitchen',
    'knee', 'knee_full', 'letter_a', 'letter_k', 'lidar', 'lincoln_life_mask', 'lobster',
    'logo', 'louis_louvre', 'lshape', 'lucy', 'm4_total_density', 'mars_surface',
    'masonry_texture', 'mercury_surface', 'meshio_xdmf', 'milk_truck',
    'milkyway_sky_background', 'model_with_variance', 'moon_surface', 'moonlanding_image',
    'motor', 'mount_damavand', 'mug', 'naca', 'nefertiti', 'nek5000', 'neptune_surface',
    'notch_displacement', 'notch_stress', 'nut', 'oblique_cone', 'office',
    'openfoam_tubes', 'owl', 'parallel_exodus', 'parched_canal_4k', 'particles',
    'particles_lethe', 'pepper', 'pine_roots', 'planet', 'planet_rings', 'plastic_vase',
    'pluto_surface', 'poly_line', 'prism', 'prostar', 'prostate', 'pump_bracket', 'puppy',
    'puppy_texture', 'quadratic_pyramid', 'random_hills', 'rectilinear',
    'rectilinear_grid', 'reservoir', 'rgba_texture', 'room_cff', 'room_surface_mesh',
    'saddle_surface', 'saturn_rings', 'saturn_surface', 'sea_vase', 'sextant', 'shark',
    'single_sphere_animation', 'sky_box_cube_map', 'sky_box_nz', 'sky_box_nz_texture',
    'sparse_points', 'sphere', 'sphere_vectors', 'spider', 'spline', 'st_helens',
    'stars_cloud_hyg', 'stars_sky_background', 'structured', 'structured_grid',
    'structured_grid_two', 'sun_surface', 't3_grid_0', 'teapot', 'teapot_vrml',
    'tecplot_ascii', 'tensors', 'tetbeam', 'tetra_dc_mesh', 'tetrahedron',
    'thermal_probes', 'topo_global', 'topo_land', 'torso', 'tri_quadratic_hexahedron',
    'trumpet', 'turbine_blade', 'uniform', 'unstructured_grid', 'uranus_surface', 'urn',
    'usa', 'usa_texture', 'venus_surface', 'victorian_goblet_face_illusion', 'vtk',
    'vtk_logo', 'warping_spheres', 'washington_bust', 'wavy', 'whole_body_ct_female',
    'whole_body_ct_male', 'woman', 'yinyang',
]
@overload
def get_example(name: Literal['3gqp'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PDBReader]]: ...
@overload
def get_example(name: Literal['action_figure'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.OBJReader]]: ...
@overload
def get_example(name: Literal['aero_bracket'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['airplane'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['angular_sector'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['ant'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['antarctica_velocity'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['armadillo'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['avocado'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.GLTFReader]]: ...
@overload
def get_example(name: Literal['backward_facing_step'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.EnSightReader]]: ...
@overload
def get_example(name: Literal['beach'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.NRRDReader]]: ...
@overload
def get_example(name: Literal['biplane'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.ExodusIIReader]]: ...
@overload
def get_example(name: Literal['bird'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['bird_bath'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['bird_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['black_vase'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['blood_vessels'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLPUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['blow'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['bolt_nut'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.SLCReader, pv.SLCReader]]: ...
@overload
def get_example(name: Literal['brain'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['brain_atlas_with_sides'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.NIFTIReader]]: ...
@overload
def get_example(name: Literal['bunny'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['bunny_coarse'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['cad_model'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['cad_model_case'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['caffeine'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PDBReader]]: ...
@overload
def get_example(name: Literal['cake_easy'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['cake_easy_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['can_crushed_hdf'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.HDFReader]]: ...
@overload
def get_example(name: Literal['can_crushed_vtu'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['carburetor'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['carotid'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['cavity'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.POpenFOAMReader]]: ...
@overload
def get_example(name: Literal['cells_nd'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.AVSucdReader]]: ...
@overload
def get_example(name: Literal['cgns_multi'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.CGNSReader]]: ...
@overload
def get_example(name: Literal['cgns_structured'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.CGNSReader]]: ...
@overload
def get_example(name: Literal['channels'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.XMLImageDataReader]]: ...
@overload
def get_example(name: Literal['chest'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.MetaImageReader]]: ...
@overload
def get_example(name: Literal['cloud_dark_matter'], *, download: bool = ...) -> Example[pv.PointSet, tuple[()]]: ...
@overload
def get_example(name: Literal['cloud_dark_matter_dense'], *, download: bool = ...) -> Example[pv.PointSet, tuple[()]]: ...
@overload
def get_example(name: Literal['clown'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.FacetReader]]: ...
@overload
def get_example(name: Literal['coastlines'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['coil_magnetic_field'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.XMLImageDataReader]]: ...
@overload
def get_example(name: Literal['cow'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['cow_head'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['crater_imagery'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.TIFFReader]]: ...
@overload
def get_example(name: Literal['crater_topo'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['cubemap_park'], *, download: bool = ...) -> Example[pv.Texture, tuple[()]]: ...
@overload
def get_example(name: Literal['cubemap_space_16k'], *, download: bool = ...) -> Example[pv.Texture, tuple[()]]: ...
@overload
def get_example(name: Literal['cubemap_space_4k'], *, download: bool = ...) -> Example[pv.Texture, tuple[()]]: ...
@overload
def get_example(name: Literal['cylinder_crossflow'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.EnSightReader]]: ...
@overload
def get_example(name: Literal['damaged_helmet'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.GLTFReader]]: ...
@overload
def get_example(name: Literal['damavand_volcano'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['delaunay_example'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['dicom_stack'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.DICOMReader]]: ...
@overload
def get_example(name: Literal['dikhololo_night'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.HDRReader]]: ...
@overload
def get_example(name: Literal['disc_quads'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['dolfin'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[()]]: ...
@overload
def get_example(name: Literal['doorman'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.OBJReader]]: ...
@overload
def get_example(name: Literal['dragon'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['drill'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.OBJReader]]: ...
@overload
def get_example(name: Literal['dual_sphere_animation'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.PVDReader]]: ...
@overload
def get_example(name: Literal['e07733s002i009'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.GESignaReader]]: ...
@overload
def get_example(name: Literal['electronics_cooling'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.XMLPolyDataReader, pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['embryo'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.SLCReader]]: ...
@overload
def get_example(name: Literal['emoji'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['emoji_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['exodus'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.ExodusIIReader]]: ...
@overload
def get_example(name: Literal['explicit_structured'], *, download: bool = ...) -> Example[pv.ExplicitStructuredGrid, tuple[()]]: ...
@overload
def get_example(name: Literal['face'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['face2'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['faults'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['fea_bracket'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['fea_hertzian_contact_cylinder'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLPUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['filled_contours'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['flamingo'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.ThreeDSReader]]: ...
@overload
def get_example(name: Literal['foot_bones'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['frd'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[()]]: ...
@overload
def get_example(name: Literal['frog'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.MetaImageReader]]: ...
@overload
def get_example(name: Literal['frog_tissues'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.XMLImageDataReader]]: ...
@overload
def get_example(name: Literal['full_head'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.MetaImageReader]]: ...
@overload
def get_example(name: Literal['gearbox'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.GLTFReader]]: ...
@overload
def get_example(name: Literal['gears'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['gif_simple'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.GIFReader]]: ...
@overload
def get_example(name: Literal['globe'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['globe_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['gourds'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.PNGReader]]: ...
@overload
def get_example(name: Literal['gourds_pnm'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.PNMReader]]: ...
@overload
def get_example(name: Literal['gourds_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.PNGReader]]: ...
@overload
def get_example(name: Literal['gpr_data_array'], *, download: bool = ...) -> Example[pv.NumpyArray[Any], tuple[()]]: ...
@overload
def get_example(name: Literal['gpr_path'], *, download: bool = ...) -> Example[pv.PolyData, tuple[()]]: ...
@overload
def get_example(name: Literal['grasshopper'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.VRMLReader]]: ...
@overload
def get_example(name: Literal['great_white_shark'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['grey_nurse_shark'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['guitar'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['head'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.MetaImageReader]]: ...
@overload
def get_example(name: Literal['head_2'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.XMLImageDataReader]]: ...
@overload
def get_example(name: Literal['headsq'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.NRRDReader]]: ...
@overload
def get_example(name: Literal['hexbeam'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['honolulu'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['horse'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['horse_points'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['human'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['hydrogen_orbital'], *, download: bool = ...) -> Example[pv.ImageData, tuple[()]]: ...
@overload
def get_example(name: Literal['iron_protein'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['ivan_angel'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['jupiter_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['kitchen'], *, download: bool = ...) -> Example[pv.StructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['knee'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.DICOMReader]]: ...
@overload
def get_example(name: Literal['knee_full'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.SLCReader]]: ...
@overload
def get_example(name: Literal['letter_a'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['letter_k'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['lidar'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['lincoln_life_mask'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.OBJReader]]: ...
@overload
def get_example(name: Literal['lobster'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['logo'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.PNGReader]]: ...
@overload
def get_example(name: Literal['louis_louvre'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['lshape'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.EnSightReader]]: ...
@overload
def get_example(name: Literal['lucy'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['m4_total_density'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.GaussianCubeReader]]: ...
@overload
def get_example(name: Literal['mars_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['masonry_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.BMPReader]]: ...
@overload
def get_example(name: Literal['mercury_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['meshio_xdmf'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.XdmfReader]]: ...
@overload
def get_example(name: Literal['milk_truck'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.GLTFReader]]: ...
@overload
def get_example(name: Literal['milkyway_sky_background'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['model_with_variance'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['moon_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['moonlanding_image'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.PNGReader]]: ...
@overload
def get_example(name: Literal['motor'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.BYUReader]]: ...
@overload
def get_example(name: Literal['mount_damavand'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['mug'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.ExodusIIReader]]: ...
@overload
def get_example(name: Literal['naca'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.EnSightReader]]: ...
@overload
def get_example(name: Literal['nefertiti'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['nek5000'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.Nek5000Reader]]: ...
@overload
def get_example(name: Literal['neptune_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['notch_displacement'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['notch_stress'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['nut'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['oblique_cone'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['office'], *, download: bool = ...) -> Example[pv.StructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['openfoam_tubes'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.POpenFOAMReader]]: ...
@overload
def get_example(name: Literal['owl'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['parallel_exodus'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.PExodusIIReader]]: ...
@overload
def get_example(name: Literal['parched_canal_4k'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.HDRReader]]: ...
@overload
def get_example(name: Literal['particles'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.ParticleReader]]: ...
@overload
def get_example(name: Literal['particles_lethe'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['pepper'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['pine_roots'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.BinaryMarchingCubesReader]]: ...
@overload
def get_example(name: Literal['planet'], *, download: bool = ...) -> Example[pv.PolyData, tuple[()]]: ...
@overload
def get_example(name: Literal['planet_rings'], *, download: bool = ...) -> Example[pv.PolyData, tuple[()]]: ...
@overload
def get_example(name: Literal['plastic_vase'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['pluto_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['poly_line'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['prism'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.GambitReader]]: ...
@overload
def get_example(name: Literal['prostar'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.ProStarReader]]: ...
@overload
def get_example(name: Literal['prostate'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.DICOMReader]]: ...
@overload
def get_example(name: Literal['pump_bracket'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['puppy'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['puppy_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['quadratic_pyramid'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['random_hills'], *, download: bool = ...) -> Example[pv.PolyData, tuple[()]]: ...
@overload
def get_example(name: Literal['rectilinear'], *, download: bool = ...) -> Example[pv.RectilinearGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['rectilinear_grid'], *, download: bool = ...) -> Example[pv.RectilinearGrid, tuple[pv.XMLRectilinearGridReader]]: ...
@overload
def get_example(name: Literal['reservoir'], *, download: bool = ...) -> Example[pv.ExplicitStructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['rgba_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.PNGReader]]: ...
@overload
def get_example(name: Literal['room_cff'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.FLUENTCFFReader]]: ...
@overload
def get_example(name: Literal['room_surface_mesh'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.OBJReader]]: ...
@overload
def get_example(name: Literal['saddle_surface'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['saturn_rings'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.PNGReader]]: ...
@overload
def get_example(name: Literal['saturn_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['sea_vase'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['sextant'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.VRMLReader]]: ...
@overload
def get_example(name: Literal['shark'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['single_sphere_animation'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.PVDReader]]: ...
@overload
def get_example(name: Literal['sky_box_cube_map'], *, download: bool = ...) -> Example[pv.Texture, tuple[()]]: ...
@overload
def get_example(name: Literal['sky_box_nz'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['sky_box_nz_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['sparse_points'], *, download: bool = ...) -> Example[pv.PolyData, tuple[()]]: ...
@overload
def get_example(name: Literal['sphere'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['sphere_vectors'], *, download: bool = ...) -> Example[pv.PolyData, tuple[()]]: ...
@overload
def get_example(name: Literal['spider'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['spline'], *, download: bool = ...) -> Example[pv.PolyData, tuple[()]]: ...
@overload
def get_example(name: Literal['st_helens'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.DEMReader]]: ...
@overload
def get_example(name: Literal['stars_cloud_hyg'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['stars_sky_background'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['structured'], *, download: bool = ...) -> Example[pv.StructuredGrid, tuple[()]]: ...
@overload
def get_example(name: Literal['structured_grid'], *, download: bool = ...) -> Example[pv.StructuredGrid, tuple[pv.XMLStructuredGridReader]]: ...
@overload
def get_example(name: Literal['structured_grid_two'], *, download: bool = ...) -> Example[pv.StructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['sun_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['t3_grid_0'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.MINCImageReader]]: ...
@overload
def get_example(name: Literal['teapot'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.BYUReader]]: ...
@overload
def get_example(name: Literal['teapot_vrml'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.VRMLReader]]: ...
@overload
def get_example(name: Literal['tecplot_ascii'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.TecplotReader]]: ...
@overload
def get_example(name: Literal['tensors'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['tetbeam'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[()]]: ...
@overload
def get_example(name: Literal['tetra_dc_mesh'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.XMLUnstructuredGridReader, pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['tetrahedron'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['thermal_probes'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['topo_global'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['topo_land'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['torso'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['tri_quadratic_hexahedron'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.XMLUnstructuredGridReader]]: ...
@overload
def get_example(name: Literal['trumpet'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.OBJReader]]: ...
@overload
def get_example(name: Literal['turbine_blade'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...
@overload
def get_example(name: Literal['uniform'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['unstructured_grid'], *, download: bool = ...) -> Example[pv.UnstructuredGrid, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['uranus_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['urn'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['usa'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.VTKDataSetReader]]: ...
@overload
def get_example(name: Literal['usa_texture'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['venus_surface'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.JPEGReader]]: ...
@overload
def get_example(name: Literal['victorian_goblet_face_illusion'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['vtk'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...
@overload
def get_example(name: Literal['vtk_logo'], *, download: bool = ...) -> Example[pv.Texture, tuple[pv.PNGReader]]: ...
@overload
def get_example(name: Literal['warping_spheres'], *, download: bool = ...) -> Example[pv.PartitionedDataSet, tuple[pv.HDFReader]]: ...
@overload
def get_example(name: Literal['washington_bust'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.OBJReader]]: ...
@overload
def get_example(name: Literal['wavy'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.PVDReader]]: ...
@overload
def get_example(name: Literal['whole_body_ct_female'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.XMLMultiBlockDataReader]]: ...
@overload
def get_example(name: Literal['whole_body_ct_male'], *, download: bool = ...) -> Example[pv.MultiBlock, tuple[pv.XMLMultiBlockDataReader]]: ...
@overload
def get_example(name: Literal['woman'], *, download: bool = ...) -> Example[pv.PolyData, tuple[pv.STLReader]]: ...
@overload
def get_example(name: Literal['yinyang'], *, download: bool = ...) -> Example[pv.ImageData, tuple[pv.PNGReader]]: ...
# --- end generated overloads ---
# fmt: on
@overload
def get_example(
    name: Callable[..., _DatasetT], *, download: bool = ...
) -> Example[_DatasetT, tuple[pv.BaseReader[Any], ...]]: ...
@overload
def get_example(name: str, *, download: bool = ...) -> Example: ...
def get_example(
    name: ExampleName | str | Callable[..., Any], *, download: bool = True
) -> Example[Any, Any]:
    """Look up any example dataset.

    .. versionadded:: 0.49

    This is a single entry point for every example in
    :mod:`pyvista.examples.examples`, :mod:`pyvista.examples.downloads`, and
    :mod:`pyvista.examples.planets`. It returns the example itself -- its files,
    where they come from, and the readers for them -- rather than the dataset, which
    :meth:`Example.load` reads. Reach for it to work with an example by name, or to
    get at its files or readers. Type checkers resolve the dataset and reader types
    from the name, so ``get_example('cow').load()`` is a :class:`~pyvista.PolyData`
    statically as well as at runtime.

    Parameters
    ----------
    name : str | Callable
        Name of the example, such as ``'bunny'``, or the function which returns it,
        such as ``examples.download_bunny``. A ``'download_'`` or ``'load_'``
        prefix on the name is optional.

    download : bool, default: True
        Download the example's files if they are not already present. If ``False``,
        a ``FileNotFoundError`` is raised for any example whose files are missing.
        Files which are already cached, and examples generated in memory, are
        unaffected.

    Returns
    -------
    Example
        The example, its files, and its readers.

    See Also
    --------
    :ref:`dataset_gallery`
        Browse every available example.

    Examples
    --------
    Look up an example and read it.

    >>> from pyvista import examples
    >>> uniform = examples.get_example('uniform')
    >>> mesh = uniform.load()
    >>> mesh.n_cells
    729

    Get the reader PyVista resolves for each file that has one. Most examples have
    exactly one reader.

    >>> [type(reader).__name__ for reader in uniform.readers]
    ['VTKDataSetReader']

    """
    loader, dataset_name, function = _get_dataset_loader(name)
    return Example(
        name=dataset_name,
        function=function,
        paths=_resolve_paths(loader, dataset_name, download=download),
        file_sizes=loader._file_sizes if isinstance(loader, _FileProps) else (),
        source_urls=loader.source_urls if isinstance(loader, _DOWNLOADABLE_TYPES) else (),
    )
