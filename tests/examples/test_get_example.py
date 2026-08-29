from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import re
import warnings

import pytest

import pyvista as pv
from pyvista import examples
from pyvista.examples import downloads
from pyvista.examples import planets
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader
from pyvista.examples._get_example import _example_names
from pyvista.examples._get_example import _resolve_paths

_DEPRECATED_DATASETS = ['can', 'osmnx_graph']
_SKIP_DATASETS_WINDOWS = ['biplane']


def _all_example_names():
    """Return the name of every example defined across the examples modules."""
    return sorted(
        name
        for module in (examples.examples, downloads, planets)
        for name in _example_names(module)
    )


@pytest.mark.parametrize(
    ('name', 'dataset_type', 'num_paths'),
    [
        ('uniform', pv.ImageData, 1),
        ('ant', pv.PolyData, 1),
        ('structured', pv.StructuredGrid, 0),
    ],
)
def test_get_example(name, dataset_type, num_paths):
    """An example resolves to its files and loads its dataset."""
    example = examples.get_example(name)

    assert isinstance(example, examples.Example)
    assert example.name == name
    assert isinstance(example.paths, tuple)
    assert len(example.paths) == num_paths
    assert all(isinstance(path, str) and Path(path).exists() for path in example.paths)
    assert isinstance(example.load(), dataset_type)


def test_get_example_from_function():
    """An example may be named by its public function instead of by string."""
    assert examples.get_example('uniform') == examples.get_example(examples.load_uniform)


@pytest.mark.parametrize('name', ['uniform', 'load_uniform'])
def test_get_example_ignores_function_prefix(name):
    """A ``'download_'`` or ``'load_'`` prefix on the name is optional."""
    assert examples.get_example(name).name == 'uniform'


def test_get_example_load_matches_the_example_function():
    """``load`` returns what the example's own function returns."""
    assert examples.get_example('uniform').load() == examples.load_uniform()


def test_get_example_fields():
    """Every tuple field has one entry per path, in the same order."""
    example = examples.get_example('uniform')

    assert example.function is examples.load_uniform
    assert example.source_urls == (
        'https://github.com/pyvista/pyvista/raw/main/pyvista/examples/uniform.vtk',
    )
    assert len(example.file_sizes) == len(example.source_urls) == len(example.paths) == 1
    assert example.file_sizes[0] == Path(example.paths[0]).stat().st_size


def test_get_example_in_memory():
    """An example generated in memory has no files, and still loads."""
    example = examples.get_example('structured')

    for empty in (example.paths, example.file_sizes, example.source_urls, example.readers):
        assert empty == ()
    assert isinstance(example.load(), pv.StructuredGrid)


def test_get_example_readers():
    """A reader is returned for each file which has one."""
    (reader,) = examples.get_example('uniform').readers

    assert isinstance(reader, pv.VTKDataSetReader)
    # the reader takes the same `str` path the example reports
    assert reader.path == examples.get_example('uniform').paths[0]


@pytest.mark.needs_download
def test_get_example_readers_multiple():
    """An example read by several readers returns all of them."""
    readers = examples.get_example('electronics_cooling').readers

    assert [type(reader).__name__ for reader in readers] == [
        'XMLPolyDataReader',
        'XMLUnstructuredGridReader',
    ]


@pytest.mark.needs_download
def test_get_example_readers_skips_files_without_one():
    """Files with no reader are left out rather than returned as ``None``."""
    example = examples.get_example('frog')

    # two files, but only the header is read
    assert len(example.paths) == 2
    assert [reader.path for reader in example.readers] == [example.paths[0]]


@pytest.mark.needs_download
def test_get_example_readers_empty_for_custom_read():
    """An example read by a custom function has no reader, but still loads."""
    example = examples.get_example('sky_box_cube_map')

    assert len(example.paths) == 6
    assert example.readers == ()
    assert isinstance(example.load(), pv.Texture)


@pytest.mark.needs_download
def test_get_example_folder_is_one_path():
    """A folder is one path, sized by what it contains."""
    example = examples.get_example('cubemap_park')

    assert len(example.paths) == len(example.file_sizes) == 1
    assert Path(example.paths[0]).is_dir()
    assert example.file_sizes[0] > 0


@pytest.mark.needs_download
def test_get_example_file_sizes_compare():
    """Sizes are bytes, so examples compare without parsing anything."""
    frog = examples.get_example('frog')
    bunny = examples.get_example('bunny')

    assert sum(frog.file_sizes) > sum(bunny.file_sizes)


def test_get_example_download_false_uses_local_files():
    """Built-in examples are available with ``download=False``."""
    example = examples.get_example('uniform', download=False)

    assert Path(example.paths[0]).is_file()


def test_get_example_download_true_downloads(monkeypatch):
    """``download=True`` calls the loader's download; the warm cache hides this otherwise."""
    calls = []
    loader = downloads._dataset_bunny
    monkeypatch.setattr(type(loader), 'download', lambda self: calls.append(self) or (self._path,))

    examples.get_example('bunny')
    assert len(calls) == 1

    calls.clear()
    examples.get_example('bunny', download=False)
    assert calls == []


def test_get_example_download_false_rejects_unresolved_archive():
    """An archive member is a relative path until extracted, and must not pass as present."""
    # `Path('data').exists()` is true whenever the working directory holds a `data/`,
    # and `Path('').exists()` is always true -- neither means the example is available
    for target in ('data', ''):
        loader = _SingleFileDownloadableDatasetLoader('not_downloaded.zip', target_file=target)
        assert not Path(loader.path[0]).is_absolute()
        with pytest.raises(FileNotFoundError, match='not available locally'):
            _resolve_paths(loader, 'archived_example', download=False)


def test_get_example_from_function_rejects_a_foreign_function():
    """A function which does not own the example is refused rather than mispaired."""
    # `planets` has both, and only `download_saturn_rings` owns `_dataset_saturn_rings`
    match = "is not the function for example 'saturn_rings'; that is 'download_saturn_rings'"
    with pytest.raises(ValueError, match=match):
        examples.get_example(planets.load_saturn_rings)


def test_example_has_no_private_loader_field():
    """The loader is derived from ``function``, so it cannot be passed or left unset."""
    assert {f.name for f in fields(examples.Example)} == {
        'name',
        'function',
        'paths',
        'file_sizes',
        'source_urls',
    }


def test_get_example_download_false_raises(monkeypatch):
    """A missing file raises rather than downloading when ``download=False``."""
    loader = _SingleFileDownloadableDatasetLoader('missing_example.vtk')
    monkeypatch.setattr(downloads, '_dataset_missing_example', loader, raising=False)
    monkeypatch.setattr(downloads, 'download_missing_example', lambda: None, raising=False)

    match = 'not available locally'
    with pytest.raises(FileNotFoundError, match=match):
        examples.get_example('missing_example', download=False)


def test_get_example_unknown_name_raises():
    """An unknown name raises and suggests close matches."""
    match = re.escape("Example 'bunni' does not exist. Did you mean: 'bunny'")
    with pytest.raises(ValueError, match=match):
        examples.get_example('bunni')

    with pytest.raises(ValueError, match=re.escape("Example 'zzzz' does not exist.")):
        examples.get_example('zzzz')


def test_get_example_function_without_dataset_raises():
    """A function with no dataset loader raises."""
    match = "Function 'load_earth' does not have an example dataset."
    with pytest.raises(ValueError, match=match):
        examples.get_example(planets.load_earth)


@pytest.mark.needs_download
@pytest.mark.parametrize('name', _all_example_names())
def test_get_example_all(name):
    """Every example resolves by name and by function, reports its files, and loads."""
    if name in _DEPRECATED_DATASETS:
        pytest.skip('Dataset is deprecated.')
    if os.name == 'nt' and name in _SKIP_DATASETS_WINDOWS:
        pytest.skip('Error loading on Windows')

    with warnings.catch_warnings():
        # a few examples warn on their own account, and the nefertiti licence fires
        # from its loader, so it reaches every route taken here
        warnings.simplefilter('ignore')
        try:
            example = examples.get_example(name)
            loaded = example.load()
            from_function = example.function()
        except pv.VTKVersionError:
            pytest.skip('VTK version not supported.')
        # looking the example up by its own function finds the same example, which is
        # what makes the two forms interchangeable -- comparing the loaded datasets
        # would rest on `DataObject.__eq__`, which is not what this is testing
        assert examples.get_example(example.function) == example

    # `load` returns what the example's own function returns. Compare the types
    # rather than the datasets: `DataObject.__eq__` walks every property, and
    # `polyhedron_faces` raises on VTK before 9.4 for a grid holding no polyhedra.
    # The declared return annotation is not usable here either, since the type can
    # differ by VTK version while the annotation cannot.
    assert type(loaded) is type(from_function)
    # every tuple field is one entry per path, including the filtered one
    assert len(example.file_sizes) == len(example.paths)
    assert len(example.source_urls) == len(example.paths)
    assert all(Path(path).is_absolute() for path in example.paths)
    assert all(Path(path).is_file() or Path(path).is_dir() for path in example.paths)
    # readers are a subset of the example's own files, never invented
    assert len(example.readers) <= len(example.paths)
    assert {reader.path for reader in example.readers} <= set(example.paths)
