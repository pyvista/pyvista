from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.examples import downloads
from pyvista.examples import planets
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader
from pyvista.examples._get_example import _example_names

# Datasets which return a different mesh each time they are loaded,
# see https://github.com/pyvista/pyvista/issues/7634
_NON_DETERMINISTIC_DATASETS = [
    '3gqp',
    'biplane',
    'caffeine',
    'damavand_volcano',
    'embryo',
    'frog_tissue',
    'notch_stress',
    'particles',
]
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
    """Every example resolves, reports its files, and loads the same by name or function."""
    if name in _DEPRECATED_DATASETS:
        pytest.skip('Dataset is deprecated.')
    if os.name == 'nt' and name in _SKIP_DATASETS_WINDOWS:
        pytest.skip('Error loading on Windows')

    try:
        example = examples.get_example(name)
        from_name = example.load()
        from_function = examples.get_example(example.function).load()
    except pv.VTKVersionError:
        pytest.skip('VTK version not supported.')

    assert len(example.file_sizes) == len(example.paths)
    assert all(Path(path).is_file() or Path(path).is_dir() for path in example.paths)
    assert all(reader is not None for reader in example.readers)

    if name in _NON_DETERMINISTIC_DATASETS:
        return
    if isinstance(from_name, np.ndarray):
        assert np.array_equal(from_name, from_function, equal_nan=True)
    else:
        assert from_name == from_function
