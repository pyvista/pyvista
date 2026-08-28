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
    """Local examples resolve to a dataset and to their files."""
    assert isinstance(examples.get_example(name), dataset_type)

    paths = examples.get_example(name, output='paths')
    assert isinstance(paths, tuple)
    assert len(paths) == num_paths
    assert all(isinstance(path, str) and Path(path).exists() for path in paths)


def test_get_example_from_function():
    """An example may be named by its public function instead of by string."""
    from_name = examples.get_example('uniform')
    from_function = examples.get_example(examples.load_uniform)
    assert from_name == from_function


@pytest.mark.parametrize('name', ['uniform', 'load_uniform'])
def test_get_example_ignores_function_prefix(name):
    """A ``'download_'`` or ``'load_'`` prefix on the name is optional."""
    assert isinstance(examples.get_example(name), pv.ImageData)


def test_get_example_readers():
    """A reader is returned for each file which has one."""
    (reader,) = examples.get_example('uniform', output='readers')
    assert isinstance(reader, pv.VTKDataSetReader)
    assert Path(reader.path).name == 'uniform.vtk'
    # the reader takes the same `str` path the example reports
    assert reader.path == examples.get_example('uniform', output='paths')[0]


@pytest.mark.needs_download
def test_get_example_readers_multiple():
    """An example read by several readers returns all of them."""
    readers = examples.get_example('electronics_cooling', output='readers')
    assert [type(reader).__name__ for reader in readers] == [
        'XMLPolyDataReader',
        'XMLUnstructuredGridReader',
    ]


@pytest.mark.needs_download
def test_get_example_readers_skips_files_without_one():
    """Files with no reader are left out rather than returned as ``None``."""
    # `frog` is two files, but only the header is read
    assert len(examples.get_example('frog', output='paths')) == 2
    assert len(examples.get_example('frog', output='readers')) == 1


@pytest.mark.parametrize('name', ['structured', 'sky_box_cube_map'])
def test_get_example_readers_empty(name):
    """An example with no reader returns an empty tuple rather than raising."""
    assert examples.get_example(name, output='readers', download=False) == ()


def test_get_example_metadata():
    """Metadata reports the example's files and their source."""
    metadata = examples.get_example('uniform', output='metadata')

    assert isinstance(metadata, examples.ExampleMetadata)
    assert metadata.name == 'uniform'
    assert metadata.is_builtin
    assert metadata.function is examples.load_uniform
    assert metadata.num_files == 1
    assert metadata.extensions == ('.vtk',)
    assert metadata.reader_types == (pv.VTKDataSetReader,)
    assert metadata.total_size == sum(metadata.file_sizes) > 0
    assert metadata.source_urls == (
        'https://github.com/pyvista/pyvista/raw/main/pyvista/examples/uniform.vtk',
    )

    assert metadata.paths == metadata.loadable_paths
    assert len(metadata.paths) == len(metadata.file_sizes) == metadata.num_files


def test_get_example_metadata_in_memory():
    """An example generated in memory has no files and no source."""
    metadata = examples.get_example('structured', output='metadata')

    assert metadata.num_files == 0
    assert metadata.total_size == 0
    for empty in (
        metadata.paths,
        metadata.loadable_paths,
        metadata.extensions,
        metadata.file_sizes,
        metadata.reader_types,
        metadata.source_urls,
    ):
        assert empty == ()


@pytest.mark.needs_download
def test_get_example_metadata_folder():
    """A folder counts as one path but reports the files it contains."""
    metadata = examples.get_example('cubemap_park', output='metadata')

    assert len(metadata.paths) == 1
    assert Path(metadata.paths[0]).is_dir()
    assert metadata.num_files == 6


@pytest.mark.needs_download
def test_get_example_metadata_multiple_files():
    """A multi-file example reports every file, but only the ones it reads as loadable."""
    metadata = examples.get_example('frog', output='metadata')

    assert metadata.num_files == 2
    assert len(metadata.paths) == 2
    assert metadata.extensions == ('.mhd', '.zraw')
    assert metadata.loadable_paths == (metadata.paths[0],)
    assert metadata.reader_types == (pv.MetaImageReader,)


@pytest.mark.needs_download
def test_get_example_is_builtin_only_for_packaged_files():
    """Only examples shipped inside the package report ``is_builtin``."""
    # `uniform` is a `_Downloadable` loader whose file ships with PyVista
    assert examples.get_example('uniform', output='metadata').is_builtin
    assert not examples.get_example('bunny', output='metadata').is_builtin
    assert not examples.get_example('frog', output='metadata').is_builtin


def test_get_example_in_memory_is_not_builtin():
    """An example generated in memory has no files, so it is not built-in."""
    assert not examples.get_example('structured', output='metadata').is_builtin


def test_get_example_download_false_uses_local_files():
    """Built-in examples are available with ``download=False``."""
    paths = examples.get_example('uniform', output='paths', download=False)
    assert Path(paths[0]).is_file()


def test_get_example_download_false_raises(monkeypatch):
    """A missing file raises rather than downloading when ``download=False``."""
    loader = _SingleFileDownloadableDatasetLoader('missing_example.vtk')
    monkeypatch.setattr(downloads, '_dataset_missing_example', loader, raising=False)
    monkeypatch.setattr(downloads, 'download_missing_example', lambda: None, raising=False)

    match = 'not available locally'
    with pytest.raises(FileNotFoundError, match=match):
        examples.get_example('missing_example', output='paths', download=False)


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


def test_get_example_invalid_output_raises():
    """An unsupported ``output`` value raises."""
    with pytest.raises(ValueError, match="Invalid output 'mesh'"):
        examples.get_example('uniform', output='mesh')


@pytest.mark.needs_download
@pytest.mark.parametrize('name', _all_example_names())
def test_get_example_all(name):
    """Every example is reachable by name and by function, and reports its files."""
    if name in _DEPRECATED_DATASETS:
        pytest.skip('Dataset is deprecated.')
    if os.name == 'nt' and name in _SKIP_DATASETS_WINDOWS:
        pytest.skip('Error loading on Windows')

    try:
        metadata = examples.get_example(name, output='metadata')
        from_name = examples.get_example(name)
        from_function = examples.get_example(metadata.function)
    except pv.VTKVersionError:
        pytest.skip('VTK version not supported.')

    assert len(metadata.paths) == len(metadata.file_sizes)
    assert all(Path(path).is_file() or Path(path).is_dir() for path in metadata.paths)

    if name in _NON_DETERMINISTIC_DATASETS:
        return
    if isinstance(from_name, np.ndarray):
        assert np.array_equal(from_name, from_function, equal_nan=True)
    else:
        assert from_name == from_function
