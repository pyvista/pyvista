from __future__ import annotations

from dataclasses import fields
import inspect
import os
from pathlib import Path
import re
from typing import get_args
import warnings

import pytest
from typing_extensions import get_overloads

import pyvista as pv
from pyvista import examples
from pyvista.examples import _get_example
from pyvista.examples import downloads
from pyvista.examples import planets
from pyvista.examples._dataset_loader import _SingleFileDatasetLoader
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader
from pyvista.examples._get_example import _example_names
from pyvista.examples._get_example import _get_dataset_loader
from pyvista.examples._get_example import _resolve_paths
from pyvista.examples._get_example import _supported_modules

_SKIP_DATASETS_WINDOWS = ['biplane']

_OVERLOADS_FILE = Path(_get_example.__file__)
_GENERATED_START = '# --- generated overloads ---\n'
_GENERATED_END = '# --- end generated overloads ---\n'
# the `load=False` half of a function's return annotation, which is not a dataset
_PATH_TYPES = {'str', 'list[str]', 'tuple[str, ...]'}
# every other dataset type name is an attribute of `pv`
_DATASET_TYPE_NAMES = {'ndarray': 'pv.NumpyArray[Any]'}


def _all_example_names():
    """Return the name of every example defined across the examples modules."""
    return sorted(name for module in _supported_modules() for name in _example_names(module))


def _dataset_annotation(function):
    """Render the dataset type a type checker resolves for a plain call of ``function``.

    That is the first overload, or the function's own annotation when it has none, less
    the members which stand for a path.
    """
    overloads = get_overloads(function)
    annotation = inspect.signature(overloads[0] if overloads else function).return_annotation
    members = [member.strip() for member in str(annotation).split('|')]
    return ' | '.join(
        _DATASET_TYPE_NAMES.get(member, f'pv.{member}')
        for member in members
        if member not in _PATH_TYPES
    )


def _readers_annotation(example):
    """Render the exact tuple type of ``example.readers``."""
    names = ', '.join(f'pv.{type(reader).__name__}' for reader in example.readers)
    return f'tuple[{names}]' if names else 'tuple[()]'


def _declared_overloads():
    """Return ``{name: (dataset, readers)}`` from the ``Literal``-name overloads in place."""
    declared = {}
    for stub in get_overloads(examples.get_example):
        annotations = stub.__annotations__
        if (name := re.fullmatch(r"Literal\['([^']+)'\]", annotations['name'])) is None:
            continue
        returns = re.fullmatch(r'Example\[(.+), (tuple\[.*\])\]', annotations['return'])
        assert returns is not None, annotations['return']
        assert name[1] not in declared, f'duplicate overload for {name[1]!r}'
        declared[name[1]] = (returns[1], returns[2])
    return declared


def _current_overloads():
    """Return ``{name: (dataset, readers)}`` from the examples themselves (downloads them)."""
    current = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # the nefertiti licence, and a few others
        for name in _all_example_names():
            example = examples.get_example(name)
            current[name] = (_dataset_annotation(example.function), _readers_annotation(example))
    return current


def _format_overloads(overloads):
    """Render the generated block: the ``ExampleName`` literal, then one overload per line."""
    lines = ['ExampleName = Literal[', *(f"    '{name}'," for name in sorted(overloads)), ']']
    for name, (dataset, readers) in sorted(overloads.items()):
        lines.append('@overload')
        lines.append(
            f"def get_example(name: Literal['{name}'], *, download: bool = ...)"
            f' -> Example[{dataset}, {readers}]: ...'
        )
    return '\n'.join(lines) + '\n'


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


@pytest.mark.needs_download
def test_get_example_nefertiti_warns_without_download():
    """The licence warning fires on the load-only route too."""
    with pytest.warns(UserWarning, match='CC BY-NC-SA'):
        examples.download_nefertiti(load=False)  # ensure the files are cached

    example = examples.get_example('nefertiti', download=False)
    with pytest.warns(UserWarning, match='CC BY-NC-SA'):
        example.load()


def test_get_example_overloads_cover_every_example():
    """One ``Literal`` overload per example, and none for an example which does not exist."""
    assert sorted(_declared_overloads()) == _all_example_names()


@pytest.mark.parametrize('name', _all_example_names())
def test_get_example_overload_dataset_type_matches_the_function(name):
    """The overload promises the dataset type the example's own function is annotated with."""
    _, _, function = _get_dataset_loader(name)
    assert _declared_overloads()[name][0] == _dataset_annotation(function)


@pytest.mark.parametrize('name', _all_example_names())
def test_get_example_function_overloads_accept_a_plain_call(name):
    """A function with a ``load`` parameter has overloads, one of which a plain call matches.

    Without them a type checker sees the ``dataset | str`` union for ``load=True``, and
    the function form of ``get_example`` inherits it.
    """
    _, _, function = _get_dataset_loader(name)
    if 'load' not in inspect.signature(function).parameters:
        pytest.skip('no `load` parameter')
    overloads = get_overloads(function)
    assert overloads, f'{function.__name__} has a `load` parameter but no overloads'
    assert any(
        all(p.default is not p.empty for p in inspect.signature(o).parameters.values())
        for o in overloads
    ), f'no overload of {function.__name__} accepts a plain call'


def test_example_name_literal_lists_every_example():
    """``ExampleName`` is the ``Literal`` of every example name, so editors can complete it."""
    assert get_args(_get_example.ExampleName) == tuple(_all_example_names())


def test_format_overloads_renders_one_line_per_stub():
    """The generated block is the name literal, then an ``@overload`` and a one-line stub each."""
    block = _format_overloads(
        {
            'cow': ('pv.PolyData', 'tuple[pv.XMLPolyDataReader]'),
            'ant': ('pv.PolyData', 'tuple[pv.PLYReader]'),
        }
    )
    assert block == (
        'ExampleName = Literal[\n'
        "    'ant',\n"
        "    'cow',\n"
        ']\n'
        '@overload\n'
        "def get_example(name: Literal['ant'], *, download: bool = ...)"
        ' -> Example[pv.PolyData, tuple[pv.PLYReader]]: ...\n'
        '@overload\n'
        "def get_example(name: Literal['cow'], *, download: bool = ...)"
        ' -> Example[pv.PolyData, tuple[pv.XMLPolyDataReader]]: ...\n'
    )


@pytest.mark.needs_download
def test_get_example_overloads_current(request):
    """The generated overloads match every example; ``--regenerate_overloads`` rewrites them."""
    current = _current_overloads()
    if request.config.getoption('--regenerate_overloads'):  # pragma: no cover -- maintainer path
        source = _OVERLOADS_FILE.read_text()
        start = source.index(_GENERATED_START) + len(_GENERATED_START)
        end = source.index(_GENERATED_END)
        _OVERLOADS_FILE.write_text(source[:start] + _format_overloads(current) + source[end:])
        return

    declared = _declared_overloads()
    stale = sorted(name for name in current if declared.get(name) != current[name])
    stale += sorted(set(declared) - set(current))
    assert not stale, (
        f'The generated `get_example` overloads are stale for: {stale}. Regenerate with\n'
        '  pytest tests/examples/test_get_example.py -k overloads_current '
        '--test_downloads --regenerate_overloads\n'
        'and run pre-commit afterwards.'
    )


def test_get_example_download_false_uses_local_files():
    """Built-in examples are available with ``download=False``."""
    example = examples.get_example('uniform', download=False)

    assert Path(example.paths[0]).is_file()


def test_get_example_download_true_downloads(monkeypatch, tmp_path):
    """``download=True`` calls the loader's download, which provides the files."""
    target = tmp_path / 'spy_example.vtk'
    loader = _SingleFileDownloadableDatasetLoader('spy_example.vtk')
    calls = []

    def fake_download():
        calls.append(True)
        target.write_bytes(b'')
        return (str(target),)

    monkeypatch.setattr(loader, '_path', str(target))
    monkeypatch.setattr(loader, 'download', fake_download)
    monkeypatch.setattr(downloads, '_dataset_spy_example', loader, raising=False)
    monkeypatch.setattr(downloads, 'download_spy_example', lambda: None, raising=False)

    example = examples.get_example('spy_example')
    assert calls == [True]
    assert example.paths == (str(target),)

    calls.clear()
    assert examples.get_example('spy_example', download=False) == example
    assert calls == []


def test_get_example_download_that_provides_nothing_raises(monkeypatch, tmp_path):
    """A download which does not produce the files raises with the right reason."""
    loader = _SingleFileDownloadableDatasetLoader('spy_example.vtk')
    monkeypatch.setattr(loader, '_path', str(tmp_path / 'spy_example.vtk'))
    monkeypatch.setattr(loader, 'download', lambda: ())
    monkeypatch.setattr(downloads, '_dataset_spy_example', loader, raising=False)
    monkeypatch.setattr(downloads, 'download_spy_example', lambda: None, raising=False)

    with pytest.raises(FileNotFoundError, match='even after downloading'):
        examples.get_example('spy_example')


def test_get_example_download_false_rejects_unresolved_archive():
    """An archive member is a relative path until extracted, and must not pass as present."""
    # `Path('data').exists()` is true whenever the working directory holds a `data/`,
    # and `Path('').exists()` is always true -- neither means the example is available
    for target in ('data', ''):
        loader = _SingleFileDownloadableDatasetLoader('not_downloaded.zip', target_file=target)
        assert not Path(loader.paths[0]).is_absolute()
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


def test_get_example_without_public_function_raises(monkeypatch):
    """A loader with no ``download_``/``load_`` function is an error, not a silent skip."""
    loader = _SingleFileDownloadableDatasetLoader('orphan_example.vtk')
    monkeypatch.setattr(downloads, '_dataset_orphan_example', loader, raising=False)

    match = "Example 'orphan_example' has no public function in 'pyvista.examples.downloads'"
    with pytest.raises(ValueError, match=match):
        examples.get_example('orphan_example')


def test_get_example_missing_file_that_cannot_be_downloaded_raises(monkeypatch, tmp_path):
    """A built-in whose file is missing says so, rather than blaming ``download=False``."""
    loader = _SingleFileDatasetLoader(str(tmp_path / 'ghost_example.vtk'))
    monkeypatch.setattr(examples.examples, '_dataset_ghost_example', loader, raising=False)
    monkeypatch.setattr(examples.examples, 'load_ghost_example', lambda: None, raising=False)

    with pytest.raises(FileNotFoundError, match='and cannot be downloaded'):
        examples.get_example('ghost_example')


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
    if os.name == 'nt' and name in _SKIP_DATASETS_WINDOWS:  # pragma: no cover -- Windows only
        pytest.skip('Error loading on Windows')

    with warnings.catch_warnings():
        # a few examples warn on their own account, and the nefertiti licence fires
        # from its loader, so it reaches every route taken here
        warnings.simplefilter('ignore')
        try:
            example = examples.get_example(name)
            loaded = example.load()
            from_function = example.function()
        except pv.VTKVersionError:  # pragma: no cover -- only on older VTK
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
    # the generated overload for this example still promises its exact readers
    assert _declared_overloads().get(name, (None, None))[1] == _readers_annotation(example)
