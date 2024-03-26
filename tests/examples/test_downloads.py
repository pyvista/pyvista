from dataclasses import dataclass
import inspect
import os
from pathlib import Path, PureWindowsPath
import shutil
from types import FunctionType
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
import pyvista.examples
from pyvista.examples import downloads
from pyvista.examples._example_loader import (
    _Downloadable,
    _load_and_merge,
    _load_as_multiblock,
    _Loadable,
    _MultiFileDownloadableLoadable,
    _SingleFileDownloadable,
    _SingleFileDownloadableLoadable,
    _SingleFileLoadable,
)


@dataclass
class ExampleTestCaseData:
    name: str
    download_func: Tuple[str, FunctionType]
    load_func: Tuple[str, Union[_SingleFileDownloadableLoadable, _MultiFileDownloadableLoadable]]


def _generate_example_loader_test_cases() -> List[ExampleTestCaseData]:
    """Generate a list of test cases with all download functions and file loaders"""

    test_cases_dict: Dict = {}

    def add_to_dict(func_name: str, func: Callable[[], Any]) -> List[ExampleTestCaseData]:
        # Function for stuffing example functions into a dict.
        # We use a dict to allow for any entry to be made based on example name alone.
        # This way, we can defer checking for any mismatch between the download functions
        # and file loaders to test time.
        nonlocal test_cases_dict
        if func_name.startswith('_example_'):
            case_name = func_name.split('_example_')[1]
            key = 'load_func'
        elif func_name.startswith('download_'):
            case_name = func_name.split('download_')[1]
            key = 'download_func'
        else:
            raise RuntimeError(f'Invalid case specified: {(func_name, func)}')
        test_cases_dict.setdefault(case_name, {})
        test_cases_dict[case_name][key] = (func_name, func)

    module_members = dict(inspect.getmembers(pv.examples.downloads))

    # Collect all `download_<name>` functions
    download_example_functions = {
        name: item
        for name, item in module_members.items()
        if name.startswith('download_') and isinstance(item, FunctionType)
    }
    del download_example_functions['download_file']
    [add_to_dict(name, func) for name, func in download_example_functions.items()]

    # Collect all `_example_<name>` file loaders
    example_file_loaders = {
        name: item
        for name, item in module_members.items()
        if name.startswith('_example_')
        and isinstance(item, (_SingleFileDownloadableLoadable, _MultiFileDownloadableLoadable))
    }
    [add_to_dict(name, func) for name, func in example_file_loaders.items()]

    # Flatten dict
    test_cases_list: List[ExampleTestCaseData] = []
    for name, content in sorted(test_cases_dict.items()):
        download_func = content.setdefault('download_func', None)
        load_func = content.setdefault('load_func', None)
        test_case = ExampleTestCaseData(name=name, download_func=download_func, load_func=load_func)
        test_cases_list.append(test_case)

    return test_cases_list


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'test_case' in metafunc.fixturenames:
        # Generate a separate test case for each downloadable example
        test_cases = _generate_example_loader_test_cases()
        ids = [case.name for case in test_cases]
        metafunc.parametrize('test_case', test_cases, ids=ids)


def _get_mismatch_fail_msg(test_case: ExampleTestCaseData):
    if test_case.download_func is None:
        return (
            f"A file loader:\n\t\'{test_case.load_func[0]}\'\n\t{test_case.load_func[1]}\n"
            f"was found but is missing a corresponding download function.\n\n"
            f"Expected to find a function named:\n\t\'download_{test_case.name}\'\nGot: {test_case.download_func}"
        )
    elif test_case.load_func is None:
        return (
            f"A download function:\n\t\'{test_case.download_func[0]}\'\n\t{test_case.download_func[1]}\n"
            f"was found but is missing a corresponding file loader.\n\n"
            f"Expected to find a loader named:\n\t\'_example_{test_case.name}\'\nGot: {test_case.load_func}"
        )
    else:
        return None


def test_example_loader_name_matches_download_name(test_case: ExampleTestCaseData):
    if (msg := _get_mismatch_fail_msg(test_case)) is not None:
        pytest.fail(msg)


def test_delete_downloads(tmpdir):
    # change the path so we don't delete the examples cache
    old_path = examples.downloads.USER_DATA_PATH
    try:
        examples.downloads.USER_DATA_PATH = str(tmpdir.mkdir("tmpdir"))
        assert Path(examples.downloads.USER_DATA_PATH).is_dir()
        tmp_file = str(Path(examples.downloads.USER_DATA_PATH) / 'tmp.txt')
        with Path(tmp_file).open('w') as fid:
            fid.write('test')
        examples.delete_downloads()
        assert Path(examples.downloads.USER_DATA_PATH).is_dir()
        assert not Path(tmp_file).is_file()
    finally:
        examples.downloads.USER_DATA_PATH = old_path


def test_delete_downloads_does_not_exist(tmpdir):
    # change the path so we don't delete the examples cache
    old_path = examples.downloads.USER_DATA_PATH
    new_path = str(tmpdir.join('doesnotexist'))

    try:
        # delete_downloads for a missing directory should not fail.
        examples.downloads.USER_DATA_PATH = new_path
        assert not Path(examples.downloads.USER_DATA_PATH).is_dir()
        examples.delete_downloads()
    finally:
        examples.downloads.USER_DATA_PATH = old_path


def test_file_from_files(tmpdir):
    path = str(tmpdir)
    fnames = [
        str(Path(path) / 'tmp2.txt'),
        str(Path(path) / 'tmp1.txt'),
        str(Path(path) / 'tmp0.txt'),
        str(Path(path) / 'tmp' / 'tmp2.txt'),
        str(Path(path) / '/__MACOSX/'),
    ]

    with pytest.raises(FileNotFoundError):
        fname = examples.downloads.file_from_files('potato.txt', fnames)

    fname = examples.downloads.file_from_files('tmp1.txt', fnames)
    if os.name == 'nt':
        assert PureWindowsPath(fname) == PureWindowsPath(fnames[1])
    else:
        assert fname == fnames[1]

    with pytest.raises(RuntimeError, match='Ambiguous'):
        fname = examples.downloads.file_from_files('tmp2.txt', fnames)


def test_file_copier(tmpdir):
    input_file = str(tmpdir.join('tmp0.txt'))
    output_file = str(tmpdir.join('tmp1.txt'))

    with Path(input_file).open('w') as fid:
        fid.write('hello world')

    examples.downloads._file_copier(input_file, output_file, None)
    assert Path(output_file).is_file()

    with pytest.raises(FileNotFoundError):
        examples.downloads._file_copier('not a file', output_file, None)


def test_local_file_cache(tmpdir):
    """Ensure that pyvista.examples.downloads can work with a local cache."""
    basename = Path(examples.planefile).name
    dirname = str(Path(examples.planefile).parent)
    downloads.FETCHER.registry[basename] = None

    try:
        downloads.FETCHER.base_url = dirname + '/'
        downloads.FETCHER.registry[basename] = None
        downloads._FILE_CACHE = True
        filename = downloads._download_and_read(basename, load=False)
        assert Path(filename).is_file()

        dataset = downloads._download_and_read(basename, load=True)
        assert isinstance(dataset, pv.DataSet)
        Path(filename).unlink()

    finally:
        downloads.FETCHER.base_url = "https://github.com/pyvista/vtk-data/raw/master/Data/"
        downloads._FILE_CACHE = False
        downloads.FETCHER.registry.pop(basename, None)


@pytest.fixture()
def examples_local_repository_tmp_dir(tmp_path):
    """Create a local repository with a bunch of examples available for download."""

    # setup
    repository_path = str(Path(tmp_path) / 'repo')
    Path(repository_path).mkdir()

    downloadable_basenames = [
        'airplane.ply',
        'hexbeam.vtk',
        'sphere.ply',
        'uniform.vtk',
        'rectilinear.vtk',
        'globe.vtk',
        '2k_earth_daymap.jpg',
        'channels.vti',
        'pyvista_logo.png',
    ]

    # copy datasets from the pyvista repo to the local repository
    [
        shutil.copyfile(
            str(Path(pyvista.examples.dir_path) / base), str(Path(repository_path) / base)
        )
        for base in downloadable_basenames
    ]

    # create a zipped copy of the datasets and include the zip with repository
    shutil.make_archive(str(Path(tmp_path) / 'archive'), 'zip', repository_path)
    shutil.move(str(Path(tmp_path) / 'archive.zip'), str(Path(repository_path) / 'archive.zip'))
    downloadable_basenames.append('archive.zip')

    # initialize downloads fetcher
    for base in downloadable_basenames:
        downloads.FETCHER.registry[base] = None
    downloads.FETCHER.base_url = str(repository_path) + '/'
    downloads._FILE_CACHE = True

    # make sure any "downloaded" files (moved from repo -> cache) are cleared
    cached_filenames = [str(Path(downloads.FETCHER.path) / base) for base in downloadable_basenames]
    [Path(file).unlink() for file in cached_filenames if Path(file).is_file()]

    yield repository_path

    # teardown
    downloads.FETCHER.base_url = "https://github.com/pyvista/vtk-data/raw/master/Data/"
    downloads._FILE_CACHE = False
    [downloads.FETCHER.registry.pop(base, None) for base in downloadable_basenames]

    # make sure any "downloaded" files (moved from repo -> cache) are cleared afterward
    [Path(file).unlink() for file in cached_filenames if Path(file).is_file()]


@pytest.mark.parametrize('use_archive', [True, False])
@pytest.mark.parametrize(
    'FileLoader', [_SingleFileLoadable, _SingleFileDownloadable, _SingleFileDownloadableLoadable]
)
def test_single_file_loader(FileLoader, use_archive, examples_local_repository_tmp_dir):
    basename = 'pyvista_logo.png'
    if use_archive and isinstance(FileLoader, _Downloadable):
        file_loader = FileLoader('archive.zip', target_file=basename)
        expected_path_is_absolute = False
    else:
        file_loader = FileLoader(basename)
        expected_path_is_absolute = True

    # test initial filename
    filename = file_loader.filename
    assert Path(filename).name == basename
    assert not Path(filename).is_file()

    if expected_path_is_absolute:
        assert Path(filename).is_absolute()
    else:
        assert not Path(filename).is_absolute()

    # test download
    if isinstance(file_loader, (_SingleFileDownloadable, _SingleFileDownloadableLoadable)):
        assert isinstance(file_loader, _Downloadable)
        filename_download = file_loader.download()
        assert Path(filename_download).is_file()
        assert Path(filename_download).is_absolute()
        assert file_loader.filename == filename_download
    else:
        with pytest.raises(AttributeError):
            file_loader.download()
        # download manually to continue test
        downloads.download_file(basename)

    # test load
    if isinstance(file_loader, (_SingleFileLoadable, _SingleFileDownloadableLoadable)):
        assert isinstance(file_loader, _Loadable)
        dataset = file_loader.load()
        assert isinstance(dataset, pv.DataSet)
    else:
        with pytest.raises(AttributeError):
            file_loader.load()

    assert Path(file_loader.filename).is_file()


@pytest.mark.parametrize('load_func', [_load_as_multiblock, _load_and_merge, None])
def test_multi_file_loader(examples_local_repository_tmp_dir, load_func):
    basename_loaded1 = 'airplane.ply'
    basename_loaded2 = 'hexbeam.vtk'
    basename_not_loaded = 'pyvista_logo.png'

    file_loaded1 = _SingleFileDownloadableLoadable(basename_loaded1)
    file_loaded2 = _SingleFileDownloadableLoadable(basename_loaded2)
    file_not_loaded = _SingleFileDownloadable(basename_not_loaded)

    expected_airplane = examples.load_airplane()
    expected_hexbeam = examples.load_hexbeam()

    def files_func():
        return file_loaded1, file_loaded2, file_not_loaded

    multi_file_loader = _MultiFileDownloadableLoadable(files_func, load_func=load_func)
    # test files func is not called when initialized
    assert multi_file_loader._file_loaders_ is None

    filename = multi_file_loader.filename
    assert multi_file_loader._file_loaders_ is not None
    assert isinstance(filename, tuple)
    assert [Path(file).is_absolute() for file in filename]
    assert len(filename) == 3

    filename_loadable = multi_file_loader.filename_loadable
    assert isinstance(filename_loadable, tuple)
    assert [Path(file).is_absolute() for file in filename_loadable]
    assert len(filename_loadable) == 2
    assert basename_not_loaded not in filename_loadable

    # test download
    filename_download = multi_file_loader.download()
    assert filename_download == filename
    assert [Path(file).is_file() for file in filename_download]

    # test load
    dataset = multi_file_loader.load()
    if load_func is None:
        assert isinstance(dataset, tuple)
        assert np.array_equal(dataset[0].points, expected_airplane.points)
        assert np.array_equal(dataset[1].points, expected_hexbeam.points)
        assert len(dataset) == 2
    elif load_func is _load_as_multiblock:
        assert isinstance(dataset, pv.MultiBlock)
        assert dataset.keys() == ['airplane', 'hexbeam']
        assert np.array_equal(dataset[0].points, expected_airplane.points)
        assert np.array_equal(dataset[1].points, expected_hexbeam.points)
        assert len(dataset) == 2
    elif load_func is _load_and_merge:
        assert isinstance(dataset, pv.UnstructuredGrid)
        expected = pv.merge((expected_airplane, expected_hexbeam))
        assert np.array_equal(dataset.points, expected.points)


def test_file_loader_file_props():
    # test single file
    example = downloads._example_cow
    example.download()
    assert Path(example.filename).is_file()
    assert example.total_size == '59.0 KiB'
    assert example.extension == '.vtp'
    assert type(example.reader) is pv.XMLPolyDataReader

    # test multiple files, but only one is loaded
    example = downloads._example_head
    example.download()
    assert all(Path(file).is_file() for file in example.filename)
    assert example.total_size == '122.3 KiB'
    assert example.extension == ('.mhd', '.raw')
    assert pv.get_ext(example.filename[0]) == '.mhd'
    assert pv.get_ext(example.filename[1]) == '.raw'
    assert type(example.reader) is pv.MetaImageReader

    # test directory (cubemap)
    example = downloads._example_cubemap_park
    example.download()
    assert Path(example.filename).is_dir()
    assert example.total_size == '591.9 KiB'
    assert example.extension == '.jpg'
    assert example.reader is None

    # test directory (dicom stack)
    example = downloads._example_dicom_stack
    example.download()
    assert Path(example.filename).is_dir()
    assert example.total_size == '1.5 MiB'
    assert example.extension == '.dcm'
    assert type(example.reader) is pv.DICOMReader
