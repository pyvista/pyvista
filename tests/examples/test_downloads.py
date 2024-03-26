from dataclasses import dataclass
import inspect
import os
from pathlib import PureWindowsPath
import shutil
from types import FunctionType
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
import pyvista.examples
from pyvista.examples import downloads
from pyvista.examples._dataset_loader import (
    _Downloadable,
    _format_file_size,
    _load_and_merge,
    _load_as_cubemap,
    _load_as_multiblock,
    _Loadable,
    _MultiFileDownloadableLoadable,
    _MultiFileLoadable,
    _SingleFile,
    _SingleFileDownloadable,
    _SingleFileDownloadableLoadable,
    _SingleFileLoadable,
)


@dataclass
class ExampleTestCaseData:
    name: str
    download_func: Tuple[str, FunctionType]
    load_func: Tuple[str, Union[_SingleFileDownloadableLoadable, _MultiFileDownloadableLoadable]]


def _generate_dataset_loader_test_cases() -> List[ExampleTestCaseData]:
    """Generate a list of test cases with all download functions and file loaders"""

    test_cases_dict: Dict = {}

    def add_to_dict(func_name: str, func: Callable[[], Any]) -> List[ExampleTestCaseData]:
        # Function for stuffing example functions into a dict.
        # We use a dict to allow for any entry to be made based on example name alone.
        # This way, we can defer checking for any mismatch between the download functions
        # and file loaders to test time.
        nonlocal test_cases_dict
        if func_name.startswith('_dataset_'):
            case_name = func_name.split('_dataset_')[1]
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
    download_dataset_functions = {
        name: item
        for name, item in module_members.items()
        if name.startswith('download_') and isinstance(item, FunctionType)
    }
    del download_dataset_functions['download_file']
    [add_to_dict(name, func) for name, func in download_dataset_functions.items()]

    # Collect all `_dataset_<name>` file loaders
    example_file_loaders = {
        name: item
        for name, item in module_members.items()
        if name.startswith('_dataset_')
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
        test_cases = _generate_dataset_loader_test_cases()
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
            f"Expected to find a loader named:\n\t\'_dataset_{test_case.name}\'\nGot: {test_case.load_func}"
        )
    else:
        return None


def test_dataset_loader_name_matches_download_name(test_case: ExampleTestCaseData):
    if (msg := _get_mismatch_fail_msg(test_case)) is not None:
        pytest.fail(msg)


def test_delete_downloads(tmpdir):
    # change the path so we don't delete the examples cache
    old_path = examples.downloads.USER_DATA_PATH
    try:
        examples.downloads.USER_DATA_PATH = str(tmpdir.mkdir("tmpdir"))
        assert os.path.isdir(examples.downloads.USER_DATA_PATH)
        tmp_file = os.path.join(examples.downloads.USER_DATA_PATH, 'tmp.txt')
        with open(tmp_file, 'w') as fid:
            fid.write('test')
        examples.delete_downloads()
        assert os.path.isdir(examples.downloads.USER_DATA_PATH)
        assert not os.path.isfile(tmp_file)
    finally:
        examples.downloads.USER_DATA_PATH = old_path


def test_delete_downloads_does_not_exist(tmpdir):
    # change the path so we don't delete the examples cache
    old_path = examples.downloads.USER_DATA_PATH
    new_path = str(tmpdir.join('doesnotexist'))

    try:
        # delete_downloads for a missing directory should not fail.
        examples.downloads.USER_DATA_PATH = new_path
        assert not os.path.isdir(examples.downloads.USER_DATA_PATH)
        examples.delete_downloads()
    finally:
        examples.downloads.USER_DATA_PATH = old_path


def test_file_from_files(tmpdir):
    path = str(tmpdir)
    fnames = [
        os.path.join(path, 'tmp2.txt'),
        os.path.join(path, 'tmp1.txt'),
        os.path.join(path, 'tmp0.txt'),
        os.path.join(path, 'tmp', 'tmp2.txt'),
        os.path.join(path, '/__MACOSX/'),
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

    with open(input_file, 'w') as fid:
        fid.write('hello world')

    examples.downloads._file_copier(input_file, output_file, None)
    assert os.path.isfile(output_file)

    with pytest.raises(FileNotFoundError):
        examples.downloads._file_copier('not a file', output_file, None)


def test_local_file_cache(tmpdir):
    """Ensure that pyvista.examples.downloads can work with a local cache."""
    basename = os.path.basename(examples.planefile)
    dirname = os.path.dirname(examples.planefile)
    downloads.FETCHER.registry[basename] = None

    try:
        downloads.FETCHER.base_url = dirname + '/'
        downloads.FETCHER.registry[basename] = None
        downloads._FILE_CACHE = True
        path = downloads._download_and_read(basename, load=False)
        assert os.path.isfile(path)

        dataset = downloads._download_and_read(basename, load=True)
        assert isinstance(dataset, pv.DataSet)
        os.remove(path)

    finally:
        downloads.FETCHER.base_url = "https://github.com/pyvista/vtk-data/raw/master/Data/"
        downloads._FILE_CACHE = False
        downloads.FETCHER.registry.pop(basename, None)


@pytest.fixture()
def examples_local_repository_tmp_dir(tmp_path):
    """Create a local repository with a bunch of examples available for download."""

    # setup
    repository_path = os.path.join(tmp_path, 'repo')
    os.mkdir(repository_path)

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
            os.path.join(pyvista.examples.dir_path, base), os.path.join(repository_path, base)
        )
        for base in downloadable_basenames
    ]

    # create a zipped copy of the datasets and include the zip with repository
    shutil.make_archive(os.path.join(tmp_path, 'archive'), 'zip', repository_path)
    shutil.move(os.path.join(tmp_path, 'archive.zip'), os.path.join(repository_path, 'archive.zip'))
    downloadable_basenames.append('archive.zip')

    # initialize downloads fetcher
    for base in downloadable_basenames:
        downloads.FETCHER.registry[base] = None
    downloads.FETCHER.base_url = str(repository_path) + '/'
    downloads._FILE_CACHE = True

    # make sure any "downloaded" files (moved from repo -> cache) are cleared
    cached_paths = [os.path.join(downloads.FETCHER.path, base) for base in downloadable_basenames]
    [os.remove(file) for file in cached_paths if os.path.isfile(file)]

    yield repository_path

    # teardown
    downloads.FETCHER.base_url = "https://github.com/pyvista/vtk-data/raw/master/Data/"
    downloads._FILE_CACHE = False
    [downloads.FETCHER.registry.pop(base, None) for base in downloadable_basenames]

    # make sure any "downloaded" files (moved from repo -> cache) are cleared afterward
    [os.remove(file) for file in cached_paths if os.path.isfile(file)]


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

    # test initial path
    path = file_loader.path
    assert os.path.basename(path) == basename
    assert not os.path.isfile(path)

    if expected_path_is_absolute:
        assert os.path.isabs(path)
    else:
        assert not os.path.isabs(path)

    # test download
    if isinstance(file_loader, (_SingleFileDownloadable, _SingleFileDownloadableLoadable)):
        assert isinstance(file_loader, _Downloadable)
        path_download = file_loader.download()
        assert os.path.isfile(path_download)
        assert os.path.isabs(path_download)
        assert file_loader.path == path_download
    else:
        with pytest.raises(AttributeError):
            file_loader.download()
        # download manually to continue test
        downloads.download_file(basename)

    # test load
    if isinstance(file_loader, (_SingleFileLoadable, _SingleFileDownloadableLoadable)):
        assert isinstance(file_loader, _Loadable)
        dataset = file_loader.load()
        assert dataset is file_loader.dataset
        assert isinstance(dataset, pv.DataSet)
    else:
        with pytest.raises(AttributeError):
            file_loader.load()

    assert os.path.isfile(file_loader.path)


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

    path = multi_file_loader.path
    assert multi_file_loader._file_loaders_ is not None
    assert isinstance(path, tuple)
    assert [os.path.isabs(file) for file in path]
    assert len(path) == 3

    path_loadable = multi_file_loader.path_loadable
    assert isinstance(path_loadable, tuple)
    assert [os.path.isabs(file) for file in path_loadable]
    assert len(path_loadable) == 2
    assert basename_not_loaded not in path_loadable

    # test download
    path_download = multi_file_loader.download()
    assert path_download == path
    assert [os.path.isfile(file) for file in path_download]

    # test load
    dataset = multi_file_loader.load()
    assert multi_file_loader.dataset is dataset
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


@pytest.fixture()
def loadable_vtp() -> _SingleFileLoadable:
    return _SingleFileLoadable(downloads.download_file('cow.vtp'))


@pytest.fixture()
def loadable_mhd() -> _MultiFileLoadable:
    def _head_files_func():
        # Multiple files needed for read, but only one gets loaded
        head_raw = _SingleFile('HeadMRVolume.raw')
        head_mhd = _SingleFileLoadable('HeadMRVolume.mhd')
        return head_mhd, head_raw

    return _MultiFileLoadable(_head_files_func)


@pytest.fixture()
def loadable_slc() -> _MultiFileLoadable:
    def _bolt_nut_files_func():
        bolt = _SingleFileLoadable(downloads.download_file('bolt.slc'))
        nut = _SingleFileLoadable(downloads.download_file('nut.slc'))
        return bolt, nut

    return _MultiFileLoadable(_bolt_nut_files_func, load_func=_load_as_multiblock)


@pytest.fixture()
def loadable_cubemap() -> _SingleFileLoadable:
    return _SingleFileLoadable(
        downloads._download_archive_file_or_folder('cubemap_park/cubemap_park.zip', target_file=''),
        read_func=_load_as_cubemap,
    )


@pytest.fixture()
def loadable_dicom() -> _SingleFileLoadable:
    return _SingleFileLoadable(
        downloads._download_archive_file_or_folder('DICOM_Stack/data.zip', target_file='data')
    )


def test_file_loader_file_props_from_one_file(loadable_vtp):
    # test single file
    example = loadable_vtp
    assert os.path.isfile(example.path)
    assert example.num_files == 1
    assert example._total_size_bytes == 60449
    assert example.total_size == '60.4 KB'
    assert example.unique_extension == '.vtp'
    assert isinstance(example._reader, pv.XMLPolyDataReader)
    assert example.unique_reader_type is pv.XMLPolyDataReader
    assert example.dataset is None
    assert example.unique_dataset_type is None
    example.load()
    assert type(example.dataset) is pv.PolyData
    assert example.unique_dataset_type is pv.PolyData


def test_file_loader_file_props_from_two_files_one_loaded(loadable_mhd):
    # test multiple files, but only one is loaded
    example = downloads._dataset_head
    example.download()
    assert all(os.path.isfile(file) for file in example.path)
    assert example.num_files == 2
    assert example._total_size_bytes == 125223
    assert example.total_size == '125.2 KB'
    assert example.unique_extension == ('.mhd', '.raw')
    assert pv.get_ext(example.path[0]) == '.mhd'
    assert pv.get_ext(example.path[1]) == '.raw'
    assert len(example._reader) == 2
    assert isinstance(example._reader[0], pv.MetaImageReader)
    assert example._reader[1] is None
    assert example.unique_reader_type is pv.MetaImageReader
    assert example.dataset is None
    assert example.unique_dataset_type is None
    example.load()
    assert type(example.dataset) is pv.ImageData
    assert example.unique_dataset_type is pv.ImageData


def test_file_loader_file_props_from_two_files_both_loaded(loadable_slc):
    # test multiple files, both have same ext and reader,
    # both of which are loaded as a multiblock
    example = loadable_slc
    assert len(example.path) == 2
    assert example.num_files == 2
    assert os.path.isfile(example.path[0])
    assert os.path.isfile(example.path[1])
    assert example._total_size_bytes == 132818
    assert example.total_size == '132.8 KB'
    assert example.unique_extension == '.slc'
    assert len(example._reader) == 2
    assert isinstance(example._reader[0], pv.SLCReader)
    assert isinstance(example._reader[1], pv.SLCReader)
    assert example.unique_reader_type is pv.SLCReader
    assert example.dataset is None
    assert example.unique_dataset_type is None
    example.load()
    assert type(example.dataset) is pv.MultiBlock
    assert example.unique_dataset_type == (pv.MultiBlock, pv.ImageData)


def test_file_loader_file_props_from_directory_cubemap(loadable_cubemap):
    # test directory (cubemap)
    example = loadable_cubemap
    assert os.path.isdir(example.path)
    assert example.num_files == 6
    assert example._total_size_bytes == 606113
    assert example.total_size == '606.1 KB'
    assert example.unique_extension == '.jpg'
    assert example._reader is None
    assert example.unique_reader_type is None
    assert example.dataset is None
    assert example.unique_dataset_type is None
    example.load()
    assert type(example.dataset) is pv.Texture
    assert example.unique_dataset_type is pv.Texture


def test_file_loader_file_props_from_directory_dicom(loadable_dicom):
    # test directory (dicom stack)
    example = loadable_dicom
    assert os.path.isdir(example.path)
    assert example.num_files == 3
    assert example._total_size_bytes == 1583688
    assert example.total_size == '1.6 MB'
    assert example.unique_extension == '.dcm'
    assert isinstance(example._reader, pv.DICOMReader)
    assert example.unique_reader_type is pv.DICOMReader
    assert example.dataset is None
    assert example.unique_dataset_type is None
    example.load()
    assert type(example.dataset) is pv.ImageData
    assert example.unique_dataset_type is pv.ImageData


def test_file_loader_file_props_from_nested_files_and_directory(
    loadable_vtp, loadable_mhd, loadable_dicom
):
    # test complex multiple file case with separate ext and reader, which are loaded as a tuple
    # piece together new dataset from existing ones
    def files_func():
        return loadable_vtp, loadable_mhd, loadable_dicom

    example = _MultiFileLoadable(files_func)
    assert len(example.path) == 4
    assert example.num_files == 6
    assert os.path.isfile(example.path[0])
    assert os.path.isfile(example.path[1])
    assert os.path.isfile(example.path[2])
    assert os.path.isdir(example.path[3])
    assert example._total_size_bytes == 1769360
    assert example.total_size == '1.8 MB'
    assert example.unique_extension == ('.dcm', '.mhd', '.raw', '.vtp')
    assert len(example._reader) == 4
    assert isinstance(example._reader[0], pv.XMLPolyDataReader)
    assert isinstance(example._reader[1], pv.MetaImageReader)
    assert example._reader[2] is None
    assert isinstance(example._reader[3], pv.DICOMReader)
    assert example.unique_reader_type == (pv.XMLPolyDataReader, pv.MetaImageReader, pv.DICOMReader)
    assert example.dataset is None
    assert example.unique_dataset_type is None
    example.load()
    assert type(example.dataset) is tuple
    assert set(example.unique_dataset_type) == {pv.PolyData, pv.ImageData}


def test_format_file_size():
    assert _format_file_size(999) == '999 B'
    assert _format_file_size(1000) == '1.0 KB'

    assert _format_file_size(999949) == '999.9 KB'
    assert _format_file_size(999950) == '1.0 MB'
    assert _format_file_size(1000000) == '1.0 MB'

    assert _format_file_size(999949000) == '999.9 MB'
    assert _format_file_size(999950000) == '1.0 GB'
    assert _format_file_size(1000000000) == '1.0 GB'

    assert _format_file_size(999949000000) == '999.9 GB'
    assert _format_file_size(999950000000) == '1000.0 GB'
    assert _format_file_size(1000000000000) == '1000.0 GB'
