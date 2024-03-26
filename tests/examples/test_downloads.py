from dataclasses import dataclass
import inspect
import os
from pathlib import Path, PureWindowsPath
from types import FunctionType
from typing import Any, Callable, Dict, List, Tuple, Union

import pytest

import pyvista as pv
from pyvista import examples
from pyvista.examples import downloads
from pyvista.examples._dataset_loader import (
    _MultiFileDownloadableLoadable,
    _SingleFileDownloadableLoadable,
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
