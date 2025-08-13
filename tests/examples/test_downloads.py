from __future__ import annotations

import os
from pathlib import Path
from pathlib import PureWindowsPath

import pytest
import requests
from retry_requests import retry

import pyvista as pv
from pyvista import examples
from pyvista.examples import downloads
from tests.conftest import flaky_test
from tests.examples.test_dataset_loader import DatasetLoaderTestCase
from tests.examples.test_dataset_loader import _generate_dataset_loader_test_cases_from_module
from tests.examples.test_dataset_loader import _get_mismatch_fail_msg


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'test_case' in metafunc.fixturenames:
        # Generate a separate test case for each downloadable dataset
        test_cases_downloads = _generate_dataset_loader_test_cases_from_module(
            pv.examples.downloads
        )
        test_cases_planets = _generate_dataset_loader_test_cases_from_module(pv.examples.planets)
        # Exclude `load` functions
        test_cases_planets = [
            case for case in test_cases_planets if case.dataset_function[0].startswith('download')
        ]
        test_cases = [*test_cases_downloads, *test_cases_planets]
        ids = [case.dataset_name for case in test_cases]
        metafunc.parametrize('test_case', test_cases, ids=ids)


def test_dataset_loader_name_matches_download_name(test_case: DatasetLoaderTestCase):
    if (msg := _get_mismatch_fail_msg(test_case)) is not None:
        pytest.fail(msg)


def _is_valid_url(url):
    session = retry(
        status_to_retry=[500, 502, 504, 403, 429],  # default + GH rate limit (403, 429)
        retries=5,
        backoff_factor=2.0,
    )
    try:
        session.get(url)
    except requests.RequestException:
        return False
    else:
        return True


# sometimes fails due to GitHub server issues
@flaky_test(times=3, exceptions=(AssertionError, pytest.fail.Exception))
def test_dataset_loader_source_url_blob(test_case: DatasetLoaderTestCase):
    try:
        # Skip test if not loadable
        sources = test_case.dataset_loader[1].source_url_blob
    except pv.VTKVersionError as e:
        reason = e.args[0]
        pytest.skip(reason)

    # Test valid url
    sources = [sources] if isinstance(sources, str) else sources  # Make iterable
    for url in sources:
        if not _is_valid_url(url):
            pytest.fail(f'Invalid blob URL for {test_case.dataset_name}:\n{url}')


def test_delete_downloads(tmpdir):
    # change the path so we don't delete the examples cache
    old_path = examples.downloads.USER_DATA_PATH
    try:
        examples.downloads.USER_DATA_PATH = str(tmpdir.mkdir('tmpdir'))
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


def test_local_file_cache():
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
        downloads.FETCHER.base_url = 'https://github.com/pyvista/vtk-data/raw/master/Data/'
        downloads._FILE_CACHE = False
        downloads.FETCHER.registry.pop(basename, None)
