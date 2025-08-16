from __future__ import annotations

import os
import pathlib
from pathlib import Path
from pathlib import PureWindowsPath
import re

import pytest
import requests
from retry_requests import retry

import pyvista as pv
from pyvista import examples
from pyvista.examples import downloads
from pyvista.examples.downloads import _get_user_data_path
from pyvista.examples.downloads import _get_vtk_data_source
from pyvista.examples.downloads import _warn_if_path_not_accessible
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
        # Check is_file() in case local cache of vtk-data is used
        if not (Path(url).is_file() or _is_valid_url(url)):
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


@pytest.mark.parametrize('endswith', ['', 'Data', 'Data/'])
def test_get_vtk_data_path_with_env_var(monkeypatch, endswith, tmp_path):
    path = (tmp_path / 'mypath').as_posix()
    if endswith:
        path = path + '/' + endswith
    monkeypatch.setenv(downloads._VTK_DATA_VARNAME, path)
    path_no_trailing_slash = path.removesuffix('/')
    match = (
        f'The given {downloads._VTK_DATA_VARNAME} is not a valid directory '
        f'and will not be used:\n{path_no_trailing_slash}'
    )
    with pytest.warns(UserWarning, match=re.escape(match)):
        _ = _get_vtk_data_source()
    Path(path).mkdir(parents=True)
    source, file_cache = _get_vtk_data_source()
    assert source.endswith('/Data/')  # it should append Data and /
    assert file_cache is True


def test_get_vtk_data_path_without_env_var(monkeypatch):
    monkeypatch.delenv(downloads._VTK_DATA_VARNAME, raising=False)
    source, file_cache = _get_vtk_data_source()
    assert source == downloads._DEFAULT_VTK_DATA_SOURCE
    assert file_cache is False


def test_get_user_data_path_env_var_valid(monkeypatch, tmp_path):
    valid_dir = tmp_path / 'valid'
    valid_dir.mkdir()
    monkeypatch.setenv(downloads._USERDATA_PATH_VARNAME, str(valid_dir))
    result = _get_user_data_path()
    assert result == str(valid_dir)


def test_get_user_data_path_env_var_invalid(monkeypatch, tmp_path):
    not_a_dir = tmp_path / 'file'
    not_a_dir.write_text('not a directory')
    monkeypatch.setenv(downloads._USERDATA_PATH_VARNAME, not_a_dir.as_posix())
    match = (
        f'The given {downloads._USERDATA_PATH_VARNAME} is not a valid directory '
        f'and will not be used:\n{not_a_dir.as_posix()}'
    )
    with pytest.warns(UserWarning, match=re.escape(match)):
        result = _get_user_data_path()
    # should fall back to pooch path
    assert result == downloads._DEFAULT_USER_DATA_PATH


def test_get_user_data_path_no_env_var(monkeypatch):
    monkeypatch.delenv(downloads._USERDATA_PATH_VARNAME, raising=False)
    result = _get_user_data_path()
    assert result == downloads._DEFAULT_USER_DATA_PATH


def test_warn_if_path_not_accessible_creates_dir(tmp_path):
    path = tmp_path / 'newdir'
    assert not path.exists()
    # Should create without warning
    _warn_if_path_not_accessible(path, 'MY_ENV')
    assert path.is_dir()


def test_warn_if_path_not_accessible_file_blocks(tmp_path):
    blocked_path = tmp_path / 'blocked'
    blocked_path.write_text('not a dir')
    match = (
        f'Unable to access path: {blocked_path.as_posix()}\nManually specify the PyVista '
        f'examples cache with the PYVISTA_USERDATA_PATH environment variable.'
    )
    with pytest.warns(UserWarning, match=re.escape(match)):
        _warn_if_path_not_accessible(blocked_path.as_posix(), downloads._user_data_path_warn_msg)


@pytest.mark.skip_windows(reason='CI has admin rights and can write to system dirs.')
def test_warn_if_path_not_accessible_no_write_permission():
    system_dir = pathlib.Path('/etc')
    assert system_dir.exists()
    assert not os.access(system_dir, os.W_OK)
    blocked_dir = system_dir / 'blocked'
    with pytest.warns(UserWarning, match='Unable to access'):
        _warn_if_path_not_accessible(blocked_dir, downloads._user_data_path_warn_msg)
