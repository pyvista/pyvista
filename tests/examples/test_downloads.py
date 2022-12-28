import os
from pathlib import PureWindowsPath

import pytest

import pyvista as pv
from pyvista import examples
from pyvista.examples import downloads


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

    with pytest.raises(RuntimeError, match='Ambigious'):
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
    basename = os.path.basename(examples.mapfile)
    dirname = os.path.dirname(examples.mapfile)
    downloads.FETCHER.registry[basename] = None

    try:
        downloads.FETCHER.base_url = dirname + '/'
        downloads.FETCHER.registry[basename] = None
        downloads._FILE_CACHE = True
        filename = downloads._download_and_read(basename, load=False)
        assert os.path.isfile(filename)

        dataset = downloads._download_and_read(basename, load=True)
        assert isinstance(dataset, pv.DataSet)
        texture = downloads._download_and_read(basename, texture=True)
        assert isinstance(texture, pv.Texture)
        os.remove(filename)

    finally:
        downloads.FETCHER.base_url = "https://github.com/pyvista/vtk-data/raw/master/Data/"
        downloads._FILE_CACHE = False
        downloads.FETCHER.registry.pop(basename, None)
