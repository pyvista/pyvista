import os

import pytest

from pyvista import examples


def test_delete_downloads(tmpdir):
    # change the path so we don't delete the examples cache
    old_path = examples.downloads.PATH
    try:
        examples.downloads.PATH = str(tmpdir.mkdir("tmpdir"))
        assert os.path.isdir(examples.downloads.PATH)
        tmp_file = os.path.join(examples.downloads.PATH, 'tmp.txt')
        with open(tmp_file, 'w') as fid:
            fid.write('test')
        examples.delete_downloads()
        assert os.path.isdir(examples.downloads.PATH)
        assert not os.path.isfile(tmp_file)
    finally:
        examples.downloads.PATH = old_path


def test_delete_downloads_does_not_exist(tmpdir):
    # change the path so we don't delete the examples cache
    old_path = examples.downloads.PATH
    new_path = str(tmpdir.join('doesnotexist'))

    try:
        # delete_downloads for a missing directory should not fail.
        examples.downloads.PATH = new_path
        assert not os.path.isdir(examples.downloads.PATH)
        examples.delete_downloads()
    finally:
        examples.downloads.PATH = old_path


def test_file_from_files(tmpdir):
    path = str(tmpdir)
    fnames = [
        os.path.join(path, 'tmp2.txt'),
        os.path.join(path, 'tmp1.txt'),
        os.path.join(path, 'tmp0.txt'),
    ]

    with pytest.raises(FileNotFoundError):
        fname = examples.downloads.file_from_files('potato.txt', fnames)

    fname = examples.downloads.file_from_files('tmp1.txt', fnames)
    assert fname == fnames[1]
