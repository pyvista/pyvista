import os
from zipfile import ZipFile

import pytest

import pyvista
from pyvista import examples


def test_invalid_dir():
    old_path = pyvista.EXAMPLES_PATH
    try:
        pyvista.EXAMPLES_PATH = None
        with pytest.raises(FileNotFoundError):
            examples.downloads._check_examples_path()
    finally:
        pyvista.EXAMPLES_PATH = old_path


def test_decompress(tmpdir):
    old_path = pyvista.EXAMPLES_PATH
    try:
        pyvista.EXAMPLES_PATH = str(tmpdir.mkdir("tmpdir"))
        assert os.path.isdir(pyvista.EXAMPLES_PATH)
        tmp_file = os.path.join(pyvista.EXAMPLES_PATH, 'tmp.txt')
        with open(tmp_file, 'w') as fid:
            fid.write('test')

        tmp_file2 = os.path.join(pyvista.EXAMPLES_PATH, 'tmp2.txt')
        with open(tmp_file2, 'w') as fid:
            fid.write('test')

        zipfilename = os.path.join(pyvista.EXAMPLES_PATH, 'sample.zip')
        with ZipFile(zipfilename, 'w') as zip_obj:
            zip_obj.write(tmp_file, os.path.basename(tmp_file))
            zip_obj.write(tmp_file2, os.path.basename(tmp_file2))

        os.remove(tmp_file)
        os.remove(tmp_file2)

        examples.downloads._decompress(zipfilename, pyvista.EXAMPLES_PATH)
        assert os.path.isfile(tmp_file)
        assert os.path.isfile(tmp_file2)
    finally:
        pyvista.EXAMPLES_PATH = old_path


def test_delete_downloads(tmpdir):
    # change the path so we don't delete the examples cache
    old_path = pyvista.EXAMPLES_PATH
    try:
        pyvista.EXAMPLES_PATH = str(tmpdir.mkdir("tmpdir"))
        assert os.path.isdir(pyvista.EXAMPLES_PATH)
        tmp_file = os.path.join(pyvista.EXAMPLES_PATH, 'tmp.txt')
        with open(tmp_file, 'w') as fid:
            fid.write('test')
        examples.delete_downloads()
        assert os.path.isdir(pyvista.EXAMPLES_PATH)
        assert not os.path.isfile(tmp_file)
    finally:
        pyvista.EXAMPLES_PATH = old_path
