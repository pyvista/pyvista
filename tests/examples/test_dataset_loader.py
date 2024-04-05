from pathlib import Path
import shutil

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
import pyvista.examples
from pyvista.examples import downloads
from pyvista.examples._dataset_loader import (
    _Downloadable,
    _load_and_merge,
    _load_as_multiblock,
    _Loadable,
    _MultiFileDownloadableLoadable,
    _SingleFileDownloadable,
    _SingleFileDownloadableLoadable,
    _SingleFileLoadable,
)


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
    example = downloads._dataset_cow
    example.download()
    assert Path(example.filename).is_file()
    assert example.total_size == '59.0 KiB'
    assert example.extension == '.vtp'
    assert type(example.reader) is pv.XMLPolyDataReader

    # test multiple files, but only one is loaded
    example = downloads._dataset_head
    example.download()
    assert all(Path(file).is_file() for file in example.filename)
    assert example.total_size == '122.3 KiB'
    assert example.extension == ('.mhd', '.raw')
    assert pv.get_ext(example.filename[0]) == '.mhd'
    assert pv.get_ext(example.filename[1]) == '.raw'
    assert type(example.reader) is pv.MetaImageReader

    # test directory (cubemap)
    example = downloads._dataset_cubemap_park
    example.download()
    assert Path(example.filename).is_dir()
    assert example.total_size == '591.9 KiB'
    assert example.extension == '.jpg'
    assert example.reader is None

    # test directory (dicom stack)
    example = downloads._dataset_dicom_stack
    example.download()
    assert Path(example.filename).is_dir()
    assert example.total_size == '1.5 MiB'
    assert example.extension == '.dcm'
    assert type(example.reader) is pv.DICOMReader
