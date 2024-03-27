# ruff: noqa: PTH102,PTH103,PTH107,PTH112,PTH113,PTH117,PTH118,PTH119,PTH122,PTH123,PTH202
import os
import shutil
from typing import Tuple

import numpy as np
import pytest

import pyvista as pv
from pyvista.examples import downloads, examples
from pyvista.examples._dataset_loader import (
    _Downloadable,
    _format_file_size,
    _load_and_merge,
    _load_as_cubemap,
    _load_as_multiblock,
    _Loadable,
    _MultiFileDownloadableLoadable,
    _MultiFileLoadable,
    _SingleFileDownloadable,
    _SingleFileDownloadableLoadable,
    _SingleFileLoadable,
)


@pytest.fixture()
def examples_local_repository_tmp_dir(tmp_path):
    """Create a local repository with a bunch of datasets available for download."""

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
        shutil.copyfile(os.path.join(examples.dir_path, base), os.path.join(repository_path, base))
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
        assert 'https://github.com/pyvista/vtk-data/raw/master/Data/' in file_loader.source_url_raw
        assert (
            'https://github.com/pyvista/vtk-data/blob/master/Data/' in file_loader.source_url_blob
        )
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
    assert [
        'https://github.com/pyvista/vtk-data/raw/master/Data/' in url
        for url in multi_file_loader.source_url_raw
    ]
    assert [
        'https://github.com/pyvista/vtk-data/blob/master/Data/' in url
        for url in multi_file_loader.source_url_blob
    ]

    # test load
    assert multi_file_loader.dataset is None
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
def dataset_loader_one_file():
    loader = _SingleFileDownloadableLoadable('cow.vtp')
    loader.download()
    loader.load()
    return loader


def test_dataset_loader_one_file(dataset_loader_one_file):
    loader = dataset_loader_one_file
    assert isinstance(loader.path, str)
    assert loader.num_files == 1
    assert loader._total_size_bytes == 60449
    assert loader.total_size == '60.4 KB'
    assert loader.unique_extension == '.vtp'
    assert isinstance(loader._reader, pv.XMLPolyDataReader)
    assert loader.unique_reader_type is pv.XMLPolyDataReader
    assert type(loader.dataset) is pv.PolyData
    assert loader.unique_dataset_type is pv.PolyData
    assert loader.source_url_raw == 'https://github.com/pyvista/vtk-data/raw/master/Data/cow.vtp'
    assert loader.source_url_blob == 'https://github.com/pyvista/vtk-data/blob/master/Data/cow.vtp'


@pytest.fixture()
def dataset_loader_two_files_one_loadable():
    def _files_func():
        loadable = _SingleFileDownloadableLoadable('HeadMRVolume.mhd')
        not_loadable = _SingleFileDownloadable('HeadMRVolume.raw')
        return loadable, not_loadable

    loader = _MultiFileDownloadableLoadable(_files_func)
    loader.download()
    loader.load()
    return loader


def test_dataset_loader_two_files_one_loadable(dataset_loader_two_files_one_loadable):
    loader = dataset_loader_two_files_one_loadable
    assert len(loader.path) == 2
    assert loader.num_files == 2
    assert loader._total_size_bytes == 125223
    assert loader.total_size == '125.2 KB'
    assert loader.unique_extension == ('.mhd', '.raw')
    assert pv.get_ext(loader.path[0]) == '.mhd'
    assert pv.get_ext(loader.path[1]) == '.raw'
    assert len(loader._reader) == 2
    assert isinstance(loader._reader[0], pv.MetaImageReader)
    assert loader._reader[1] is None
    assert loader.unique_reader_type is pv.MetaImageReader
    assert isinstance(loader.dataset, pv.ImageData)
    assert loader.unique_dataset_type is pv.ImageData
    assert loader.source_url_raw == (
        'https://github.com/pyvista/vtk-data/raw/master/Data/HeadMRVolume.mhd',
        'https://github.com/pyvista/vtk-data/raw/master/Data/HeadMRVolume.raw',
    )
    assert loader.source_url_blob == (
        'https://github.com/pyvista/vtk-data/blob/master/Data/HeadMRVolume.mhd',
        'https://github.com/pyvista/vtk-data/blob/master/Data/HeadMRVolume.raw',
    )


@pytest.fixture()
def dataset_loader_two_files_both_loadable():
    def _files_func():
        loadable1 = _SingleFileDownloadableLoadable('bolt.slc')
        loadable2 = _SingleFileDownloadableLoadable('nut.slc')
        return loadable1, loadable2

    loader = _MultiFileDownloadableLoadable(_files_func)
    loader.download()
    loader.load()
    return loader


def test_dataset_loader_two_files_both_loadable(dataset_loader_two_files_both_loadable):
    loader = dataset_loader_two_files_both_loadable
    assert len(loader.path) == 2
    assert loader.num_files == 2
    assert loader._total_size_bytes == 132818
    assert loader.total_size == '132.8 KB'
    assert loader.unique_extension == '.slc'
    assert pv.get_ext(loader.path[0]) == '.slc'
    assert pv.get_ext(loader.path[1]) == '.slc'
    assert len(loader._reader) == 2
    assert isinstance(loader._reader[0], pv.SLCReader)
    assert isinstance(loader._reader[1], pv.SLCReader)
    assert loader.unique_reader_type is pv.SLCReader
    assert isinstance(loader.dataset, Tuple)
    dataset1, dataset2 = loader.dataset
    assert isinstance(dataset1, pv.ImageData)
    assert isinstance(dataset2, pv.ImageData)
    assert loader.unique_dataset_type is pv.ImageData
    assert loader.source_url_raw == (
        'https://github.com/pyvista/vtk-data/raw/master/Data/bolt.slc',
        'https://github.com/pyvista/vtk-data/raw/master/Data/nut.slc',
    )
    assert loader.source_url_blob == (
        'https://github.com/pyvista/vtk-data/blob/master/Data/bolt.slc',
        'https://github.com/pyvista/vtk-data/blob/master/Data/nut.slc',
    )


@pytest.fixture()
def dataset_loader_cubemap():
    loader = _SingleFileDownloadableLoadable(
        'cubemap_park/cubemap_park.zip',
        read_func=_load_as_cubemap,
    )
    loader.download()
    loader.load()
    return loader


def test_dataset_loader_cubemap(dataset_loader_cubemap):
    loader = dataset_loader_cubemap
    assert os.path.isdir(loader.path)
    assert loader.num_files == 6
    assert loader._total_size_bytes == 606113
    assert loader.total_size == '606.1 KB'
    assert loader.unique_extension == '.jpg'
    assert type(loader.dataset) is pv.Texture
    assert loader.unique_dataset_type is pv.Texture
    assert (
        loader.source_url_raw
        == 'https://github.com/pyvista/vtk-data/raw/master/Data/cubemap_park/cubemap_park.zip'
    )
    assert (
        loader.source_url_blob
        == 'https://github.com/pyvista/vtk-data/blob/master/Data/cubemap_park/cubemap_park.zip'
    )


@pytest.fixture()
def dataset_loader_dicom():
    loader = _SingleFileDownloadableLoadable('DICOM_Stack/data.zip', target_file='data')
    loader.download()
    loader.load()
    return loader


def test_dataset_loader_dicom(dataset_loader_dicom):
    loader = dataset_loader_dicom
    assert os.path.isdir(loader.path)
    assert loader.num_files == 3
    assert loader._total_size_bytes == 1583688
    assert loader.total_size == '1.6 MB'
    assert loader.unique_extension == '.dcm'
    assert isinstance(loader._reader, pv.DICOMReader)
    assert loader.unique_reader_type is pv.DICOMReader
    assert isinstance(loader.dataset, pv.ImageData)
    assert loader.unique_dataset_type is pv.ImageData
    assert (
        loader.source_url_raw
        == 'https://github.com/pyvista/vtk-data/raw/master/Data/DICOM_Stack/data.zip'
    )
    assert (
        loader.source_url_blob
        == 'https://github.com/pyvista/vtk-data/blob/master/Data/DICOM_Stack/data.zip'
    )


def test_dataset_loader_from_nested_files_and_directory(
    dataset_loader_one_file, dataset_loader_two_files_one_loadable, dataset_loader_dicom
):
    # test complex multiple file case with separate ext and reader, which are loaded as a tuple
    # piece together new dataset from existing ones
    def files_func():
        return dataset_loader_one_file, dataset_loader_two_files_one_loadable, dataset_loader_dicom

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


def test_reader_returns_none(dataset_loader_one_file):
    dataset = downloads._dataset_cloud_dark_matter
    match = '`pyvista.get_reader` does not support a file with the .npy extension'
    with pytest.raises(ValueError, match=match):
        pv.get_reader(dataset.path)
    assert dataset.unique_extension == '.npy'
    assert dataset._reader is None


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
