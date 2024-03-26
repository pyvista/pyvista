from pathlib import Path
import shutil

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
    _SingleFile,
    _SingleFileDownloadable,
    _SingleFileDownloadableLoadable,
    _SingleFileLoadable,
)


@pytest.fixture()
def examples_local_repository_tmp_dir(tmp_path):
    """Create a local repository with a bunch of datasets available for download."""

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
        shutil.copyfile(str(Path(examples.dir_path) / base), str(Path(repository_path) / base))
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
    cached_paths = [Path(downloads.FETCHER.path) / base for base in downloadable_basenames]
    [path.unlink() for path in cached_paths if path.is_file()]

    yield repository_path

    # teardown
    downloads.FETCHER.base_url = "https://github.com/pyvista/vtk-data/raw/master/Data/"
    downloads._FILE_CACHE = False
    [downloads.FETCHER.registry.pop(base, None) for base in downloadable_basenames]

    # make sure any "downloaded" files (moved from repo -> cache) are cleared afterward
    [path.unlink() for path in cached_paths if path.is_file()]


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
    assert Path(path).name == basename
    assert not Path(path).is_file()

    if expected_path_is_absolute:
        assert Path(path).is_absolute()
    else:
        assert not Path(path).is_absolute()

    # test download
    if isinstance(file_loader, (_SingleFileDownloadable, _SingleFileDownloadableLoadable)):
        assert isinstance(file_loader, _Downloadable)
        path_download = file_loader.download()
        assert Path(path_download).is_file()
        assert Path(path_download).is_absolute()
        assert file_loader.path == path_download
        assert 'https://github.com/pyvista/vtk-data/raw/master/Data/' in file_loader.download_url
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

    assert Path(file_loader.path).is_file()


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
    assert all(Path(file).is_absolute() for file in path)
    assert len(path) == 3

    path_loadable = multi_file_loader.path_loadable
    assert isinstance(path_loadable, tuple)
    assert all(Path(file).is_absolute() for file in path_loadable)
    assert len(path_loadable) == 2
    assert basename_not_loaded not in path_loadable

    # test download
    path_download = multi_file_loader.download()
    assert path_download == path
    assert [Path(file).is_file() for file in path_download]
    assert [
        'https://github.com/pyvista/vtk-data/raw/master/Data/' in url
        for url in multi_file_loader.download_url
    ]

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
    assert Path(example.path).is_file()
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
    assert all(Path(file).is_file() for file in example.path)
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
    example = downloads._dataset_bolt_nut
    assert len(example.path) == 2
    assert example.num_files == 2
    assert Path(example.path[0]).is_file()
    assert Path(example.path[1]).is_file()
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
    assert Path(example.path).is_dir()
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
    assert Path(example.path).is_dir()
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
    assert Path(example.path[0]).is_file()
    assert Path(example.path[1]).is_file()
    assert Path(example.path[2]).is_file()
    assert Path(example.path[3]).is_dir()
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
